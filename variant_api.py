"""Small construction hooks for a single CellFoundry variant.

Variants use native FLAMEGPU descriptions and own configure_layers(ctx).
This module deliberately has no schedule builder or biological model knowledge.
Importing it (or a variant's parameter defaults) does not require CUDA.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from numbers import Integral
import sys
from types import MappingProxyType


def load_variant(root, name):
    """Load a package or legacy flat variant, including relative imports."""
    if not name:
        return None
    if not name.isidentifier():
        raise ValueError(f"Invalid variant name: {name!r}")
    directory = Path(root) / "variants"
    package = directory / name / "__init__.py"
    flat = directory / f"{name}.py"
    path = package if package.is_file() else flat
    if not path.is_file():
        available = sorted({p.stem for p in directory.glob("*.py") if p.stem != "__init__"}
                           | {p.parent.name for p in directory.glob("*/__init__.py")})
        raise FileNotFoundError(f"Variant {name!r} not found in {directory}; available: {available}")
    module_name = f"variants.{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if not callable(getattr(module, "configure_layers", None)):
            raise TypeError(f"Variant {name!r} must define configure_layers(ctx)")
        if hasattr(module, "configure_globals"):
            raise TypeError(f"Variant {name!r}: replace configure_globals with PARAM_DEFAULTS and declare_model(ctx)")
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def register_parameter_defaults(namespace, variant):
    """Declare new variant parameters before PARAMS and CLI/optimizer overrides.

    PARAM_DEFAULTS introduces names; PARAMS overrides existing core/default names.
    Copies prevent indexed overrides from modifying a module's baseline lists.
    """
    defaults = getattr(variant, "PARAM_DEFAULTS", {})
    for name, value in defaults.items():
        if not name.isupper() or name.startswith("_"):
            raise ValueError(f"Invalid variant parameter name: {name!r}")
        if name in namespace:
            raise ValueError(f"Variant parameter {name!r} already exists; override it in PARAMS instead")
        namespace[name] = deepcopy(value)
    return tuple(defaults)


def configuration_snapshot(namespace):
    """Expose effective values, without handing a variant model.py globals()."""
    return MappingProxyType({name: deepcopy(value) for name, value in namespace.items()
                             if name.isupper() and not name.startswith("_")})


def call_hook(variant, name, context):
    hook = getattr(variant, name, None)
    if hook is not None:
        hook(context)


@dataclass(frozen=True)
class PopulationIDs:
    """A half-open custom-id range, not a FLAMEGPU internal-ID range.

    Dense macro-array indices are ``id - begin``, never the custom id itself.
    Reservations are disjoint during initialization. Core birth counters remain
    type-specific, so cross-type references must also identify the agent type.
    """

    agent: str
    begin: int
    count: int
    capacity: int
    state: str = "default"

    @property
    def end(self):
        return self.begin + self.capacity

    @property
    def counter(self):
        return f"VARIANT_{self.agent}_LAST_ID"

    @property
    def exhaustion_flag(self):
        return f"VARIANT_{self.agent}_ID_EXHAUSTED"

    @property
    def begin_property(self):
        return f"VARIANT_{self.agent}_ID_BEGIN"

    @property
    def end_property(self):
        return f"VARIANT_{self.agent}_ID_END"


def core_initial_population_counts(config):
    """Core's fixed initialization order. Counts are resolved before declarations.

    This describes initial populations only: it must not be used as a bound on
    CELL/FNODE/LUMEN births. LUMEN has no initial population in the core.
    """
    cells = config["N_CELLS"] if config["INCLUDE_CELLS"] else 0
    return {
        "BCORNER": 8,
        "FNODE": config["N_NODES"] if config["INCLUDE_FIBRE_NETWORK"] else 0,
        "CELL": cells,
        "FOCAD": cells * config["INIT_N_FOCAD_PER_CELL"] if config["INCLUDE_FOCAL_ADHESIONS"] else 0,
        "ECM": config["ECM_POPULATION_SIZE"],
        "VASC": config["N_VASC_NODES"] if config["INCLUDE_VASCULARIZATION"] else 0,
    }


def _nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass
class VariantContext:
    """Native construction objects, initialization hooks and per-run outputs.

    config is a detached snapshot; its top-level mapping is read-only. Treat nested
    values as read-only too. Host callbacks receive FLAMEGPU's HostAPI directly.
    """

    model: object
    config: object
    root: Path
    name: str | None = None
    agents: dict = field(default_factory=dict)
    messages: dict = field(default_factory=dict)
    functions: dict = field(default_factory=dict)
    agent_initializers: dict = field(default_factory=dict)
    initial_ids: dict = field(default_factory=dict)
    cell_vtk_scalars: list = field(default_factory=list)
    cell_vtk_vectors: list = field(default_factory=list)
    add_multiscale_diffusion_layers: object = None
    _callbacks: list = field(default_factory=list, repr=False)
    _results: dict = field(default_factory=dict, repr=False)
    _populations: dict = field(default_factory=dict, repr=False)
    _core_last_id: int | None = field(default=None, repr=False)
    _reserved_last_id: int = field(default=0, repr=False)
    _populations_sealed: bool = field(default=False, repr=False)

    @property
    def env(self):
        return self.model.Environment()

    def add_agent_initializer(self, agent_name, callback):
        """Extend initialization for any declared agent; callbacks take (agent, rng).

        Hooks may initialize additional variables, but must not change ``id``.
        GPU births do not call Python: their kernels must initialize extensions.
        """
        if agent_name not in self.agents:
            raise ValueError(f"Initializer targets undeclared agent {agent_name!r}")
        if not callable(callback):
            raise TypeError("Agent initializer must be callable")
        self.agent_initializers.setdefault(agent_name, []).append(callback)

    def initialize_agent(self, agent_name, instance, rng):
        callbacks = self.agent_initializers.get(agent_name, ())
        if not callbacks:
            return
        has_id = self.agents[agent_name].hasVariable("id")
        identity = instance.getVariableInt("id") if has_id else None
        for initialize in callbacks:
            initialize(instance, rng)
            if has_id and instance.getVariableInt("id") != identity:
                raise ValueError(f"{agent_name} initializer must not change its allocated id")

    def configure_initial_ids(self, counts):
        """Called once by the core, before declare_model(), with actual counts."""
        if self._core_last_id is not None:
            raise RuntimeError("Core initial ID layout is already configured")
        cursor = 0
        for agent, count in counts.items():
            count = _nonnegative_int(f"{agent} count", count)
            self.initial_ids[agent] = PopulationIDs(agent, cursor + 1, count, count)
            cursor += count
        if cursor >= 2**31 - 1:
            raise ValueError("Initial population exceeds the signed-int custom-ID space")
        self._core_last_id = self._reserved_last_id = cursor

    def add_population(self, agent_name, count, *, capacity=None, state="default"):
        """Plan a new agent population in declare_model(), before bucket setup.

        Declare the agent and its Int ``id`` variable first. The core creates the
        population after all core agents, runs its registered initializers, seeds
        a per-type macro counter, and advances CURRENT_ID across the *reservation*.
        Capacity covers all IDs ever issued, including dead agents. It is not a
        live-population limit. Empty populations still require positive capacity.
        """
        if self._core_last_id is None or self._populations_sealed:
            raise RuntimeError("Population reservations must be made during declare_model()")
        if agent_name not in self.agents:
            raise ValueError(f"Population targets undeclared agent {agent_name!r}")
        if agent_name in self.initial_ids or agent_name == "LUMEN":
            raise ValueError(f"Population {agent_name!r} is already owned by the core or variant")
        if not agent_name.isidentifier():
            raise ValueError("Managed population names must be identifiers")
        agent = self.agents[agent_name]
        if not agent.hasVariable("id") or agent.getVariableLength("id") != 1:
            raise ValueError(f"Managed population {agent_name!r} requires a scalar Int id variable")
        if not agent.hasState(state):
            raise ValueError(f"Managed population {agent_name!r} has no state {state!r}")
        count = _nonnegative_int("count", count)
        capacity = count if capacity is None else _nonnegative_int("capacity", capacity)
        if capacity < max(1, count):
            raise ValueError("capacity must be positive and at least count")
        if self._reserved_last_id + capacity >= 2**31 - 1:
            raise ValueError("Population reservation exceeds the signed-int custom-ID space")
        ids = PopulationIDs(agent_name, self._reserved_last_id + 1, count, capacity, state)
        self.env.newPropertyInt(ids.begin_property, ids.begin)
        self.env.newPropertyInt(ids.end_property, ids.end)
        self.env.newMacroPropertyInt(ids.counter, 1)
        self.env.newMacroPropertyInt(ids.exhaustion_flag, 1)
        self._populations[agent_name] = self.initial_ids[agent_name] = ids
        self._reserved_last_id = ids.end - 1
        return ids

    def seal_populations(self):
        """Freeze bounds and install the capacity check before core step outputs."""
        if self._populations_sealed:
            return
        if self._populations:
            self.add_step_function(self.check_population_capacity)
        self._populations_sealed = True

    def check_population_capacity(self, flamegpu):
        """Reject an under-capacity run instead of silently suppressing births.

        The device helper sets sticky flags atomically. Reading them on the host
        after GPU layers is safe, including in builds without FLAMEGPU seatbelts.
        Filling the final slot is valid; only a further allocation attempt fails.
        """
        exhausted = [ids for ids in self._populations.values()
                     if int(flamegpu.environment.getMacroPropertyInt(ids.exhaustion_flag))]
        if exhausted:
            details = "; ".join(f"{ids.agent}: capacity={ids.capacity}, "
                                f"reserved IDs [{ids.begin}, {ids.end})" for ids in exhausted)
            step = flamegpu.getStepCounter() + 1
            raise RuntimeError(
                f"Variant agent ID capacity exhausted at step {step}: {details}. "
                "At least one requested birth could not be created. "
                "Increase capacity in add_population(), keep bucket/macro-array bounds "
                "consistent, and rerun. Capacity covers all IDs issued, including dead "
                "agents. This run is invalid; earlier output files are partial results.")

    def initialize_populations(self, flamegpu, rng):
        """Called at the end of core population initialization, before LUMEN seeding.

        No mutable allocation cursor is stored in this context at run time: every
        simulation/ensemble run gets the same layout and its own macro counters.
        """
        actual = flamegpu.environment.getPropertyUInt("CURRENT_ID")
        if actual != self._core_last_id:
            raise ValueError(f"Core initial ID layout mismatch: expected {self._core_last_id}, got {actual}")
        for name, ids in self._populations.items():
            population = flamegpu.agent(name, ids.state)
            for offset in range(ids.count):
                instance = population.newAgent()
                instance.setVariableInt("id", ids.begin + offset)
                self.initialize_agent(name, instance, rng)
            counter = flamegpu.environment.getMacroPropertyInt(ids.counter)
            counter[0] = ids.begin + ids.count - 1
            flamegpu.environment.getMacroPropertyInt(ids.exhaustion_flag)[0] = 0
        flamegpu.environment.setPropertyUInt("CURRENT_ID", self._reserved_last_id)

    def results_for(self, run_index=0):
        return self._results.setdefault(int(run_index), {})

    def runtime_results(self, flamegpu):
        # FLAMEGPU returns UINT_MAX for a non-ensemble simulation.
        run_index = flamegpu.getEnsembleRunIndex() if self.config.get("ENSEMBLE") else 0
        return self.results_for(run_index)

    def _add_host_function(self, kind, callback):
        import pyflamegpu

        class Callback(pyflamegpu.HostFunction):
            def run(self, FLAMEGPU):
                callback(FLAMEGPU)

        wrapped = Callback()
        getattr(self.model, f"add{kind}Function")(wrapped)
        self._callbacks.append(wrapped)  # SWIG callbacks must outlive simulation.

    def add_init_function(self, callback):
        self._add_host_function("Init", callback)

    def add_step_function(self, callback):
        self._add_host_function("Step", callback)

    def add_exit_function(self, callback):
        self._add_host_function("Exit", callback)

    def merge_results(self, payload, run_index=0):
        extra = self.results_for(run_index)
        collisions = payload.keys() & extra.keys()
        if collisions:
            raise ValueError(f"Variant output overwrites core result keys: {sorted(collisions)}")
        return {**payload, **extra}

    def print_results(self, run_index=0):
        for name, value in self.results_for(run_index).items():
            if hasattr(value, "__len__") and len(value) == 0:
                continue
            print("============================")
            print(name)
            print(value)
            print()
