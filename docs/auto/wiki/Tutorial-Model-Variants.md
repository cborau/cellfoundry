# Tutorial: Model variants

A variant is a Python package under `variants/<name>/`. It owns model-specific parameters, additions to agents and messages, custom kernels, initialization, analysis, and its **complete execution schedule**. `model.py` supplies the generic CellFoundry model and invokes a small set of construction hooks from `variant_api.py`.

There are three starting points:

| Starting point | Use it for |
| --- | --- |
| [`variant_template`](../../../variants/variant_template/__init__.py) | Copyable scaffold containing every recognized hook, explanations, commented examples and the full generic schedule. |
| [`simple_signal`](../../../variants/simple_signal/__init__.py) | First exercise: two CELL variables and two functions, with an answer calculable by hand. |
| [`cell_markers`](Tutorial-Variant-Cell-Markers.md) | Complete worked example: extend CELL and ECM, create a new agent type from CELL, communicate, expire agents, collect results and retain the full generic schedule. |

Run an existing variant with:

```sh
python model.py --variant organoid
python model.py --variant radial_glia --overrides configs/my_overrides.json
python model.py --variant radial_glia --result-dir "results/radial_glia_test"
```

Use a Python environment with CellFoundry's dependencies and a compatible FLAMEGPU/CUDA installation (`flamegpu_py310` in the supplied installation). The optimizer selects variants through the same `--variant`, `--overrides`, and `--result-dir` interfaces.

### Command-line arguments

Run `python model.py --help` (or `-h`) for the complete list. Help exits before importing FLAMEGPU, constructing the model, checking kernels or creating output directories; it does not require CUDA.

| Argument | Purpose |
| --- | --- |
| `-h`, `--help` | Print usage and exit. |
| `--variant NAME` | Select a package under `variants/<NAME>/`. Omit it to use the generic core model. |
| `--overrides FILE.json` | Read a JSON object containing parameter overrides. Precedence is JSON > variant `PARAMS` > core defaults. |
| `--result-dir DIR` | Set the results directory. Quote paths containing spaces. Relative paths are relative to the working directory from which Python was launched. The default is `result_files` beside `model.py`. This option takes precedence over a parameter override of `RES_PATH`. |

These are the model's command-line options. Parameters such as `STEPS`, `N_CELLS` and `SAVE_EVERY_N_STEPS` belong in `PARAMS` or an overrides JSON object, for example `{"STEPS": 10, "N_CELLS": 4}`; they are not separate command-line flags. Use a different result directory for each run whose output you want to retain. A variant may provide its own output settings through `PARAMS`.

## Ownership

- Core agents such as CELL belong to the central FLAMEGPU `ModelDescription`. Their extensions can be declared in a variant using the same native description.
- An entirely variant-specific agent can be created with `ctx.model.newAgent(...)`. Declare its variables/states/messages, reserve its initial population with `ctx.add_population(...)`, bind its functions, schedule it, and provide any specialized output inside the variant.
- Generic mechanics, diffusion, assay initialization, and output infrastructure remain in the core.

## Structural parameters remain core-controlled

**Define initial `BOUNDARY_COORDS` in `model.py`. A differing value in variant `PARAMS` or JSON is rejected.** Bounds use the order `[+X, -X, +Y, -Y, +Z, -Z]`, in micrometres. An identical repeated value is allowed, but omitting it from the variant is clearer. Runtime boundary motion prescribed by an assay remains supported; this restriction concerns the initial domain configuration.

The ECM grid is constructed from the core `N` and `BOUNDARY_COORDS` **before** variant/JSON overrides are applied. Parameter recomputation deliberately preserves `ECM_AGENTS_PER_DIR` and `ECM_POPULATION_SIZE`, because those dimensions also occur in hard-coded RTC constants, message arrays and macro arrays.

| Setting in variant `PARAMS` or overrides JSON | Effect / constraint |
| --- | --- |
| `BOUNDARY_COORDS` | Core-controlled initial geometry. Differing variant/JSON overrides, including indexed changes, stop startup before downstream assay setup. The optimizer rejects it as a search parameter. |
| `N` | Does not rebuild the grid. An override differing from the core value is a startup error. Set `N` in the core when changing resolution. |
| `ECM_AGENTS_PER_DIR`, `ECM_POPULATION_SIZE`, `N_SPECIES`, `N_CELL_TYPES`, `MAX_CONNECTIVITY`, `N_ANCHOR_POINTS`, `MAX_VASC_CONNECTIVITY` | Structural settings: configure these consistently in the core and kernels. Overrides that differ from the core values are rejected; they cannot resize dependent structures or synchronize RTC constants. The optimizer also rejects these names as search parameters. |

For example, with core `N=11` and core bounds `±50` on all axes, the fixed grid is `11 × 11 × 11` (1,331 nodes). Putting `BOUNDARY_COORDS=[500,-500,500,-500,25,-25]` or `N=6` only in the variant stops startup; it cannot construct the intended `101 × 101 × 6` grid. Recomputing lengths alone is insufficient: some assay initialization, including the oscillatory message-domain envelope, occurs before overrides.

To obtain that uniform `10 µm` grid deliberately:

1. Set both `BOUNDARY_COORDS=[500,-500,500,-500,25,-25]` and `N=6` in `model.py`. Keep the variant's values consistent with them. The core construction then derives `ECM_AGENTS_PER_DIR=[101,101,6]` and `ECM_POPULATION_SIZE=61206`.
2. Synchronize constants in both the core and active variant kernels, for example from the repository root: `python check_hard_coded_values.py --model-file model.py --scan-root . --scan-root variants/radial_glia --no-recursive`. The checker reads core literals and prompts before updating mismatching constants; inspect its proposed values and changes. Add `--fail-on-mismatch` for a read-only check (nonzero exit on mismatch), or `--fix` to explicitly repair without a prompt. These modes are mutually exclusive. All JSON override runs, including an empty `{}` and optimizer trials, use read-only checking and stop on mismatches or checker errors. Optimization also checks constants before launching any trials; it never rewrites kernels.
3. Recheck spatial-message bounds/radii, fixed arrays, diffusion stability and initialization geometry, then validate with a short run before a long experiment. A different node count changes resolution and memory use. Do not vary structural settings independently in optimizer trials.

At startup, the `[GEOMETRY]` lines show the bounds, lengths, fixed grid counts and nominal spacings before the checker can prompt to update kernels. `[KERNEL CHECK]` then labels the checker's **core reference values**. Its `The domain is cubical` message describes the core reference geometry used to derive constants. Saved `MODEL_CONFIG.BOUNDARY_COORDS` and the first row of `BPOS_OVER_TIME` record the actual starting bounds after the core's grid-alignment adjustment.

A new core geometry still requires model-specific consistency checks. Check grid spacing, message limits, pinned search radii, oscillatory amplitude/envelope and external fibre/vascular coordinates; these external files are not rescaled automatically. The current oscillatory envelope uses a core expression containing `0.25`, so changing the maximum strain or oscillation amplitude also requires reviewing that expression. Initial-domain overrides remain disabled until the complete geometry/assay setup is resolved after configuration and tested together. Successful initialization is not evidence that a geometry is biologically calibrated or numerically suitable.

Do not use `PARAM_DEFAULTS` to redeclare core parameters such as `N_SPECIES`. It introduces only genuinely new variant parameters. The registration hooks do not provide automatic resizing or rewriting of kernels.

## Variant interface and construction order

**Only `configure_layers(ctx)` is mandatory.** Loading a variant without it fails. Every other entry below may be omitted entirely; a no-op implementation is useful in a template but is not a requirement.

These names are the interface recognized by CellFoundry. They are not arbitrary function names discovered by FLAMEGPU. The loader reads the dictionaries, and `model.py` explicitly calls the named hooks at the appropriate construction stages.

| Export | Required? | Purpose and time of execution |
| --- | --- | --- |
| `PARAM_DEFAULTS` | No | Dictionary introducing new parameter names before overrides. It does not declare GPU environment properties. |
| `PARAMS` | No | Dictionary assigning variant defaults to existing core/new parameters. User JSON overrides these values. |
| `FILES` | No | Dictionary selecting replacement files for existing core RTC functions, before core function registration. |
| `validate_config(config)` | No; recommended | Called once with effective configuration to reject incompatible features/values. Raise a clear error; do not mutate configuration. |
| `declare_model(ctx)` | No; needed for schema additions | Called once after core descriptions exist. Add variables, states, agents, messages and GPU properties; reserve new populations. Descriptions are being built, not live agents. |
| `register_functions(ctx)` | No; needed for extra kernels | Called once after declarations. Register RTC functions, message bindings, states, death permission and birth targets. This does not execute or schedule them. |
| `register_runtime(ctx)` | No; needed for custom host work | Called once during construction to register per-agent initializers, Python callbacks and output fields. The registered callbacks execute later. |
| `configure_layers(ctx)` | **Yes** | Called once to build the **entire** GPU execution order. The resulting layers are executed each simulation step. No automatic merge with a core schedule occurs. |

For example, a parameter-only variant can omit the optional function hooks, but must still supply its complete schedule. A variant adding a constant-default variable needs `declare_model()` but no per-agent initializer. A variant adding an RTC kernel also needs its registration and a layer that executes it. Optional hooks become necessary when their responsibilities are needed, not because every variant must contain boilerplate.

Core functions are registered alongside core descriptions. Variant declarations extend those shared descriptions before the simulation is constructed and RTC executes. Select existing core function replacements through `FILES`; do not register a second function with an existing name. Additional functions use `register_functions()`. Either choice still requires the relevant function in the selected schedule.

The lifecycle is:

```text
Construction, once:
  import package -> read defaults/overrides -> validate_config
  -> build core descriptions/functions using FILES
  -> declare_model -> seal population reservations -> register_functions
  -> register_runtime -> configure_layers -> construct simulation

At simulation initialization:
  core population loops call registered per-agent initializers
  -> managed variant populations and their initializers
  -> core macro initialization -> registered variant init callbacks

At each step:
  execute the selected variant's GPU layers in order
  -> capacity/error checks and registered end-of-step host callbacks

At normal completion:
  registered exit callbacks -> final result collection/pickle writing
```

`initialize_cell`, `initialize_probe`, `Metrics.step` and similar names are ordinary Python functions/methods. Their names have no special meaning to the loader. They run only because a recognized hook registers them, for example `ctx.add_agent_initializer("CELL", initialize_cell)`. You may write other helper functions such as `declare_my_agents(ctx)`, but must call them from a recognized hook yourself.

### Why radial glia variant example has a runtime.py

`runtime.py` is an **optional organization choice**, not another framework hook or a second model. Radial glia has enough CPU initialization and analysis code to benefit from a separate file: initial polarity/anchors, VTK field definitions, rosette metrics and diagnostic output. Its `__init__.py` keeps declarations, registration and the full schedule together.

The connection is an ordinary Python import and explicit registration:

```python
def register_runtime(ctx):
    from .runtime import initialize_cell, Metrics
    ctx.add_agent_initializer("CELL", initialize_cell)
    metrics = Metrics(ctx)
    ctx.add_init_function(metrics.initialize)
    ctx.add_step_function(metrics.run)
```

The leading dot means “this variant package.” `model.py` does not search for `runtime.py`. Keeping the same definitions in `__init__.py` would work; renaming the file requires updating its imports. `simple_signal` keeps its small callbacks in `__init__.py`; `cell_markers` demonstrates a small separate module. The registration hook runs once; the callback registered with `add_step_function()` runs every step.

### Start from variant_template

Copy the entire `variants/variant_template/` folder to a new folder such as `variants/my_model/`. It contains all five function hooks, all three configuration dictionaries, an optional `runtime.py`, and the complete generic `configure_layers()` implementation. The optional hook bodies are documented no-ops with commented examples; the generic schedule is active code, so the unchanged template runs the generic model using core defaults.

1. Edit `PARAMS` and introduce any new names through `PARAM_DEFAULTS`.
2. Add validation for your supported feature combinations and parameter ranges.
3. Add declarations in `declare_model()`. Constants can use declaration defaults. Reserve custom-ID populations here if needed.
4. Write the corresponding C++ files, register them and bind messages/states/birth targets. Use `FILES` only when replacing a core function; update any literal variant path after renaming the folder.
5. Enable only the initialization/output callbacks you need in `register_runtime()`. The supplied `runtime.py` has no effect until imported and registered.
6. Edit the full `configure_layers()` explicitly. Include each enabled process and put producers before consumers. Uncommenting a declaration alone does not supply a kernel or schedule it.
7. Run a small case and verify an expected result before adding another mechanism.

```sh
python model.py --variant my_model --result-dir results/my_model
```

Paths built from `ctx.root / "variants" / ctx.name` follow the copied folder's name. Keep the full schedule in the variant; the template does not delegate it to another variant or patch it invisibly. The runnable `cell_markers` example shows these steps assembled into a working model.

## Walkthrough: create a small signal variant

The runnable example is in `variants/simple_signal/`. It adds two CELL variables and two GPU functions. Cells remain stationary: a dimensionless signal increases at a constant rate, and an integer flag switches on when the signal reaches a threshold. This is a programming example, not a biological model. No mechanics, diffusion or division are scheduled. Core agents still exist and receive their normal initialization.

**Its two layers are the whole schedule for this minimal model. No additional core layers run implicitly.** The example deliberately disables physical processes to isolate the registration mechanism. Use `variant_template` or the [cell-markers walkthrough](Tutorial-Variant-Cell-Markers.md) when you want the full generic schedule as a starting point.

### Step 1: create the package and files

From the repository root, create this structure (the supplied example already has these files):

```text
variants/
  simple_signal/
    __init__.py
    signal_accumulate.cpp
    signal_switch.cpp
```

The folder name is the `--variant` name. To create your own copy, copy this folder to `variants/my_signal/`; the function paths below use `ctx.name`, so the folder can be renamed without modifying `model.py`. Start `__init__.py` with `import math` and the following dictionaries.

### Step 2: choose the parameters and supported features

```python
PARAM_DEFAULTS = {"SIGNAL_RATE": 0.1, "SIGNAL_THRESHOLD": 0.25}
PARAMS = {
    "N_CELLS": 4, "STEPS": 4, "TIME_STEP": 1.0,
    "INCLUDE_CELLS": True,
    "INCLUDE_CELL_CELL_INTERACTION": False, "INCLUDE_CELL_CYCLE": False,
    "INCLUDE_DIFFUSION": False, "INCLUDE_FIBRE_NETWORK": False,
    "INCLUDE_FOCAL_ADHESIONS": False, "INCLUDE_NETWORK_REMODELING": False,
    "INCLUDE_CELL_FNODE_REPULSION": False, "INCLUDE_VASCULARIZATION": False,
    "INCLUDE_VASCULAR_CELL_RECRUITMENT": False, "INCLUDE_LUMEN": False,
    "ORGANOID_ASSAY": False, "MONOLAYER_ASSAY": False,
    "MOVING_BOUNDARIES": False,
    "VISUALISATION": False, "SHOW_PLOTS": False,
    "SAVE_PICKLE": True, "SAVE_DATA_TO_FILE": True, "SAVE_EVERY_N_STEPS": 1,
}


def validate_config(config):
    if not config["INCLUDE_CELLS"]:
        raise ValueError("simple_signal requires INCLUDE_CELLS=True")
    for name, value in PARAMS.items():
        if value is False and (name.startswith("INCLUDE_") or
                               name in ("MOVING_BOUNDARIES", "ORGANOID_ASSAY", "MONOLAYER_ASSAY")):
            if config[name]:
                raise ValueError(f"simple_signal does not schedule {name}; keep it False")
    for name in PARAM_DEFAULTS:
        value = config[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
```

`SIGNAL_RATE` has units of signal per second; `SIGNAL_THRESHOLD` has signal units. The validator prevents someone enabling a feature for which this deliberately small schedule has no functions. `N_CELLS` is the initial CELL count, distinct from the structural ECM grid parameter `N`. The example leaves grid dimensions, species counts, domain size and compiled arrays alone.

### Step 3: declare the additional variables and environment properties

Append this to `__init__.py`:

```python
def declare_model(ctx):
    cell = ctx.agents["CELL"]
    cell.newVariableFloat("signal", 0.0)
    cell.newVariableInt("signal_on", 0)
    for name in PARAM_DEFAULTS:
        ctx.env.newPropertyFloat(name, ctx.config[name])
```

The native CELL description is extended before simulation construction. Parameters in a Python dictionary do not automatically become GPU environment properties: the two `newPropertyFloat` calls make them available to the kernels. `TIME_STEP` is already a core environment property.

### Step 4: implement two GPU functions

Put this in `signal_accumulate.cpp`:

```cpp
FLAMEGPU_AGENT_FUNCTION(signal_accumulate, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float rate = FLAMEGPU->environment.getProperty<float>("SIGNAL_RATE");
    const float dt = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
    const float signal = FLAMEGPU->getVariable<float>("signal");
    FLAMEGPU->setVariable<float>("signal", signal + rate * dt);
    return flamegpu::ALIVE;
}
```

Put this in `signal_switch.cpp`:

```cpp
FLAMEGPU_AGENT_FUNCTION(signal_switch, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float threshold = FLAMEGPU->environment.getProperty<float>("SIGNAL_THRESHOLD");
    const float signal = FLAMEGPU->getVariable<float>("signal");
    FLAMEGPU->setVariable<int>("signal_on", signal >= threshold ? 1 : 0);
    return flamegpu::ALIVE;
}
```

Both functions operate on one CELL at a time, need no messages and return `ALIVE`. Python declarations and C++ accesses must agree on names and types.

### Step 5: register the functions

Append to `__init__.py`:

```python
def register_functions(ctx):
    directory = ctx.root / "variants" / ctx.name
    cell = ctx.agents["CELL"]
    cell.newRTCFunctionFile("signal_accumulate", str(directory / "signal_accumulate.cpp"))
    cell.newRTCFunctionFile("signal_switch", str(directory / "signal_switch.cpp"))
```

Registration makes functions available; it does not schedule them. `FILES` is unnecessary because neither function replaces a core function.

### Step 6: initialize the variables and collect results

Append to `__init__.py`:

```python
def initialize_cell(instance, rng):
    instance.setVariableFloat("signal", 0.0)
    instance.setVariableInt("signal_on", 0)


def register_runtime(ctx):
    ctx.add_agent_initializer("CELL", initialize_cell)
    ctx.cell_vtk_scalars.extend([
        ("signal", "signal", "float"), ("signal_on", "signal_on", "int"),
    ])

    def initialize_results(host):
        ctx.runtime_results(host)["SIGNAL_OVER_TIME"] = []

    def record_signal(host):
        cells = host.agent("CELL").getPopulationData()
        ctx.runtime_results(host)["SIGNAL_OVER_TIME"].append({
            "step": host.getStepCounter() + 1,
            "cells": [{"id": cell.getVariableInt("id"),
                       "signal": cell.getVariableFloat("signal"),
                       "signal_on": cell.getVariableInt("signal_on")} for cell in cells],
        })

    ctx.add_init_function(initialize_results)
    ctx.add_step_function(record_signal)
```

The explicit initializer illustrates the hook even though the declaration defaults are also zero. `rng` can be used for randomized initialization, but this example is deterministic. A step callback copies a tiny population to the host for teaching purposes; for a large model, sample less often or use reductions instead of exporting every agent every step. Each run has its own result list.

### Step 7: write the complete schedule

Append to `__init__.py`:

```python
def configure_layers(ctx):
    ctx.model.newLayer("Signal_Accumulate").addAgentFunction("CELL", "signal_accumulate")
    ctx.model.newLayer("Signal_Switch").addAgentFunction("CELL", "signal_switch")
```

This is the entire GPU schedule for this stationary teaching model. Switching uses the signal computed in the same step because it is in the next layer. Keep these functions in separate layers: both access the same CELL population. Host initialization and the registered end-of-step/output callbacks run through the core host-function infrastructure.

### Step 8: run and check the numerical result

From the repository root in the configured environment:

```sh
python model.py --variant simple_signal --result-dir results/simple_signal
```

All four cells should follow this trajectory (allow floating-point tolerance):

| Completed step | signal | signal_on |
| --- | --- | --- |
| 1 | 0.1 | 0 |
| 2 | 0.2 | 0 |
| 3 | 0.3 | 1 |
| 4 | 0.4 | 1 |

Inspect `SIGNAL_OVER_TIME` in `output_data_0.pickle`; the CELL VTK files also contain `signal` and `signal_on`. For a parameter experiment, create `signal_overrides.json` containing `{"SIGNAL_RATE": 0.2}` and run:

```sh
python model.py --variant simple_signal --overrides signal_overrides.json --result-dir results/simple_signal_fast
```

The flag should now turn on at step 2. This verifies declarations, parameter overrides, RTC registration, initialization, ordering and output with an independently calculable answer. The GPU regression script also runs both parameter cases. If you subsequently add division, implement daughter initialization in the birth kernel and add the cycle layers explicitly; enabling its flag alone is rejected.

## Parameters and optimizer overrides

```python
PARAM_DEFAULTS = {
    "MY_RESPONSE_RATE": 0.02,
}

PARAMS = {
    "CELL_SPEED_REF": 0.005,
    "SAVE_PICKLE": True,
}

FILES = {
    "cell_cycle_file": "variants/my_variant/cell_cycle.cpp",
}
```

Precedence is core/default declarations, then `PARAMS`, then user JSON. Indexed overrides and scalar-to-list broadcasting retain the existing `apply_param_overrides()` semantics. New parameters exist before JSON is applied, so optimizer values are not overwritten by a later registration hook.

`validate_config(config)` should reject incompatible combinations rather than allow a schedule to omit required behaviour silently. For example, the current radial-glia schedule does not implement fibre-network, focal-adhesion, lumen, or vascular dynamics and rejects those enabled flags. Its kernels require three species and three cell types.

The results pickle's `MODEL_CONFIG` records `VARIANT_NAME` and the resolved `VARIANT_PARAMETERS` introduced by `PARAM_DEFAULTS`, in addition to the existing core configuration.

An optimizer configuration uses:

```yaml
model:
  variant: radial_glia
  extra_overrides:
    SAVE_PICKLE: true
parameters:
  RG_COMMIT_RATE:
    type: float
    low: 0.000001
    high: 0.00001
```

See the existing optimizer YAML files for complete study definitions and their fixed-override format.

## Context and native declarations

`ctx.config` is a detached effective-configuration snapshot with a read-only top-level mapping. Treat nested lists as read-only as well. It contains values, not access to the `model.py` namespace.

`ctx.model`, `ctx.env`, `ctx.agents`, and `ctx.messages` expose native FLAMEGPU objects. Agent and message dictionaries use their actual FLAMEGPU names. `ctx.root` is the project directory. `ctx.functions` can retain additional function handles; existing core functions are also accessible through the native agent description.

```python
def declare_model(ctx):
    cell = ctx.agents["CELL"]
    cell.newVariableFloat("response", 0.0)

    message = ctx.messages["cell_spatial_location_message"]
    message.newVariableFloat("response")
    ctx.env.newPropertyFloat("MY_RESPONSE_RATE", ctx.config["MY_RESPONSE_RATE"])


def register_functions(ctx):
    function = ctx.agents["CELL"].newRTCFunctionFile(
        "cell_response", str(ctx.root / "variants/my_variant/cell_response.cpp"))
    function.setMessageInput("cell_spatial_location_message")
    ctx.functions["CELL.cell_response"] = function
```

Adding a message field also requires updating its publisher. Declaring a field does not automatically write it. Each spatial message must cover every consumer's required radius, with kernels applying their own distance cutoffs. For example, radial glia extends the CELL message and supplies its publisher through `FILES`.

Adding agent states also requires ensuring that the applicable generic and variant functions cover those states.

## Full schedule ownership

Variants must contain their full `configure_layers(ctx)` implementation. There is no schedule patcher, insertion registry, or automatic merge with the core schedule. A run with no selected variant uses `_build_default_layers()` in `model.py`.

“Full” means all processes intended for **that model**, not every optional CellFoundry feature. For example, `simple_signal` variant intentionally defines only accumulation and switching. `variant_template` contains all generic conditional branches. `cell_markers` contains those same branches plus its marker sequence. A disabled branch does not create a layer for that run. Enabling a flag can create agents/functions in the core, but those functions execute only if the chosen variant schedules them; flags are not an implicit schedule.

Inside a variant:

```python
def configure_layers(ctx):
    model, config = ctx.model, ctx.config
    # Define every layer required by this variant here, in execution order.
    # See organoid/__init__.py for a complete generic-feature schedule,
    # or radial_glia/__init__.py for the full RG-specific sequence.
```

Copy a complete existing schedule as a starting point and then edit it explicitly. Register functions in `register_functions()`.

For multiscale diffusion, explicitly call `ctx.add_multiscale_diffusion_layers()` at the appropriate point after cellular exchange and C_sp/D_sp preparation. This shared helper owns the solver's internal substeps and commit. The parent schedule still owns broadcasts, boundary handling, mechanics, and movement. Follow the complete examples for the L0 boundary/L1 publication ordering.

For example, the radial glia variant schedules division before metabolism, then differentiation before polarity and movement. Organoid defines the generic sequence explicitly in its own module. When generic scheduling changes, review variant schedules explicitly; the schedule regression tests help identify divergence.

## Initialization, birth and runtime callbacks

Register a per-agent initializer in `register_runtime()`. The same API accepts any declared agent name:

```python
def initialize_cell(instance, rng):
    instance.setVariableFloat("response", rng.uniform(0.0, 1.0))


def register_runtime(ctx):
    ctx.add_agent_initializer("CELL", initialize_cell)
    ctx.cell_vtk_scalars.append(("response", "response", "float"))
```

Callbacks receive `(instance, rng)` and run in registration order. They initialize additional variables; they must not change the allocated `id` (checked by the API). The RNG is the core NumPy random source. Core loops dispatch callbacks for BCORNER, FNODE, CELL, FOCAD, ECM and VASC when their initial populations are created. CELL callbacks run after position/type/basic state are set, before anchor and stress initialization. Other core-agent callbacks run after their core fields are set. For example, radial glia initializes substrate anchors from the CELL position. Avoid modifying generic geometry or consuming random numbers unless the variant intends that change.

For new types, the managed population initializer sets `id`, then calls the same per-agent callbacks. There is no core initial LUMEN population: registering a LUMEN initializer alone does not create droplets. These callbacks do not run for GPU-born agents. Initialize or inherit every extended variable explicitly in their birth kernels, as in radial glia's `cell_cycle.cpp`.

Use `ctx.add_init_function(callback)`, `ctx.add_step_function(callback)`, and `ctx.add_exit_function(callback)` for Python callbacks accepting one native FLAMEGPU HostAPI argument. Init callbacks run after the core population/macro initialization; step callbacks run after the GPU layers. The context wraps these in native HostFunctions and retains the wrappers for their required lifetime. Mid-step host work belongs in an explicit layer in `configure_layers()`.

Use `add_population()` for initial custom-ID populations, rather than calling `newAgent()` in an arbitrary init callback. The managed path reserves bounds before simulation construction and updates initialization counters in the correct order. Ordinary init callbacks are appropriate for analysis buffers and post-initialization work; all managed populations exist by then. Per-instance initializers must not assume another pending population is already visible through HostAPI. Loading a saved population or adding host-side births during a run requires an explicit policy for restoring/allocating counters; the initial-population helper does not implement either workflow.

## New agent populations, identifiers and bucket messages

### Keep identity, bucket keys and array indices distinct

| Quantity | Meaning and ownership |
| --- | --- |
| FLAMEGPU internal ID (`getID()`) | Assigned by FLAMEGPU; not the user-defined `id` variable. Do not assume it is dense or matches core offsets. |
| Custom Int `id` | Used by CellFoundry kernels and outputs. Initial core populations have fixed consecutive ranges. Runtime CELL/FNODE/LUMEN counters are independent, so a reference must identify the target **agent type as well as its id**. |
| `CURRENT_ID` | Initial-population/reservation high-water mark. It is not a live count and does not track GPU births. |
| Bucket key | Chosen by the publisher. Its bounds belong to that message, not to the whole model. FNODE buckets use FNODE `id`; FOCAD buckets use the owning `cell_id`. |
| Dense array index | A bounded index into a specific allocation. For a managed range it can be `id - range.begin`; ECM uses its own `grid_lin_id` for `C_SP_MACRO`. Never index a macro array directly with a global-looking custom id. |

The core initialization prefix is fixed: BCORNER, FNODE (when enabled), CELL, FOCAD (when enabled), ECM, then VASC (when enabled). Absent populations contribute zero to the offsets. Variant populations are appended **after this complete prefix**; they cannot insert themselves between core populations. `ctx.initial_ids["CELL"]`, for example, exposes `.begin`, `.count` and `.end` for the **initial** CELL range. It is not a bound on later CELL births; core bucket capacities account for those separately.

At construction time, `add_population()` reserves an explicit custom-ID range and declares its environment bounds and birth counter. At initialization time, the core verifies its cursor against the planned prefix, creates all managed populations in declaration order, assigns IDs, runs their per-agent initializers, seeds their counters and advances `CURRENT_ID` past **all reserved slots**. The LUMEN counter is seeded afterward, including for the monolayer assay when lumen is enabled. Core CELL/FNODE counters retain their own population endpoints.

For example, if the last core initial id is 100, a new population with `count=2, capacity=5` receives initial IDs 101 and 102, reserves `[101, 106)`, and advances `CURRENT_ID` to 105. Its last-issued-ID counter starts at 102; its next birth receives 103. A second managed population starts at 106. Empty populations use `count=0` and a positive capacity; their counters start at `begin - 1`.

These reservations protect initial offsets and separate the new populations' allocated ranges. They do **not** turn the existing core birth counters into one globally unique allocator. A later CELL/FNODE birth can have the same numeric custom id as another type. Store typed references and use type-specific messages/arrays. A model requiring globally unique custom IDs across every type would need a coordinated core migration of birth kernels, links, bucket bounds and exports; changing `CURRENT_ID` alone cannot provide that guarantee.

### Declare a new type and reserve its population

For a small additional `PROBE` agent, add the following inside your variant's hooks. Extend your existing hook bodies; a Python module must have only one definition of each hook.

```python
# In declare_model(ctx), after any CELL extensions:
probe = ctx.model.newAgent("PROBE")
probe.newVariableInt("id")
probe.newVariableFloat("value", 0.0)
ctx.agents["PROBE"] = probe
ids = ctx.add_population("PROBE", count=2, capacity=5)

message = ctx.model.newMessageBucket("probe_values")
message.setBounds(ids.begin, ids.end)  # lower inclusive, upper exclusive
message.newVariableFloat("value")
ctx.messages["probe_values"] = message
```

The reservation requires an existing agent with a scalar Int `id`. Registering the agent in `ctx.agents` makes it available to generic initialization. A completely new agent with only a native FLAMEGPU ID can still use `initialize_agent()` explicitly, but it is outside the managed custom-ID population allocator. The minimal managed path above covers normal CellFoundry custom-ID agents.

`capacity` counts **all IDs that may be issued during the run**, including agents which subsequently die. Slots are not recycled. Choose it deliberately: oversized bucket ranges/macro arrays consume memory, while undersized ranges prevent further births. Use `capacity=count` when there are no births. A zero initial count needs an explicit positive capacity. Counts and capacities must be integers and fit the signed Int ID space. Reservations must be made in `declare_model()`; bounds are sealed before function/runtime registration.

Register its initializer in `register_runtime(ctx)`:

```python
def initialize_probe(instance, rng):
    instance.setVariableFloat("value", rng.uniform(0.0, 1.0))

# In register_runtime(ctx):
ctx.add_agent_initializer("PROBE", initialize_probe)
```

Do not manually advance `CURRENT_ID` or manually create the initial PROBE population. The core does both through the reservation. For additional states, declare them natively first, then pass the desired initial state, for example `state="growing"`, to `add_population()`. One reservation and counter cover that agent type across its states; the helper creates its initial population in one state. State transitions, appropriate function bindings and outputs remain explicit variant code.

A publisher in `probe_publish.cpp` could be:

```cpp
FLAMEGPU_AGENT_FUNCTION(probe_publish, flamegpu::MessageNone, flamegpu::MessageBucket) {
    FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("value", FLAMEGPU->getVariable<float>("value"));
    return flamegpu::ALIVE;
}
```

Register and bind it in `register_functions(ctx)`:

```python
function = ctx.agents["PROBE"].newRTCFunctionFile(
    "probe_publish", str(ctx.root / "variants" / ctx.name / "probe_publish.cpp"))
function.setMessageOutput("probe_values")
```

In your full `configure_layers(ctx)`, add a publication layer and place any consuming functions in subsequent layers:

```python
ctx.model.newLayer("Probe_Publication").addAgentFunction("PROBE", "probe_publish")
```

A consumer must use `MessageBucket` as its input type and bind `setMessageInput("probe_values")`. Query a PROBE's custom id, not a CELL id or a dense slot. Bounds apply even when that bucket is empty. For arrays of capacity 5, calculate `slot = probe_id - VARIANT_PROBE_ID_BEGIN`, check `0 <= slot < 5`, then use the slot. A macro property's C++ template extent must match its Python declaration; changing a capacity does not automatically update a hard-coded template argument.

### Allocate IDs for GPU births without overrunning the reservation

`add_population("PROBE", ...)` declares four native environment entries:

- Int `VARIANT_PROBE_ID_BEGIN`: first reserved id.
- Int `VARIANT_PROBE_ID_END`: exclusive upper bound.
- Int macro `VARIANT_PROBE_LAST_ID`, dimension 1: last issued id for this type.
- Int macro `VARIANT_PROBE_ID_EXHAUSTED`, dimension 1: sticky error flag, initially zero.

The names are also available as `ids.begin_property`, `ids.end_property`, `ids.counter` and `ids.exhaustion_flag`. Use the counter and error flag with `variant_ids.cuh` in a birth kernel:

```cpp
#include "variant_ids.cuh"

FLAMEGPU_AGENT_FUNCTION(probe_birth, flamegpu::MessageNone, flamegpu::MessageNone) {
    auto counter = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_PROBE_LAST_ID");
    auto exhausted = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_PROBE_ID_EXHAUSTED");
    const int end = FLAMEGPU->environment.getProperty<int>("VARIANT_PROBE_ID_END");
    const int id = cellfoundry_claim_variant_id(counter, end, exhausted);
    if (id < 0) {
        // Error flag is set; the host will reject this run after the GPU layers.
        // Do not write agent_out or index a message/array with this invalid id.
        return flamegpu::ALIVE;
    }
    FLAMEGPU->agent_out.setVariable<int>("id", id);
    FLAMEGPU->agent_out.setVariable<float>("value", 0.0);
    return flamegpu::ALIVE;
}
```

Register the function on its parent agent, bind `setAgentOutput("PROBE")`, and schedule it explicitly. This illustrative function attempts one birth per caller per step; the biological eligibility rule belongs before allocation. Allocate only when the birth will be committed and initialize every required daughter field. The helper returns `-1` on exhaustion, sets the error flag atomically, and never moves the counter outside its reservation. It does not read or update `CURRENT_ID`. Normal (non-atomic) counter reads must be in a separate layer from GPU writes. Make the shared header visible to RTC compilation; CellFoundry runs from the project root, or an RTC string can include the absolute path constructed from `ctx.root`.

**Why the allocation is atomic:** FLAMEGPU's `addAtomic(1)` uses CUDA's `atomicAdd()` and returns the incremented value. The helper uses FLAMEGPU's `CAS(expected, replacement)`, backed by CUDA's atomic compare-and-swap. That operation changes the counter only if it still equals `expected`, and returns the old value. If two threads both observe 44, only one can successfully replace 44 with 45. The other receives the updated value and retries for 46. Each successful CAS is an atomic allocation; the surrounding retry loop is not one indivisible GPU instruction. The helper performs the same unique increment while capacity is available, with an additional bound. A separate ordinary check followed by `addAtomic(1)` would let several threads pass the check when only one slot remains. See FLAMEGPU's [macro-property operations](https://docs.flamegpu.com/guide/agent-functions/interacting-with-environment.html).

**Exhaustion is an error, not a model rule suppressing births.** The framework installs a host check when it seals the reservations, before normal core end-of-step output callbacks. After the GPU layers, any exhaustion flag raises an error naming the affected agent, step, capacity and ID range. The simulation fails instead of continuing with missing agents. Earlier output files may remain and are partial results, not a completed run. Increase the reservation, synchronize any explicit bucket/macro-array bounds, and rerun. The last valid allocation succeeds without an error; only an additional request triggers failure. No extra check registration is needed in the variant. This works independently of FLAMEGPU's optional device seatbelts.

Returning `ALIVE` in the kernel above keeps the **parent** alive until the host reports the allocation failure. The `-1` is only an internal invalid-ID sentinel; it must never be assigned to a daughter or used as an index. The helper cannot resize a bucket message or compiled macro array while the simulation runs.

The executable native test `tools/validate_variant_populations.py` covers two new types, a non-default state, an empty starting population, concurrent births, the first/last valid bucket keys, dense macro indexing and repeated simulations. It also compares 1,024 concurrent allocations against `addAtomic(1)` and verifies that insufficient capacities (37 and 1) produce unique bounded IDs and a visible error before normal output. FLAMEGPU's [message documentation](https://docs.flamegpu.com/guide/defining-messages-communication/index.html) and [agent-ID documentation](https://docs.flamegpu.com/guide/agent-functions/modifying-agent-variables.html) describe the native mechanisms underlying this API.

## Metrics and output

Additional CELL VTK scalars use `(vtk_name, variable_name, "float" or "int")`. Vectors use `(vtk_name, x_variable, y_variable, z_variable)`. Register them through `ctx.cell_vtk_scalars` and `ctx.cell_vtk_vectors`. The output writer handles these fields with or without focal-adhesion anchors.

### Optional focal-adhesion anchors

`INCLUDE_FOCAL_ADHESIONS` determines the CELL schema when the model is constructed. When false, CELL has no `x_i`, `y_i`, `z_i`, `u_ref_x_i`, `u_ref_y_i` or `u_ref_z_i` arrays. Initialization, movement, growth and division skip anchor work. `N_ANCHOR_POINTS` remains a positive core-controlled structural constant; do not set it to zero or tune it as a workaround. With 50 anchors, omitting the six float arrays saves 1,200 bytes of raw state per cell, before FLAMEGPU buffering overhead.

Nucleus radius, stress/strain and cell mechanics remain available independently of focal adhesions. Radial-glia `substrate_anchor_x/y` describe a separate biological constraint and remain present in that variant.

Core registration of `cell_move`, `cell_cycle` and `cell_stress_state_update`, including their `FILES` replacements, uses `helper_module.register_cell_rtc()`. This prefixes the RTC source with `CELLFOUNDRY_CELL_ANCHORS=0` or `1`, derived from the resolved feature flag. In a replacement kernel, enclose **every** anchor-array declaration and access, including `agent_out` initialization, in:

```cpp
#if CELLFOUNDRY_CELL_ANCHORS
// Access the optional anchor arrays here.
#endif
```

A runtime `if` can safely guard an absent-variable access when the branch is never entered; this was verified with FLAMEGPU 2.0.0-rc.5. The compile-time guard is used here to remove all anchor accesses and local arrays from the disabled kernel and keep its compiled feature set consistent with the constructed CELL schema. The per-cell storage saving comes from omitting the Python agent-variable declarations. A runtime-only design would need to guard every anchor access and handle temporary-array scope; its performance would need measurement rather than assumptions. For an additional anchor-aware function registered by the variant, use the same helper explicitly:

```python
from helper_module import register_cell_rtc
from pathlib import Path

fn = register_cell_rtc(ctx.agents["CELL"], "my_anchor_function",
                       Path(__file__).with_name("my_anchor_function.cpp"),
                       ctx.config["INCLUDE_FOCAL_ADHESIONS"])
# Bind messages/output and add fn in the variant's configure_layers().
```

When focal adhesions are off, `cells_t*.vtk` contains one point per cell and `SAVE_NO_ANCHOR_CELL_FILES` requires no additional action. When on, the original CELL files include anchors and that option can generate the extra centre-only files. Nucleus VTK output remains enabled whenever a supported deformation mechanism is active. Removing unnecessary anchor initialization also removes its random draws; stochastic trajectories should not be expected to match older anchor-disabled runs exactly.

For parameter-dependent instability checks, see [optimizer failure handling](Tutorial-Parameter-Optimization.md#configuration-checks-and-stopping-on-errors). Use `helper_module.reject_trial()` for an explicitly detected infeasible sample; ordinary validation/programming errors stop the study.

For pickle results, use `ctx.runtime_results(FLAMEGPU)` from a host callback. It returns a dictionary for that simulation/ensemble run. Initialize each owned result key in an init callback, then collect data in step/exit callbacks. Avoid module-level mutable result buffers.

The core merges these entries into its pickle payload and rejects collisions with core result keys. For example, radial glia emits `RG_FINAL_METRICS` and `RG_ROSETTE_METRICS_OVER_TIME`, with column names consumed by the optimizer and diagnostics. Specialized output for new agent types belongs in the variant; the CELL VTK extension lists do not automatically export other agents.
