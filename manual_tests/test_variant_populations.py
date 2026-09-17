"""CPU contracts for managed variant initialization and custom-ID reservations."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from variant_api import VariantContext, core_initial_population_counts


class Instance:
    def __init__(self): self.values = {}
    def setVariableInt(self, key, value): self.values[key] = value
    def getVariableInt(self, key): return self.values[key]


class Environment:
    def __init__(self, current=0):
        self.values = {"CURRENT_ID": current}
        self.macros = {}
    def newPropertyInt(self, name, value): self.values[name] = value
    def newMacroPropertyInt(self, name, size): self.macros[name] = Macro([0] * size)
    def getPropertyUInt(self, name): return self.values[name]
    def setPropertyUInt(self, name, value): self.values[name] = value
    def getMacroPropertyInt(self, name): return self.macros[name]


class Macro(list):
    def __int__(self): return self[0]


class Host:
    def __init__(self, env):
        self.environment = env
        self.populations = {}
    def getStepCounter(self): return 2
    def agent(self, name, state="default"):
        instances = self.populations.setdefault((name, state), [])
        def create():
            instance = Instance()
            instances.append(instance)
            return instance
        return SimpleNamespace(newAgent=create)


class PopulationContracts(unittest.TestCase):
    def context(self):
        env = Environment(23)
        ctx = VariantContext(SimpleNamespace(Environment=lambda: env), {}, Path("."))
        ctx._test_step_callbacks = []
        ctx.add_step_function = ctx._test_step_callbacks.append
        ctx.configure_initial_ids({"BCORNER": 8, "FNODE": 3, "CELL": 2, "FOCAD": 4, "ECM": 5, "VASC": 1})
        agent = SimpleNamespace(hasVariable=lambda name: name == "id",
                                getVariableLength=lambda name: 1,
                                hasState=lambda state: state in ("default", "growing"))
        ctx.agents.update({name: agent for name in ("CELL", "ECM", "PROBE", "TIP")})
        return ctx, Host(env)

    def test_core_order_and_disabled_populations(self):
        config = dict(N_CELLS=4, N_NODES=None, INIT_N_FOCAD_PER_CELL=2,
                      ECM_POPULATION_SIZE=27, N_VASC_NODES=3, INCLUDE_CELLS=True,
                      INCLUDE_FIBRE_NETWORK=False, INCLUDE_FOCAL_ADHESIONS=False,
                      INCLUDE_VASCULARIZATION=False)
        counts = core_initial_population_counts(config)
        self.assertEqual(list(counts), ["BCORNER", "FNODE", "CELL", "FOCAD", "ECM", "VASC"])
        self.assertEqual(list(counts.values()), [8, 0, 4, 0, 27, 0])
        config.update(INCLUDE_FIBRE_NETWORK=True, N_NODES=6,
                      INCLUDE_FOCAL_ADHESIONS=True, INCLUDE_VASCULARIZATION=True)
        self.assertEqual(list(core_initial_population_counts(config).values()), [8, 6, 4, 8, 27, 3])

    def test_generic_initializer_order_and_identity_guard(self):
        ctx, host = self.context()
        cell = Instance()
        cell.setVariableInt("id", 12)
        events = []
        ctx.add_agent_initializer("CELL", lambda instance, rng: events.append(("CELL", rng)))
        ctx.add_agent_initializer("ECM", lambda instance, rng: events.append(("ECM", rng)))
        ctx.initialize_agent("CELL", cell, "rng")
        self.assertEqual(events, [("CELL", "rng")])
        ctx.add_agent_initializer("CELL", lambda instance, rng: instance.setVariableInt("id", 999))
        with self.assertRaisesRegex(ValueError, "must not change"):
            ctx.initialize_agent("CELL", cell, None)
        with self.assertRaisesRegex(ValueError, "undeclared"):
            ctx.add_agent_initializer("UNKNOWN", lambda instance, rng: None)

    def test_initializer_can_also_extend_agents_without_a_custom_id(self):
        ctx, host = self.context()
        ctx.agents["NATIVE_ID_ONLY"] = SimpleNamespace(hasVariable=lambda name: False)
        ctx.add_agent_initializer("NATIVE_ID_ONLY", lambda instance, rng: instance.setVariableInt("tag", 3))
        instance = Instance()
        ctx.initialize_agent("NATIVE_ID_ONLY", instance, None)
        self.assertEqual(instance.values, {"tag": 3})

    def test_two_populations_keep_core_offsets_and_seed_separate_counters(self):
        ctx, host = self.context()
        core = dict(ctx.initial_ids)
        probe = ctx.add_population("PROBE", 2, capacity=5)
        tip = ctx.add_population("TIP", 0, capacity=3, state="growing")
        ctx.add_agent_initializer("PROBE", lambda instance, rng: instance.setVariableInt("value", rng))
        self.assertEqual((probe.begin, probe.end, tip.begin, tip.end), (24, 29, 29, 32))
        ctx.seal_populations()
        ctx.initialize_populations(host, 7)
        self.assertEqual(host.environment.values["CURRENT_ID"], 31)
        self.assertEqual(host.environment.macros[probe.counter], [25])
        self.assertEqual(host.environment.macros[tip.counter], [28])
        self.assertEqual(host.environment.macros[probe.exhaustion_flag], [0])
        self.assertEqual([i.values for i in host.populations[("PROBE", "default")]],
                         [{"id": 24, "value": 7}, {"id": 25, "value": 7}])
        self.assertEqual(host.populations[("TIP", "growing")], [])
        self.assertEqual({key: ctx.initial_ids[key] for key in core}, core)
        # A second simulation has its own HostAPI/environment and starts afresh.
        second = Host(Environment(23))
        for ids in (probe, tip):
            second.environment.newMacroPropertyInt(ids.counter, 1)
            second.environment.newMacroPropertyInt(ids.exhaustion_flag, 1)
            second.environment.macros[ids.exhaustion_flag][0] = 1
        ctx.initialize_populations(second, 9)
        self.assertEqual(second.environment.macros[probe.counter], [25])
        self.assertEqual(second.environment.macros[probe.exhaustion_flag], [0])
        self.assertEqual(host.populations[("PROBE", "default")][0].values["value"], 7)

    def test_invalid_reservations_and_layout_fail_early(self):
        ctx, host = self.context()
        for count, capacity in [(3, 2), (0, 0), (-1, 2), (1.5, 2), (True, 2), (1, 2**31)]:
            with self.subTest(count=count, capacity=capacity), self.assertRaises(ValueError):
                ctx.add_population("PROBE", count, capacity=capacity)
        with self.assertRaisesRegex(ValueError, "owned"):
            ctx.add_population("CELL", 1)
        with self.assertRaisesRegex(ValueError, "no state"):
            ctx.add_population("TIP", 1, state="unknown")
        ctx.add_population("PROBE", 1)
        with self.assertRaisesRegex(ValueError, "owned"):
            ctx.add_population("PROBE", 1)
        ctx.seal_populations()
        with self.assertRaisesRegex(RuntimeError, "declare_model"):
            ctx.add_population("TIP", 1)
        host.environment.values["CURRENT_ID"] = 22
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            ctx.initialize_populations(host, None)

    def test_capacity_error_is_registered_once_and_reports_all_failed_types(self):
        ctx, host = self.context()
        probe = ctx.add_population("PROBE", 1, capacity=2)
        tip = ctx.add_population("TIP", 0, capacity=1)
        ctx.seal_populations()
        ctx.seal_populations()
        self.assertEqual(len(ctx._test_step_callbacks), 1)
        ctx.initialize_populations(host, None)
        # Consuming the last valid ID is not itself an error.
        host.environment.macros[probe.counter][0] = probe.end - 1
        ctx._test_step_callbacks[0](host)
        host.environment.macros[probe.exhaustion_flag][0] = 1
        host.environment.macros[tip.exhaustion_flag][0] = 1
        with self.assertRaisesRegex(RuntimeError, "capacity exhausted at step 3") as error:
            ctx._test_step_callbacks[0](host)
        for text in ("PROBE: capacity=2", "TIP: capacity=1", "Increase capacity", "run is invalid"):
            self.assertIn(text, str(error.exception))


if __name__ == "__main__":
    unittest.main()
