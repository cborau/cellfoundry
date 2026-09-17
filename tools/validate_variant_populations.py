"""Native GPU test of variant IDs, births, bucket endpoints and dense macro slots.

Run with flamegpu_py310: python tools/validate_variant_populations.py
Uses a tiny synthetic model, with no files written and no biological assumptions.
"""
from pathlib import Path
import sys

import numpy as np
import pyflamegpu as fg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from variant_api import VariantContext


def validate():
    model = fg.ModelDescription("variant_population_contract")
    env = model.Environment()
    env.newPropertyUInt("CURRENT_ID", 0)
    for name in ("CELL", "FNODE", "LUMEN"):
        env.newMacroPropertyInt(f"MACRO_MAX_GLOBAL_{name}_ID", 1)
    env.newMacroPropertyInt("PROBE_SLOTS", 4)
    counts = {"BCORNER": 8, "FNODE": 3, "CELL": 2, "FOCAD": 4, "ECM": 5, "VASC": 1}
    ctx = VariantContext(model, {}, ROOT)
    for name in (*counts, "PROBE", "TIP"):
        agent = model.newAgent(name)
        agent.newVariableInt("id")
        agent.newVariableInt("found")
        agent.newVariableInt("tag")
        ctx.agents[name] = agent
    ctx.agents["TIP"].newState("growing")
    ctx.configure_initial_ids(counts)
    probe = ctx.add_population("PROBE", 2, capacity=4)
    tip = ctx.add_population("TIP", 0, capacity=2, state="growing")
    ctx.seal_populations()
    ctx.add_agent_initializer("ECM", lambda instance, rng: instance.setVariableInt("tag", 42))
    ctx.add_agent_initializer("PROBE", lambda instance, rng: instance.setVariableInt("tag", 77))

    def initialize(host):
        cursor = 0
        for name, count in counts.items():
            for _ in range(count):
                cursor += 1
                instance = host.agent(name).newAgent()
                instance.setVariableInt("id", cursor)
                ctx.initialize_agent(name, instance, np.random)
            if name in ("CELL", "FNODE"):
                host.environment.getMacroPropertyInt(f"MACRO_MAX_GLOBAL_{name}_ID")[0] = cursor
        host.environment.setPropertyUInt("CURRENT_ID", cursor)
        ctx.initialize_populations(host, np.random)
        host.environment.getMacroPropertyInt("MACRO_MAX_GLOBAL_LUMEN_ID")[0] = host.environment.getPropertyUInt("CURRENT_ID")
    ctx.add_init_function(initialize)

    header = (ROOT / "variant_ids.cuh").as_posix()
    for target in ("PROBE", "TIP"):
        code = f'''
#include "{header}"
FLAMEGPU_AGENT_FUNCTION(birth_{target}, flamegpu::MessageNone, flamegpu::MessageNone) {{
    if (FLAMEGPU->getVariable<int>("tag") == 99) return flamegpu::ALIVE;
    auto counter = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_{target}_LAST_ID");
    auto exhausted = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_{target}_ID_EXHAUSTED");
    const int end = FLAMEGPU->environment.getProperty<int>("VARIANT_{target}_ID_END");
    const int id = cellfoundry_claim_variant_id(counter, end, exhausted);
    if (id >= 0) {{
        FLAMEGPU->agent_out.setVariable<int>("id", id);
        FLAMEGPU->agent_out.setVariable<int>("tag", 99);
    }}
    return flamegpu::ALIVE;
}}'''
        fn = ctx.agents["PROBE"].newRTCFunction(f"birth_{target}", code)
        fn.setAgentOutput(target, "growing" if target == "TIP" else "default")
        model.newLayer(f"Birth_{target}").addAgentFunction("PROBE", f"birth_{target}")

    for ids in (probe, tip):
        name = ids.agent
        message = model.newMessageBucket(f"{name}_bucket")
        message.setBounds(ids.begin, ids.end)
        message.newVariableInt("id")
        publish = ctx.agents[name].newRTCFunction(f"publish_{name}", f'''
FLAMEGPU_AGENT_FUNCTION(publish_{name}, flamegpu::MessageNone, flamegpu::MessageBucket) {{
    const int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->message_out.setKey(id);
    FLAMEGPU->message_out.setVariable<int>("id", id);
    return flamegpu::ALIVE;
}}''')
        publish.setMessageOutput(f"{name}_bucket")
        read = ctx.agents[name].newRTCFunction(f"read_{name}", f'''
FLAMEGPU_AGENT_FUNCTION(read_{name}, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    const int id = FLAMEGPU->getVariable<int>("id");
    int found = 0;
    for (const auto &message : FLAMEGPU->message_in(id)) {{
        if (message.getVariable<int>("id") == id) ++found;
    }}
    FLAMEGPU->setVariable<int>("found", found);
    return flamegpu::ALIVE;
}}''')
        read.setMessageInput(f"{name}_bucket")
        if name == "TIP":
            for function in (publish, read):
                function.setInitialState("growing")
                function.setEndState("growing")
        model.newLayer(f"Publish_{name}").addAgentFunction(name, f"publish_{name}")
        model.newLayer(f"Read_{name}").addAgentFunction(name, f"read_{name}")

    slot = ctx.agents["PROBE"].newRTCFunction("write_slot", '''
FLAMEGPU_AGENT_FUNCTION(write_slot, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int begin = FLAMEGPU->environment.getProperty<int>("VARIANT_PROBE_ID_BEGIN");
    const int slot = FLAMEGPU->getVariable<int>("id") - begin;
    auto slots = FLAMEGPU->environment.getMacroProperty<int, 4>("PROBE_SLOTS");
    slots[slot].exchange(100 + slot);
    return flamegpu::ALIVE;
}''')
    model.newLayer("Dense_slots").addAgentFunction("PROBE", "write_slot")

    def verify(host):
        for name, ids in ctx.initial_ids.items():
            agents = host.agent(name, ids.state).getPopulationData()
            expected = list(range(ids.begin, ids.end))
            assert sorted(a.getVariableInt("id") for a in agents) == expected, name
            if name in ("PROBE", "TIP"):
                assert all(a.getVariableInt("found") == 1 for a in agents), name
                assert int(host.environment.getMacroPropertyInt(ids.counter)) == ids.end - 1
                assert int(host.environment.getMacroPropertyInt(ids.exhaustion_flag)) == 0
            if name == "ECM":
                assert all(a.getVariableInt("tag") == 42 for a in agents)
        assert host.environment.getPropertyUInt("CURRENT_ID") == tip.end - 1
        for name in ("CELL", "FNODE"):
            assert int(host.environment.getMacroPropertyInt(f"MACRO_MAX_GLOBAL_{name}_ID")) == ctx.initial_ids[name].end - 1
        assert int(host.environment.getMacroPropertyInt("MACRO_MAX_GLOBAL_LUMEN_ID")) == tip.end - 1
        slots = host.environment.getMacroPropertyInt("PROBE_SLOTS")
        assert [int(slots[i]) for i in range(4)] == [100, 101, 102, 103]
    ctx.add_exit_function(verify)

    # Two distinct simulations ensure no mutable allocation cursor leaks between runs.
    for _ in range(2):
        simulation = fg.CUDASimulation(model)
        simulation.SimulationConfig().steps = 1
        simulation.simulate()
    print("PASS: generic initialization; fixed core offsets/counters; two populations/states; "
          "empty population births; bucket endpoints; concurrent bounded allocation; "
          "exact capacity without false alarms; dense macro slots; repeated simulations")


def validate_atomic_contention(capacity):
    """1024 concurrent requests: compare with addAtomic, or require a visible error."""
    requests = 1024
    model = fg.ModelDescription("variant_atomic_contention")
    env = model.Environment()
    env.newPropertyUInt("CURRENT_ID", 0)
    env.newMacroPropertyInt("ADD_ATOMIC_COUNTER", 1)
    parent = model.newAgent("PARENT")
    for name in ("id", "claimed", "add_atomic_id"):
        parent.newVariableInt(name)
    child = model.newAgent("PROBE")
    child.newVariableInt("id")
    ctx = VariantContext(model, {}, ROOT)
    ctx.agents.update(PARENT=parent, PROBE=child)
    ctx.configure_initial_ids({"PARENT": requests})
    ids = ctx.add_population("PROBE", 0, capacity=capacity)
    observations = {}

    # Test-only observation before the guard: inspect bounded allocation even if
    # the mandatory guard raises before the model's normal output callbacks.
    def inspect(host):
        population = host.agent("PARENT").getPopulationData()
        observations["claimed"] = [a.getVariableInt("claimed") for a in population]
        observations["atomic"] = [a.getVariableInt("add_atomic_id") for a in population]
        observations["children"] = [a.getVariableInt("id") for a in host.agent("PROBE").getPopulationData()]
        observations["last"] = int(host.environment.getMacroPropertyInt(ids.counter))
        observations["exhausted"] = int(host.environment.getMacroPropertyInt(ids.exhaustion_flag))
    ctx.add_step_function(inspect)
    ctx.seal_populations()

    def initialize(host):
        for i in range(requests):
            host.agent("PARENT").newAgent().setVariableInt("id", i + 1)
        host.environment.setPropertyUInt("CURRENT_ID", requests)
        host.environment.getMacroPropertyInt("ADD_ATOMIC_COUNTER")[0] = requests
        ctx.initialize_populations(host, np.random)
    ctx.add_init_function(initialize)
    header = (ROOT / "variant_ids.cuh").as_posix()
    function = parent.newRTCFunction("claim_id", f'''
#include "{header}"
FLAMEGPU_AGENT_FUNCTION(claim_id, flamegpu::MessageNone, flamegpu::MessageNone) {{
    auto last = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_PROBE_LAST_ID");
    auto exhausted = FLAMEGPU->environment.getMacroProperty<int, 1>("VARIANT_PROBE_ID_EXHAUSTED");
    const int end = FLAMEGPU->environment.getProperty<int>("VARIANT_PROBE_ID_END");
    const int id = cellfoundry_claim_variant_id(last, end, exhausted);
    FLAMEGPU->setVariable<int>("claimed", id);
    auto atomic = FLAMEGPU->environment.getMacroProperty<int, 1>("ADD_ATOMIC_COUNTER");
    FLAMEGPU->setVariable<int>("add_atomic_id", atomic.addAtomic(1));
    if (id >= 0) FLAMEGPU->agent_out.setVariable<int>("id", id);
    return flamegpu::ALIVE;
}}''')
    function.setAgentOutput("PROBE")
    model.newLayer("Concurrent_requests").addAgentFunction("PARENT", "claim_id")
    ctx.add_step_function(lambda host: observations.update(normal_output_reached=True))
    simulation = fg.CUDASimulation(model)
    simulation.SimulationConfig().steps = 1
    error = None
    try:
        simulation.simulate()
    except RuntimeError as caught:
        error = caught
    valid = sorted(value for value in observations["claimed"] if value >= 0)
    assert valid == list(range(ids.begin, ids.begin + min(capacity, requests)))
    assert sorted(observations["children"]) == valid
    assert observations["last"] == ids.begin + min(capacity, requests) - 1
    assert sorted(observations["atomic"]) == list(range(ids.begin, ids.begin + requests))
    if capacity < requests:
        assert error is not None, "Under-capacity run must fail visibly"
        assert f"PROBE: capacity={capacity}" in str(error), str(error)
        assert "step 1" in str(error)
        assert observations["exhausted"] == 1
        assert observations["claimed"].count(-1) == requests - capacity
        assert "normal_output_reached" not in observations
        print(f"PASS: {requests} concurrent requests, capacity {capacity}: "
              f"{capacity} unique IDs, bounded counter, explicit error before normal output")
    else:
        assert error is None, str(error)
        assert observations["exhausted"] == 0
        assert observations["normal_output_reached"]
        assert valid == sorted(observations["atomic"])
        print(f"PASS: {requests} concurrent requests: CAS helper and addAtomic both "
              "allocate the complete unique ID range")


if __name__ == "__main__":
    validate()
    for capacity in (1024, 37, 1):
        validate_atomic_contention(capacity)
