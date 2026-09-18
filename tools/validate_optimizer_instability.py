"""Exercise production instability guards through native FLAMEGPU and Optuna.

Run in flamegpu_py310. Each study rejects a bad sample then completes a good
sample. No checked-in model parameters or kernels are modified.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import optuna
from optimizer.optimize import make_objective
from optimizer.objectives import OBJECTIVE_REGISTRY


SCRIPT = '''
N = 2
N_SPECIES = 3
N_CELL_TYPES = 3
MAX_CONNECTIVITY = 8
N_ANCHOR_POINTS = 50
MAX_VASC_CONNECTIVITY = 2
BOUNDARY_COORDS = [1, -1, 1, -1, 1, -1]
import ast, json, pickle, sys
from pathlib import Path
sys.path.insert(0, ROOT_PATH)
import pyflamegpu
from simulation_errors import reject_trial
overrides = json.loads(Path(sys.argv[sys.argv.index("--overrides") + 1]).read_text())
output = Path(sys.argv[sys.argv.index("--result-dir") + 1])
tree = ast.parse((Path(ROOT_PATH) / "model.py").read_text(encoding="utf-8"))
definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("CheckFNODEStability", "CheckDiffusionStability")]
exec(compile(ast.Module(body=definitions, type_ignores=[]), "model.py", "exec"))
INCLUDE_FIBRE_NETWORK = True
ABORT_ON_UNSTABLE_FNODE_MOVE = overrides.get("ABORT_ON_UNSTABLE_FNODE_MOVE", True)
FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE = 10.
model = pyflamegpu.ModelDescription("guard_test")
agent = model.newAgent("FNODE")
agent.newVariableUInt8("unstable_move", 1)
agent.newRTCFunction("noop", "FLAMEGPU_AGENT_FUNCTION(noop, flamegpu::MessageNone, flamegpu::MessageNone) { return flamegpu::ALIVE; }")
model.newLayer().addAgentFunction(agent.getFunction("noop"))
ecm = model.newAgent("ECM")
ecm.newVariableUInt("diffusion_error", overrides.get("diffusion_error", 0))
guard = CheckDiffusionStability() if overrides.get("test_diffusion") else CheckFNODEStability()
model.addStepFunction(guard)
simulation = pyflamegpu.CUDASimulation(model)
simulation.SimulationConfig().steps = 2
simulation.setPopulationData(pyflamegpu.AgentVector(agent, 1))
simulation.setPopulationData(pyflamegpu.AgentVector(ecm, 1))
simulation.simulate()
pickle.dump({"score": 1.}, open(output / "output_data_0.pickle", "wb"))
'''


def validate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = output / "model.py"
    script.write_text(SCRIPT.replace("ROOT_PATH", repr(str(ROOT))), encoding="utf-8")
    OBJECTIVE_REGISTRY["guard_test"] = lambda results, reference, **kwargs: (results["score"], "ok")
    report = {}
    for name, parameter, bad, good, fixed in (
        ("fnode_abort_flag", "ABORT_ON_UNSTABLE_FNODE_MOVE", True, False, {}),
        ("diffusion_invalid_data", "diffusion_error", 1, 0, {"test_diffusion": True}),
        ("diffusion_cfl", "diffusion_error", 2, 0, {"test_diffusion": True}),
    ):
        config = {"parameters": {parameter: {"type": "categorical", "choices": [bad, good]}},
                  "model": {"extra_overrides": fixed, "timeout": 120},
                  "objective": {"function": "guard_test"}}
        objective, _, _ = make_objective(config, str(script), str(output / name))
        study = optuna.create_study()
        study.enqueue_trial({parameter: bad})
        study.enqueue_trial({parameter: good})
        study.optimize(objective, n_trials=2)
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.COMPLETE]
        assert study.trials[0].user_attrs["prune_reason"]
        report[name] = [t.state.name for t in study.trials]
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/anchor_refactor/instability")
    validate(parser.parse_args().output)
