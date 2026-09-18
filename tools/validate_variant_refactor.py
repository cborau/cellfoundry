"""GPU integration checks through the real optimizer subprocess/output path.

Run in flamegpu_py310:
    python tools/validate_variant_refactor.py --output tmp/variant_refactor/gpu

Uses an isolated source copy and explicitly synchronizes its RTC constants
before validation. Production batch runs only check constants and never repair
them. No kernel or initialization cache in the checkout is modified. Tiny
synthetic runs validate wiring, not biological calibration.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.optimize import run_trial_subprocess, make_objective, _load_pickle
from optimizer.objectives import organoid_error
from check_hard_coded_values import main as check_constants


def prepare_workspace(output):
    """Copy only model sources to an isolated checkout for GPU checks."""
    output = Path(output).resolve()
    workspace = output / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".cuh")]
    files.append(ROOT / "tools/remove_anchors_from_cell_vtks.py")
    for directory in ("variants", "optimizer"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".cuh")
                     and "__pycache__" not in p.parts)
    for source in files:
        target = workspace / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Prepare the core geometry itself. Runtime bounds overrides cannot rebuild
    # all dependent setup. This small domain belongs only to the test copy.
    model_path = workspace / "model.py"
    source, count = re.subn(r"^BOUNDARY_COORDS = .*?$", "BOUNDARY_COORDS = [50., -50., 50., -50., 50., -50.]",
                            model_path.read_text(encoding="utf-8"), count=1, flags=re.M)
    assert count == 1
    model_path.write_text(source, encoding="utf-8")
    # Keep the RG fixture's explicit PARAMS equal to its selected core build.
    rg_path = workspace / "variants/radial_glia/__init__.py"
    source, count = re.subn(r'("BOUNDARY_COORDS"\s*:\s*)\[[^\]]+\]',
                            r'\g<1>[50., -50., 50., -50., 50., -50.]',
                            rg_path.read_text(encoding="utf-8"), count=1)
    assert count == 1
    rg_path.write_text(source, encoding="utf-8")
    # Fixture preparation may repair its private copy; optimizer runs must not.
    roots = [workspace, *(workspace / "variants").iterdir()]
    args = ["--model-file", str(workspace / "model.py"), "--no-recursive", "--fix"]
    for root in roots:
        if root.is_dir() or root.suffix == ".py":
            args.extend(["--scan-root", str(root)])
    if check_constants(args):
        raise RuntimeError("Could not synchronize constants in the isolated validation workspace")
    return workspace


def validate(output):
    output = Path(output).resolve()
    workspace = prepare_workspace(output)

    common = dict(STEPS=3, TIME_STEP=1.0, N_CELLS=6,
                  BOUNDARY_COORDS=[50., -50., 50., -50., 50., -50.],
                  BOUNDARY_DISP_RATES=[0.] * 6, BOUNDARY_DISP_RATES_PARALLEL=[0.] * 12,
                  INCLUDE_CELLS=True, INCLUDE_CELL_CELL_INTERACTION=True,
                  INCLUDE_CELL_CYCLE=True, INCLUDE_FIBRE_NETWORK=False,
                  INCLUDE_FOCAL_ADHESIONS=False, INCLUDE_VASCULARIZATION=False,
                  INCLUDE_NETWORK_REMODELING=False, INCLUDE_LUMEN=False,
                  INCLUDE_CELL_FNODE_REPULSION=False,
                  INCLUDE_CHEMOTAXIS=False, INCLUDE_CHEMOKINESIS=False,
                  VISUALISATION=False, SHOW_PLOTS=False, DEBUG_PRINTING=False,
                  SAVE_PICKLE=True, SAVE_DATA_TO_FILE=True, SAVE_NO_ANCHOR_CELL_FILES=True,
                  SAVE_EVERY_N_STEPS=1, DEBUG_PRINT_INTERVAL=1,
                  CELL_RADIUS=[3., 3., 3.], ORGANOID_INIT_RADIUS=12.,
                  MONOLAYER_CLUSTER_RADIUS=20., MONOLAYER_Z=0.,
                  CELL_HYPOXIA_DAMAGE_RATE=[0.] * 3, CELL_NUTRIENT_DAMAGE_RATE=[0.] * 3,
                  CELL_STRESS_DAMAGE_RATE=[0.] * 3)
    cases = [
        ("base", None, {"INCLUDE_DIFFUSION": False}),
        ("organoid_assay", None, {"INCLUDE_DIFFUSION": False, "ORGANOID_ASSAY": True,
                                  "MONOLAYER_ASSAY": False}),
        ("organoid", "organoid", {"ORGANOID_CONTACT_INHIBIT_SIGMA": 2.75}),
        ("radial_glia", "radial_glia", {"RG_COMMIT_RATE": 0.000123, "MIN_ROSETTE_SIZE": 2}),
        ("radial_glia_multiscale", "radial_glia", {
            "TIME_STEP_DIFFUSION": [.25, .5, 1.], "RG_COMMIT_RATE": 0.000321,
            "RG_ADHESION_MATRIX[8]": 1.75, "MIN_ROSETTE_SIZE": 2,
        }),
    ]
    report = {}
    original_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        for name, variant, extra in cases:
            run_dir = output / name
            run_dir.mkdir(exist_ok=True)
            if variant:
                # Exercise the actual Optuna dispatch, including new parameters,
                # indexed matrix tuning, target CSV synthesis and pickle loading.
                import optuna
                overrides = {**common, **extra}
                parameter = ("RG_COMMIT_RATE" if variant == "radial_glia"
                             else "ORGANOID_CONTACT_INHIBIT_SIGMA")
                value = overrides.pop(parameter)
                parameters = {parameter: {"type": "float", "low": value, "high": value}}
                if "RG_ADHESION_MATRIX[8]" in overrides:
                    value = overrides.pop("RG_ADHESION_MATRIX[8]")
                    parameters["RG_ADHESION_MATRIX"] = {
                        "type": "array_float", "elements": {"8": {"low": value, "high": value}}}
                objective = ({"function": "rg_rosette_2d_error",
                              "kwargs": {"metric": "rg_fraction", "target_value": .3}}
                             if variant == "radial_glia" else
                             {"function": "organoid_error",
                              "kwargs": {"metric": "radius_of_gyration", "target_metric": 12.}})
                study = optuna.create_study(direction="minimize")
                objective_fn, _, _ = make_objective({
                    "model": {"variant": variant, "extra_overrides": overrides, "timeout": 300},
                    "parameters": parameters, "objective": objective,
                }, str(workspace / "model.py"), str(run_dir))
                study.optimize(objective_fn, n_trials=1)
                assert study.trials[0].state == optuna.trial.TrialState.COMPLETE, name
                run_dir = run_dir / "trial_00000"
                results = _load_pickle(str(run_dir / "output_data_0.pickle"))
            else:
                results = run_trial_subprocess(
                    {**common, **extra}, str(workspace / "model.py"), str(run_dir), timeout=300)
            assert len(results["CELL_SPEED_METRICS"]) >= common["N_CELLS"]
            vtk_files = list(run_dir.rglob("*.vtk"))
            assert vtk_files, f"{name}: VTK output missing"
            # Focal adhesions are disabled: the original CELL files are already
            # centre-only; no duplicate stripped files should be generated.
            assert not any("no_anchor" in str(p).lower() for p in vtk_files), name
            for vtk in run_dir.glob("cells_t*.vtk"):
                content = vtk.read_text()
                points = int(re.search(r"^POINTS (\d+)", content, re.M).group(1))
                assert points == common["N_CELLS"], (name, vtk, points)
            config = results["MODEL_CONFIG"]
            assert config.VARIANT_NAME == variant
            if variant == "radial_glia":
                metrics = results["RG_ROSETTE_METRICS_OVER_TIME"]
                assert metrics["step"].tolist() == [1, 2, 3]
                assert len(results["RG_FINAL_METRICS"]) >= common["N_CELLS"]
                assert config.VARIANT_PARAMETERS["RG_COMMIT_RATE"] == extra["RG_COMMIT_RATE"]
                assert config.VARIANT_PARAMETERS["MIN_ROSETTE_SIZE"] == 2
                if "RG_ADHESION_MATRIX[8]" in extra:
                    assert config.VARIANT_PARAMETERS["RG_ADHESION_MATRIX"][8] == 1.75
                assert any("apical_vector" in p.read_text() and "rg_committed" in p.read_text()
                           for p in vtk_files)
                report[name] = {"rosette_rows": len(metrics), "objective": study.best_value,
                                "optuna_state": "COMPLETE"}
            elif name in ("organoid", "organoid_assay"):
                assert len(results["ORGANOID_METRICS_OVER_TIME"]) == 3
                score = organoid_error(results, metric="radius_of_gyration", target_metric=12.)
                if variant:
                    assert config.VARIANT_PARAMETERS["ORGANOID_CONTACT_INHIBIT_SIGMA"] == 2.75
                assert "RG_FINAL_METRICS" not in results
                report[name] = {"organoid_rows": 3, "objective": str(score)}
            else:
                assert "RG_FINAL_METRICS" not in results
                report[name] = {"cell_rows": len(results["CELL_SPEED_METRICS"])}
            report[name]["vtk_files"] = len(vtk_files)
            (output / "report.json").write_text(json.dumps(report, indent=2))
            print(f"PASS: {name}", flush=True)
        report.update(validate_signal_examples(workspace, output))
        (output / "report.json").write_text(json.dumps(report, indent=2))
    finally:
        os.chdir(original_cwd)
    return report


def validate_signal_examples(workspace, output):
    """Check the tutorial's calculable answer and real core population integration."""
    import math
    # The complete tutorial variant, through the optimizer's actual subprocess path.
    report = {}
    for rate, first_on in ((.1, 3), (.2, 2)):
        name = "simple_signal" if rate == .1 else "simple_signal_fast"
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        results = run_trial_subprocess(
            {"SIGNAL_RATE": rate}, str(workspace / "model.py"), str(directory),
            timeout=300, variant="simple_signal")
        rows = results["SIGNAL_OVER_TIME"]
        assert len(rows) == 4
        for row in rows:
            assert len(row["cells"]) == 4
            for cell in row["cells"]:
                assert math.isclose(cell["signal"], row["step"] * rate, abs_tol=1e-6)
                assert cell["signal_on"] == int(row["step"] >= first_on)
        assert any("SCALARS signal float" in p.read_text() and "SCALARS signal_on int" in p.read_text()
                   for p in directory.rglob("*.vtk"))
        report[name] = {"steps": 4, "first_on": first_on, "vtk_fields": True}
        print(f"PASS: {name}", flush=True)

    # Synthetic variant only in the isolated checkout. It exercises model.py's
    # actual creation order, non-CELL hooks, reservation tail and monolayer LUMEN
    # counter seeding without adding a biological example to the public variants.
    fixture = workspace / "variants/population_contract"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "__init__.py").write_text('''
from variants import simple_signal as signal
PARAM_DEFAULTS = signal.PARAM_DEFAULTS
PARAMS = {**signal.PARAMS, "INCLUDE_LUMEN": True, "MONOLAYER_ASSAY": True}
configure_layers = signal.configure_layers

def register_functions(ctx):
    # The test kernels belong to the supplied signal example.
    for name in ("signal_accumulate", "signal_switch"):
        ctx.agents["CELL"].newRTCFunctionFile(name, str(ctx.root / "variants/simple_signal" / (name + ".cpp")))

def declare_model(ctx):
    signal.declare_model(ctx)
    ctx.agents["ECM"].newVariableInt("variant_tag")
    for name, count, capacity in (("PROBE", 2, 5), ("TIP", 0, 2)):
        agent = ctx.model.newAgent(name)
        agent.newVariableInt("id")
        agent.newVariableInt("variant_tag")
        ctx.agents[name] = agent
        ctx.add_population(name, count, capacity=capacity)

def register_runtime(ctx):
    signal.register_runtime(ctx)
    for name in ("ECM", "PROBE"):
        ctx.add_agent_initializer(name, lambda instance, rng: instance.setVariableInt("variant_tag", 37))
    def verify(host):
        for name, ids in ctx.initial_ids.items():
            if name not in ctx.agents:
                continue
            population = host.agent(name).getPopulationData()
            assert sorted(a.getVariableInt("id") for a in population) == list(range(ids.begin, ids.begin + ids.count)), name
            if name in ("ECM", "PROBE"):
                assert all(a.getVariableInt("variant_tag") == 37 for a in population)
            if name == "ECM":
                assert sorted(a.getVariableInt("grid_lin_id") for a in population) == list(range(ids.count))
        end = ctx.initial_ids["TIP"].end - 1
        assert host.environment.getPropertyUInt("CURRENT_ID") == end
        assert int(host.environment.getMacroPropertyInt("MACRO_MAX_GLOBAL_LUMEN_ID")) == end
        assert int(host.environment.getMacroPropertyInt("MACRO_MAX_GLOBAL_CELL_ID")) == ctx.initial_ids["CELL"].end - 1
        ctx.runtime_results(host)["POPULATION_CONTRACT"] = {"current_id": end, "passed": True}
    ctx.add_init_function(verify)
''', encoding="utf-8")
    (output / "population_contract").mkdir(parents=True, exist_ok=True)
    results = run_trial_subprocess({}, str(workspace / "model.py"), str(output / "population_contract"),
                                   timeout=300, variant="population_contract")
    assert results["POPULATION_CONTRACT"]["passed"]
    report["population_contract"] = results["POPULATION_CONTRACT"]
    print("PASS: population_contract", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/variant_refactor/gpu")
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2))
