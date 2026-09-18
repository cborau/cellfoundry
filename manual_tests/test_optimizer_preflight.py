"""Read-only checker and fail-fast study integration, using tiny subprocesses."""

import contextlib
import io
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optuna
from optimizer import optimize
from optimizer.objectives import OBJECTIVE_REGISTRY


REFERENCE = """N = 2
N_SPECIES = 3
N_CELL_TYPES = 1
MAX_CONNECTIVITY = 8
N_ANCHOR_POINTS = 2
MAX_VASC_CONNECTIVITY = 4
BOUNDARY_COORDS = [1, -1, 1, -1, 1, -1]
"""


class OptimizerPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model.py"
        self.model.write_text(REFERENCE, encoding="utf-8")
        self.kernel = self.root / "kernel.cpp"
        self.kernel.write_text("const int N_SPECIES = 3;\n", encoding="utf-8")
        self.results = self.root / "results"
        self.results.mkdir()
        self.config = {"parameters": {}, "objective": {"function": "checker_test"}}
        registry = patch.dict(OBJECTIVE_REGISTRY, {"checker_test": lambda results, reference, **kw: (results["score"], "ok")})
        registry.start()
        self.addCleanup(registry.stop)
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    def objective(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return optimize.make_objective(self.config, str(self.model), str(self.results))[0]

    def write_runner(self, body):
        # stdlib-only model stand-in: exercise real subprocess exit/log/output handling.
        self.model.write_text(REFERENCE + "\nimport sys, pathlib, pickle\n"
                             "output = pathlib.Path(sys.argv[sys.argv.index('--result-dir') + 1])\n"
                             + body, encoding="utf-8")

    def test_invalid_constants_stop_before_any_trial_and_do_not_write(self):
        self.kernel.write_text("const int N_SPECIES = 99;\n", encoding="utf-8")
        before = self.kernel.read_bytes()
        with patch.object(optimize, "run_trial_subprocess") as run, self.assertRaisesRegex(optimize.OptimizationError, "expected 3"):
            self.objective()
        run.assert_not_called()
        self.assertEqual(list(self.results.iterdir()), [])
        self.assertEqual(self.kernel.read_bytes(), before)

    def test_preflight_selects_only_active_package_or_flat_variant(self):
        active = self.root / "variants" / "active"
        active.mkdir(parents=True)
        (active / "__init__.py").write_text("")
        (active / "kernel.cpp").write_text("N_SPECIES = 3\n")
        flat = self.root / "variants" / "legacy.py"
        flat.write_text("N_SPECIES = 99\n")
        with contextlib.redirect_stdout(io.StringIO()):
            optimize.check_model_preflight(str(self.model), "active")
            optimize.check_model_preflight(str(self.model))
        with self.assertRaisesRegex(optimize.OptimizationError, "legacy.py"):
            optimize.check_model_preflight(str(self.model), "legacy")
        flat.write_text("N_SPECIES = 3\n")
        (active / "kernel.cpp").write_text("N_SPECIES = 99\n")
        with contextlib.redirect_stdout(io.StringIO()):
            optimize.check_model_preflight(str(self.model), "legacy")
        with self.assertRaisesRegex(optimize.OptimizationError, "kernel.cpp"):
            optimize.check_model_preflight(str(self.model), "active")

    def test_invalid_variant_and_reference_fail_closed(self):
        for variant in ("missing", "../active"):
            with self.subTest(variant=variant), self.assertRaises(optimize.OptimizationError):
                optimize.check_model_preflight(str(self.model), variant)
        self.model.write_text("# no reference dimensions\n")
        with self.assertRaisesRegex(optimize.OptimizationError, "literal assignment"):
            optimize.check_model_preflight(str(self.model))

    def test_structural_search_parameters_and_missing_pickle_setting_rejected(self):
        for name in ("N", "N_SPECIES", "ECM_AGENTS_PER_DIR[0]", "BOUNDARY_COORDS", "BOUNDARY_COORDS[0]"):
            self.config["parameters"] = {name: {"type": "int", "low": 1, "high": 2}}
            with self.subTest(name=name), self.assertRaisesRegex(optimize.OptimizationError, "structural dimensions"):
                self.objective()
        self.config["parameters"] = {}
        self.config["model"] = {"extra_overrides": {"SAVE_PICKLE": False}}
        with self.assertRaisesRegex(optimize.OptimizationError, "SAVE_PICKLE"):
            self.objective()

    def test_failed_model_stops_study_after_first_failure_with_both_logs(self):
        self.write_runner("print('invalid dimension details', flush=True)\nsys.exit('bad model configuration')\n")
        objective = self.objective()
        study = optuna.create_study()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(optimize.OptimizationError) as error:
            study.optimize(objective, n_trials=5)
        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.FAIL)
        self.assertIn("invalid dimension details", str(error.exception))
        self.assertIn("bad model configuration", str(error.exception))
        self.assertTrue((self.results / "trial_00000" / "stderr.log").is_file())
        self.assertFalse((self.results / "trial_00001").exists())

    def test_late_model_failure_also_stops_remaining_trials(self):
        self.write_runner("pickle.dump({'score': 2.}, open(output / 'output_data_0.pickle', 'wb'))\n")
        objective = self.objective()
        study = optuna.create_study()
        def break_model(study, trial):
            self.write_runner("sys.exit('configuration changed during study')\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(optimize.OptimizationError, "configuration changed"):
            study.optimize(objective, n_trials=5, callbacks=[break_model])
        self.assertEqual([t.state for t in study.trials],
                         [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL])

    def test_successful_exit_without_pickle_stops_study(self):
        self.write_runner("print('no results produced')\n")
        objective = self.objective()
        study = optuna.create_study()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(optimize.OptimizationError, "SAVE_PICKLE"):
            study.optimize(objective, n_trials=5)
        self.assertEqual(len(study.trials), 1)

    def test_objective_error_stops_instead_of_pruning_repeatedly(self):
        self.write_runner("pickle.dump({}, open(output / 'output_data_0.pickle', 'wb'))\n")
        objective = self.objective()
        study = optuna.create_study()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(optimize.OptimizationError, "Objective 'checker_test'"):
            study.optimize(objective, n_trials=5)
        self.assertEqual(len(study.trials), 1)

    def test_timeout_remains_trial_local(self):
        objective = self.objective()
        study = optuna.create_study()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(optimize, "run_trial_subprocess", side_effect=subprocess.TimeoutExpired("model", 1)):
            study.optimize(objective, n_trials=2)
        self.assertEqual([t.state for t in study.trials], [optuna.trial.TrialState.PRUNED] * 2)

    def test_explicit_instability_prunes_then_next_trial_completes(self):
        shutil.copy2(ROOT / "simulation_errors.py", self.root)
        self.write_runner("from simulation_errors import reject_trial\n"
                          "if output.name == 'trial_00000': reject_trial('excessive cell displacement')\n"
                          "pickle.dump({'score': 2.}, open(output / 'output_data_0.pickle', 'wb'))\n")
        study = optuna.create_study()
        with contextlib.redirect_stdout(io.StringIO()):
            study.optimize(self.objective(), n_trials=2)
        self.assertEqual([t.state for t in study.trials],
                         [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.COMPLETE])
        self.assertIn("excessive cell displacement", study.trials[0].user_attrs["prune_reason"])

    def test_stale_rejection_cannot_mask_a_new_error(self):
        run = self.results / "trial_00000"
        run.mkdir()
        (run / "trial_rejection.json").write_text(json.dumps({"token": "old", "reason": "old instability"}))
        self.write_runner("sys.exit('new programming error')\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(optimize.OptimizationError, "new programming error"):
            optimize.run_trial_subprocess({}, str(self.model), str(run))

    def test_objective_can_explicitly_reject_a_parameter_combination(self):
        self.write_runner("pickle.dump({'score': 2.}, open(output / 'output_data_0.pickle', 'wb'))\n")
        def reject(results, reference, **kwargs):
            raise optimize.TrialRejected("population exceeds calibrated observable range")
        with patch.dict(OBJECTIVE_REGISTRY, {"checker_test": reject}):
            objective = self.objective()
        study = optuna.create_study()
        with contextlib.redirect_stdout(io.StringIO()):
            study.optimize(objective, n_trials=2)
        self.assertEqual([t.state for t in study.trials], [optuna.trial.TrialState.PRUNED] * 2)

    def test_valid_subprocesses_complete_multiple_trials_with_empty_overrides(self):
        self.write_runner("assert sys.stdin.read() == ''\npickle.dump({'score': 2.}, open(output / 'output_data_0.pickle', 'wb'))\n")
        objective = self.objective()
        study = optuna.create_study()
        with contextlib.redirect_stdout(io.StringIO()):
            study.optimize(objective, n_trials=2)
        self.assertEqual([t.state for t in study.trials], [optuna.trial.TrialState.COMPLETE] * 2)
        self.assertEqual(study.best_value, 2.)

    def test_cli_preflight_failure_creates_no_study_or_trials(self):
        self.kernel.write_text("N_SPECIES = 99\n")
        config = {**self.config, "objective": {"function": "final_cell_count_error"},
                  "study": {"storage": "sqlite:///must_not_create.db", "n_trials": 5}}
        path = self.root / "config.json"
        path.write_text(json.dumps(config))  # JSON is a YAML subset.
        for cwd, command in ((ROOT, ["-m", "optimizer.optimize"]),
                             (ROOT / "optimizer", ["optimize.py"])):
            with self.subTest(command=command):
                proc = subprocess.run([sys.executable, *command, "--config", str(path),
                                       "--model", str(self.model), "--result-dir", str(self.results)],
                                      cwd=cwd, capture_output=True, text=True, timeout=30)
                self.assertEqual(proc.returncode, 1)
                self.assertIn("No trials were launched", proc.stderr)
                self.assertFalse((cwd / "must_not_create.db").exists())
                self.assertEqual(list(self.results.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
