"""CLI checks, including --help without any third-party packages installed."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from model_cli import parse_model_args
from helper_module import load_param_overrides_from_cli


ROOT = Path(__file__).resolve().parents[1]


class ModelCliTests(unittest.TestCase):
    def test_defaults_and_all_options(self):
        self.assertEqual(vars(parse_model_args([])),
                         {"variant": None, "overrides": None, "result_dir": None})
        args = parse_model_args(["--variant", "radial_glia", "--overrides", "params.json",
                                 "--result-dir", "results/my run"])
        self.assertEqual(vars(args), {"variant": "radial_glia", "overrides": "params.json",
                                     "result_dir": "results/my run"})

    def test_help_and_errors_exit_before_model_imports_or_writes(self):
        # Only the entry point and its stdlib parser are available. -S disables
        # site-packages, so a FLAMEGPU/numpy import would fail this test.
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            for name in ("model.py", "model_cli.py"):
                shutil.copy2(ROOT / name, destination / name)
            for arguments, exit_code in ((["--help"], 0), (["-h"], 0),
                                         (["--variant"], 2), (["--overrides"], 2),
                                         (["--result-dir"], 2), (["--steps", "1"], 2)):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [sys.executable, "-B", "-S", "model.py", *arguments],
                        cwd=destination, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, exit_code, result.stderr)
                    if exit_code == 0:
                        for flag in ("--variant", "--overrides", "--result-dir", "--help"):
                            self.assertIn(flag, result.stdout)
                        self.assertIn("does not resize the fixed ECM grid", result.stdout)
                    else:
                        self.assertIn("error:", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual({p.name for p in destination.iterdir()},
                                     {"model.py", "model_cli.py"})

    def test_unknown_and_abbreviated_flags_are_rejected(self):
        for args in (["--results-dir", "out"], ["--vari", "radial_glia"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_model_args(args)
                self.assertEqual(error.exception.code, 2)

    def test_json_loading_supports_saved_argv_and_preparsed_args(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "my overrides.json"
            values = {"STEPS": 2, "BOUNDARY_COORDS": [500, -500, 500, -500, 25, -25]}
            path.write_text(json.dumps(values), encoding="utf-8")
            args = ["--variant=radial_glia", f"--overrides={path}", "--result-dir=results/my run"]
            expected = (values, "results/my run")
            self.assertEqual(load_param_overrides_from_cli(["model.py", *args]), expected)
            self.assertEqual(load_param_overrides_from_cli(parsed_args=parse_model_args(args)), expected)

    def test_missing_or_non_object_json_reports_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.json"
            args = parse_model_args(["--overrides", str(path)])
            with self.assertRaises(FileNotFoundError):
                load_param_overrides_from_cli(parsed_args=args)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "object mapping parameter names"):
                load_param_overrides_from_cli(parsed_args=args)


if __name__ == "__main__":
    unittest.main()
