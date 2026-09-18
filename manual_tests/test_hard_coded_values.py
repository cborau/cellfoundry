"""Check scan boundaries and automatic fixes without running FLAMEGPU."""

import ast
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
import check_hard_coded_values as checker


class HardCodedScanScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "model.py").write_text(
            "N = 2\nN_SPECIES = 3\nN_CELL_TYPES = 1\n"
            "MAX_CONNECTIVITY = 8\nN_ANCHOR_POINTS = 2\n"
            "MAX_VASC_CONNECTIVITY = 4\n"
            "BOUNDARY_COORDS = [1, -1, 1, -1, 1, -1]\n",
            encoding="utf-8",
        )
        self.files = [
            "kernel.cpp",
            "helpers/kernel.cpp",
            "variants/radial_glia/kernel.cpp",
            "variants/other/nested/kernel.cpp",
            "variants/legacy.py",
        ]
        for name in self.files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("N_SPECIES = 9\n", encoding="utf-8")

    def run_checker(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            status = checker.main([
                "--model-file", str(self.root / "model.py"),
                "--fix", *args,
            ])
        self.assertEqual(status, 0)

    def assert_fixed_only(self, *fixed):
        for name in self.files:
            with self.subTest(file=name):
                expected = "N_SPECIES = 3\n" if name in fixed else "N_SPECIES = 9\n"
                self.assertEqual((self.root / name).read_text(encoding="utf-8"), expected)

    def test_default_manual_scan_includes_subdirectories(self):
        self.run_checker()
        self.assert_fixed_only(*self.files)

    def test_exclude_variants_skips_entire_tree(self):
        self.run_checker("--exclude-variants")
        self.assert_fixed_only("kernel.cpp", "helpers/kernel.cpp")

    def test_exclude_variants_alias_skips_entire_tree(self):
        self.run_checker("--exclude-variants")
        self.assert_fixed_only("kernel.cpp", "helpers/kernel.cpp")

    def test_variant_exclusion_combines_with_other_exclusions(self):
        self.run_checker("--exclude-variants", "--exclude-dir", "helpers")
        self.assert_fixed_only("kernel.cpp")

    def test_relative_exclusion_preserves_other_variants(self):
        self.run_checker("--exclude-dir", "variants/radial_glia")
        self.assert_fixed_only(
            "kernel.cpp", "helpers/kernel.cpp",
            "variants/other/nested/kernel.cpp", "variants/legacy.py",
        )

    def test_nonrecursive_scan_only_fixes_root_files(self):
        self.run_checker("--no-recursive")
        self.assert_fixed_only("kernel.cpp")

    def test_multiple_scan_roots_fix_only_selected_directories(self):
        self.run_checker(
            "--scan-root", str(self.root),
            "--scan-root", str(self.root / "variants" / "radial_glia"),
            "--no-recursive",
        )
        self.assert_fixed_only("kernel.cpp", "variants/radial_glia/kernel.cpp")

    def test_file_scan_root_does_not_scan_siblings(self):
        self.run_checker("--scan-root", str(self.root / "variants" / "legacy.py"))
        self.assert_fixed_only("variants/legacy.py")

    def test_overlapping_scan_roots_do_not_duplicate_fixes(self):
        with patch.object(checker, "apply_fixes", wraps=checker.apply_fixes) as apply_fixes:
            self.run_checker(
                "--scan-root", str(self.root),
                "--scan-root", str(self.root / "variants" / "radial_glia"),
            )
        self.assertEqual(len(apply_fixes.call_args.args[0]), len(self.files))
        self.assert_fixed_only(*self.files)

    def test_read_only_failure_never_prompts_or_writes(self):
        before = {name: (self.root / name).read_bytes() for name in self.files}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch("builtins.input") as prompt:
            status = checker.main(["--model-file", str(self.root / "model.py"), "--fail-on-mismatch"])
        self.assertEqual(status, 2)
        prompt.assert_not_called()
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in self.files})

    def test_read_only_success_never_prompts_or_writes(self):
        self.run_checker()
        with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input") as prompt, patch.object(checker, "apply_fixes") as fix:
            status = checker.main(["--model-file", str(self.root / "model.py"), "--fail-on-mismatch"])
        self.assertEqual(status, 0)
        prompt.assert_not_called()
        fix.assert_not_called()

    def test_interactive_yes_repairs_and_no_preserves_files(self):
        for answer, expected in (("n", 2), ("y", 0)):
            with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input", return_value=answer) as prompt:
                status = checker.main(["--model-file", str(self.root / "model.py")])
            prompt.assert_called_once()
            self.assertEqual(status, expected)
            self.assert_fixed_only(*(self.files if answer == "y" else []))

    def test_missing_stdin_fails_without_writing(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch("builtins.input", side_effect=EOFError):
            status = checker.main(["--model-file", str(self.root / "model.py")])
        self.assertEqual(status, 2)
        self.assert_fixed_only()

    def test_fix_and_check_only_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            checker.main(["--fix", "--fail-on-mismatch"])
        self.assertEqual(error.exception.code, 2)

    def test_missing_scan_root_is_an_error(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = checker.main(["--model-file", str(self.root / "model.py"),
                                   "--scan-root", str(self.root / "missing"), "--fail-on-mismatch"])
        self.assertEqual(status, 1)

    def test_scan_io_error_does_not_pass_or_write(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch.object(checker, "find_mismatches", side_effect=PermissionError("locked")):
            status = checker.main(["--model-file", str(self.root / "model.py"), "--fail-on-mismatch"])
        self.assertEqual(status, 1)
        self.assert_fixed_only()

    def test_invalid_reference_is_an_error(self):
        path = self.root / "model.py"
        original = path.read_text()
        for source in ("N = 2\n", original.replace("N = 2", "N = 1"),
                       original.replace("[1, -1, 1, -1, 1, -1]", "[0, 0, 0, 0, 0, 0]")):
            path.write_text(source)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = checker.main(["--model-file", str(path), "--fail-on-mismatch"])
            self.assertEqual(status, 1)
            self.assert_fixed_only()

    def test_empty_json_still_selects_noninteractive_mode(self):
        tree = ast.parse((REPOSITORY_ROOT / "model.py").read_text(encoding="utf-8"))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "_OPTUNA_QUIET" for t in node.targets))
        scope = {"_CLI_ARGS": SimpleNamespace(overrides="empty.json"), "_PARAM_OVERRIDES": {}}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "model.py", "exec"), scope)
        self.assertTrue(scope["_OPTUNA_QUIET"])

    def test_model_rejects_changed_bounds_and_dimensions_before_assay_setup(self):
        tree = ast.parse((REPOSITORY_ROOT / "model.py").read_text(encoding="utf-8"))
        guard = next(node for node in tree.body if isinstance(node, ast.For)
                     and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)
                     and isinstance(node.iter.func.value, ast.Name)
                     and node.iter.func.value.id == "_CORE_STRUCTURAL_SETTINGS")
        code = compile(ast.Module(body=[guard], type_ignores=[]), "model.py", "exec")
        for name, core, changed in (("N", 6, 7), ("N_SPECIES", 3, 4),
                                    ("BOUNDARY_COORDS", [50., -50.] * 3, [100., -100.] * 3)):
            scope = {"sys": sys, "_CORE_STRUCTURAL_SETTINGS": {name: core}, name: core}
            exec(code, scope)  # Equal declarations are harmless.
            scope[name] = changed
            with self.subTest(name=name), self.assertRaises(SystemExit) as error:
                exec(code, scope)
            self.assertIn(name, str(error.exception))
        derived_setup = next(node for node in tree.body if isinstance(node, ast.If)
                             and isinstance(node.test, ast.UnaryOp)
                             and isinstance(node.test.operand, ast.Name)
                             and node.test.operand.id == "OSCILLATORY_SHEAR_ASSAY")
        self.assertLess(guard.lineno, derived_setup.lineno)

    def test_model_invocation_scans_base_and_active_variant_during_optimization(self):
        # Execute the real checker invocation without importing the GPU model.
        model_path = REPOSITORY_ROOT / "model.py"
        tree = ast.parse(model_path.read_text(encoding="utf-8"))
        check_block = next(
            node for node in tree.body
            if isinstance(node, ast.Try) and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "check_hard_coded_values"
                and child.func.attr == "main"
                for child in ast.walk(node)
            )
        )
        code = compile(ast.Module(body=[check_block], type_ignores=[]), str(model_path), "exec")
        cases = [
            (None, []),
            ("variants/radial_glia/__init__.py", ["variants/radial_glia/kernel.cpp"]),
            ("variants/legacy.py", ["variants/legacy.py"]),
        ]
        for variant, variant_files in cases:
            for quiet in (False, True):
                with self.subTest(variant=variant, quiet=quiet):
                    for name in self.files:
                        (self.root / name).write_text("N_SPECIES = 9\n", encoding="utf-8")
                    namespace = {
                        "CURR_PATH": self.root,
                        "_ACTIVE_VARIANT": object() if variant else None,
                        "_variant_path": self.root / variant if variant else None,
                        "_OPTUNA_QUIET": quiet,
                        "check_hard_coded_values": checker,
                        "critical_error": False,
                        "sys": sys,
                    }
                    with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input", return_value="y") as prompt:
                        if quiet:
                            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                                exec(code, namespace)
                            self.assertNotEqual(error.exception.code, 0)
                        else:
                            exec(code, namespace)
                    if quiet:
                        prompt.assert_not_called()
                    else:
                        prompt.assert_called_once()
                    self.assertEqual(namespace["hard_coded_check_exit_code"], 2 if quiet else 0)
                    self.assertFalse(namespace["critical_error"])
                    self.assert_fixed_only(*([] if quiet else ["kernel.cpp", *variant_files]))


if __name__ == "__main__":
    unittest.main()
