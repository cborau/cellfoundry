"""Check scan boundaries and automatic fixes without running FLAMEGPU."""

import ast
import contextlib
import io
from pathlib import Path
import sys
import tempfile
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
                "--fail-on-mismatch", *args,
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

    def test_exlude_variants_skips_entire_tree(self):
        self.run_checker("--exlude-variants")
        self.assert_fixed_only("kernel.cpp", "helpers/kernel.cpp")

    def test_exclude_variants_alias_skips_entire_tree(self):
        self.run_checker("--exclude-variants")
        self.assert_fixed_only("kernel.cpp", "helpers/kernel.cpp")

    def test_variant_exclusion_combines_with_other_exclusions(self):
        self.run_checker("--exlude-variants", "--exclude-dir", "helpers")
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
                    }
                    with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input", return_value="y") as prompt:
                        exec(code, namespace)
                    if quiet:
                        prompt.assert_not_called()
                    else:
                        prompt.assert_called_once()
                    self.assertEqual(namespace["hard_coded_check_exit_code"], 0)
                    self.assertFalse(namespace["critical_error"])
                    self.assert_fixed_only("kernel.cpp", *variant_files)


if __name__ == "__main__":
    unittest.main()
