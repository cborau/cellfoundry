# check_hard_coded_varaibles.py
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

Number = Union[int, float]

EXCLUDED_DIRS_DEFAULT = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

EXCLUDED_FILE_BASENAMES_RE = re.compile(
    r"^check_hard_coded_.*\.py$", re.IGNORECASE
)

DEFAULT_EXTS = {".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".hxx", ".cu", ".cuh"}


@dataclass(frozen=True)
class ReferenceValues:
    model_path: str
    n: int
    n_species: int
    max_connectivity: int
    n_anchor_points: int
    boundary_coords: List[Number]
    ecm_agents_per_dir: List[int]
    ecm_population_size: int
    is_cubical: bool


@dataclass(frozen=True)
class Mismatch:
    file_path: str
    line: int
    var_name: str
    found_value: int
    expected_value: int
    line_text: str


# -----------------------------
# Reference parsing (AST, Python only)
# -----------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _safe_parse_py(path: str) -> ast.AST:
    return ast.parse(_read_text(path), filename=path)


def _extract_literal_assignments(tree: ast.AST) -> Dict[str, object]:
    out: Dict[str, object] = {}

    def is_num(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return isinstance(node.operand.value, (int, float))
        return False

    def get_num(node: ast.AST) -> Number:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            v = node.operand.value
            if isinstance(v, (int, float)):
                return -v
        raise ValueError("Not a numeric literal")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue

        name = tgt.id
        val = node.value

        if isinstance(val, ast.Constant):
            out[name] = val.value
            continue

        if isinstance(val, ast.UnaryOp) and isinstance(val.op, ast.USub) and isinstance(val.operand, ast.Constant):
            v = val.operand.value
            if isinstance(v, (int, float)):
                out[name] = -v
            continue

        if isinstance(val, (ast.List, ast.Tuple)) and all(is_num(e) for e in val.elts):
            out[name] = [get_num(e) for e in val.elts]

    return out


def load_reference_values(model_path: str) -> ReferenceValues:
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Reference file not found: {model_path}")

    tree = _safe_parse_py(model_path)
    assigns = _extract_literal_assignments(tree)

    def need_int(name: str) -> int:
        if name not in assigns:
            raise RuntimeError(f"Could not find literal assignment {name} = <int> in {model_path}")
        v = assigns[name]
        if not isinstance(v, int):
            raise RuntimeError(f"{name} must be an integer literal in {model_path}, found: {v!r}")
        return v

    def need_boundary() -> List[Number]:
        if "BOUNDARY_COORDS" not in assigns:
            raise RuntimeError(f"Could not find literal assignment BOUNDARY_COORDS = [..] in {model_path}")
        v = assigns["BOUNDARY_COORDS"]
        if not isinstance(v, list) or len(v) != 6 or not all(isinstance(x, (int, float)) for x in v):
            raise RuntimeError(
                f"BOUNDARY_COORDS must be a list of 6 numeric literals in {model_path}, found: {v!r}"
            )
        return v

    N = need_int("N")
    N_SPECIES = need_int("N_SPECIES")
    MAX_CONNECTIVITY = need_int("MAX_CONNECTIVITY")
    N_ANCHOR_POINTS = need_int("N_ANCHOR_POINTS")
    BOUNDARY_COORDS = need_boundary()

    diff_x = abs(BOUNDARY_COORDS[0] - BOUNDARY_COORDS[1])
    diff_y = abs(BOUNDARY_COORDS[2] - BOUNDARY_COORDS[3])
    diff_z = abs(BOUNDARY_COORDS[4] - BOUNDARY_COORDS[5])

    if diff_x == diff_y == diff_z:
        ECM_AGENTS_PER_DIR = [N, N, N]
        is_cubical = True
        print("The domain is cubical.")
    else:
        is_cubical = False
        print("The domain is not cubical.")
        min_length = min(diff_x, diff_y, diff_z)
        if N <= 1:
            raise RuntimeError("N must be >= 2 for dist_agents = min_length / (N - 1).")
        dist_agents = min_length / (N - 1)

        ECM_AGENTS_PER_DIR = [
            int(diff_x / dist_agents) + 1,
            int(diff_y / dist_agents) + 1,
            int(diff_z / dist_agents) + 1,
        ]

        diff_x = dist_agents * (ECM_AGENTS_PER_DIR[0] - 1)
        diff_y = dist_agents * (ECM_AGENTS_PER_DIR[1] - 1)
        diff_z = dist_agents * (ECM_AGENTS_PER_DIR[2] - 1)
        BOUNDARY_COORDS = [
            round(diff_x / 2, 2), -round(diff_x / 2, 2),
            round(diff_y / 2, 2), -round(diff_y / 2, 2),
            round(diff_z / 2, 2), -round(diff_z / 2, 2),
        ]

    ECM_POPULATION_SIZE = ECM_AGENTS_PER_DIR[0] * ECM_AGENTS_PER_DIR[1] * ECM_AGENTS_PER_DIR[2]

    return ReferenceValues(
        model_path=model_path,
        n=N,
        n_species=N_SPECIES,
        max_connectivity=MAX_CONNECTIVITY,
        n_anchor_points=N_ANCHOR_POINTS,
        boundary_coords=BOUNDARY_COORDS,
        ecm_agents_per_dir=ECM_AGENTS_PER_DIR,
        ecm_population_size=ECM_POPULATION_SIZE,
        is_cubical=is_cubical,
    )


# -----------------------------
# Plain-text scanning (all files)
# -----------------------------

def iter_project_files(root: str, exts: set, excluded_dirs: set) -> List[str]:
    root = os.path.abspath(root)
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        for fn in filenames:
            _, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                files.append(os.path.join(dirpath, fn))
    return files


def scan_plain_text_assignments(path: str, var_name: str):
    """
    Plain-text scan for hard-coded integer assignments in mixed Python/C++ files.

    Matches:
      N_SPECIES = 6
      const uint8_t N_SPECIES = 3;
      static constexpr std::uint32_t ECM_POPULATION_SIZE = 125; // comment
      #define N_SPECIES 3

    Ignores:
      N_SPECIES = something
      N_SPECIES = 6 + 1
      N_SPECIES = foo(3)
    """
    # Capture the integer literal as GROUP 1 (positional), not a named group.
    assign_pat = re.compile(
        rf"""^\s*[^=]*\b{re.escape(var_name)}\b[^=]*=\s*(-?\d+)\s*;?\s*$"""
    )
    define_pat = re.compile(
        rf"""^\s*#\s*define\s+{re.escape(var_name)}\s+(-?\d+)\s*$"""
    )

    def strip_comments(line: str) -> str:
        line = re.sub(r"//.*$", "", line)          # remove // comments
        line = re.sub(r"/\*.*?\*/", "", line)      # remove inline /* ... */ (single-line)
        return line

    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, raw in enumerate(f, start=1):
            line = strip_comments(raw).strip()
            if not line:
                continue

            m = assign_pat.match(line)
            if m:
                out.append((i, int(m.group(1)), raw.rstrip("\n")))
                continue

            m = define_pat.match(line)
            if m:
                out.append((i, int(m.group(1)), raw.rstrip("\n")))
                continue

    return out




def find_mismatches(
    scan_root: str,
    expected: Dict[str, int],
    excluded_files_abs: set,
    exts: set,
    excluded_dirs: set,
) -> List[Mismatch]:
    scan_root = os.path.abspath(scan_root)

    mismatches: List[Mismatch] = []
    for path in iter_project_files(scan_root, exts=exts, excluded_dirs=excluded_dirs):
        abs_path = os.path.abspath(path)
        base = os.path.basename(abs_path)

        # Exclude by exact path and also by filename pattern (robust on Windows)
        if abs_path in excluded_files_abs:
            continue
        if EXCLUDED_FILE_BASENAMES_RE.match(base):
            continue

        for var, exp_val in expected.items():
            for line_no, found_val, line_text in scan_plain_text_assignments(path, var):
                if found_val != exp_val:
                    mismatches.append(
                        Mismatch(
                            file_path=path,
                            line=line_no,
                            var_name=var,
                            found_value=found_val,
                            expected_value=exp_val,
                            line_text=line_text,
                        )
                    )
    return mismatches

def apply_fixes(mismatches: List[Mismatch]) -> None:
    """
    Apply in-place fixes for mismatched integer assignments.
    Only replaces the numeric literal, preserving the rest of the line.
    """
    # Group mismatches per file
    by_file: Dict[str, List[Mismatch]] = {}
    for mm in mismatches:
        by_file.setdefault(mm.file_path, []).append(mm)

    for path, items in by_file.items():
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for mm in items:
            idx = mm.line - 1
            original = lines[idx]

            # Replace only the integer literal after "="
            assign_pattern = re.compile(
                rf"(\b{re.escape(mm.var_name)}\b[^=]*=\s*)(-?\d+)"
            )
            define_pattern = re.compile(
                rf"(^\s*#\s*define\s+{re.escape(mm.var_name)}\s+)(-?\d+)"
            )

            if assign_pattern.search(original):
                lines[idx] = assign_pattern.sub(
                    rf"\g<1>{mm.expected_value}",
                    original
                )
            elif define_pattern.search(original):
                lines[idx] = define_pattern.sub(
                    rf"\g<1>{mm.expected_value}",
                    original
                )

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    print("Files successfully updated.")

# -----------------------------
# CLI / Output
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default="model.py")
    parser.add_argument(
        "--scan-root",
        default=None,
        help="Directory to scan. Default: the directory containing the model file.",
    )
    parser.add_argument(
        "--exts",
        default=",".join(sorted(DEFAULT_EXTS)),
        help="Comma-separated extensions to scan.",
    )
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args(argv)

    model_file = os.path.abspath(args.model_file)
    checker_file = os.path.abspath(__file__)

    print(f"Reading reference from: {model_file}")

    try:
        ref = load_reference_values(model_file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"N = {ref.n}")
    print(f"N_SPECIES = {ref.n_species}")
    print(f"MAX_CONNECTIVITY = {ref.max_connectivity}")
    print(f"N_ANCHOR_POINTS = {ref.n_anchor_points}")
    print(f"ECM_AGENTS_PER_DIR = {ref.ecm_agents_per_dir}")
    print(f"ECM_POPULATION_SIZE = {ref.ecm_population_size}")
    print("")

    # Default scan root: SAME folder as model.py (not parent)
    scan_root = os.path.abspath(args.scan_root) if args.scan_root else os.path.dirname(ref.model_path)

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    expected = {
        "N_SPECIES": ref.n_species,
        "ECM_POPULATION_SIZE": ref.ecm_population_size,
        "MAX_CONNECTIVITY": ref.max_connectivity,
        "N_ANCHOR_POINTS": ref.n_anchor_points,
    }

    # Exclude:
    # - model.py itself
    # - the checker script itself
    excluded_files_abs = {os.path.abspath(ref.model_path), checker_file}

    print(f"Scanning (plain text) under: {scan_root}")
    print(f"Extensions: {', '.join(sorted(exts))}")
    print("")

    mismatches = find_mismatches(
        scan_root=scan_root,
        expected=expected,
        excluded_files_abs=excluded_files_abs,
        exts=exts,
        excluded_dirs=EXCLUDED_DIRS_DEFAULT,
    )

    if not mismatches:
        print("No mismatches found (hard-coded integer assignments only).")
        return 0

    print("Mismatches found:")
    for mm in sorted(mismatches, key=lambda x: (x.file_path, x.line, x.var_name)):
        rel = os.path.relpath(mm.file_path, scan_root)
        print(f"  - {rel}:{mm.line}: {mm.var_name} = {mm.found_value} (expected {mm.expected_value})")
        print(f"    {mm.line_text}")

    # Collect unique expected values summary
    unique_expected = {}
    for mm in mismatches:
        unique_expected[mm.var_name] = mm.expected_value

    print("\nExpected values summary:")
    for var, val in unique_expected.items():
        print(f"  {var} = {val}")

    answer = input(
        "\nDo you want to automatically update the variables to their expected values "
        f"{list(unique_expected.items())}? [y/N]: "
    ).strip().lower()

    if answer in {"y", "yes"}:
        apply_fixes(mismatches)
        return 0
    else:
        print("No changes applied.")
        return 2 
   

if __name__ == "__main__":
    raise SystemExit(main())
