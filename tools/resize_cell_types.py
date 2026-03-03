#!/usr/bin/env python
"""
resize_cell_types.py  –  Expand (or shrink) per-cell-type arrays to a new N_CELL_TYPES.

Usage
-----
    python tools/resize_cell_types.py <new_N>

What it does
------------
1. **model.py**
   - Updates the ``N_CELL_TYPES = <old>`` declaration to the new value.
   - For every ``newPropertyArrayFloat("<NAME>", <VAR>)`` registration whose
     Python variable is a per-type list, it locates the corresponding
     declaration(s) in the file and resizes them:
       * Extending → appends copies of the *last* element.
       * Shrinking → truncates from the right.
   - Also resizes list literals of the form ``[val] * N_CELL_TYPES`` and
     ``[expr] * N_CELL_TYPES`` by simply rewriting N_CELL_TYPES (which
     automatically adjusts them).

2. **C++ agent functions (*.cpp)**
   - Updates every ``const uint8_t N_CELL_TYPES = <old>;`` line.

Safety
------
- Creates timestamped backups of every file it modifies
  (``<file>.bak.<timestamp>``).
- Dry-run mode (``--dry-run``) prints changes without writing.
- Rejects a new N < 1.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # workspace root
MODEL_PY = ROOT / "model.py"
CPP_GLOB = "*.cpp"

# Pattern that finds  N_CELL_TYPES = <int>  in model.py
RE_NCELL_PY = re.compile(r"^(N_CELL_TYPES\s*=\s*)(\d+)", re.MULTILINE)

# Pattern that finds  const uint8_t N_CELL_TYPES = <int>;  in C++ files
RE_NCELL_CPP = re.compile(
    r"(const\s+uint8_t\s+N_CELL_TYPES\s*=\s*)(\d+)(\s*;)"
)

# Pattern that finds env registration:  newPropertyArrayFloat("NAME", VAR)
RE_ENV_REG = re.compile(
    r'newPropertyArrayFloat\(\s*"([^"]+)"\s*,\s*([A-Za-z_]\w*)\s*\)'
)

# A Python list literal like  [1.0, 2.0, 3.0]  or  [True, False, True]
# Non-greedy (.+?) so that comments containing brackets (e.g.  # [nN/um])
# do not swallow the real list content.
RE_EXPLICIT_LIST = re.compile(
    r"^(\s*[A-Za-z_]\w*\s*=\s*)\[(.+?)\](\s*(?:#.*)?)$"
)

# A replicated-list like  [expr] * N_CELL_TYPES  (or * 3)
RE_REPLICATED_LIST = re.compile(
    r"^(\s*[A-Za-z_]\w*\s*=\s*\[.+\]\s*\*\s*)(?:N_CELL_TYPES|\d+)(\s*(?:#.*)?)$"
)

# A list-comprehension like  [0.2 * k for k in CELL_K_ELAST]
RE_LIST_COMP = re.compile(
    r"^(\s*[A-Za-z_]\w*\s*=\s*\[.+\bfor\b.+\])(\s*(?:#.*)?)$"
)


def _backup(path: Path, dry: bool) -> None:
    """Create a timestamped backup."""
    if dry:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, dst)


def _detect_old_N(text: str) -> int:
    """Read current N_CELL_TYPES from model.py content."""
    m = RE_NCELL_PY.search(text)
    if not m:
        sys.exit("ERROR: could not find  N_CELL_TYPES = <int>  in model.py")
    return int(m.group(2))


def _resize_list(elements: list, new_n: int) -> list:
    """Extend (repeat last) or truncate a list to length new_n.

    If the last element contains subscript references like ``VAR[idx]``
    where *idx* equals its 0-based position, new elements are generated
    with incremented indices (e.g. ``VAR[3]``, ``VAR[4]``) rather than
    blindly copying the last element.
    """
    if len(elements) == 0:
        return elements
    if len(elements) >= new_n:
        return elements[:new_n]

    last = elements[-1]
    last_idx = len(elements) - 1
    idx_tag = f"[{last_idx}]"

    if idx_tag in last:
        # Index-based padding: increment subscripts for new positions
        result = list(elements)
        for j in range(len(elements), new_n):
            result.append(last.replace(idx_tag, f"[{j}]"))
        return result

    # Simple pad: repeat last element
    return elements + [last] * (new_n - len(elements))


def _parse_list_elements(raw: str) -> list[str] | None:
    """Try to split the inside of a Python list literal into element strings.

    Returns None if the content looks too complex (nested brackets, etc.).
    """
    # Quick bracket-depth safety check
    depth = 0
    for ch in raw:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth < 0:
            return None
    if depth != 0:
        return None
    # Split on commas that are at depth 0
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in raw:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts if parts else None


# ------------------------------------------------------------------
# Main resize logic for model.py
# ------------------------------------------------------------------

def resize_model_py(new_n: int, dry: bool) -> list[str]:
    """Resize all per-type declarations in model.py.  Returns change log."""
    text = MODEL_PY.read_text(encoding="utf-8")
    old_n = _detect_old_N(text)

    if old_n == new_n:
        return [f"model.py: N_CELL_TYPES already {new_n}, nothing to do."]

    log: list[str] = []
    lines = text.splitlines(keepends=True)

    # 1. Collect variable names that are registered as per-type arrays.
    per_type_vars: set[str] = set()
    for m in RE_ENV_REG.finditer(text):
        per_type_vars.add(m.group(2))   # the Python variable name

    log.append(
        f"model.py: {old_n} → {new_n}  ({len(per_type_vars)} per-type vars detected)"
    )

    # 2. Walk every line and rewrite where necessary.
    new_lines: list[str] = []
    for i, line in enumerate(lines):
        lineno = i + 1
        stripped = line.rstrip("\n\r")

        # --- 2a. N_CELL_TYPES = <old> ---
        m_nct = RE_NCELL_PY.match(stripped)
        if m_nct:
            new_line = f"{m_nct.group(1)}{new_n}\n"
            new_lines.append(new_line)
            log.append(f"  L{lineno}: N_CELL_TYPES = {old_n} → {new_n}")
            continue

        # Determine the LHS variable name (if assignment)
        eq_pos = stripped.find("=")
        if eq_pos > 0 and stripped[eq_pos - 1] not in "!<>=":
            lhs = stripped[:eq_pos].strip()
        else:
            lhs = None

        if lhs and lhs not in per_type_vars:
            # Not a per-type variable: keep as-is
            new_lines.append(line)
            continue

        # --- 2b. [expr] * N_CELL_TYPES  or  [expr] * <old_n> ---
        m_rep = RE_REPLICATED_LIST.match(stripped)
        if m_rep and lhs in per_type_vars:
            # Rewrite the multiplier to N_CELL_TYPES (which was already
            # updated in step 2a, but the literal reference is fine).
            new_line = f"{m_rep.group(1)}N_CELL_TYPES{m_rep.group(2)}\n"
            if new_line != line:
                log.append(f"  L{lineno}: replicated-list → * N_CELL_TYPES")
            new_lines.append(new_line)
            continue

        # --- 2c. Explicit list  [a, b, c] ---
        m_expl = RE_EXPLICIT_LIST.match(stripped)
        if m_expl and lhs in per_type_vars:
            elems = _parse_list_elements(m_expl.group(2))
            if elems is not None and len(elems) == old_n:
                resized = _resize_list(elems, new_n)
                inner = ", ".join(resized)
                new_line = f"{m_expl.group(1)}[{inner}]{m_expl.group(3)}\n"
                log.append(
                    f"  L{lineno}: {lhs} list {old_n} → {new_n} elems"
                )
                new_lines.append(new_line)
                continue

        # --- 2d. List comprehension  [f(x) for x in OTHER_LIST] ---
        # These derive their length from another per-type list, so they
        # automatically resize once the source list is resized.  No action.

        # --- default: keep line unchanged ---
        new_lines.append(line)

    result = "".join(new_lines)

    if not dry:
        _backup(MODEL_PY, dry)
        MODEL_PY.write_text(result, encoding="utf-8")
        log.append("  ✓ model.py written")
    else:
        log.append("  (dry-run – no file written)")

    return log


# ------------------------------------------------------------------
# C++ files
# ------------------------------------------------------------------

def resize_cpp_files(new_n: int, dry: bool) -> list[str]:
    """Update ``const uint8_t N_CELL_TYPES = <old>;`` in every .cpp file."""
    log: list[str] = []
    for cpp in sorted(ROOT.glob(CPP_GLOB)):
        text = cpp.read_text(encoding="utf-8")
        new_text, count = RE_NCELL_CPP.subn(rf"\g<1>{new_n}\3", text)
        if count:
            log.append(f"{cpp.name}: {count} N_CELL_TYPES occurrence(s) → {new_n}")
            if not dry:
                _backup(cpp, dry)
                cpp.write_text(new_text, encoding="utf-8")
        # else: no match → skip silently
    if not log:
        log.append("(no C++ files needed updating)")
    return log


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize per-cell-type arrays to a new N_CELL_TYPES."
    )
    parser.add_argument(
        "new_n",
        type=int,
        help="New value for N_CELL_TYPES (must be >= 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without modifying any files.",
    )
    args = parser.parse_args()

    if args.new_n < 1:
        sys.exit("ERROR: N_CELL_TYPES must be >= 1")

    print(f"{'DRY RUN: ' if args.dry_run else ''}Resizing N_CELL_TYPES → {args.new_n}\n")

    # 1. model.py
    for msg in resize_model_py(args.new_n, args.dry_run):
        print(msg)

    print()

    # 2. C++ files
    for msg in resize_cpp_files(args.new_n, args.dry_run):
        print(msg)

    print("\nDone." if not args.dry_run else "\n(dry-run complete, no files modified)")


if __name__ == "__main__":
    main()
