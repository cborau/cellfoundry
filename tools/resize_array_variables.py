#!/usr/bin/env python
"""
resize_array_variables.py  –  Expand (or shrink) per-cell-type and/or per-species
arrays to a new N_CELL_TYPES and/or N_SPECIES value. USE WITH CAUTION AND REVIEW CHANGES CAREFULLY.

Usage
-----
    python tools/resize_array_variables.py --cell-types <N>
    python tools/resize_array_variables.py --species <N>
    python tools/resize_array_variables.py --cell-types <N> --species <M> [--dry-run]

What it does
------------
1. **model.py**
   - Updates ``N_CELL_TYPES`` and/or ``N_SPECIES`` to the new value(s).
   - Resizes every variable in the hard-coded whitelists (PER_CELL_TYPE_VARS,
     PER_SPECIES_VARS) using extend-by-last / truncate logic.
   - Normalises ``[expr] * N_CELL_TYPES`` / ``[expr] * N_SPECIES`` replicated
     lists regardless of whether they are in the whitelist (they carry an
     explicit reference to the dimension constant so resizing is safe).
   - Handles BOUNDARY_CONC_INIT_MULTI / BOUNDARY_CONC_FIXED_MULTI, which are
     multi-line list-of-lists with outer dim = N_SPECIES and inner dim = 6.
   - For any OTHER list whose length happens to equal the old value, the user
     is asked interactively whether to resize it before any file is written
     (dry-run just reports candidates without asking).

2. **C++ agent functions (*.cpp)**
   - Updates every ``const uint8_t N_CELL_TYPES = <old>;`` line.
   - Updates every ``const uint8_t N_SPECIES = <old>;`` line.

Safety
------
- Creates timestamped backups of every file it modifies.
- Dry-run mode (``--dry-run``) prints changes without writing.
- Rejects N < 1.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PY = ROOT / "model.py"
CPP_GLOB = "*.cpp"

# ---------------------------------------------------------------------------
# Hard-coded variable whitelists
# Update these whenever a new per-type or per-species array is added.
# ---------------------------------------------------------------------------

# Variables whose Python length == N_CELL_TYPES
PER_CELL_TYPE_VARS: frozenset[str] = frozenset({
    # --- Fibre-network remodelling ---
    "FNODE_DEGRADATION_RATE",
    "FNODE_DEPOSITION_RATE",
    "FNODE_BIRTH_K_0",
    "FNODE_BIRTH_K_MAX",
    "FNODE_BIRTH_K_C",
    "FNODE_BIRTH_HILL_CONC",
    "FNODE_BIRTH_K_SIGMA",
    "FNODE_BIRTH_HILL_SIGMA",
    "FNODE_BIRTH_RADIUS",
    "FNODE_BIRTH_LINK_MAX_DISTANCE",
    "FNODE_BIRTH_REFRACTORY",
    # --- Mechanical & morphological ---
    "CELL_K_ELAST",
    "CELL_D_DUMPING",
    "CELL_RADIUS",
    "CELL_NUCLEUS_RADIUS",
    "CELL_SPEED_REF",
    "BROWNIAN_MOTION_STRENGTH_FACTOR",
    "BROWNIAN_MOTION_STRENGTH",
    "ROTATIONAL_DIFFUSION_RATE",
    "CELL_CELL_REPULSION_K",
    "CELL_CELL_ADHESION_K",
    "CELL_CELL_ADHESION_RANGE",
    "CELL_CELL_DV_MAX",
    "CELL_FNODE_REPULSION_K",
    "CELL_FNODE_EXCLUSION_DISTANCE",
    "CELL_FNODE_DV_MAX",
    # --- Cell-cycle timing ---
    "CYCLE_PHASE_G1_DURATION",
    "CYCLE_PHASE_S_DURATION",
    "CYCLE_PHASE_G2_DURATION",
    "CYCLE_PHASE_M_DURATION",
    "CYCLE_PHASE_G1_START",
    "CYCLE_PHASE_S_START",
    "CYCLE_PHASE_G2_START",
    "CYCLE_PHASE_M_START",
    "CELL_CYCLE_DURATION",
    # --- Cell-cycle multipliers ---
    "DIVISION_RATE_MULTIPLIER",
    "DAMAGE_ACCUMULATION_MULTIPLIER",
    "DAMAGE_REPAIR_MULTIPLIER",
    "DAMAGE_DEATH_THRESHOLD",
    # --- Per-type species multipliers ---
    "CELL_CONSUMPTION_MULTIPLIER",
    "CELL_PRODUCTION_MULTIPLIER",
    "CELL_REACTION_MULTIPLIER",
    "CELL_INIT_CONCENTRATION_MULTIPLIER",
    # --- Damage / death pathways ---
    "CELL_HYPOXIA_THRESHOLD",
    "CELL_NUTRIENT_THRESHOLD",
    "CELL_STRESS_THRESHOLD",
    "CELL_HYPOXIA_DAMAGE_RATE",
    "CELL_NUTRIENT_DAMAGE_RATE",
    "CELL_STRESS_DAMAGE_RATE",
    "CELL_BASAL_DAMAGE_REPAIR_RATE",
    "CELL_ACUTE_HYPOXIA_THRESHOLD",
    "CELL_ACUTE_NUTRIENT_THRESHOLD",
    "CELL_ACUTE_STRESS_THRESHOLD",
    # --- Focal adhesion ---
    "FOCAD_K_FA",
    "FOCAD_F_MAX",
    "FOCAD_V_C",
    "FOCAD_K_ON",
    "FOCAD_K_OFF_0",
    "FOCAD_F_C",
    "CATCH_BOND_CATCH_SCALE",
    "CATCH_BOND_SLIP_SCALE",
    "CATCH_BOND_F_CATCH",
    "CATCH_BOND_F_SLIP",
    "FOCAD_K_REINF",
    "FOCAD_F_REINF",
    "FOCAD_K_FA_MAX",
    "FOCAD_K_FA_DECAY",
    "FOCAD_POLARITY_KON_FRONT_GAIN",
    "FOCAD_POLARITY_KOFF_FRONT_REDUCTION",
    "FOCAD_POLARITY_KOFF_REAR_GAIN",
    "FOCAD_F_MATURE",
    "FOCAD_T_NASCENT_MAX",
    "FOCAD_T_DETACHED_GRACE",
    "FOCAD_T_DISASSEMBLY",
    "FOCAD_BIRTH_N_MIN",
    "FOCAD_BIRTH_N_MAX",
    "FOCAD_BIRTH_K_0",
    "FOCAD_BIRTH_K_MAX",
    "FOCAD_BIRTH_K_SIGMA",
    "FOCAD_BIRTH_HILL_SIGMA",
    "FOCAD_BIRTH_K_C",
    "FOCAD_BIRTH_HILL_CONC",
    "FOCAD_BIRTH_REFRACTORY",
    # --- LINC / Nucleus ---
    "LINC_K_ELAST",
    "LINC_D_DUMPING",
    "LINC_REST_LENGTH",
    "NUCLEUS_E",
    "NUCLEUS_NU",
    "NUCLEUS_TAU",
    "NUCLEUS_EPS_CLAMP",
    # --- Chemotaxis / chemokinesis (per cell-type dimension) ---
    "CHEMOTAXIS_CHI",
    "CHEMOKINESIS_ALPHA",
    "CHEMOKINESIS_K",
    "CHEMOKINESIS_HILL_N",
    "CHEMOKINESIS_ADAPT_TAU",
    "CHEMOKINESIS_SIGNAL_SAT_MULTIPLIER",
    # --- Cell migration ---
    "FOCAD_MOBILITY_MU",
    "ORIENTATION_ALIGN_RATE",
    "DUROTAXIS_BLEND_BETA",
})

# Variables whose Python length == N_SPECIES (simple 1-D lists)
PER_SPECIES_VARS: frozenset[str] = frozenset({
    "DIFFUSION_COEFF_MULTI",
    "CHEMOTAXIS_SENSITIVITY",
    "CHEMOKINESIS_SENSITIVITY",
    "CHEMOKINESIS_SIGNAL_SAT",
    "INIT_ECM_CONCENTRATION_VALS",
    "INIT_CELL_CONCENTRATION_VALS",
    "INIT_CELL_CONC_MASS_VALS",      # list-comp, auto-scales; listed to suppress false alarm
    "INIT_ECM_SAT_CONCENTRATION_VALS",
    "INIT_CELL_CONSUMPTION_RATES",
    "INIT_CELL_PRODUCTION_RATES",
    "INIT_CELL_REACTION_RATES",
    "LUMEN_DIFFUSION_COEFF_MULTI",
    "INIT_VASCULARIZATION_CONCENTRATION_VALS",
})

# List-of-6 variables: outer dim = N_SPECIES, inner dim = 6 (boundaries).
# These span multiple lines and need special handling.
PER_SPECIES_6_VARS: frozenset[str] = frozenset({
    "BOUNDARY_CONC_INIT_MULTI",
    "BOUNDARY_CONC_FIXED_MULTI",
})

# All known vars combined (for whitelist checks / suspicious detection)
_ALL_KNOWN_VARS: frozenset[str] = PER_CELL_TYPE_VARS | PER_SPECIES_VARS | PER_SPECIES_6_VARS

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# N_CELL_TYPES = <int>  in model.py
RE_NCELL_PY = re.compile(r"^(N_CELL_TYPES\s*=\s*)(\d+)", re.MULTILINE)
# N_SPECIES = <int>  in model.py
RE_NSPEC_PY = re.compile(r"^(N_SPECIES\s*=\s*)(\d+)", re.MULTILINE)

# const uint8_t N_CELL_TYPES = <int>;  in C++
RE_NCELL_CPP = re.compile(r"(const\s+uint8_t\s+N_CELL_TYPES\s*=\s*)(\d+)(\s*;)")
# const uint8_t N_SPECIES = <int>;  in C++
RE_NSPEC_CPP = re.compile(r"(const\s+uint8_t\s+N_SPECIES\s*=\s*)(\d+)(\s*;)")

# Explicit list on a single line: VAR = [a, b, c]  (non-greedy inner)
RE_EXPLICIT_LIST = re.compile(
    r"^(\s*[A-Za-z_]\w*\s*=\s*)\[(.+?)\](\s*(?:#.*)?)$"
)
# Replicated list: VAR = [expr] * N_CELL_TYPES  or  * <digit>
RE_REPLICATED_CT = re.compile(
    r"^(\s*[A-Za-z_]\w*\s*=\s*\[.+\]\s*\*\s*)(?:N_CELL_TYPES|\d+)(\s*(?:#.*)?)$"
)
# Replicated list: VAR = [expr] * N_SPECIES  or  * <digit>
RE_REPLICATED_SP = re.compile(
    r"^(\s*[A-Za-z_]\w*\s*=\s*\[.+\]\s*\*\s*)(?:N_SPECIES|\d+)(\s*(?:#.*)?)$"
)
# List comprehension: VAR = [... for ... in ...]
RE_LIST_COMP = re.compile(
    r"^\s*[A-Za-z_]\w*\s*=\s*\[.+\bfor\b.+\]"
)
# Replicated list whose multiplier is literally N_CELL_TYPES
RE_REPLICATED_CT_NAMED = re.compile(
    r"^(\s*[A-Za-z_]\w*\s*=\s*\[.+\]\s*\*\s*)(N_CELL_TYPES)(\s*(?:#.*)?)$"
)
RE_REPLICATED_SP_NAMED = re.compile(
    r"^(\s*[A-Za-z_]\w*\s*=\s*\[.+\]\s*\*\s*)(N_SPECIES)(\s*(?:#.*)?)$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backup(path: Path, dry: bool) -> None:
    if dry:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, dst)


def _detect_old_N(text: str, pattern: re.Pattern, name: str) -> int:
    m = pattern.search(text)
    if not m:
        sys.exit(f"ERROR: could not find  {name} = <int>  in model.py")
    return int(m.group(2))


def _get_lhs(stripped: str) -> str | None:
    """Return LHS variable name from a simple assignment, or None."""
    eq_pos = stripped.find("=")
    if eq_pos > 0 and stripped[eq_pos - 1] not in "!<>=" and (
        eq_pos + 1 >= len(stripped) or stripped[eq_pos + 1] != "="
    ):
        lhs = stripped[:eq_pos].strip()
        if re.fullmatch(r"[A-Za-z_]\w*", lhs):
            return lhs
    return None


def _resize_list(elements: list, new_n: int) -> list:
    """Extend (repeat last, with index-bump if applicable) or truncate."""
    if not elements:
        return elements
    if len(elements) >= new_n:
        return elements[:new_n]
    last = elements[-1]
    last_idx = len(elements) - 1
    idx_tag = f"[{last_idx}]"
    if idx_tag in last:
        result = list(elements)
        for j in range(len(elements), new_n):
            result.append(last.replace(idx_tag, f"[{j}]"))
        return result
    return elements + [last] * (new_n - len(elements))


def _parse_list_elements(raw: str) -> list[str] | None:
    """Split the inside of a list literal into element strings at depth-0 commas."""
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


# ---------------------------------------------------------------------------
# Multi-line list-of-6 handler (BOUNDARY_CONC_*_MULTI)
# ---------------------------------------------------------------------------

def _collect_multiline_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect lines from `start` until outer brackets are balanced.
    Returns (block_lines, last_index_inclusive)."""
    depth = 0
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
        if depth == 0:
            return lines[start : i + 1], i
    return lines[start:], len(lines) - 1


def _parse_species6_block(block: list[str]) -> list[list] | None:
    """Parse a multi-line list-of-lists block using ast, ignoring comment lines."""
    data_parts: list[str] = []
    for line in block:
        s = line.strip()
        if s.startswith("#"):
            continue
        data_parts.append(re.sub(r"\s*#.*$", "", line))
    joined = "".join(data_parts)
    eq = joined.find("=")
    if eq == -1:
        return None
    try:
        val = ast.literal_eval(joined[eq + 1 :].strip())
        if isinstance(val, list) and all(isinstance(r, list) for r in val):
            return val
    except (ValueError, SyntaxError):
        pass
    return None


def _fmt_val(v: object) -> str:
    """Format a numeric value for insertion into source."""
    if isinstance(v, float):
        r = repr(v)
        return r
    return repr(v)


def _resize_species6_block(
    block: list[str], var: str, old_sp: int, new_sp: int
) -> tuple[list[str] | None, list[str]]:
    """Resize the outer (species) dimension of a N_SPECIES×6 multi-line variable."""
    log: list[str] = []
    parsed = _parse_species6_block(block)
    if parsed is None:
        log.append(f"  WARNING: {var}: could not parse multi-line block, skipping")
        return None, log
    if len(parsed) != old_sp:
        log.append(
            f"  WARNING: {var}: expected {old_sp} rows but found {len(parsed)}, skipping"
        )
        return None, log

    # Compute new value
    if new_sp > old_sp:
        new_val = parsed + [list(parsed[-1])] * (new_sp - old_sp)
    else:
        new_val = parsed[:new_sp]

    # --- Reconstruct ---
    # Find the continuation indentation from the second non-comment line after the first.
    # The first line is  "VAR = [[row0...],"
    first_line = block[0]
    idx_open_outer = first_line.find("[[")
    if idx_open_outer == -1:
        log.append(f"  WARNING: {var}: cannot find [[ in opening line, skipping")
        return None, log
    cont_indent = " " * (idx_open_outer + 1)

    # Collect trailing comment from last original line
    last_orig = block[-1].rstrip("\n\r")
    tc_m = re.search(r"\]\]\s*(#.*)", last_orig)
    trailing_comment = ("  " + tc_m.group(1)) if tc_m else ""

    # Rebuild
    new_block: list[str] = []
    prefix = first_line[: idx_open_outer + 1]  # up to and including the first [

    for j, row in enumerate(new_val):
        vals_str = ", ".join(_fmt_val(v) for v in row)
        if j == 0:
            if len(new_val) == 1:
                new_block.append(f"{prefix}[{vals_str}]]{trailing_comment}\n")
            else:
                new_block.append(f"{prefix}[{vals_str}],\n")
        elif j < len(new_val) - 1:
            new_block.append(f"{cont_indent}[{vals_str}],\n")
        else:
            new_block.append(f"{cont_indent}[{vals_str}]]{trailing_comment}\n")

    direction = "extended" if new_sp > old_sp else "shrunk"
    log.append(f"  {var}: {old_sp} → {new_sp} rows ({direction})")
    if new_sp > old_sp:
        log.append(
            f"    NOTE: new species row(s) copied from last row — review values manually"
        )
    return new_block, log


# ---------------------------------------------------------------------------
# Suspicious-variable scanner
# ---------------------------------------------------------------------------

def _scan_suspects(
    lines: list[str],
    old_ct: int | None,
    old_sp: int | None,
    do_ct: bool,
    do_sp: bool,
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Return (suspects_ct, suspects_sp) as lists of (lineno, var_name, reason)."""
    suspects_ct: list[tuple[int, str, str]] = []
    suspects_sp: list[tuple[int, str, str]] = []

    i = 0
    while i < len(lines):
        stripped = lines[i].rstrip("\n\r")
        lhs = _get_lhs(stripped)

        # Skip multi-line blocks (PER_SPECIES_6_VARS) — not suspicious
        if lhs in PER_SPECIES_6_VARS:
            _, end_i = _collect_multiline_block(lines, i)
            i = end_i + 1
            continue

        # Skip already-known vars and list comps
        if lhs in _ALL_KNOWN_VARS or RE_LIST_COMP.match(stripped):
            i += 1
            continue

        lineno = i + 1

        if lhs is not None:
            # Replicated list with a numeric literal (not N_CELL_TYPES / N_SPECIES)
            if do_ct and old_ct is not None:
                m_rep = RE_REPLICATED_CT.match(stripped)
                if m_rep and not RE_REPLICATED_CT_NAMED.match(stripped):
                    # Multiplier is a digit matching old_ct?
                    digit_m = re.search(r"\*\s*(\d+)\s*(?:#.*)?$", stripped)
                    if digit_m and int(digit_m.group(1)) == old_ct:
                        suspects_ct.append(
                            (lineno, lhs, f"replicated list * {old_ct}")
                        )
                        i += 1
                        continue

            if do_sp and old_sp is not None:
                m_rep = RE_REPLICATED_SP.match(stripped)
                if m_rep and not RE_REPLICATED_SP_NAMED.match(stripped):
                    digit_m = re.search(r"\*\s*(\d+)\s*(?:#.*)?$", stripped)
                    if digit_m and int(digit_m.group(1)) == old_sp:
                        suspects_sp.append(
                            (lineno, lhs, f"replicated list * {old_sp}")
                        )
                        i += 1
                        continue

            # Explicit list
            m_expl = RE_EXPLICIT_LIST.match(stripped)
            if m_expl:
                elems = _parse_list_elements(m_expl.group(2))
                if elems is not None:
                    if do_ct and old_ct is not None and len(elems) == old_ct:
                        suspects_ct.append(
                            (lineno, lhs, f"explicit list with {old_ct} element(s)")
                        )
                    # Note: a var could match both if old_ct == old_sp
                    elif do_sp and old_sp is not None and len(elems) == old_sp:
                        suspects_sp.append(
                            (lineno, lhs, f"explicit list with {old_sp} element(s)")
                        )

        i += 1

    return suspects_ct, suspects_sp


def _confirm_suspects(
    suspects_ct: list[tuple[int, str, str]],
    suspects_sp: list[tuple[int, str, str]],
    dry: bool,
) -> tuple[set[str], set[str]]:
    """Ask the user which suspects to include; return (extra_ct, extra_sp)."""
    extra_ct: set[str] = set()
    extra_sp: set[str] = set()

    if dry:
        for lineno, var, reason in suspects_ct:
            print(
                f"  [DRY-RUN] Suspicious N_CELL_TYPES candidate: {var} "
                f"(L{lineno}, {reason}) — skipped (dry-run, use interactive mode to confirm)"
            )
        for lineno, var, reason in suspects_sp:
            print(
                f"  [DRY-RUN] Suspicious N_SPECIES candidate: {var} "
                f"(L{lineno}, {reason}) — skipped (dry-run, use interactive mode to confirm)"
            )
        return extra_ct, extra_sp

    if suspects_ct:
        print(
            "\nThe following variables were NOT found in the N_CELL_TYPES whitelist\n"
            "but their current length matches old N_CELL_TYPES.\n"
            "Answer 'y' to include them in the resize.\n"
        )
        for lineno, var, reason in suspects_ct:
            ans = input(f"  Resize {var!r} (L{lineno}, {reason}) as per-cell-type? [y/N] ")
            if ans.strip().lower() == "y":
                extra_ct.add(var)

    if suspects_sp:
        print(
            "\nThe following variables were NOT found in the N_SPECIES whitelist\n"
            "but their current length matches old N_SPECIES.\n"
            "Answer 'y' to include them in the resize.\n"
        )
        for lineno, var, reason in suspects_sp:
            ans = input(f"  Resize {var!r} (L{lineno}, {reason}) as per-species? [y/N] ")
            if ans.strip().lower() == "y":
                extra_sp.add(var)

    return extra_ct, extra_sp


# ---------------------------------------------------------------------------
# Main model.py rewrite
# ---------------------------------------------------------------------------

def resize_model_py(
    new_ct: int | None,
    new_sp: int | None,
    dry: bool,
) -> list[str]:
    """Resize arrays in model.py.  Returns change log."""
    text = MODEL_PY.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    log: list[str] = []

    old_ct = _detect_old_N(text, RE_NCELL_PY, "N_CELL_TYPES") if new_ct is not None else None
    old_sp = _detect_old_N(text, RE_NSPEC_PY, "N_SPECIES") if new_sp is not None else None

    do_ct = new_ct is not None and new_ct != old_ct
    do_sp = new_sp is not None and new_sp != old_sp

    if not do_ct and not do_sp:
        msgs = []
        if new_ct is not None:
            msgs.append(f"N_CELL_TYPES already {new_ct}")
        if new_sp is not None:
            msgs.append(f"N_SPECIES already {new_sp}")
        return [f"model.py: {', '.join(msgs)}, nothing to do."]

    if do_ct:
        log.append(f"model.py: N_CELL_TYPES {old_ct} → {new_ct}")
    if do_sp:
        log.append(f"model.py: N_SPECIES {old_sp} → {new_sp}")

    # --- Pre-scan for suspicious unknown variables ---
    suspects_ct, suspects_sp = _scan_suspects(
        lines, old_ct, old_sp, do_ct, do_sp
    )
    extra_ct, extra_sp = _confirm_suspects(suspects_ct, suspects_sp, dry)

    # Effective resize sets
    ct_vars = PER_CELL_TYPE_VARS | extra_ct
    sp_vars = PER_SPECIES_VARS | extra_sp

    # --- Rewrite ---
    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n\r")
        lineno = i + 1

        lhs = _get_lhs(stripped)

        # --- Multi-line species-6 blocks ---
        if lhs in PER_SPECIES_6_VARS:
            block, end_i = _collect_multiline_block(lines, i)
            if do_sp:
                new_block, block_log = _resize_species6_block(
                    block, lhs, old_sp, new_sp  # type: ignore[arg-type]
                )
                log.extend(block_log)
                new_lines.extend(new_block if new_block is not None else block)
            else:
                new_lines.extend(block)
            i = end_i + 1
            continue

        # --- N_CELL_TYPES = <old> ---
        if do_ct:
            m = RE_NCELL_PY.match(stripped)
            if m:
                new_lines.append(f"{m.group(1)}{new_ct}\n")
                log.append(f"  L{lineno}: N_CELL_TYPES = {old_ct} → {new_ct}")
                i += 1
                continue

        # --- N_SPECIES = <old> ---
        if do_sp:
            m = RE_NSPEC_PY.match(stripped)
            if m:
                new_lines.append(f"{m.group(1)}{new_sp}\n")
                log.append(f"  L{lineno}: N_SPECIES = {old_sp} → {new_sp}")
                i += 1
                continue

        # Skip list comprehensions — they auto-resize from their sources
        if RE_LIST_COMP.match(stripped):
            new_lines.append(line)
            i += 1
            continue

        # --- Replicated list: [expr] * N_CELL_TYPES  (named reference) ---
        if do_ct and RE_REPLICATED_CT_NAMED.match(stripped):
            # Already correct, just keep (multiplier already says N_CELL_TYPES)
            new_lines.append(line)
            i += 1
            continue

        # --- Replicated list: [expr] * N_SPECIES  (named reference) ---
        if do_sp and RE_REPLICATED_SP_NAMED.match(stripped):
            new_lines.append(line)
            i += 1
            continue

        # --- Replicated list: [expr] * <digit>  (normalise to named constant) ---
        if do_ct and lhs and lhs in ct_vars:
            m_rep = RE_REPLICATED_CT.match(stripped)
            if m_rep:
                new_line = f"{m_rep.group(1)}N_CELL_TYPES{m_rep.group(2)}\n"
                if new_line != line:
                    log.append(f"  L{lineno}: {lhs} replicated-list → * N_CELL_TYPES")
                new_lines.append(new_line)
                i += 1
                continue

        if do_sp and lhs and lhs in sp_vars:
            m_rep = RE_REPLICATED_SP.match(stripped)
            if m_rep:
                new_line = f"{m_rep.group(1)}N_SPECIES{m_rep.group(2)}\n"
                if new_line != line:
                    log.append(f"  L{lineno}: {lhs} replicated-list → * N_SPECIES")
                new_lines.append(new_line)
                i += 1
                continue

        # --- Explicit list [a, b, c] ---
        m_expl = RE_EXPLICIT_LIST.match(stripped)
        if m_expl:
            inner = m_expl.group(2)
            elems = _parse_list_elements(inner)
            if elems is not None:
                if do_ct and lhs and lhs in ct_vars and len(elems) == old_ct:
                    resized = _resize_list(elems, new_ct)  # type: ignore[arg-type]
                    new_line = f"{m_expl.group(1)}[{', '.join(resized)}]{m_expl.group(3)}\n"
                    log.append(f"  L{lineno}: {lhs} list {old_ct} → {new_ct} elem(s)")
                    new_lines.append(new_line)
                    i += 1
                    continue
                if do_sp and lhs and lhs in sp_vars and len(elems) == old_sp:
                    resized = _resize_list(elems, new_sp)  # type: ignore[arg-type]
                    new_line = f"{m_expl.group(1)}[{', '.join(resized)}]{m_expl.group(3)}\n"
                    log.append(f"  L{lineno}: {lhs} list {old_sp} → {new_sp} elem(s)")
                    new_lines.append(new_line)
                    i += 1
                    continue

        # Default: keep line unchanged
        new_lines.append(line)
        i += 1

    result = "".join(new_lines)

    if not dry:
        _backup(MODEL_PY, dry)
        MODEL_PY.write_text(result, encoding="utf-8")
        log.append("  ✓ model.py written")
    else:
        log.append("  (dry-run – model.py not written)")

    return log


# ---------------------------------------------------------------------------
# C++ files
# ---------------------------------------------------------------------------

def resize_cpp_files(
    new_ct: int | None,
    new_sp: int | None,
    dry: bool,
) -> list[str]:
    """Update N_CELL_TYPES and/or N_SPECIES in every .cpp file."""
    log: list[str] = []

    for cpp in sorted(ROOT.glob(CPP_GLOB)):
        text = cpp.read_text(encoding="utf-8")
        new_text = text
        file_log: list[str] = []

        if new_ct is not None:
            new_text, n = RE_NCELL_CPP.subn(rf"\g<1>{new_ct}\3", new_text)
            if n:
                file_log.append(f"  N_CELL_TYPES → {new_ct} ({n} occurrence(s))")

        if new_sp is not None:
            new_text, n = RE_NSPEC_CPP.subn(rf"\g<1>{new_sp}\3", new_text)
            if n:
                file_log.append(f"  N_SPECIES → {new_sp} ({n} occurrence(s))")

        if file_log:
            log.append(f"{cpp.name}:")
            log.extend(file_log)
            if not dry:
                _backup(cpp, dry)
                cpp.write_text(new_text, encoding="utf-8")

    if not log:
        log.append("(no C++ files needed updating)")
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resize per-cell-type and/or per-species arrays to new N_CELL_TYPES "
            "and/or N_SPECIES values."
        )
    )
    parser.add_argument(
        "--cell-types",
        type=int,
        metavar="N",
        default=None,
        help="New value for N_CELL_TYPES (>= 1).",
    )
    parser.add_argument(
        "--species",
        type=int,
        metavar="N",
        default=None,
        help="New value for N_SPECIES (>= 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without modifying any files.",
    )
    args = parser.parse_args()

    if args.cell_types is None and args.species is None:
        parser.error("Provide at least one of --cell-types or --species.")

    if args.cell_types is not None and args.cell_types < 1:
        sys.exit("ERROR: N_CELL_TYPES must be >= 1")
    if args.species is not None and args.species < 1:
        sys.exit("ERROR: N_SPECIES must be >= 1")

    tag = "DRY RUN: " if args.dry_run else ""
    parts = []
    if args.cell_types is not None:
        parts.append(f"N_CELL_TYPES → {args.cell_types}")
    if args.species is not None:
        parts.append(f"N_SPECIES → {args.species}")
    print(f"{tag}Resizing {', '.join(parts)}\n")

    # 1. model.py
    for msg in resize_model_py(args.cell_types, args.species, args.dry_run):
        print(msg)

    print()

    # 2. C++ files
    for msg in resize_cpp_files(args.cell_types, args.species, args.dry_run):
        print(msg)

    print(
        "\nDone."
        if not args.dry_run
        else "\n(dry-run complete – no files modified)"
    )


if __name__ == "__main__":
    main()
