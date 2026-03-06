"""
Extract FLAMEGPU2 agent variable and RTC function registrations from a Python model file
and generate LaTeX tables (one section per agent).

What it looks for
- Agent creation:
    CELL_agent = model.newAgent("CELL")

- Variable declarations:
    CELL_agent.newVariableFloat("clock", 0.0) # internal clock
    CELL_agent.newVariableArrayFloat("x_i", N_ANCHOR_POINTS) # anchor x positions

- RTC functions (including chained message IO, possibly multi-line):
    CELL_agent.newRTCFunctionFile("cell_bucket_location_data", file_var).setMessageOutput("cell_bucket_location_message")
    CELL_agent.newRTCFunctionFile("cell_focad_update", file_var).setMessageInput("focad_bucket_location_message")

Output
- A LaTeX file containing, for each agent:
  1) Variables table: (Variable, Description)
     Description = "type, comment" or just "type" when comment is missing.
  2) Functions table: (Function, Input, Output)

Standalone vs fragment
- Default output is a LaTeX fragment intended to be included with:
    \\input{agent_api_tables.tex}

- With --standalone, the script emits a complete LaTeX document including a preamble
  and required packages, so it can be compiled directly:
    pdflatex agent_api_tables_standalone.tex
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


TARGET_AGENTS_DEFAULT = ["CELL", "FOCAD", "FNODE", "ECM", "BCORNER"]


@dataclass
class VarInfo:
    name: str
    desc: str  # combined: "type, comment"


@dataclass
class FuncInfo:
    name: str
    msg_in: str
    msg_out: str


_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(s: str) -> str:
    out = []
    for ch in s:
        out.append(_LATEX_ESCAPE_MAP.get(ch, ch))
    return "".join(out)


# Agent declaration:
#   CELL_agent = model.newAgent("CELL")
_AGENT_DECL_RE = re.compile(
    r'^\s*(?P<varname>[A-Za-z_]\w*)\s*=\s*model\.newAgent\(\s*"(?P<agent>[A-Za-z0-9_]+)"\s*\)\s*(?:#.*)?$'
)

# Variable line:
#   CELL_agent.newVariableFloat("x", 0.0) # comment
#   CELL_agent.newVariableArrayFloat("x_i", N_ANCHOR_POINTS) # comment
_VAR_RE = re.compile(
    r'^\s*(?P<agentvar>[A-Za-z_]\w*)\s*\.newVariable(?P<array>Array)?(?P<type>[A-Za-z0-9_]+)\('
    r'\s*"(?P<name>[^"]+)"\s*(?:,\s*(?P<arg2>[^)]*?))?\)\s*(?P<trailing>.*)$'
)

_RTC_START_RE = re.compile(r"\.newRTCFunctionFile\s*\(")
_RTC_NAME_RE = re.compile(r'\.newRTCFunctionFile\s*\(\s*"(?P<fname>[^"]+)"')
_MSG_IN_RE = re.compile(r'\.setMessageInput\s*\(\s*"(?P<min>[^"]+)"\s*\)')
_MSG_OUT_RE = re.compile(r'\.setMessageOutput\s*\(\s*"(?P<mout>[^"]+)"\s*\)')


def _strip_inline_comment(line: str) -> Tuple[str, str]:
    if "#" not in line:
        return line.rstrip("\n"), ""
    code, comment = line.split("#", 1)
    return code.rstrip("\n").rstrip(), comment.strip()


def _count_parens_delta(s: str) -> int:
    return s.count("(") - s.count(")")


def _normalize_type(array: bool, t: str, arg2: Optional[str]) -> str:
    base = t.strip()

    lower_map = {
        "Int": "int",
        "UInt": "uint",
        "UInt8": "uint8",
        "UInt16": "uint16",
        "UInt32": "uint32",
        "UInt64": "uint64",
        "Int8": "int8",
        "Int16": "int16",
        "Int32": "int32",
        "Int64": "int64",
        "Float": "float",
        "Double": "double",
    }
    base_norm = lower_map.get(base, base)

    if not array:
        return base_norm

    size_expr = (arg2 or "").strip()
    if size_expr:
        return f"{base_norm}[{size_expr}]"
    return f"{base_norm}[]"


def _combine_desc(vtype: str, comment: str) -> str:
    # Single column: "type, comment"
    if comment:
        return f"{vtype}, {comment}"
    return vtype


def parse_model(
    path: str, target_agents: List[str]
) -> Tuple[Dict[str, List[VarInfo]], Dict[str, List[FuncInfo]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    agent_varnames: Dict[str, str] = {}
    vars_by_agent: Dict[str, List[VarInfo]] = {a: [] for a in target_agents}
    funcs_by_agent: Dict[str, List[FuncInfo]] = {a: [] for a in target_agents}

    # Pass 1: find agent variable names, e.g. CELL_agent
    for line in lines:
        m = _AGENT_DECL_RE.match(line)
        if not m:
            continue
        agent = m.group("agent")
        if agent in target_agents:
            agent_varnames[agent] = m.group("varname")

    # Pass 2: parse variable + function calls
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i].rstrip("\n")

        # Variables
        vm = _VAR_RE.match(raw)
        if vm:
            agentvar = vm.group("agentvar")
            name = vm.group("name")
            array = vm.group("array") is not None
            vtype_raw = vm.group("type")
            arg2 = vm.group("arg2")

            agent_name_match = None
            for agent_name, varname in agent_varnames.items():
                if agentvar == varname:
                    agent_name_match = agent_name
                    break

            if agent_name_match is not None:
                _, comment = _strip_inline_comment(raw)
                vtype = _normalize_type(array=array, t=vtype_raw, arg2=arg2)
                desc = _combine_desc(vtype=vtype, comment=comment)
                vars_by_agent[agent_name_match].append(VarInfo(name=name, desc=desc))

            i += 1
            continue

        # RTC functions (possibly multi-line)
        if _RTC_START_RE.search(raw):
            stmt_lines = [raw]
            delta = _count_parens_delta(raw)
            j = i + 1
            while j < n and delta > 0:
                nxt = lines[j].rstrip("\n")
                stmt_lines.append(nxt)
                delta += _count_parens_delta(nxt)
                j += 1

            stmt = " ".join(s.strip() for s in stmt_lines)

            prefix = stmt.split(".newRTCFunctionFile", 1)[0].strip()
            agentvar = prefix

            agent_name_match = None
            for agent_name, varname in agent_varnames.items():
                if agentvar == varname:
                    agent_name_match = agent_name
                    break

            if agent_name_match is not None:
                fn_m = _RTC_NAME_RE.search(stmt)
                fname = fn_m.group("fname") if fn_m else "(unknown)"

                in_m = _MSG_IN_RE.search(stmt)
                out_m = _MSG_OUT_RE.search(stmt)
                msg_in = in_m.group("min") if in_m else "None"
                msg_out = out_m.group("mout") if out_m else "None"

                funcs_by_agent[agent_name_match].append(FuncInfo(name=fname, msg_in=msg_in, msg_out=msg_out))

            i = j
            continue

        i += 1

    return vars_by_agent, funcs_by_agent


def _latex_agent_section(agent_name: str, var_rows: List[VarInfo], func_rows: List[FuncInfo]) -> str:
    lines: List[str] = []

    lines.append(r"\subsection*{" + latex_escape(agent_name) + r"}")
    lines.append("")

    # ========================
    # VARIABLES TABLE
    # ========================
    lines.append(r"\noindent\textbf{Variables}")
    lines.append(r"\par\smallskip")

    # p{...} gives fixed width columns that can wrap text
    lines.append(r"\begin{longtable}{@{} p{0.25\textwidth} p{0.75\textwidth} @{} }")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{2}{@{}l@{}}{\textbf{" + latex_escape(agent_name) + r"}} \\")
    lines.append(r"\midrule")
    lines.append(r"\textbf{Variable} & \textbf{Description} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    lines.append(r"\toprule")
    lines.append(r"\multicolumn{2}{@{}l@{}}{\textbf{" + latex_escape(agent_name) + r" (continued)}} \\")
    lines.append(r"\midrule")
    lines.append(r"\textbf{Variable} & \textbf{Description} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    if not var_rows:
        lines.append(r"\multicolumn{2}{l}{(none found)} \\")
    else:
        for r in var_rows:
            lines.append(f"{latex_escape(r.name)} & {latex_escape(r.desc)} \\\\")

    lines.append(r"\end{longtable}")
    lines.append("")
    lines.append(r"\bigskip")

    # ========================
    # FUNCTIONS TABLE
    # ========================
    lines.append(r"\noindent\textbf{Functions}")
    lines.append(r"\par\smallskip")

    lines.append(r"\begin{longtable}{@{} p{0.4\textwidth} p{0.3\textwidth} p{0.3\textwidth} @{} }")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{3}{@{}l@{}}{\textbf{" + latex_escape(agent_name) + r"}} \\")
    lines.append(r"\midrule")
    lines.append(r"\textbf{Function} & \textbf{Input} & \textbf{Output} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    lines.append(r"\toprule")
    lines.append(r"\multicolumn{3}{@{}l@{}}{\textbf{" + latex_escape(agent_name) + r" (continued)}} \\")
    lines.append(r"\midrule")
    lines.append(r"\textbf{Function} & \textbf{Input} & \textbf{Output} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    if not func_rows:
        lines.append(r"\multicolumn{3}{l}{(none found)} \\")
    else:
        for r in func_rows:
            lines.append(
                f"{latex_escape(r.name)} & {latex_escape(r.msg_in)} & {latex_escape(r.msg_out)} \\\\"
            )

    lines.append(r"\end{longtable}")
    lines.append("")
    lines.append(r"\bigskip")

    return "\n".join(lines)


def emit_latex(
    out_path: str,
    vars_by_agent: Dict[str, List[VarInfo]],
    funcs_by_agent: Dict[str, List[FuncInfo]],
    target_agents: List[str],
    standalone: bool,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    parts: List[str] = []
    if standalone:
        parts.append(r"\documentclass{article}")
        # Smaller margins + wider usable page area
        parts.append(r"\usepackage[margin=1.6cm]{geometry}")
        parts.append(r"\usepackage{booktabs}")
        parts.append(r"\usepackage{tabularx}")
        parts.append(r"\usepackage{parskip}")  # nicer spacing, avoids indentation
        parts.append(r"\renewcommand{\arraystretch}{1.15}")
        parts.append(r"\setlength{\tabcolsep}{8pt}")
        parts.append(r"\begin{document}")
        parts.append("")

    parts.append(r"\section*{Agent API Summary}")
    parts.append("")

    for agent in target_agents:
        parts.append(_latex_agent_section(agent, vars_by_agent.get(agent, []), funcs_by_agent.get(agent, [])))

    if standalone:
        parts.append(r"\end{document}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts).rstrip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Parse a FLAMEGPU2 Python model file to extract agent variables and RTC function registrations, "
            "then generate LaTeX tables."
        )
    )
    ap.add_argument(
        "--model",
        default="model.py",
        help='Path to the model file to parse (default: "model.py").',
    )
    ap.add_argument(
        "--out",
        default="agent_api_tables.tex",
        help='Output LaTeX file path (default: "agent_api_tables.tex").',
    )
    ap.add_argument(
        "--agents",
        nargs="*",
        default=TARGET_AGENTS_DEFAULT,
        help="List of agent names to extract (default: CELL FOCAD FNODE ECM BCORNER).",
    )
    ap.add_argument(
        "--standalone",
        action="store_true",
        help=(
            "Emit a complete LaTeX document including preamble and packages. "
            "Without this flag, the output is a fragment intended for \\input{} inclusion."
        ),
    )

    args = ap.parse_args()

    vars_by_agent, funcs_by_agent = parse_model(args.model, args.agents)
    emit_latex(args.out, vars_by_agent, funcs_by_agent, args.agents, args.standalone)

    print(f"Wrote LaTeX tables to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())