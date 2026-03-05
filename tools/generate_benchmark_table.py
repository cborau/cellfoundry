"""One-shot script to generate docs/benchmark_results_table.tex from the CSV."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "tools" / "benchmark_results.csv"
DEFAULT_OUT = ROOT / "docs" / "benchmark_results_table.tex"

COLS = [
    "run", "N", "ECM_POPULATION_SIZE", "N_CELLS", "INIT_N_FOCAD_PER_CELL",
    "FOCAD_count_init", "N_FNODES", "CELL_RADIUS", "MAX_SEARCH_RADIUS",
    "steps", "init_time_s", "simulation_time_s", "rtc_time_s",
    "init_functions_time_s", "exit_functions_time_s", "total_time_s",
    "time_per_step_s",
]

HEADERS = {
    "run": "Run",
    "N": r"\(N\)",
    "ECM_POPULATION_SIZE": "ECM",
    "N_CELLS": "CELL",
    "INIT_N_FOCAD_PER_CELL": "FOCAD/CELL",
    "FOCAD_count_init": "FOCAD",
    "N_FNODES": "FNODE",
    "CELL_RADIUS": r"\(R_c\)",
    "MAX_SEARCH_RADIUS": r"\(R_s\)",
    "steps": "Steps",
    "init_time_s": r"\(t_\text{init}\)",
    "simulation_time_s": r"\(t_\text{sim}\)",
    "rtc_time_s": r"\(t_\text{RTC}\)",
    "init_functions_time_s": r"\(t_\text{initfn}\)",
    "exit_functions_time_s": r"\(t_\text{exit}\)",
    "total_time_s": r"\(t_\text{total}\)",
    "time_per_step_s": r"\(t_\text{step}\)",
}

TIME_COLS = {
    "init_time_s", "simulation_time_s", "rtc_time_s",
    "init_functions_time_s", "exit_functions_time_s",
    "total_time_s", "time_per_step_s",
}


def _fmt(col, val):
    if pd.isna(val):
        return "--"
    if col in ("CELL_RADIUS", "MAX_SEARCH_RADIUS"):
        return f"{val:.1f}"
    if col in TIME_COLS:
        return f"{val:.3f}"
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def generate(csv_path: Path = DEFAULT_CSV, out_path: Path = DEFAULT_OUT):
    df = pd.read_csv(csv_path)
    df = df[df["status"].str.startswith("OK", na=False)].copy()

    ncols = len(COLS)
    col_spec = " ".join(["r"] * ncols)
    hdr_row = " & ".join(HEADERS[c] for c in COLS) + r" \\"

    lines = [
        r"% ---------------------------------------------------------",
        r"% Benchmark raw results table  (auto-generated)",
        r"% Requires: booktabs, longtable, lscape (or pdflscape)",
        r"% ---------------------------------------------------------",
        r"",
        r"\begin{landscape}",
        r"\begingroup",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\scriptsize",
        r"\begin{longtable}{@{} " + col_spec + r" @{}}",
        r"\caption{Raw benchmark results (single-step runs). Times are in seconds.}",
        r"\label{tab:benchmark-raw} \\",
        r"\toprule",
        hdr_row,
        r"\midrule",
        r"\endfirsthead",
        r"",
        r"\multicolumn{" + str(ncols) + r"}{c}{\emph{(continued from previous page)}} \\",
        r"\toprule",
        hdr_row,
        r"\midrule",
        r"\endhead",
        r"",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r@{}}{\emph{Continued on next page}} \\",
        r"\endfoot",
        r"",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for _, row in df.iterrows():
        vals = [_fmt(c, row[c]) for c in COLS]
        lines.append(" & ".join(vals) + r" \\")

    lines += [
        r"\end{longtable}",
        r"\endgroup",
        r"\end{landscape}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a LaTeX longtable from benchmark_results.csv",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help=f"Path to the benchmark CSV. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help=f"Output .tex file. Default: {DEFAULT_OUT}",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    out_path = Path(args.out) if args.out else DEFAULT_OUT
    generate(csv_path, out_path)
