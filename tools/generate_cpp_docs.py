#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
GITHUB_REPO = "https://github.com/cborau/cellfoundry"
GITHUB_REF = "master"


@dataclass
class FunctionDoc:
    file: str
    kind: str
    name: str
    line: int
    purpose: str
    inputs: List[str]
    outputs: List[str]
    notes: List[str]


def clean_docblock(raw: str) -> List[str]:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^/\*\*?", "", line)
        line = re.sub(r"\*/$", "", line)
        line = re.sub(r"^\*\s?", "", line)
        line = line.rstrip()
        if line:
            lines.append(line)
    return lines


def parse_sections(lines: List[str]) -> Tuple[str, List[str], List[str], List[str]]:
    purpose = ""
    inputs: List[str] = []
    outputs: List[str] = []
    notes: List[str] = []

    section = None
    for idx, line in enumerate(lines):
        text = line.strip()
        low = text.lower()
        if low.startswith("purpose:"):
            section = "purpose"
            purpose = text.split(":", 1)[1].strip()
            continue
        if low.startswith("inputs:"):
            section = "inputs"
            rest = text.split(":", 1)[1].strip()
            if rest:
                inputs.append(rest)
            continue
        if low.startswith("outputs:"):
            section = "outputs"
            rest = text.split(":", 1)[1].strip()
            if rest:
                outputs.append(rest)
            continue
        if low.startswith("notes:"):
            section = "notes"
            rest = text.split(":", 1)[1].strip()
            if rest:
                notes.append(rest)
            continue

        if text.startswith("-"):
            item = text[1:].strip()
            if section == "inputs":
                inputs.append(item)
            elif section == "outputs":
                outputs.append(item)
            elif section == "notes":
                notes.append(item)
            elif section == "purpose" and not purpose:
                purpose = item
            continue

        if section == "purpose":
            if purpose:
                purpose = f"{purpose} {text}".strip()
            else:
                purpose = text
        elif section == "notes":
            notes.append(text)

    if not purpose:
        fallback_lines = [
            line for line in lines
            if not line.lower().startswith(("inputs:", "outputs:", "notes:", "purpose:"))
            and not line.startswith("-")
        ]
        if fallback_lines:
            if len(fallback_lines) > 1 and fallback_lines[0].replace("_", "").isalnum():
                purpose = fallback_lines[1]
            else:
                purpose = fallback_lines[0]

    return purpose, inputs, outputs, notes


def extract_name_and_kind(signature: str) -> Tuple[str, str]:
    m_agent = re.search(r"FLAMEGPU_AGENT_FUNCTION\s*\(\s*([A-Za-z0-9_]+)", signature)
    if m_agent:
        return m_agent.group(1), "agent"

    m_helper = re.search(
        r"FLAMEGPU_(?:DEVICE|HOST_DEVICE)_FUNCTION\s+[A-Za-z0-9_:<>\*&\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        signature,
    )
    if m_helper:
        return m_helper.group(1), "helper"

    return "unknown", "unknown"


def parse_file(path: pathlib.Path) -> List[FunctionDoc]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    docs: List[FunctionDoc] = []

    pattern = re.compile(
        r"/\*\*(.*?)\*/\s*\n\s*(FLAMEGPU_(?:AGENT_FUNCTION|DEVICE_FUNCTION|HOST_DEVICE_FUNCTION)[^\n]*\n?(?:[^\{\n]*\{)?)",
        re.DOTALL,
    )

    for m in pattern.finditer(text):
        raw_doc = m.group(1)
        signature = m.group(2)
        signature_line = text.count("\n", 0, m.start(2)) + 1

        name, kind = extract_name_and_kind(signature)
        lines = clean_docblock(raw_doc)
        purpose, inputs, outputs, notes = parse_sections(lines)

        docs.append(
            FunctionDoc(
                file=path.name,
                kind=kind,
                name=name,
                line=signature_line,
                purpose=purpose or "(not specified)",
                inputs=inputs,
                outputs=outputs,
                notes=notes,
            )
        )

    return docs


def render_reference(all_docs: Dict[str, List[FunctionDoc]], github_repo: str, github_ref: str) -> str:
    out: List[str] = []
    out.append("# C++ Function Reference\n")
    out.append("Generated automatically from Doxygen-style docblocks in `.cpp` files.\n")
    out.append(
        "**Legend:** 🔸 Purpose  |  ⬇️ Inputs  |  ⬆️ Outputs  |  📝 Notes  |  🔗 Click function names to open source\n"
    )

    for file_name in sorted(all_docs.keys()):
        entries = all_docs[file_name]
        if not entries:
            continue
        out.append(f"## 📄 {file_name}\n")
        for entry in entries:
            source_link = f"{github_repo}/blob/{github_ref}/{entry.file}"
            out.append(f"### 🔹 [{entry.name}]({source_link})")
            out.append(f"**Type:** `{entry.kind}`  ")
            out.append(f"**Source:** [Open {entry.file}]({source_link})\n")
            out.append(f"- 🔸 **Purpose:** {entry.purpose}")
            if entry.inputs:
                out.append("- ⬇️ **Inputs:**")
                out.extend([f"  - {x}" for x in entry.inputs])
            if entry.outputs:
                out.append("- ⬆️ **Outputs:**")
                out.extend([f"  - {x}" for x in entry.outputs])
            if entry.notes:
                out.append("- 📝 **Notes:**")
                out.extend([f"  - {x}" for x in entry.notes])
            out.append("- - -")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_wiki_home() -> str:
    return (
        "# CellFoundry Wiki\n\n"
        "This wiki content is generated from source code and project metadata.\n\n"
        "## Pages\n\n"
        "- [What is CellFoundry](What-is-CellFoundry)\n"
        "- [C++ Function Reference](Function-Reference)\n"
        "- [Model Editor](Model-Editor)\n"
        "- [Post Processing](Post-Processing)\n"
    )


def github_blob_url(github_repo: str, github_ref: str, repo_path: str) -> str:
    return f"{github_repo}/blob/{github_ref}/{repo_path}"


def github_raw_url(github_repo: str, github_ref: str, repo_path: str) -> str:
    m = re.match(r"^https://github\.com/([^/]+)/([^/]+)$", github_repo.rstrip("/"))
    if m:
        owner, repo = m.group(1), m.group(2)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{github_ref}/{repo_path}"
    return github_blob_url(github_repo, github_ref, repo_path)


def render_what_is_cellfoundry(github_repo: str, github_ref: str) -> str:
    logo_url = github_raw_url(github_repo, github_ref, "assets/logo_cellfoundry.png")
    return (
        "# What is CellFoundry\n\n"
        "<p align=\"center\">\n"
        f"  <img src=\"{logo_url}\" alt=\"CellFoundry logo\" width=\"360\">\n"
        "</p>\n\n"
        "## Overview\n\n"
        "CellFoundry is a multi-physics, agent-based simulation framework for studying the cellular microenvironment. "
        "It combines interacting cells, extracellular matrix, fibre networks, diffusing chemical species, "
        "mechanical coupling, and more, in a single GPU-accelerated model.\n\n"
        "The framework is designed for in vitro and organoid-scale studies where transport, mechanics, and "
        "microstructure jointly affect cell behaviour. Its modular structure also makes it suitable for parameter "
        "sweeps, digital twin prototyping, and mechanobiology hypothesis testing.\n\n"
        "Model structure and initialization are contained in a single Python file (model.py), while agent interaction implementation is separated into single C++ files. "
        "Agent functions are fully customizable and can be used to simulate a wide range of biological processes"
        "## Core Model Components\n\n"
        "- **Cells (CELL)**: migration, metabolism, stress updates, and interactions with ECM and adhesions.\n"
        "- **Extracellular matrix (ECM)**: concentration fields, diffusion, and voxel-level mechanics.\n"
        "- **Fibre nodes (FNODE)**: network mechanics and boundary interactions.\n"
        "- **Focal adhesions (FOCAD)**: attachment dynamics and force transmission between cells and fibres.\n"
        "- **Boundary/corner agents (BCORNER)**: domain constraints and boundary condition enforcement.\n\n"
        "## Outputs and Analysis\n\n"
        "CellFoundry produces a range of output data that can be analyzed to extract biological insights:\n\n"
        "- **VTK files**: 3D visualization of cell and ECM states over time.\n"
        "- **Pickle snapshots**: Complete model state at specified intervals for detailed analysis.\n"
        "- **Custom output functions**: User-defined functions that extract specific metrics or generate reports.\n\n"
        "## Built on FLAME GPU 2\n\n"
        "CellFoundry is implemented on top of FLAME GPU 2, which provides high-performance GPU execution for "
        "agent-based models.\n\n"
        "- FLAME GPU 2 repository: <https://github.com/FLAMEGPU/FLAMEGPU2>\n"
        "- FLAME GPU 2 documentation: <https://docs.flamegpu.com/>\n"
        "- FLAME GPU 2 examples: <https://github.com/FLAMEGPU/FLAMEGPU2/tree/master/examples>\n\n"
        "## Typical Workflow\n\n"
        "1. Configure model parameters in `model.py` (or through the Model Editor UI).\n"
        "2. Run simulation to produce VTK outputs and optional pickle snapshots.\n"
        "3. Analyze dynamics using scripts in `postprocessing/`.\n"
        "4. Use the generated function reference to inspect model behavior and implementation details.\n"
    )


def render_model_editor_page(github_repo: str, github_ref: str) -> str:
    icon_url = github_raw_url(github_repo, github_ref, "assets/icon.png")
    screenshot_url = github_raw_url(github_repo, github_ref, "assets/parameter_editor.png")
    return (
        "# Model Editor\n\n"
        "<p align=\"left\">\n"
        f"  <img src=\"{icon_url}\" alt=\"Model Editor icon\" width=\"72\">\n"
        "</p>\n\n"
        "## Purpose\n\n"
        "`param_ui.py` provides a custom desktop interface to inspect and edit simulation parameters in `model.py` "
        "without manually searching through the source file. It is designed to speed up model iteration and reduce "
        "editing errors during parameter tuning.\n\n"
        "## What it does\n\n"
        "- Loads `model.py` and indexes key configuration variables.\n"
        "- Exposes grouped controls for frequently tuned parameters (time stepping, boundaries, feature toggles, etc.).\n"
        "- Preserves comments and formatting while patching variable assignments.\n"
        "- Provides a code editor view with syntax highlighting for quick navigation.\n"
        "- Supports launching model runs from the same interface to shorten edit-run cycles.\n\n"
        "## Why use it\n\n"
        "- Faster navigation across large configuration sections.\n"
        "- Reduced risk of syntax mistakes in manual edits.\n"
        "- Better reproducibility when adjusting many parameters across experiments.\n\n"
        "## Interface Preview\n\n"
        f"![Model Parameter Editor screenshot]({screenshot_url})\n"
    )


def render_post_processing_page(github_repo: str, github_ref: str) -> str:
    focad_report = github_blob_url(github_repo, github_ref, "postprocessing/focad_report.py")
    compare_linc = github_blob_url(github_repo, github_ref, "postprocessing/compare_linc_runs.py")
    plot_boundary = github_blob_url(github_repo, github_ref, "postprocessing/plot_boundary_results.py")
    plot_diffusion = github_blob_url(github_repo, github_ref, "postprocessing/plot_diffusion_results.py")
    return (
        "# Post Processing\n\n"
        "The `postprocessing/` folder contains analysis and plotting utilities for simulation outputs in "
        "`result_files/` (CSV, pickle, and VTK-derived signals).\n\n"
        "## Main Scripts\n\n"
        f"- [`focad_report.py`]({focad_report})\n"
        "  - Loads simulation pickle data and exports focal adhesion metrics/polarity time series to CSV.\n"
        "  - Builds summary tables (including last-20% statistics) and generates diagnostic plots.\n"
        f"- [`compare_linc_runs.py`]({compare_linc})\n"
        "  - Compares tagged LINC OFF/ON CSV outputs and produces comparison tables and figures.\n"
        f"- [`plot_boundary_results.py`]({plot_boundary})\n"
        "  - Loads boundary-related outputs from pickle and produces force/position/shear visualizations.\n"
        f"- [`plot_diffusion_results.py`]({plot_diffusion})\n"
        "  - Demonstrates time-series plotting for concentration variables from VTK-derived datasets.\n\n"
        "## Typical Outputs\n\n"
        "- Time-series CSV files for metrics and polarity indicators.\n"
        "- Summary CSV files for run-level comparison.\n"
        "- PNG figures for trends, diagnostics, and side-by-side run comparisons.\n\n"
        "## Recommended Usage\n\n"
        "1. Run simulation and generate outputs in `result_files/`.\n"
        "2. Use `focad_report.py` for baseline reports.\n"
        "3. Use `compare_linc_runs.py` for condition-to-condition analysis.\n"
        "4. Use boundary/diffusion plotting scripts for targeted diagnostics.\n"
    )


def write_if_changed(path: pathlib.Path, content: str) -> bool:
    if path.exists():
        old = path.read_text(encoding="utf-8", errors="ignore")
        if old == content:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs from FLAMEGPU C++ docblocks")
    parser.add_argument("--check", action="store_true", help="Fail if generated files are not up-to-date")
    parser.add_argument("--github-ref", default=GITHUB_REF, help="Git reference for source links (e.g. main, master, v1.0.0)")
    parser.add_argument("--github-repo", default=GITHUB_REPO, help="GitHub repository URL used in source links")
    args = parser.parse_args()

    cpp_files = sorted([p for p in ROOT.glob("*.cpp") if p.is_file()])
    all_docs: Dict[str, List[FunctionDoc]] = {}

    for cpp in cpp_files:
        docs = parse_file(cpp)
        all_docs[cpp.name] = docs

    github_repo = args.github_repo.rstrip("/")
    reference_md = render_reference(all_docs, github_repo, args.github_ref)
    wiki_reference_md = render_reference(all_docs, github_repo, args.github_ref)
    wiki_home_md = render_wiki_home()
    wiki_what_md = render_what_is_cellfoundry(github_repo, args.github_ref)
    wiki_editor_md = render_model_editor_page(github_repo, args.github_ref)
    wiki_post_md = render_post_processing_page(github_repo, args.github_ref)

    out_ref = ROOT / "docs" / "auto" / "Function-Reference.md"
    out_wiki_ref = ROOT / "docs" / "auto" / "wiki" / "Function-Reference.md"
    out_wiki_home = ROOT / "docs" / "auto" / "wiki" / "Home.md"
    out_wiki_what = ROOT / "docs" / "auto" / "wiki" / "What-is-CellFoundry.md"
    out_wiki_editor = ROOT / "docs" / "auto" / "wiki" / "Model-Editor.md"
    out_wiki_post = ROOT / "docs" / "auto" / "wiki" / "Post-Processing.md"

    changed = []
    for path, content in [
        (out_ref, reference_md),
        (out_wiki_ref, wiki_reference_md),
        (out_wiki_home, wiki_home_md),
        (out_wiki_what, wiki_what_md),
        (out_wiki_editor, wiki_editor_md),
        (out_wiki_post, wiki_post_md),
    ]:
        did_change = write_if_changed(path, content)
        if did_change:
            changed.append(str(path.relative_to(ROOT)))

    if args.check and changed:
        print("Generated documentation is out of date:")
        for c in changed:
            print(f" - {c}")
        print("Run: python tools/generate_cpp_docs.py")
        return 1

    if changed:
        print("Updated docs:")
        for c in changed:
            print(f" - {c}")
    else:
        print("Documentation already up to date.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
