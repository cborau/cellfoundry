#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class FunctionDoc:
    file: str
    kind: str
    name: str
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
        low = line.lower()
        if low.startswith("purpose:"):
            section = "purpose"
            purpose = line.split(":", 1)[1].strip()
            continue
        if low.startswith("inputs:"):
            section = "inputs"
            rest = line.split(":", 1)[1].strip()
            if rest:
                inputs.append(rest)
            continue
        if low.startswith("outputs:"):
            section = "outputs"
            rest = line.split(":", 1)[1].strip()
            if rest:
                outputs.append(rest)
            continue
        if low.startswith("notes:"):
            section = "notes"
            rest = line.split(":", 1)[1].strip()
            if rest:
                notes.append(rest)
            continue

        if line.startswith("-"):
            item = line[1:].strip()
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
                purpose = f"{purpose} {line}".strip()
            else:
                purpose = line
        elif section == "notes":
            notes.append(line)

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

        name, kind = extract_name_and_kind(signature)
        lines = clean_docblock(raw_doc)
        purpose, inputs, outputs, notes = parse_sections(lines)

        docs.append(
            FunctionDoc(
                file=path.name,
                kind=kind,
                name=name,
                purpose=purpose or "(not specified)",
                inputs=inputs,
                outputs=outputs,
                notes=notes,
            )
        )

    return docs


def render_reference(all_docs: Dict[str, List[FunctionDoc]]) -> str:
    out: List[str] = []
    out.append("# C++ Function Reference\n")
    out.append("Generated automatically from Doxygen-style docblocks in `.cpp` files.\n")

    for file_name in sorted(all_docs.keys()):
        entries = all_docs[file_name]
        if not entries:
            continue
        out.append(f"## {file_name}\n")
        for entry in entries:
            out.append(f"### {entry.name} ({entry.kind})\n")
            out.append(f"- **Purpose:** {entry.purpose}")
            if entry.inputs:
                out.append("- **Inputs:**")
                out.extend([f"  - {x}" for x in entry.inputs])
            if entry.outputs:
                out.append("- **Outputs:**")
                out.extend([f"  - {x}" for x in entry.outputs])
            if entry.notes:
                out.append("- **Notes:**")
                out.extend([f"  - {x}" for x in entry.notes])
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_wiki_home() -> str:
    return (
        "# CellFoundry Wiki\n\n"
        "This wiki content is generated from source-code docblocks.\n\n"
        "- [C++ Function Reference](Function-Reference)\n"
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
    args = parser.parse_args()

    cpp_files = sorted([p for p in ROOT.glob("*.cpp") if p.is_file()])
    all_docs: Dict[str, List[FunctionDoc]] = {}

    for cpp in cpp_files:
        docs = parse_file(cpp)
        all_docs[cpp.name] = docs

    reference_md = render_reference(all_docs)
    wiki_home_md = render_wiki_home()

    out_ref = ROOT / "docs" / "auto" / "Function-Reference.md"
    out_wiki_ref = ROOT / "docs" / "auto" / "wiki" / "Function-Reference.md"
    out_wiki_home = ROOT / "docs" / "auto" / "wiki" / "Home.md"

    changed = []
    for path, content in [
        (out_ref, reference_md),
        (out_wiki_ref, reference_md),
        (out_wiki_home, wiki_home_md),
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
