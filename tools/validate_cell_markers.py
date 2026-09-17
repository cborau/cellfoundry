"""Validate the complete marker tutorial through model.py and saved outputs.

Run: python tools/validate_cell_markers.py --output tmp/cell_markers_validation
Use flamegpu_py310. The model/checker runs in an isolated source copy.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_variant_refactor import prepare_workspace
from optimizer.optimize import run_trial_subprocess


def validate(output):
    output = Path(output).resolve()
    workspace = prepare_workspace(output)
    original_cwd = Path.cwd()
    report = {}
    try:
        os.chdir(workspace)
        for name, overrides in (
            ("default", {}),
            ("multiscale", {"INCLUDE_DIFFUSION": True, "TIME_STEP_DIFFUSION": [.25, .5, 1.],
                            "N_CELLS": 6, "MARKER_LIFETIME_STEPS": 3,
                            "MARKER_DETECTION_RADIUS": 12.0}),
        ):
            directory = output / name
            directory.mkdir(parents=True, exist_ok=True)
            results = run_trial_subprocess(overrides, str(workspace / "model.py"),
                                           str(directory), timeout=300, variant="cell_markers")
            history = results["MARKER_HISTORY"]
            summary = results["MARKER_SUMMARY"]
            final_ecm = results["ECM_MARKER_FINAL"]
            count = overrides.get("N_CELLS", 4)
            lifetime = overrides.get("MARKER_LIFETIME_STEPS", 2)
            radius = overrides.get("MARKER_DETECTION_RADIUS", 15.)
            assert len(history) == 4
            assert summary["live_markers"].tolist() == [count if s < lifetime else 0 for s in range(1, 5)]
            assert summary["owner_reports"].tolist() == [count if s <= lifetime else 0 for s in range(1, 5)]
            assert summary["emitted_total"].tolist() == [count] * 4
            first_markers = history[0]["markers"]
            assert len({m["id"] for m in first_markers}) == count
            assert {m["owner_cell_id"] for m in first_markers} == {c["id"] for c in history[0]["cells"]}
            for frame in history:
                for cell in frame["cells"]:
                    assert cell["marker_emitted"] == 1
                    assert cell["own_marker_count"] == int(frame["step"] <= lifetime)
                assert all(m["age_steps"] == frame["step"] for m in frame["markers"])

            # Independent geometric check of every ECM node's exposure, not just
            # agreement between two versions of the GPU accumulation code.
            points = final_ecm[["x", "y", "z"]].to_numpy()
            markers = np.array([[m[axis] for axis in ("x", "y", "z")] for m in first_markers])
            distances2 = ((points[:, None, :] - markers[None, :, :]) ** 2).sum(axis=2)
            expected_counts = (distances2 <= radius * radius).sum(axis=1)
            np.testing.assert_allclose(final_ecm["marker_exposure"], expected_counts * lifetime, atol=1e-5)
            assert float(final_ecm["marker_exposure"].sum()) > 0
            assert (final_ecm["marker_count"] == 0).all()  # Empty message list after death.
            np.testing.assert_allclose(summary["ecm_exposure_total"],
                                       [expected_counts.sum() * min(s, lifetime) for s in range(1, 5)])
            assert results["MODEL_CONFIG"].VARIANT_NAME == "cell_markers"
            assert results["MODEL_CONFIG"].VARIANT_PARAMETERS["MARKER_LIFETIME_STEPS"] == lifetime
            assert any("SCALARS marker_emitted int" in p.read_text() and "SCALARS own_marker_count int" in p.read_text()
                       for p in directory.glob("cells_*.vtk"))
            report[name] = {"steps": 4, "markers_born": count, "lifetime_steps": lifetime,
                            "owner_feedback": True, "ecm_exposure": True, "vtk": True, "pickle": True}
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"PASS: cell_markers {name}", flush=True)
    finally:
        os.chdir(original_cwd)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/cell_markers_validation")
    print(json.dumps(validate(parser.parse_args().output), indent=2))
