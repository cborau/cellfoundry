"""GPU checks of optional anchors, division, nucleus mechanics and VTK output.

Run with flamegpu_py310: python tools/validate_cell_anchors.py
Uses a private source copy with explicitly synchronized fixture constants.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_variant_refactor import prepare_workspace
from optimizer.optimize import run_trial_subprocess, OptimizationError


# Appended only to the copied variants. Production schedules stay intact.
FIXTURE = '''
_production_runtime = globals().get("register_runtime")
def register_runtime(ctx):
    import math
    if _production_runtime:
        _production_runtime(ctx)
    enabled = ctx.config["INCLUDE_FOCAL_ADHESIONS"]
    arrays = ("x_i", "y_i", "z_i", "u_ref_x_i", "u_ref_y_i", "u_ref_z_i")
    assert all(ctx.agents["CELL"].hasVariable(name) == enabled for name in arrays)
    def initialize(cell, rng):
        rng.seed(cell.getVariableInt("id"))
        ct = 0
        cell.setVariableInt("cell_type", ct)
        cell.setVariableFloat("clock", ctx.config["CELL_CYCLE_DURATION"][ct] - ctx.config["TIME_STEP"])
        # Two overlapping cells exercise nucleus deformation; division moves
        # parent/daughter apart. Fix their geometry to keep the fixture stable.
        i = cell.getVariableInt("id") - ctx.initial_ids["CELL"].begin
        for axis, value in zip("xyz", (float(i), 0., 0.)):
            cell.setVariableFloat(axis, value)
        for name, value in zip(("orx", "ory", "orz"), (1., 0., 0.)):
            cell.setVariableFloat(name, value)
        if enabled:
            count = ctx.config["N_ANCHOR_POINTS"]
            angles = [2 * math.pi * j / count for j in range(count)]
            refs = ([math.cos(a) for a in angles], [math.sin(a) for a in angles], [0.] * count)
            radius = cell.getVariableFloat("nucleus_radius")
            for axis, ref in zip("xyz", refs):
                cell.setVariableArrayFloat("u_ref_" + axis + "_i", ref)
                cell.setVariableArrayFloat(axis + "_i", [cell.getVariableFloat(axis) + radius * v for v in ref])
        if ctx.agents["CELL"].hasVariable("rg_committed"):
            cell.setVariableInt("rg_committed", 0)
            cell.setVariableFloat("substrate_anchor_x", float(i))
            cell.setVariableFloat("substrate_anchor_y", 0.)
    ctx.add_agent_initializer("CELL", initialize)
    def verify(host):
        population = host.agent("CELL").getPopulationData()
        assert len(population) == 4, len(population)
        rows = []
        for cell in population:
            assert cell.getVariableInt("dead") == 0
            row = {name: cell.getVariableFloat(name) for name in ("x", "y", "z", "radius", "nucleus_radius", "sig_xx", "eps_xx")}
            assert all(math.isfinite(v) for v in row.values())
            assert row["nucleus_radius"] > 0
            row["id"] = cell.getVariableInt("id")
            row["mother_id"] = cell.getVariableInt("mother_id")
            if enabled:
                for name in arrays:
                    values = list(cell.getVariableArrayFloat(name))
                    assert len(values) == ctx.config["N_ANCHOR_POINTS"]
                    assert all(math.isfinite(v) for v in values)
                    row[name] = values
                assert any(abs(v) > 0 for v in row["u_ref_x_i"])
            if ctx.agents["CELL"].hasVariable("substrate_anchor_x"):
                assert math.isfinite(cell.getVariableFloat("substrate_anchor_x"))
            rows.append(row)
        assert any(abs(row["eps_xx"]) > 1e-8 for row in rows), "Nucleus deformation was not exercised"
        daughters = [row for row in rows if row["mother_id"] != -1]
        assert len(daughters) == 2
        if enabled:
            by_id = {row["id"]: row for row in rows}
            for daughter in daughters:
                for axis in "xyz":
                    key = "u_ref_" + axis + "_i"
                    assert daughter[key] == by_id[daughter["mother_id"]][key]
        ctx.runtime_results(host)["ANCHOR_TEST"] = {"enabled": enabled, "cells": sorted(rows, key=lambda r: r["id"])}
    ctx.add_exit_function(verify)
'''


def validate_vtk(path, expected):
    """Every scalar/vector has exactly the advertised point count."""
    lines = path.read_text().splitlines()
    points = next(int(s.split()[1]) for s in lines if s.startswith("POINTS "))
    assert points == expected, (path, points, expected)
    assert next(int(s.split()[1]) for s in lines if s.startswith("POINT_DATA ")) == points
    for i, line in enumerate(lines):
        if line.startswith(("SCALARS ", "VECTORS ")):
            width, start = (1, i + 2) if line.startswith("SCALARS ") else (3, i + 1)
            values = lines[start:start + points]
            assert len(values) == points
            assert all(len(v.split()) == width for v in values), (path, line)
            assert all(np.isfinite(float(n)) for v in values for n in v.split())


def validate(output):
    output = Path(output).resolve()
    workspace = prepare_workspace(output)
    # Use the template's explicit base schedule for the base-kernel fixture.
    import shutil
    shutil.copytree(workspace / "variants/variant_template", workspace / "variants/anchor_base", dirs_exist_ok=True)
    for variant in ("anchor_base", "organoid", "radial_glia"):
        path = workspace / "variants" / variant / "__init__.py"
        with path.open("a", encoding="utf-8") as file:
            file.write(FIXTURE)
    # A short fibre near the two cells; irregular bounds are intentional here.
    network = workspace / "anchor_test_network.pkl"
    with network.open("wb") as file:
        pickle.dump({"node_coords": np.array([[-5., 2., 0.], [5., 2., 0.]]),
                     "connectivity": {0: [1] + [-1] * 7, 1: [0] + [-1] * 7}}, file)
    common = dict(STEPS=2, TIME_STEP=1., N_CELLS=2, INCLUDE_CELLS=True,
                  INCLUDE_CELL_CELL_INTERACTION=True, INCLUDE_CELL_CYCLE=True,
                  INCLUDE_DIFFUSION=False, HETEROGENEOUS_DIFFUSION=False,
                  INCLUDE_VASCULARIZATION=False, INCLUDE_NETWORK_REMODELING=False,
                  INCLUDE_LUMEN=False, INCLUDE_CELL_FNODE_REPULSION=False,
                  INCLUDE_CHEMOTAXIS=False, INCLUDE_CHEMOKINESIS=False,
                  BOUNDARY_DISP_RATES=[0.] * 6, BOUNDARY_DISP_RATES_PARALLEL=[0.] * 12,
                  VISUALISATION=False, SHOW_PLOTS=False, DEBUG_PRINTING=False,
                  SAVE_PICKLE=True, SAVE_DATA_TO_FILE=True, SAVE_EVERY_N_STEPS=1,
                  SAVE_NO_ANCHOR_CELL_FILES=True, CELL_RADIUS=[3.] * 3,
                  ORGANOID_INIT_RADIUS=12., MONOLAYER_CLUSTER_RADIUS=20., MONOLAYER_Z=0.,
                  MONOLAYER_CELL_TYPE_RATIOS=[1., 0., 0.], DIVISION_RATE_MULTIPLIER=[1.] * 3,
                  CELL_HYPOXIA_DAMAGE_RATE=[0.] * 3, CELL_NUTRIENT_DAMAGE_RATE=[0.] * 3,
                  CELL_STRESS_DAMAGE_RATE=[0.] * 3, NETWORK_FILE=str(network),
                  ALLOW_IRREGULAR_NETWORK=True, FIBRE_SEGMENT_EQUILIBRIUM_DISTANCE=10.,
                  INIT_N_FOCAD_PER_CELL=2, ENABLE_FOCAD_BIRTH=False)
    report = {}
    previous = Path.cwd()
    try:
        os.chdir(workspace)
        for variant in ("anchor_base", "organoid", "radial_glia"):
            for enabled in (False, True):
                name = f"{variant}_{'on' if enabled else 'off'}"
                run = output / name
                run.mkdir(exist_ok=True)
                params = dict(common, INCLUDE_FOCAL_ADHESIONS=enabled, INCLUDE_FIBRE_NETWORK=enabled,
                              PERIODIC_BOUNDARIES_FOR_CELLS=not enabled,
                              INCLUDE_DIFFUSION=variant == "radial_glia")
                if variant == "radial_glia" and enabled:
                    # RG intentionally does not schedule FNODE/FOCAD. Preserve
                    # that constraint; testing anchors must not bypass it.
                    try:
                        run_trial_subprocess(params, str(workspace / "model.py"), str(run), timeout=300, variant=variant)
                    except OptimizationError as error:
                        assert "radial_glia configure_layers does not schedule" in str(error)
                    else:
                        raise AssertionError("Unsupported RG focal-adhesion setup was accepted")
                    report[name] = {"unsupported_configuration_rejected": True}
                    (output / "report.json").write_text(json.dumps(report, indent=2))
                    print(f"PASS: {name} rejected before simulation", flush=True)
                    continue
                result = run_trial_subprocess(params, str(workspace / "model.py"), str(run), timeout=300, variant=variant)
                data = result["ANCHOR_TEST"]
                for path in run.glob("cells_t*.vtk"):
                    validate_vtk(path, 4 * (51 if enabled else 1))
                assert list(run.glob("nucleus_t*.vtk")), name
                assert bool(list(run.rglob("*no_anchor*.vtk"))) == enabled, name
                report[name] = data
                (output / "report.json").write_text(json.dumps(report, indent=2))
                print(f"PASS: {name}", flush=True)
    finally:
        os.chdir(previous)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/anchor_refactor/division")
    validate(parser.parse_args().output)
