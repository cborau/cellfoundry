"""CPU-only tests for diffusion plotting data selection and CLI exports."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postprocessing import plot_diffusion_results as plotting
from postprocessing import diffusion_plane_maps as maps


def write_vtk(path, points, **scalars):
    lines = ["# vtk DataFile Version 3.0", "test", "ASCII", "DATASET POLYDATA",
             f"POINTS {len(points)} float"]
    lines.extend(" ".join(map(str, p)) for p in points)
    lines.append(f"POINT_DATA {len(points)}")
    for name, values in scalars.items():
        lines.extend([f"SCALARS {name} float 1", "LOOKUP_TABLE default",
                      " ".join(map(str, values))])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class DiffusionPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def args(self, *extra):
        return plotting.build_parser().parse_args(["--results-dir", str(self.folder), *extra])

    def test_discovery_sorts_numeric_steps_and_filters_before_loading(self):
        for step in [10, 2, 1, 20]:
            (self.folder / f"ecm_data_t{step}.vtk").touch()
        series = plotting.discover_series(self.args("--start-step", "2", "--every", "2"))
        self.assertEqual(list(series), ["ecm"])
        self.assertEqual([plotting.saved_step(p) for p in series["ecm"]], [2, 20])
        with self.assertRaisesRegex(ValueError, "No cells files"):
            plotting.discover_series(self.args("--datasets", "cells"))

    def test_corner_and_anchor_filtering_preserves_original_point_indices(self):
        path = self.folder / "ecm_data_t1.vtk"
        write_vtk(path, [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                  is_corner=[1, 0, 0], concentration_species_0=[99, 2, 4])
        ecm = plotting.read_snapshot(path, "ecm")
        self.assertEqual(ecm.point_id.tolist(), [1, 2])
        self.assertEqual(plotting.nearest_row(ecm, [0, 0, 0])["concentration_species_0"], 2)
        path = self.folder / "cells_t1.vtk"
        write_vtk(path, [[0, 0, 0], [3, 0, 0], [0.1, 0, 0], [3.1, 0, 0]],
                  id=[11, 22, 11, 22], concentration_species_0=[2, 6, 100, 100])
        cells = plotting.read_snapshot(path, "cells")
        self.assertEqual(cells.point_id.tolist(), [0, 1])
        self.assertEqual(cells.id.tolist(), [11, 22])
        self.assertEqual(plotting.scalar_summary(cells.concentration_species_0.to_numpy())["mean"], 4)

    def test_fixed_tracking_uses_agent_id_after_movement_and_reordering(self):
        first = pd.DataFrame({"point_id": [0, 1], "id": [11, 22],
                              "x": [0, 4], "y": [0, 0], "z": [0, 0]})
        later = pd.DataFrame({"point_id": [0, 1], "id": [22, 11],
                              "x": [0, 4], "y": [0, 0], "z": [0, 0]})
        key = plotting.tracking_key(first, [0, 0, 0])
        self.assertEqual(key, ("id", 11))
        self.assertEqual(plotting.sample_probe(later, [0, 0, 0], key)["id"], 11)
        self.assertEqual(plotting.sample_probe(later, [0, 0, 0], None)["id"], 22)
        self.assertIsNone(plotting.sample_probe(later.iloc[:1], [0, 0, 0], key))

    def test_fixed_point_fallback_keeps_index_after_marker_removal(self):
        frame = pd.DataFrame({"point_id": [8, 9], "x": [0, 1], "y": [0, 0], "z": [0, 0]})
        self.assertEqual(plotting.tracking_key(frame, [0, 0, 0]), ("point_id", 8))

    def test_reference_selection_and_missing_agents_leave_trace_gaps(self):
        for step, ids, concentrations in [(1, [22], [2]), (5, [22, 11], [2, 3]), (10, [22], [2])]:
            write_vtk(self.folder / f"cells_t{step}.vtk", [[3, 0, 0], [0, 0, 0]][:len(ids)],
                      id=ids, concentration_species_0=concentrations)
        args = self.args("--nearest-mode", "fixed_id", "--reference-step", "5")
        series = plotting.discover_series(args)
        snapshots = {"cells": plotting.read_snapshot(series["cells"][0], "cells")}
        with redirect_stdout(io.StringIO()):
            traces, _, _ = plotting.collect_data(args, series, snapshots, ["concentration_species_0"], 2)
        self.assertEqual(traces.step.tolist(), [1, 5, 10])
        self.assertEqual(traces.time.tolist(), [2, 10, 20])
        self.assertEqual(traces.id.iloc[1], 11)
        self.assertTrue(np.isnan(traces.value.iloc[[0, 2]]).all())

    def test_summary_counts_invalid_values_without_hiding_them(self):
        result = plotting.scalar_summary(np.array([-1, 1, 3, np.nan, np.inf]))
        self.assertEqual(result, dict(count=5, finite_count=3, nonfinite_count=2,
                                     negative_count=1, min=-1, mean=1, max=3))
        self.assertTrue(np.isnan(plotting.scalar_summary(np.array([]))["mean"]))

    def test_profile_picks_one_equidistant_line_instead_of_averaging(self):
        frame = pd.DataFrame({"x": [-1, 1, -1, 1], "y": [-1, -1, 1, 1], "z": [0] * 4,
                              "concentration_species_0": [2, 3, 8, 9]})
        line = plotting.profile_line(frame, "x", [0, 0, 0])
        self.assertEqual(line.y.tolist(), [-1, -1])
        self.assertEqual(line.concentration_species_0.tolist(), [2, 3])

    def test_cli_ecm_only_discovers_species_and_exports_correct_time_and_profiles(self):
        for step in [1, 5, 10]:
            write_vtk(self.folder / f"ecm_data_t{step}.vtk", [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
                      concentration_species_10=[0, step, 2], concentration_species_2=[1, 1, 1])
        (self.folder / "parameters.json").write_text(json.dumps({"TIME_STEP": 2, "SAVE_EVERY_N_STEPS": 5}))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(plotting.main(["--results-dir", str(self.folder), "--plots",
                                            "traces", "summary", "profiles", "--probe", "0", "0", "0",
                                            "--probe", "20", "0", "0"]), 0)
        outdir = self.folder / "diffusion_plots"
        for kind in ["traces", "summary", "profiles"]:
            self.assertGreater((outdir / f"diffusion_{kind}.png").stat().st_size, 1000)
            self.assertTrue((outdir / f"diffusion_{kind}.csv").is_file())
        traces = pd.read_csv(outdir / "diffusion_traces.csv")
        self.assertEqual(traces.variable.unique().tolist(), ["concentration_species_2", "concentration_species_10"])
        self.assertEqual(traces.time.unique().tolist(), [2, 10, 20])
        profiles = pd.read_csv(outdir / "diffusion_profiles.csv")
        self.assertEqual(profiles.step.unique().tolist(), [1, 10])
        self.assertEqual(len(profiles), 12)  # Two probes on the same grid line produce one profile.

    def test_bad_options_and_missing_species_have_actionable_errors(self):
        for arguments in [("--every", "0"), ("--species", "-1"), ("--probe", "nan", "0", "0"),
                          ("--reference-step", "1"), ("--start-step", "10", "--end-step", "1")]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                plotting.validate_args(self.args(*arguments))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            plotting.resolve_time_step(self.args("--time-step", "0"))
        with self.assertRaisesRegex(ValueError, "Scalars not found"):
            plotting.choose_variables(self.args("--species", "7"), {"ecm": pd.DataFrame({"concentration_species_0": []})})

    def test_exact_steps_are_sorted_and_missing_steps_are_reported(self):
        for step in [1, 5, 10]:
            (self.folder / f"ecm_data_t{step}.vtk").touch()
        series = plotting.discover_series(self.args("--steps", "10", "1"))
        self.assertEqual([plotting.saved_step(path) for path in series["ecm"]], [1, 10])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            plotting.discover_series(self.args("--steps", "4"))

    def plane_frame(self):
        points = [[x, y, z] for x in [-2, 0, 2] for y in [-3, 0, 3] for z in [-4, 0, 4]]
        frame = pd.DataFrame(points, columns=["x", "y", "z"])
        frame["concentration_species_0"] = frame.x + 10 * frame.y + 100 * frame.z
        return frame

    def test_plane_maps_preserve_orientation_pixel_values_and_nearest_slice(self):
        frame = self.plane_frame().sample(frac=1, random_state=2)
        for axis in "xyz":
            section = maps.extract_plane(frame, maps.Plane(axis, 0.5), ["concentration_species_0"])
            self.assertEqual(section["actual"], 0)
            horizontal, vertical = [c for c in "xyz" if c != axis]
            self.assertEqual(section["horizontal"], horizontal)
            self.assertEqual(section["vertical"], vertical)
            values = section["values"]["concentration_species_0"]
            self.assertEqual(values.shape, (3, 3))
            for j, y in enumerate(section["y"]):
                for i, x in enumerate(section["x"]):
                    point = {axis: 0, horizontal: x, vertical: y}
                    self.assertEqual(values[j, i], point["x"] + 10 * point["y"] + 100 * point["z"])

    def test_plane_validation_rejects_duplicate_directions_and_incomplete_grids(self):
        for specs in [["z=0", "z=2"], ["x=nan"], ["q=0"]]:
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                maps.resolve_planes(specs)
        frame = self.plane_frame()
        with self.assertRaisesRegex(ValueError, "outside"):
            maps.extract_plane(frame, maps.Plane("z", 99), ["concentration_species_0"])
        with self.assertRaisesRegex(ValueError, "complete axis-aligned"):
            maps.extract_plane(frame.drop(index=13), maps.Plane("z", 0), ["concentration_species_0"])

    def test_map_smoothing_changes_display_only_and_keeps_common_color_scale(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        section = maps.extract_plane(self.plane_frame(), maps.Plane("z", 0), ["concentration_species_0"])
        fig, axes = plt.subplots(1, 2)
        self.addCleanup(plt.close, fig)
        for ax, smooth in zip(axes, [False, True]):
            image = maps.draw_map(ax, section, "concentration_species_0", (-432, 432), smooth=smooth)
            self.assertEqual(image.get_interpolation(), "bilinear" if smooth else "nearest")
            self.assertEqual(image.get_clim(), (-432, 432))
            np.testing.assert_array_equal(image.get_array(), section["values"]["concentration_species_0"])

    def test_map_cli_defaults_include_all_species_planes_and_times_across_pages(self):
        frame = self.plane_frame()
        for step in [1, 5, 10]:
            write_vtk(self.folder / f"ecm_data_t{step}.vtk", frame[["x", "y", "z"]].to_numpy(),
                      concentration_species_0=frame.concentration_species_0.to_numpy() * step,
                      concentration_species_2=np.zeros(len(frame)))
        with redirect_stdout(io.StringIO()):
            plotting.run(self.args("--plots", "maps", "--map-columns", "2", "--time-step", "2"))
        outdir = self.folder / "diffusion_plots"
        manifest = json.loads((outdir / "diffusion_maps.json").read_text())
        self.assertEqual(manifest["planes"], ["x=0", "y=0", "z=0"])
        self.assertEqual(manifest["variables"], ["concentration_species_0", "concentration_species_2"])
        self.assertEqual(manifest["steps"], [1, 5, 10])
        self.assertEqual(manifest["layout"], "by_species")
        self.assertFalse(manifest["smooth"])
        self.assertEqual([page["steps"] for page in manifest["pages"]], [[1, 5], [1, 5], [10], [10]])
        self.assertEqual(manifest["color_limits"]["concentration_species_0"], [-4320, 4320])
        for page in manifest["pages"]:
            self.assertGreater((outdir / page["files"][0]).stat().st_size, 1000)

    def test_map_cli_respects_species_plane_and_step_selection(self):
        frame = self.plane_frame()
        for step in [1, 5, 10]:
            write_vtk(self.folder / f"ecm_data_t{step}.vtk", frame[["x", "y", "z"]].to_numpy(),
                      concentration_species_0=np.full(len(frame), step),
                      concentration_species_2=frame.concentration_species_0.to_numpy())
        with redirect_stdout(io.StringIO()):
            plotting.run(self.args("--plots", "maps", "--species", "2", "--planes", "z=0.5",
                                   "--steps", "1", "10", "--smooth", "--map-layout", "by_plane"))
        manifest = json.loads((self.folder / "diffusion_plots/diffusion_maps.json").read_text())
        self.assertEqual(manifest["variables"], ["concentration_species_2"])
        self.assertEqual(manifest["planes"], ["z=0.5"])
        self.assertEqual(manifest["steps"], [1, 10])
        self.assertTrue(manifest["smooth"])
        self.assertEqual(manifest["pages"][0]["sampled_planes"], [{"z=0.5": 0}, {"z=0.5": 0}])


if __name__ == "__main__":
    unittest.main()
