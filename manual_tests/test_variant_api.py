"""Variant configuration, schema, initialization, metrics and schedule contracts.

Run: python -m unittest manual_tests.test_variant_api
No CUDA is required; GPU/optimizer/output checks are in tools/validate_variant_refactor.py.
"""
from copy import deepcopy
from pathlib import Path
import ast
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from helper_module import apply_param_overrides
from variant_api import (VariantContext, configuration_snapshot, load_variant,
                         register_parameter_defaults)
from manual_tests.test_model_initialization_and_layers import _FakeModel, _run_default_layer_builder

ROOT = Path(__file__).resolve().parents[1]


def config_for(variant):
    config = dict(N_CELL_TYPES=3, N_SPECIES=3, ENSEMBLE=False,
                  MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION=30.,
                  CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER=1.,
                  MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER=2.,
                  BROWNIAN_MOTION_STRENGTH_FACTOR=[2.] * 3,
                  MOVING_BOUNDARIES=False, MULTISCALE_DIFFUSION=False,
                  INCLUDE_VASCULAR_CELL_RECRUITMENT=False,
                  INCLUDE_NETWORK_REMODELING=False,
                  INCLUDE_CELL_FNODE_REPULSION=False,
                  HETEROGENEOUS_DIFFUSION=False, SAVE_PICKLE=True)
    register_parameter_defaults(config, variant)
    config.update(deepcopy(variant.PARAMS))
    return config


class Schema:
    def __init__(self):
        self.fields = {}
        self.properties = {}
        self.functions = {}

    def _add(self, store, name, value):
        if name in store:
            raise ValueError(f"Duplicate {name}")
        store[name] = value

    def newVariableFloat(self, name, default=0.):
        self._add(self.fields, name, ("float", default))

    def newVariableInt(self, name, default=0):
        self._add(self.fields, name, ("int", default))

    def newPropertyFloat(self, name, value):
        self._add(self.properties, name, float(value))

    def newPropertyArrayFloat(self, name, value):
        self._add(self.properties, name, list(value))

    def setRadius(self, radius):
        self.radius = radius

    def newRTCFunctionFile(self, name, path):
        function = SimpleNamespace(path=path)
        function.setMessageInput = lambda message: setattr(function, "message", message)
        self._add(self.functions, name, function)
        return function


class VariantContracts(unittest.TestCase):
    def setUp(self):
        self.rg = load_variant(ROOT, "radial_glia")
        self.org = load_variant(ROOT, "organoid")

    def context(self, variant):
        env, cell, message = Schema(), Schema(), Schema()
        ctx = VariantContext(SimpleNamespace(Environment=lambda: env), config_for(variant), ROOT)
        ctx.agents["CELL"] = cell
        ctx.messages["cell_spatial_location_message"] = message
        return ctx

    def test_new_parameters_exist_before_optimizer_overrides(self):
        for variant, key, value in [(self.rg, "RG_COMMIT_RATE", .000123),
                                    (self.org, "ORGANOID_CONTACT_INHIBIT_SIGMA", 2.75)]:
            ctx = self.context(variant)
            apply_param_overrides(ctx.config, {key: value})
            variant.declare_model(ctx)
            self.assertEqual(ctx.env.properties[key], value)

    def test_indexed_and_scalar_overrides_do_not_mutate_variant_defaults(self):
        config = config_for(self.rg)
        config["CELL_SPEED_REF"] = [1., 1., 1.]
        apply_param_overrides(config, {"RG_ADHESION_MATRIX[8]": 2.25, "CELL_SPEED_REF": .01})
        self.assertEqual(config["RG_ADHESION_MATRIX"][8], 2.25)
        self.assertEqual(config["CELL_SPEED_REF"], [.01] * 3)
        self.assertEqual(self.rg.PARAM_DEFAULTS["RG_ADHESION_MATRIX"][8], 1.5)

    def test_snapshot_cannot_modify_model_namespace(self):
        namespace = {"A": [1, 2], "_PRIVATE": 3, "model": object()}
        config = configuration_snapshot(namespace)
        config["A"][0] = 9
        self.assertEqual(namespace["A"], [1, 2])
        self.assertEqual(list(config), ["A"])
        with self.assertRaises(TypeError):
            config["B"] = 3

    def test_parameter_collisions_fail(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            register_parameter_defaults({"RG_COMMIT_RATE": 1}, self.rg)

    def test_shipped_parameter_defaults_do_not_collide_with_core_names(self):
        tree = ast.parse((ROOT / "model.py").read_text(encoding="utf-8"))
        namespace = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                if "_ACTIVE_VARIANT" in names:
                    break
                namespace.update({name: None for name in names})
        for variant in (self.org, self.rg):
            register_parameter_defaults(dict(namespace), variant)

    def test_rg_schema_matches_existing_contract(self):
        ctx = self.context(self.rg)
        self.rg.declare_model(ctx)
        expected = {"rg_commit_level", "epithelialization_level", "rosette_maturity",
                    "apx", "apy", "apz", "rg_neighbour_density", "morphogen_local",
                    "rg_committed", "substrate_anchor_x", "substrate_anchor_y"}
        self.assertEqual(set(ctx.agents["CELL"].fields), expected)
        self.assertEqual(ctx.agents["CELL"].fields["rg_committed"], ("int", 0))
        self.assertEqual(set(ctx.messages["cell_spatial_location_message"].fields),
                         {"rg_commit_level", "epithelialization_level", "apx", "apy", "apz"})
        self.assertEqual(len(ctx.env.properties), 25)
        self.assertNotIn("MIN_ROSETTE_SIZE", ctx.env.properties)  # Host-only analysis parameter.

    def test_rg_message_covers_all_consumers(self):
        for interaction_radius, polarity_radius, expected in [(30., 84., 84.), (100., 84., 100.)]:
            ctx = self.context(self.rg)
            ctx.config.update(MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION=interaction_radius,
                              RG_LUMEN_SEARCH_RADIUS=polarity_radius)
            self.rg.declare_model(ctx)
            self.assertEqual(ctx.messages["cell_spatial_location_message"].radius, expected)

    def test_registration_errors_are_not_swallowed(self):
        ctx = self.context(self.org)
        self.org.declare_model(ctx)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.org.declare_model(ctx)

    def test_rg_functions_are_registered_outside_schedule(self):
        ctx = self.context(self.rg)
        self.rg.register_functions(ctx)
        self.assertEqual(set(ctx.functions), {"CELL.cell_rg_differentiation", "CELL.cell_rg_polarity_update"})
        for function in ctx.functions.values():
            self.assertTrue(Path(function.path).is_file())
            self.assertEqual(function.message, "cell_spatial_location_message")

    def test_organoid_does_not_extend_cell_or_require_assay_flag(self):
        ctx = self.context(self.org)
        ctx.config["ORGANOID_ASSAY"] = False
        self.org.validate_config(ctx.config)
        self.org.declare_model(ctx)
        self.assertFalse(ctx.agents["CELL"].fields)
        self.assertEqual(set(ctx.env.properties), set(self.org.PARAM_DEFAULTS))

    def test_rg_rejects_unsupported_features_and_invalid_arrays(self):
        config = config_for(self.rg)
        self.rg.validate_config(config)
        for overrides in ({"INCLUDE_FIBRE_NETWORK": True}, {"N_SPECIES": 2},
                          {"RG_ADHESION_MATRIX": [1.]}, {"RG_COMMIT_RATE": float("nan")},
                          {"RG_SYMMETRIC_DIVISION_PROB": 1.1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.rg.validate_config({**config, **overrides})

    def test_organoid_full_schedule_matches_core_optional_features(self):
        flags = dict(INCLUDE_VASCULARIZATION=True, INCLUDE_VASCULAR_CELL_RECRUITMENT=True,
                     INCLUDE_DIFFUSION=True, MULTISCALE_DIFFUSION=True, MONOLAYER_ASSAY=False,
                     MOVING_BOUNDARIES=True, INCLUDE_CELLS=True, INCLUDE_FOCAL_ADHESIONS=True,
                     ORGANOID_ASSAY=True, INCLUDE_LUMEN=True, INCLUDE_FIBRE_NETWORK=True,
                     INCLUDE_NETWORK_REMODELING=True, INCLUDE_CELL_CYCLE=True,
                     HETEROGENEOUS_DIFFUSION=True, INCLUDE_CELL_CELL_INTERACTION=True,
                     INCLUDE_CELL_FNODE_REPULSION=True)
        for enabled in (False, True):
            config = {name: value and enabled for name, value in flags.items()}
            model = _FakeModel()
            ctx = VariantContext(model, config, ROOT)
            ctx.add_multiscale_diffusion_layers = lambda: model.events.append(
                ("L5_Transport", "ECM", "subcycled_diffusion"))
            self.org.configure_layers(ctx)
            self.assertEqual(model.events, _run_default_layer_builder(**config))

    def test_loader_supports_relative_imports_and_requires_full_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "variants" / "example"
            package.mkdir(parents=True)
            (package / "values.py").write_text("VALUE = 7\n")
            (package / "__init__.py").write_text(
                "from .values import VALUE\ndef configure_layers(ctx): pass\n")
            self.assertEqual(load_variant(directory, "example").VALUE, 7)
            (package / "__init__.py").write_text("PARAMS = {}\n")
            with self.assertRaisesRegex(TypeError, "configure_layers"):
                load_variant(directory, "example")

    def test_diagnostics_reads_new_parameter_dictionary(self):
        from variants.radial_glia.rg_rosette_diagnostics import read_variant_config
        values = read_variant_config(ROOT / "variants/radial_glia/__init__.py")
        for name, value in self.rg.PARAM_DEFAULTS.items():
            self.assertEqual(values[name], value)


class Cell:
    def __init__(self, x=0., y=0., cell_type=2, dead=0):
        self.values = dict(x=x, y=y, z=0., id=1, cell_type=cell_type, dead=dead,
                           mother_id=-1, rg_commit_level=.9, epithelialization_level=.8,
                           rosette_maturity=.8, rg_neighbour_density=2., morphogen_local=.2,
                           rg_committed=1, apx=0., apy=0., apz=.5)

    def getVariableFloat(self, name): return float(self.values[name])
    def getVariableInt(self, name): return int(self.values[name])
    def setVariableFloat(self, name, value): self.values[name] = float(value)
    def setVariableInt(self, name, value): self.values[name] = int(value)


class RuntimeContracts(unittest.TestCase):
    def test_initializer_preserves_rng_and_position_dependent_fields(self):
        from variants.radial_glia.runtime import initialize_cell
        cell, rng, reference = Cell(4., -7.), np.random.RandomState(42), np.random.RandomState(42)
        angle = reference.uniform(0., 2. * np.pi)
        initialize_cell(cell, rng)
        self.assertAlmostEqual(cell.values["apx"], np.cos(angle))
        self.assertAlmostEqual(cell.values["apy"], np.sin(angle))
        self.assertEqual(cell.values["substrate_anchor_x"], 4.)
        self.assertEqual(cell.values["substrate_anchor_y"], -7.)
        self.assertEqual(cell.values["rg_committed"], 0)
        self.assertEqual(rng.uniform(), reference.uniform())

    def test_metrics_keep_output_schema_and_separate_ensemble_runs(self):
        from variants.radial_glia.runtime import Metrics
        cells = [Cell(x, y) for x, y in [(0, 0), (0, 1), (1, 0), (20, 20), (20, 21), (21, 20)]]
        cells += [Cell(cell_type=0), Cell(dead=1)]
        host = SimpleNamespace(getEnsembleRunIndex=lambda: 0, getStepCounter=lambda: 2,
                               agent=lambda name: SimpleNamespace(getPopulationData=lambda: cells),
                               environment=SimpleNamespace(getPropertyFloat=lambda name: 1.))
        ctx = VariantContext(None, dict(STEPS=3, SAVE_EVERY_N_STEPS=2, CELL_RADIUS=[1.] * 3,
                                       MIN_ROSETTE_SIZE=3, ENSEMBLE=True), ROOT)
        metrics = Metrics(ctx)
        metrics.initialize(host)
        metrics.run(host)
        results = ctx.merge_results({"CORE": 1}, 0)
        row = results["RG_ROSETTE_METRICS_OVER_TIME"].iloc[0]
        self.assertEqual(row["step"], 3)
        self.assertEqual(row["n_alive_total"], 7)
        self.assertEqual(row["n_alive_rg"], 6)
        self.assertEqual(row["n_large_rg_clusters"], 2)
        self.assertEqual(row["large_cluster_mean_size"], 3)
        self.assertAlmostEqual(row["mean_cluster_compactness"], 1 / 3)
        self.assertEqual(row["mean_apz"], .5)
        self.assertEqual(len(results["RG_FINAL_METRICS"]), 8)
        host.getEnsembleRunIndex = lambda: 1
        metrics.initialize(host)
        self.assertTrue(ctx.results_for(1)["RG_FINAL_METRICS"].empty)
        self.assertEqual(len(ctx.results_for(0)["RG_FINAL_METRICS"]), 8)
        with self.assertRaisesRegex(ValueError, "overwrites"):
            ctx.merge_results({"RG_FINAL_METRICS": []}, 0)


if __name__ == "__main__":
    unittest.main()
