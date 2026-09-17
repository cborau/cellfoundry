"""CPU checks for the tutorial's explicit schedule and configuration contract."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from variant_api import load_variant
from manual_tests.test_model_initialization_and_layers import _FakeModel, _run_default_layer_builder

ROOT = Path(__file__).resolve().parents[1]


class MarkerTutorialTests(unittest.TestCase):
    def setUp(self):
        self.variant = load_variant(ROOT, "cell_markers")

    def test_defaults_and_unsupported_population_changes(self):
        config = {**self.variant.PARAMS, **self.variant.PARAM_DEFAULTS}
        self.variant.validate_config(config)
        for change in ({"INCLUDE_CELLS": False}, {"INCLUDE_CELL_CYCLE": True},
                       {"INCLUDE_VASCULAR_CELL_RECRUITMENT": True}, {"MARKER_CAPACITY": 3},
                       {"MARKER_LIFETIME_STEPS": 0}, {"MARKER_LIFETIME_STEPS": 1.5},
                       {"MARKER_DETECTION_RADIUS": float("nan")}, {"MARKER_DETECTION_RADIUS": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.variant.validate_config({**config, **change})

    def test_template_has_all_entry_points_and_optional_hooks_are_noops(self):
        template = load_variant(ROOT, "variant_template")
        self.assertEqual(template.PARAM_DEFAULTS, {})
        self.assertEqual(template.PARAMS, {})
        self.assertEqual(template.FILES, {})
        template.validate_config({})
        template.declare_model(None)
        template.register_functions(None)
        template.register_runtime(None)

    def test_whole_generic_schedule_is_preserved_before_marker_lifecycle(self):
        flags = dict(INCLUDE_VASCULARIZATION=True, INCLUDE_VASCULAR_CELL_RECRUITMENT=True,
                     INCLUDE_DIFFUSION=True, MULTISCALE_DIFFUSION=True, MONOLAYER_ASSAY=False,
                     MOVING_BOUNDARIES=True, INCLUDE_CELLS=True, INCLUDE_FOCAL_ADHESIONS=True,
                     ORGANOID_ASSAY=True, INCLUDE_LUMEN=True, INCLUDE_FIBRE_NETWORK=True,
                     INCLUDE_NETWORK_REMODELING=True, INCLUDE_CELL_CYCLE=True,
                     HETEROGENEOUS_DIFFUSION=True, INCLUDE_CELL_CELL_INTERACTION=True,
                     INCLUDE_CELL_FNODE_REPULSION=True)
        for enabled in (False, True):
            config = {name: value and enabled for name, value in flags.items()}
            # CELL creation is required, while CELL birth functions are explicitly unsupported.
            config.update(INCLUDE_CELLS=True, INCLUDE_CELL_CYCLE=False, INCLUDE_VASCULAR_CELL_RECRUITMENT=False)
            model = _FakeModel()
            ctx = SimpleNamespace(model=model, config=config,
                                  add_multiscale_diffusion_layers=lambda: model.events.append(
                                      ("L5_Transport", "ECM", "subcycled_diffusion")))
            self.variant.configure_layers(ctx)
            self.assertEqual(model.events[:-5], _run_default_layer_builder(**config))
            self.assertEqual(model.events[-5:], [
                ("M1_CELL_Emit", "CELL", "cell_emit_marker"),
                ("M2_MARKER_Publish", "MARKER", "marker_publish"),
                ("M3_Read_Markers", "CELL", "cell_read_markers"),
                ("M3_Read_Markers", "ECM", "ecm_read_markers"),
                ("M4_MARKER_Age", "MARKER", "marker_age"),
            ])
            template = load_variant(ROOT, "variant_template")
            template_model = _FakeModel()
            ctx.model = template_model
            ctx.add_multiscale_diffusion_layers = lambda: template_model.events.append(
                ("L5_Transport", "ECM", "subcycled_diffusion"))
            template.configure_layers(ctx)
            self.assertEqual(template_model.events, _run_default_layer_builder(**config))


if __name__ == "__main__":
    unittest.main()
