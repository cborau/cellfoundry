"""Regression tests for model wiring that do not require FLAMEGPU/CUDA."""

import ast
from pathlib import Path
import unittest


MODEL_PATH = Path(__file__).resolve().parents[1] / "model.py"
RADIAL_GLIA_VARIANT_PATH = (
    Path(__file__).resolve().parents[1] / "variants" / "radial_glia" / "__init__.py"
)


class _FakeRTCFunction:
    def setMessageInput(self, _message):
        return self


class _FakeAgent:
    def newRTCFunctionFile(self, _name, _path):
        return _FakeRTCFunction()


class _FakeLayer:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def addAgentFunction(self, agent, function):
        self.events.append((self.name, agent, function))
        return self


class _FakeModel:
    def __init__(self):
        self.events = []
        self.layers = {}

    def newLayer(self, name):
        layer = _FakeLayer(name, self.events)
        self.layers[name] = layer
        return layer

    def Layer(self, name):
        return self.layers[name]

    def Agent(self, _name):
        return _FakeAgent()


def _run_default_layer_builder(**overrides):
    """Extract and execute only _build_default_layers from model.py."""
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_default_layers"
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)

    fake_model = _FakeModel()
    namespace = {
        "model": fake_model,
        "INCLUDE_VASCULARIZATION": False,
        "INCLUDE_VASCULAR_CELL_RECRUITMENT": False,
        "INCLUDE_DIFFUSION": False,
        "MOVING_BOUNDARIES": False,
        "INCLUDE_CELLS": False,
        "INCLUDE_FOCAL_ADHESIONS": False,
        "ORGANOID_ASSAY": False,
        "INCLUDE_LUMEN": False,
        "INCLUDE_FIBRE_NETWORK": False,
        "INCLUDE_NETWORK_REMODELING": False,
        "INCLUDE_CELL_CYCLE": False,
        "HETEROGENEOUS_DIFFUSION": False,
        "INCLUDE_CELL_CELL_INTERACTION": False,
        "INCLUDE_CELL_FNODE_REPULSION": False,
    }
    namespace.update(overrides)
    exec(compile(isolated_module, str(MODEL_PATH), "exec"), namespace)
    namespace["_build_default_layers"]()
    return fake_model.events


def _run_radial_glia_layer_builder(**overrides):
    """Extract and execute the radial-glia variant's copied layer builder."""
    tree = ast.parse(RADIAL_GLIA_VARIANT_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "configure_layers"
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)

    fake_model = _FakeModel()
    variant_globals = {
        "env": None,
        "INCLUDE_DIFFUSION": False,
        "MOVING_BOUNDARIES": False,
        "INCLUDE_CELLS": True,
        "INCLUDE_CELL_CELL_INTERACTION": False,
        "INCLUDE_CELL_CYCLE": False,
        "INCLUDE_VASCULARIZATION": False,
        "INCLUDE_FIBRE_NETWORK": False,
        "INCLUDE_FOCAL_ADHESIONS": False,
        "INCLUDE_LUMEN": False,
        "ORGANOID_ASSAY": False,
        "INCLUDE_VASCULAR_CELL_RECRUITMENT": False,
        "INCLUDE_CELL_FNODE_REPULSION": False,
        "INCLUDE_NETWORK_REMODELING": False,
        "HETEROGENEOUS_DIFFUSION": False,
    }
    variant_globals.update(overrides)
    namespace = {
        "_HERE": RADIAL_GLIA_VARIANT_PATH.parent,
        "_register_rg_env_properties": lambda _env, _globals: None,
    }
    exec(compile(isolated_module, str(RADIAL_GLIA_VARIANT_PATH), "exec"), namespace)
    namespace["configure_layers"](fake_model, variant_globals)
    return fake_model.events


class TestECMLayerDependencies(unittest.TestCase):
    def test_static_ecm_without_diffusion_skips_message_and_interaction(self):
        events = _run_default_layer_builder()
        functions = [function for _, _, function in events]
        self.assertNotIn("ecm_grid_location_data", functions)
        self.assertNotIn("ecm_ecm_interaction", functions)

    def test_moving_ecm_without_diffusion_runs_mechanics(self):
        events = _run_default_layer_builder(MOVING_BOUNDARIES=True)
        functions = [function for _, _, function in events]
        self.assertIn("ecm_grid_location_data", functions)
        self.assertIn("ecm_ecm_interaction", functions)
        self.assertIn("ecm_move", functions)
        self.assertLess(
            functions.index("ecm_grid_location_data"),
            functions.index("ecm_ecm_interaction"),
        )
        self.assertLess(
            functions.index("ecm_ecm_interaction"),
            functions.index("ecm_move"),
        )

    def test_vasc_movement_without_diffusion_has_ecm_message(self):
        events = _run_default_layer_builder(
            MOVING_BOUNDARIES=True,
            INCLUDE_VASCULARIZATION=True,
        )
        functions = [function for _, _, function in events]
        ecm_message_indices = [
            index
            for index, function in enumerate(functions)
            if function == "ecm_grid_location_data"
        ]
        self.assertEqual(len(ecm_message_indices), 2)
        self.assertIn("vasc_move", functions)
        self.assertLess(
            functions.index("ecm_move"),
            ecm_message_indices[-1],
        )
        self.assertLess(
            ecm_message_indices[-1],
            functions.index("vasc_move"),
        )

    def test_diffusion_order_remains_l4_l5_l6(self):
        events = _run_default_layer_builder(INCLUDE_DIFFUSION=True)
        functions = [function for _, _, function in events]
        self.assertLess(
            functions.index("ecm_Csp_update"),
            functions.index("ecm_ecm_interaction"),
        )
        self.assertLess(
            functions.index("ecm_ecm_interaction"),
            max(
                index
                for index, function in enumerate(functions)
                if function == "ecm_boundary_concentration_conditions"
            ),
        )

    def test_diffusion_and_moving_boundaries_run_both_paths(self):
        events = _run_default_layer_builder(
            INCLUDE_DIFFUSION=True,
            MOVING_BOUNDARIES=True,
        )
        functions = [function for _, _, function in events]
        self.assertIn("ecm_grid_location_data", functions)
        self.assertIn("ecm_Csp_update", functions)
        self.assertIn("ecm_ecm_interaction", functions)
        self.assertIn("ecm_boundary_concentration_conditions", functions)
        self.assertIn("ecm_move", functions)
        self.assertLess(
            functions.index("ecm_Csp_update"),
            functions.index("ecm_ecm_interaction"),
        )
        self.assertLess(
            functions.index("ecm_ecm_interaction"),
            functions.index("ecm_move"),
        )

    def test_radial_glia_copied_scheduler_runs_moving_ecm_mechanics(self):
        events = _run_radial_glia_layer_builder(MOVING_BOUNDARIES=True)
        functions = [function for _, _, function in events]
        self.assertIn("ecm_grid_location_data", functions)
        self.assertIn("ecm_ecm_interaction", functions)
        self.assertIn("ecm_move", functions)
        self.assertLess(
            functions.index("ecm_grid_location_data"),
            functions.index("ecm_ecm_interaction"),
        )
        self.assertLess(
            functions.index("ecm_ecm_interaction"),
            functions.index("ecm_move"),
        )


class TestFocalAdhesionTypeInheritance(unittest.TestCase):
    def test_focal_adhesion_type_is_never_recomputed_from_index(self):
        tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
        assignments = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "cell_type_i"
                for target in node.targets
            )
        ]
        modulo_assignments = [value for value in assignments if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mod)]
        inherited_assignments = [
            value
            for value in assignments
            if isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id == "cell_types"
        ]
        self.assertEqual(modulo_assignments, [])
        self.assertGreaterEqual(len(inherited_assignments), 2)


if __name__ == "__main__":
    unittest.main()
