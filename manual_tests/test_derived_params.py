"""Regression tests for radius-derived model parameters.

The tests avoid importing model.py because importing it constructs and launches
a FLAMEGPU simulation. Model startup and override recomputation instead share
the helper functions tested here.
"""

import unittest

from helper_module import (
    apply_param_overrides,
    derive_cell_cell_adhesion_range,
    derive_max_focad_arm_length,
    recompute_derived_params,
)


class RadiusDerivedParameterTests(unittest.TestCase):
    ADHESION_MULTIPLIER = 1.0
    FOCAD_MULTIPLIER = 4.0

    def test_strict_overrides_reject_misspellings_and_invalid_indices(self):
        for key in ("CELL_RADIUSS", "CELL_RADIUS[9]", "MISSING[0]", "STEPS[0]"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                apply_param_overrides({"CELL_RADIUS": [3., 3., 3.], "STEPS": 1},
                                      {key: 2}, strict=True)

    def test_strict_overrides_keep_broadcasting_and_indexed_values(self):
        ns = {"CELL_RADIUS": [3., 3., 3.], "CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER": 1.,
              "MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER": 4.}
        apply_param_overrides(ns, {"CELL_RADIUS": 5.}, strict=True)
        apply_param_overrides(ns, {"CELL_RADIUS[1]": 7.}, strict=True)
        self.assertEqual(ns["CELL_RADIUS"], [5., 7., 5.])

    def test_canonical_formulas_support_per_type_radii(self):
        radii = [5.0, 7.5, 10.0]

        self.assertEqual(
            derive_cell_cell_adhesion_range(radii, self.ADHESION_MULTIPLIER),
            [5.0, 7.5, 10.0],
        )
        self.assertEqual(
            derive_max_focad_arm_length(radii, self.FOCAD_MULTIPLIER),
            40.0,
        )

    def test_canonical_formulas_support_scalar_radius(self):
        self.assertEqual(
            derive_cell_cell_adhesion_range(8.0, self.ADHESION_MULTIPLIER),
            8.0,
        )
        self.assertEqual(
            derive_max_focad_arm_length(8.0, self.FOCAD_MULTIPLIER),
            32.0,
        )

    def test_unrelated_override_preserves_default_semantics(self):
        namespace = {
            "CELL_RADIUS": [5.0, 5.0, 5.0],
            "CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER": self.ADHESION_MULTIPLIER,
            "MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER": self.FOCAD_MULTIPLIER,
            "CELL_CELL_ADHESION_RANGE": [5.0, 5.0, 5.0],
            "MAX_FOCAD_ARM_LENGTH": 20.0,
            "N_CELL_TYPES": 3,
            "STEPS": 1,
        }

        apply_param_overrides(namespace, {"STEPS": 2})

        self.assertEqual(
            namespace["CELL_CELL_ADHESION_RANGE"],
            [5.0, 5.0, 5.0],
        )
        self.assertEqual(namespace["MAX_FOCAD_ARM_LENGTH"], 20.0)

    def test_recomputation_matches_canonical_startup_formulas(self):
        namespace = {
            "CELL_RADIUS": [5.0, 7.5, 10.0],
            "CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER": self.ADHESION_MULTIPLIER,
            "MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER": self.FOCAD_MULTIPLIER,
            "N_CELL_TYPES": 3,
        }

        recompute_derived_params(namespace)

        self.assertEqual(
            namespace["CELL_CELL_ADHESION_RANGE"],
            derive_cell_cell_adhesion_range(
                namespace["CELL_RADIUS"],
                namespace["CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER"],
            ),
        )
        self.assertEqual(
            namespace["MAX_FOCAD_ARM_LENGTH"],
            derive_max_focad_arm_length(
                namespace["CELL_RADIUS"],
                namespace["MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER"],
            ),
        )

    def test_radius_override_recomputes_all_dependent_geometry(self):
        namespace = {
            "CELL_RADIUS": [5.0, 5.0, 5.0],
            "CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER": self.ADHESION_MULTIPLIER,
            "MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER": self.FOCAD_MULTIPLIER,
            "CELL_CELL_ADHESION_RANGE": [5.0, 5.0, 5.0],
            "MAX_FOCAD_ARM_LENGTH": 20.0,
            "MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION": 15.0,
            "N_CELL_TYPES": 3,
        }

        apply_param_overrides(
            namespace,
            {"CELL_RADIUS": [4.0, 6.0, 8.0]},
        )

        self.assertEqual(
            namespace["CELL_CELL_ADHESION_RANGE"],
            [4.0, 6.0, 8.0],
        )
        self.assertEqual(namespace["MAX_FOCAD_ARM_LENGTH"], 32.0)
        self.assertEqual(
            namespace["MAX_SEARCH_RADIUS_CELL_CELL_INTERACTION"],
            24.0,
        )

    def test_explicit_derived_overrides_remain_pinned(self):
        namespace = {
            "CELL_RADIUS": [5.0, 5.0, 5.0],
            "CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER": self.ADHESION_MULTIPLIER,
            "MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER": self.FOCAD_MULTIPLIER,
            "CELL_CELL_ADHESION_RANGE": [5.0, 5.0, 5.0],
            "MAX_FOCAD_ARM_LENGTH": 20.0,
            "N_CELL_TYPES": 3,
        }

        apply_param_overrides(
            namespace,
            {
                "CELL_RADIUS": [6.0, 8.0, 10.0],
                "CELL_CELL_ADHESION_RANGE": [1.0, 2.0, 3.0],
                "MAX_FOCAD_ARM_LENGTH": 99.0,
            },
        )

        self.assertEqual(
            namespace["CELL_CELL_ADHESION_RANGE"],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(namespace["MAX_FOCAD_ARM_LENGTH"], 99.0)

    def test_pins_accumulate_across_variant_and_json_layers(self):
        namespace = {
            "CELL_RADIUS": [5.0, 5.0, 5.0],
            "CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER": self.ADHESION_MULTIPLIER,
            "MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER": self.FOCAD_MULTIPLIER,
            "CELL_CELL_ADHESION_RANGE": [5.0, 5.0, 5.0],
            "MAX_FOCAD_ARM_LENGTH": 20.0,
            "N_CELL_TYPES": 3,
            "STEPS": 1,
        }

        pins = apply_param_overrides(
            namespace,
            {"MAX_FOCAD_ARM_LENGTH": 77.0},
        )
        apply_param_overrides(
            namespace,
            {"STEPS": 2},
            pinned=pins,
        )

        self.assertEqual(namespace["MAX_FOCAD_ARM_LENGTH"], 77.0)

    def test_user_facing_multiplier_overrides_recompute_derived_values(self):
        namespace = {
            "CELL_RADIUS": [5.0, 10.0],
            "CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER": self.ADHESION_MULTIPLIER,
            "MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER": self.FOCAD_MULTIPLIER,
            "CELL_CELL_ADHESION_RANGE": [5.0, 10.0],
            "MAX_FOCAD_ARM_LENGTH": 40.0,
            "N_CELL_TYPES": 2,
        }

        apply_param_overrides(
            namespace,
            {
                "CELL_CELL_ADHESION_RANGE_RADIUS_MULTIPLIER": 0.25,
                "MAX_FOCAD_ARM_LENGTH_RADIUS_MULTIPLIER": 6.0,
            },
        )

        self.assertEqual(namespace["CELL_CELL_ADHESION_RANGE"], [1.25, 2.5])
        self.assertEqual(namespace["MAX_FOCAD_ARM_LENGTH"], 60.0)


class DomainDerivedParameterTests(unittest.TestCase):
    def test_recomputation_alone_does_not_resize_the_core_grid(self):
        # Low-level helper behavior, not permission to override core geometry.
        # model.py separately rejects changes to initial bounds/dimensions.
        namespace = {"N": 11, "BOUNDARY_COORDS": [50., -50.] * 3,
                     "ECM_AGENTS_PER_DIR": [11, 11, 11], "ECM_POPULATION_SIZE": 1331}
        bounds = [500., -500., 500., -500., 25., -25.]
        pins = apply_param_overrides(namespace, {"N": 6, "BOUNDARY_COORDS": bounds})
        self.assertEqual(namespace["BOUNDARY_COORDS"], bounds)
        self.assertEqual([namespace[key] for key in ("L0_x", "L0_y", "L0_z")],
                         [1000., 1000., 50.])
        self.assertEqual(namespace["ECM_AGENTS_PER_DIR"], [11, 11, 11])
        self.assertEqual(namespace["ECM_POPULATION_SIZE"], 1331)
        self.assertEqual(namespace["ECM_VOXEL_VOLUME"], 100. * 100. * 5.)
        self.assertEqual(namespace["ECM_ECM_EQUILIBRIUM_DISTANCE"], 100.)
        self.assertEqual(namespace["MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION"], 100.)

        # Recomputing with another geometry still cannot rebuild this grid.
        apply_param_overrides(namespace, {"BOUNDARY_COORDS": [100., -100.] * 3}, pinned=pins)
        self.assertEqual(namespace["ECM_AGENTS_PER_DIR"], [11, 11, 11])
        self.assertEqual(namespace["ECM_POPULATION_SIZE"], 1331)
        self.assertEqual(namespace["ECM_ECM_EQUILIBRIUM_DISTANCE"], 20.)
        self.assertEqual(namespace["MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION"], 20.)

    def test_geometry_changes_preserve_explicit_search_radius(self):
        namespace = {"BOUNDARY_COORDS": [50., -50.] * 3,
                     "ECM_AGENTS_PER_DIR": [11, 11, 11], "ECM_POPULATION_SIZE": 1331,
                     "MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION": 10.}
        pins = apply_param_overrides(namespace, {"MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION": 42.})
        apply_param_overrides(namespace, {"BOUNDARY_COORDS": [100., -100.] * 3}, pinned=pins)
        self.assertEqual(namespace["ECM_ECM_EQUILIBRIUM_DISTANCE"], 20.)
        self.assertEqual(namespace["MAX_SEARCH_RADIUS_CELL_ECM_INTERACTION"], 42.)


if __name__ == "__main__":
    unittest.main()
