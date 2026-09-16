"""CPU-only clock, configuration and numerical-reference regressions."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helper_module import expected_min_diffusion_spacing, plan_diffusion, substep_count, validate_diffusion_parameters
from tools.validate_multiscale_diffusion import validate


class DiffusionClockTests(unittest.TestCase):
    def test_noninteger_ratios_fill_main_step_without_overshoot(self):
        groups = plan_diffusion(1.0, [0.3, 1.0, 0.3], [0, 0, 0], [1, 1, 1])
        self.assertEqual(groups[0].species, (0, 2))
        self.assertEqual(groups[0].substeps, 4)
        self.assertEqual(groups[1].substeps, 1)
        for group in groups:
            self.assertEqual(group.dt * group.substeps, 1.0)

    def test_oxygen_and_protein_example(self):
        groups = plan_diffusion(600.0, [600.0] * 3, [1000, 50, 5], [20, 20, 20])
        self.assertEqual([g.substeps for g in groups], [10000, 500, 50])

    def test_anisotropic_cfl_is_sum_of_all_axes(self):
        group, = plan_diffusion(1.0, [1.0], [2], [1, 2, 4])
        limit = 0.9 / (4 * (1 + 1 / 4 + 1 / 16))
        self.assertEqual(group.substeps, math.ceil(1 / limit))
        self.assertLessEqual(group.dt, limit)

    def test_lumen_bound_overrides_lower_base_diffusivity(self):
        group, = plan_diffusion(1.0, [1.0], [1], [1, 1, 1], coefficient_upper_bounds=[20])
        self.assertEqual(group.substeps, 134)

    def test_integer_roundoff_does_not_add_a_step(self):
        self.assertEqual(substep_count(0.6, 0.04), 15)
        self.assertEqual(substep_count(1.0, 0.1 - 1e-10), 11)

    def test_invalid_inputs_fail_before_gpu_construction(self):
        for requested in ([0], [-1], [math.nan], [math.inf], [2], [], [True]):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                plan_diffusion(1, requested, [1], [1, 1, 1])
        for kwargs in (dict(safety=1), dict(safety=0), dict(max_substeps=0),
                       dict(max_substeps=2), dict(coefficient_upper_bounds=[-1])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_diffusion(1, [0.1], [1], [1, 1, 1], **kwargs)

    def test_boundary_speeds_predict_spacing_over_the_full_run(self):
        config = self.diffusion_config(STEPS=100, L0_x=50.0, L0_y=50.0, L0_z=50.0,
                                      ECM_AGENTS_PER_DIR=[6, 6, 6],
                                      BOUNDARY_DISP_RATES=[-0.9, 0, 0, 0, 0, 0])
        self.assertEqual(expected_min_diffusion_spacing(config), [1.0, 10.0, 10.0])

    def test_translation_and_expansion_do_not_reduce_initial_spacing(self):
        config = self.diffusion_config(STEPS=100,
                                      BOUNDARY_DISP_RATES=[3, 3, 0.1, -0.1, 0, 0])
        self.assertEqual(expected_min_diffusion_spacing(config), [1.0, 1.0, 1.0])

    def test_predicted_boundary_crossing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "boundary crossing"):
            validate_diffusion_parameters(self.diffusion_config(STEPS=100,
                                          BOUNDARY_DISP_RATES=[-1, 0, 0, 0, 0, 0]))

    def test_minimum_spacing_fixes_the_count_before_grid_contraction(self):
        config = self.diffusion_config(MOVING_BOUNDARIES=True,
                                      DIFFUSION_MIN_SPACING=[0.25, 0.25, 0.25])
        group, = validate_diffusion_parameters(config)
        self.assertEqual(group.substeps, 54)
        self.assertAlmostEqual(group.dt, 0.5 / 54)

    def test_minimum_spacing_cannot_exceed_initial_spacing(self):
        for spacing in ([2, 1, 1], [0, 1, 1], [float("nan"), 1, 1], [1, 1]):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                validate_diffusion_parameters(self.diffusion_config(DIFFUSION_MIN_SPACING=spacing))

    def test_lumen_upper_bound_is_used_by_configuration_validator(self):
        config = self.diffusion_config(INCLUDE_LUMEN=True, LUMEN_DIFFUSION_COEFF_MULTI=[20.0])
        group, = validate_diffusion_parameters(config)
        self.assertEqual(group.substeps, 67)

    @staticmethod
    def diffusion_config(**overrides):
        config = dict(N_SPECIES=1, DIFFUSION_COEFF_MULTI=[1.0], ECM_DEGRADATION_RATE_MULTI=[0.0],
                      INIT_ECM_CONCENTRATION_VALS=[0.0], INIT_ECM_SAT_CONCENTRATION_VALS=[1.0],
                      BOUNDARY_CONC_INIT_MULTI=[[-1.0] * 6], BOUNDARY_CONC_FIXED_MULTI=[[-1.0] * 6],
                      ECM_AGENTS_PER_DIR=[5, 5, 5], L0_x=4.0, L0_y=4.0, L0_z=4.0,
                      TIME_STEP=0.5, TIME_STEP_DIFFUSION=[0.5], DIFFUSION_CFL_SAFETY=0.9,
                      DIFFUSION_MAX_SUBSTEPS=1000000)
        config.update(overrides)
        return config

    def test_reference_numerical_invariants(self):
        report = validate(gpu=False)
        self.assertLess(report["checks"]["three_species_cosine_and_exact_decay"]["linf"], 1e-12)


if __name__ == "__main__":
    unittest.main()
