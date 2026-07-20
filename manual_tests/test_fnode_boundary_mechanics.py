"""Host-only regressions for FNODE elastic-boundary mechanics."""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def nearest_face(position, positive_boundary, negative_boundary):
    """Mirror the intended FNODE elastic-force attribution rule."""
    return (
        "pos"
        if abs(position - positive_boundary) < abs(position - negative_boundary)
        else "neg"
    )


def elastic_spring_force(position, initial_boundary, equilibrium_distance, stiffness, positive_face):
    """Return -k(x-x_rest) for an inward-offset boundary attachment."""
    rest_position = (
        initial_boundary - equilibrium_distance
        if positive_face
        else initial_boundary + equilibrium_distance
    )
    return -stiffness * (position - rest_position)


class TestFnodeBoundaryMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.move_source = (REPO_ROOT / "fnode_move.cpp").read_text(encoding="utf-8")
        cls.boundary_source = (
            REPO_ROOT / "fnode_boundary_interaction.cpp"
        ).read_text(encoding="utf-8")

    def test_all_six_faces_are_attributed_from_pre_move_position(self):
        boundaries = {
            "x": (500.0, -500.0),
            "y": (300.0, -300.0),
            "z": (25.0, -25.0),
        }
        for axis, (positive, negative) in boundaries.items():
            with self.subTest(axis=axis, face="pos"):
                self.assertEqual(nearest_face(positive - 0.02, positive, negative), "pos")
            with self.subTest(axis=axis, face="neg"):
                self.assertEqual(nearest_face(negative + 0.02, positive, negative), "neg")

            axis_upper = axis.upper()
            comparison = re.compile(
                rf"fabsf\(prev_agent_{axis} - COORD_BOUNDARY_{axis_upper}_POS\)\s*<\s*"
                rf"fabsf\(prev_agent_{axis} - COORD_BOUNDARY_{axis_upper}_NEG\)"
            )
            self.assertRegex(self.move_source, comparison)

    def test_nonzero_equilibrium_distance_is_force_free_on_all_faces(self):
        equilibrium_distance = 0.02
        stiffness = 10.0
        boundaries = {
            "x": (500.0, -500.0),
            "y": (300.0, -300.0),
            "z": (25.0, -25.0),
        }

        for axis, (positive, negative) in boundaries.items():
            positive_rest = positive - equilibrium_distance
            negative_rest = negative + equilibrium_distance
            with self.subTest(axis=axis, face="pos"):
                self.assertAlmostEqual(
                    elastic_spring_force(
                        positive_rest,
                        positive,
                        equilibrium_distance,
                        stiffness,
                        positive_face=True,
                    ),
                    0.0,
                )
                displacement = 0.005
                stretched = elastic_spring_force(
                    positive_rest + displacement,
                    positive,
                    equilibrium_distance,
                    stiffness,
                    positive_face=True,
                )
                compressed = elastic_spring_force(
                    positive_rest - displacement,
                    positive,
                    equilibrium_distance,
                    stiffness,
                    positive_face=True,
                )
                self.assertAlmostEqual(stretched, -stiffness * displacement)
                self.assertAlmostEqual(compressed, stiffness * displacement)
            with self.subTest(axis=axis, face="neg"):
                self.assertAlmostEqual(
                    elastic_spring_force(
                        negative_rest,
                        negative,
                        equilibrium_distance,
                        stiffness,
                        positive_face=False,
                    ),
                    0.0,
                )
                stretched = elastic_spring_force(
                    negative_rest + displacement,
                    negative,
                    equilibrium_distance,
                    stiffness,
                    positive_face=False,
                )
                compressed = elastic_spring_force(
                    negative_rest - displacement,
                    negative,
                    equilibrium_distance,
                    stiffness,
                    positive_face=False,
                )
                self.assertAlmostEqual(stretched, -stiffness * displacement)
                self.assertAlmostEqual(compressed, stiffness * displacement)

            axis_upper = axis.upper()
            positive_term = re.compile(
                rf"agent_{axis}\s*\+\s*ECM_BOUNDARY_EQUILIBRIUM_DISTANCE\s*"
                rf"-\s*INIT_COORD_BOUNDARY_{axis_upper}_POS"
            )
            negative_term = re.compile(
                rf"agent_{axis}\s*-\s*ECM_BOUNDARY_EQUILIBRIUM_DISTANCE\s*"
                rf"-\s*INIT_COORD_BOUNDARY_{axis_upper}_NEG"
            )
            self.assertRegex(self.boundary_source, positive_term)
            self.assertRegex(self.boundary_source, negative_term)

    def test_positive_face_old_sign_would_not_be_force_free(self):
        initial_positive_boundary = 500.0
        equilibrium_distance = 0.02
        stiffness = 10.0
        intended_rest = initial_positive_boundary - equilibrium_distance

        old_force = -stiffness * (
            intended_rest
            - equilibrium_distance
            - initial_positive_boundary
        )
        self.assertAlmostEqual(old_force, 0.4)
        self.assertNotEqual(old_force, 0.0)


if __name__ == "__main__":
    unittest.main()
