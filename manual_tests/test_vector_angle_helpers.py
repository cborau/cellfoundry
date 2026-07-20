"""Source-level regressions for the RTC vector-angle helpers.

The FLAMEGPU device functions cannot be imported without the CUDA runtime, so
these tests verify every checked-in implementation uses the scale-independent
atan2(cross magnitude, dot product) formula and reserves the zero result for a
genuinely zero-length input vector.
"""

import math
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HELPER_FILES = {
    "ecm_ecm_interaction.cpp",
    "fnode_fnode_bucket_interaction.cpp",
    "fnode_fnode_spatial_interaction.cpp",
    "handy_device_functions_template.cpp",
}
FUNCTION_SIGNATURE = "FLAMEGPU_DEVICE_FUNCTION float getAngleBetweenVec"


def extract_function_body(source: str) -> str:
    """Extract getAngleBetweenVec(), accounting for nested braces."""
    signature_index = source.index(FUNCTION_SIGNATURE)
    opening_brace = source.index("{", signature_index)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace : index + 1]
    raise AssertionError("Unterminated getAngleBetweenVec() implementation")


def reference_angle(first, second):
    """Host-side equivalent of the intended device helper semantics."""
    if first == (0.0, 0.0, 0.0) or second == (0.0, 0.0, 0.0):
        return 0.0
    dot = sum(left * right for left, right in zip(first, second))
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    determinant = math.sqrt(sum(component * component for component in cross))
    return math.atan2(determinant, dot)


class VectorAngleHelperTests(unittest.TestCase):
    def test_all_checked_in_copies_use_the_correct_formula(self):
        implementations = {}
        for path in REPOSITORY_ROOT.glob("*.cpp"):
            source = path.read_text(encoding="utf-8")
            if FUNCTION_SIGNATURE in source:
                implementations[path.name] = extract_function_body(source)

        self.assertEqual(set(implementations), EXPECTED_HELPER_FILES)
        for filename, body in implementations.items():
            with self.subTest(filename=filename):
                self.assertIn("x1 == 0.0f && y1 == 0.0f && z1 == 0.0f", body)
                self.assertIn("x2 == 0.0f && y2 == 0.0f && z2 == 0.0f", body)
                self.assertIn("return atan2f(det_dir, dot_dir);", body)
                self.assertNotIn("fabsf(dot_dir)", body)
                self.assertNotIn("EPSILON", body)

    def test_orthogonal_and_nearly_orthogonal_vectors_are_not_zero(self):
        self.assertAlmostEqual(
            reference_angle((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            math.pi / 2.0,
        )
        almost_orthogonal = reference_angle(
            (1.0, 0.0, 0.0),
            (1.0e-12, 1.0, 0.0),
        )
        self.assertGreater(almost_orthogonal, 1.5)
        self.assertAlmostEqual(almost_orthogonal, math.pi / 2.0, places=10)

    def test_parallel_antiparallel_and_zero_vector_conventions(self):
        self.assertEqual(reference_angle((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)), 0.0)
        self.assertAlmostEqual(
            reference_angle((1.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
            math.pi,
        )
        self.assertEqual(reference_angle((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
