"""
Tests for matvec_multiply.py.

Three groups:
  - TestDotProduct / TestMatrixVectorMultiply: normal-case correctness.
  - TestDotProductGuards / TestMatrixVectorMultiplyGuards: the input
    validation added in matvec_multiply.py (wrong types, wrong shapes,
    NaN/inf, overflow).

Run with:
    python -m unittest matvec_tests.py -v
"""

import random
import unittest

from matvec_multiply import dot_product, matrix_vector_multiply


class TestDotProduct(unittest.TestCase):

    def test_known_value(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32.0)

    def test_zero_vector(self):
        self.assertEqual(dot_product([0, 0, 0], [1, 2, 3]), 0.0)

    def test_negative_values(self):
        self.assertEqual(dot_product([-1, 2, -3], [1, -2, 3]), -14.0)

    def test_single_element_vectors(self):
        self.assertEqual(dot_product([7], [6]), 42.0)

    def test_commutative(self):
        u, v = [2.0, -3.0, 0.5], [1.0, 4.0, -2.0]
        self.assertAlmostEqual(dot_product(u, v), dot_product(v, u), places=10)

    def test_tuples_are_accepted(self):
        # A tuple is just as much a "sequence" as a list.
        self.assertEqual(dot_product((1, 2, 3), (4, 5, 6)), 32.0)

    def test_booleans_are_accepted_as_0_1(self):
        self.assertEqual(dot_product([True, False, True], [2, 3, 4]), 6.0)

    def test_matches_independent_sum_over_random_data(self):
        random.seed(7)
        n = 300
        u = [random.uniform(-10, 10) for _ in range(n)]
        v = [random.uniform(-10, 10) for _ in range(n)]
        expected = sum(a * b for a, b in zip(u, v))
        self.assertAlmostEqual(dot_product(u, v), expected, places=6)


class TestMatrixVectorMultiply(unittest.TestCase):

    def test_identity_matrix(self):
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        vector = [5, 6, 7]
        self.assertEqual(matrix_vector_multiply(identity, vector), [5, 6, 7])

    def test_zero_matrix(self):
        zeros = [[0, 0], [0, 0]]
        self.assertEqual(matrix_vector_multiply(zeros, [3, 4]), [0, 0])

    def test_non_square_matrix(self):
        matrix = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(matrix_vector_multiply(matrix, [1, 1, 1]), [6, 15])

    def test_single_row_matrix(self):
        self.assertEqual(matrix_vector_multiply([[1, 2, 3]], [1, 0, 1]), [4])

    def test_empty_matrix_returns_empty_list(self):
        self.assertEqual(matrix_vector_multiply([], [1, 2, 3]), [])

    def test_tuple_of_tuples_is_accepted(self):
        self.assertEqual(matrix_vector_multiply(((1, 2), (3, 4)), (5, 6)), [17, 39])

    def test_does_not_modify_its_inputs(self):
        matrix = [[1, 2], [3, 4]]
        vector = [5, 6]
        matrix_vector_multiply(matrix, vector)
        self.assertEqual(matrix, [[1, 2], [3, 4]])
        self.assertEqual(vector, [5, 6])

    def test_matches_independent_computation_on_random_data(self):
        random.seed(3)
        rows, cols = 30, 20
        matrix = [[random.uniform(-5, 5) for _ in range(cols)] for _ in range(rows)]
        vector = [random.uniform(-5, 5) for _ in range(cols)]
        result = matrix_vector_multiply(matrix, vector)
        expected = [sum(a * b for a, b in zip(row, vector)) for row in matrix]
        for got, want in zip(result, expected):
            self.assertAlmostEqual(got, want, places=8)


class TestDotProductGuards(unittest.TestCase):

    def test_mismatched_lengths_raise_value_error(self):
        with self.assertRaises(ValueError):
            dot_product([1, 2, 3], [1, 2])

    def test_none_argument_raises_type_error(self):
        with self.assertRaises(TypeError):
            dot_product(None, [1, 2])

    def test_scalar_argument_raises_type_error(self):
        with self.assertRaises(TypeError):
            dot_product(3, [1, 2])

    def test_string_argument_raises_type_error(self):
        with self.assertRaises(TypeError):
            dot_product("ab", "cd")

    def test_set_argument_raises_type_error(self):
        # A set has a length but no positional order, so it can't be indexed.
        with self.assertRaises(TypeError):
            dot_product({1, 2}, [1, 2])

    def test_non_numeric_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            dot_product([1, "two"], [3, 4])

    def test_nan_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            dot_product([1.0, float("nan")], [1.0, 1.0])

    def test_infinite_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            dot_product([float("inf")], [1.0])

    def test_overflow_raises_overflow_error(self):
        # Both entries are finite, but their product overflows float range.
        with self.assertRaises(OverflowError):
            dot_product([1e308, 1e308], [10.0, 10.0])


class TestMatrixVectorMultiplyGuards(unittest.TestCase):

    def test_empty_matrix_returns_empty_list(self):
        # A 0-row matrix is a valid (degenerate) input, not an error.
        self.assertEqual(matrix_vector_multiply([], [1, 2, 3]), [])

    def test_empty_matrix_still_validates_the_vector(self):
        # The empty-matrix shortcut must not let a malformed vector through.
        with self.assertRaises(TypeError):
            matrix_vector_multiply([], None)

    def test_ragged_matrix_raises_value_error(self):
        matrix = [[1, 2, 3], [4, 5]]
        with self.assertRaises(ValueError):
            matrix_vector_multiply(matrix, [1, 1, 1])

    def test_dimension_mismatch_raises_value_error(self):
        matrix = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            matrix_vector_multiply(matrix, [1, 1])

    def test_non_sequence_matrix_raises_type_error(self):
        with self.assertRaises(TypeError):
            matrix_vector_multiply(None, [1, 2])

    def test_flat_vector_passed_as_matrix_raises_type_error(self):
        # A common mistake: passing [1, 2, 3] where [[1, 2, 3]] was meant.
        # Each "row" (an int) isn't itself an indexable sequence.
        with self.assertRaises(TypeError):
            matrix_vector_multiply([1, 2, 3], [1, 2, 3])

    def test_bad_entry_reports_its_row(self):
        matrix = [[1, 2], [3, "x"]]
        with self.assertRaises(TypeError) as caught:
            matrix_vector_multiply(matrix, [1, 1])
        self.assertIn("matrix[1]", str(caught.exception))

    def test_nan_in_matrix_raises_value_error(self):
        with self.assertRaises(ValueError):
            matrix_vector_multiply([[1.0, float("nan")]], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)