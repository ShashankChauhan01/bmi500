"""
Vector and matrix-vector multiplication for BMI 500 HW3.

Both functions validate their inputs before doing any arithmetic, so a
malformed call (wrong type, wrong shape, NaN/inf entries) raises a
specific, readable error instead of failing deep inside a loop with a
confusing IndexError or silently returning a wrong answer.
"""

import math
import random
import time


def _validate_sequence(seq, label):
    """
    Confirm that `seq` is an indexable sequence of finite real numbers.

    Parameters
    ----------
    seq : object
        The value to check.
    label : str
        Name to use for `seq` in any error message (e.g. "vec1" or
        "matrix[3]"), so the caller can tell exactly which argument
        was the problem.

    Raises
    ------
    TypeError
        If `seq` isn't an indexable, sized sequence of real numbers —
        e.g. it's None, a bare number, a string, a set, or a
        generator, or one of its entries isn't a number.
    ValueError
        If any entry is NaN or +/-infinity. Both would otherwise
        propagate silently through the arithmetic below and turn into
        a wrong (but valid-looking) answer.
    """
    # Strings/bytes are technically sized and indexable, so they'd slip
    # past the check below and fail later with a confusing error about
    # multiplying two characters. Reject them explicitly first.
    if isinstance(seq, (str, bytes, bytearray)):
        raise TypeError(
            f"{label} must be a sequence of numbers, not {type(seq).__name__}"
        )

    if not hasattr(seq, "__len__") or not hasattr(seq, "__getitem__"):
        # Catches None, plain numbers, sets, dicts, and generators — all
        # of which would otherwise fail on the first len() or seq[i] call.
        raise TypeError(
            f"{label} must be an indexable sequence, got {type(seq).__name__}"
        )

    for i in range(len(seq)):
        value = seq[i]
        if isinstance(value, bool):
            continue  # bool is a subclass of int; True/False are valid 1/0 entries
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{label}[{i}] must be a real number, got {type(value).__name__}"
            )
        if not math.isfinite(value):
            raise ValueError(f"{label}[{i}] must be finite, got {value!r}")


# create a function to compute the dot product of two vectors using a for loop
# add comments for the selected function
def dot_product(vec1, vec2, _validated=False):
    """
    Compute the dot product of two vectors using a for loop.

    The dot product of a = [a0, ..., a(n-1)] and b = [b0, ..., b(n-1)]
    is a0*b0 + a1*b1 + ... + a(n-1)*b(n-1).

    Parameters
    ----------
    vec1, vec2 : sequence of float
        The two vectors. Must be the same length.
    _validated : bool, optional
        Internal flag. When True, skips re-checking vec1/vec2, since
        matrix_vector_multiply already validated every row and the
        vector once before calling this in a loop. Callers should
        leave this at its default.

    Returns
    -------
    float
        The dot product.

    Raises
    ------
    TypeError
        If either argument isn't a sequence of real numbers.
    ValueError
        If the vectors have different lengths, or contain NaN/inf.
    OverflowError
        If the true dot product is too large to represent as a float,
        even though every input value was finite.
    """
    if not _validated:
        _validate_sequence(vec1, "vec1")
        _validate_sequence(vec2, "vec2")

    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors must be the same length to compute a dot product "
            f"(got lengths {len(vec1)} and {len(vec2)})."
        )

    total = 0.0
    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]

    # Every input was finite (checked above), so a non-finite total means
    # the true result overflowed float range during accumulation.
    if not math.isfinite(total):
        raise OverflowError("dot product overflowed the range of a float")

    return total


# create a function to compute the matrix-vector product using the dot_product function
# add comments for the selected function
def matrix_vector_multiply(matrix, vector):
    """
    Compute the matrix-vector product using the dot_product function.

    For an (m x n) matrix M and a length-n vector v, entry i of the
    result is the dot product of row i of M with v.

    Parameters
    ----------
    matrix : sequence of sequence of float
        An (m x n) matrix: m rows, each of length n.
    vector : sequence of float
        A vector of length n.

    Returns
    -------
    list of float
        The resulting length-m vector. Empty if the matrix has no rows.

    Raises
    ------
    TypeError
        If matrix or vector isn't an indexable sequence, or a row/
        vector entry isn't a real number.
    ValueError
        If a row's length doesn't match the vector's length, or
        matrix/vector contain NaN/inf.
    """
    # Validate the vector once, up front, rather than re-checking it
    # inside dot_product on every row — it's the same object each time,
    # so re-validating it m times would be wasted work for large m.
    _validate_sequence(vector, "vector")

    if (
        isinstance(matrix, (str, bytes, bytearray))
        or not hasattr(matrix, "__len__")
        or not hasattr(matrix, "__getitem__")
    ):
        raise TypeError(
            f"matrix must be an indexable sequence of rows, got {type(matrix).__name__}"
        )

    if len(matrix) == 0:
        return []

    n_cols = len(vector)
    for row_index in range(len(matrix)):
        row = matrix[row_index]
        _validate_sequence(row, f"matrix[{row_index}]")
        if len(row) != n_cols:
            raise ValueError(
                f"matrix[{row_index}] has length {len(row)}, but vector "
                f"has length {n_cols}."
            )

    result = []
    for row in matrix:
        # Both row and vector were already validated above, so skip the
        # redundant re-check inside dot_product.
        result.append(dot_product(row, vector, _validated=True))
    return result


# create a main function to test the matrix-vector product function using randomly generated data of size 1000x1000
# add comments for the selected function
def main():
    """
    Test the matrix-vector product function using randomly generated
    data of size 1000x1000.

    Builds a random 1000x1000 matrix and length-1000 vector, runs
    matrix_vector_multiply, and cross-checks the result against an
    independently written computation (plain zip/sum, not calling
    dot_product or matrix_vector_multiply at all) so a bug shared
    between the implementation and a hand check wouldn't go unnoticed.
    """
    random.seed(0)  # fixed seed so results are reproducible across runs

    n = 1000
    matrix = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    vector = [random.uniform(-10, 10) for _ in range(n)]

    start_time = time.time()
    result = matrix_vector_multiply(matrix, vector)
    elapsed = time.time() - start_time

    # Independent cross-check: deliberately not using dot_product or
    # matrix_vector_multiply, so it can't share a bug with them.
    expected = [sum(a * b for a, b in zip(row, vector)) for row in matrix]
    max_error = max(abs(r - e) for r, e in zip(result, expected))

    print(f"Matrix size: {n}x{n}, vector length: {n}")
    print(f"Result length: {len(result)}")
    print(f"First 5 entries of result: {[round(v, 6) for v in result[:5]]}")
    print(f"Max absolute error vs. independent check: {max_error:.3e}")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print("PASS" if max_error < 1e-6 else "FAIL")


if __name__ == "__main__":
    main()