"""
Maor Malka 31494407
Topaz Natan 311561567
Nadav Mozeson 211810874
"""
import numpy as np

def gauss_jordan_inverse(matrix):
    """Find the inverse of a matrix using Gauss-Jordan elimination."""
    n = len(matrix)
    I = np.eye(n)
    A = np.hstack((matrix, I))

    for i in range(n):
        # Make the diagonal contain all 1's
        A[i] = A[i] / A[i, i]

        # Make the other rows contain 0's
        for j in range(n):
            if i != j:
                A[j] = A[j] - A[j, i] * A[i]

    return A[:, n:]


def max_row_norm(matrix):
    """Calculate the maximum row norm of a matrix."""
    return np.max(np.sum(np.abs(matrix), axis=1))


def main():
    # Example 3x3 matrix
    matrix = np.array([[1.0, -1.0, -2.0],
                       [2.0, -3.0, -5.0],
                       [-1.0, 3.0, 5]])

    # Check if the matrix is invertible (i.e., determinant is not zero)
    if matrix[0, 0] == 0.0 or matrix[1, 1] == 0.0 or matrix[2, 2] == 0.0:
        print("The matrix is not invertible.")
        exit()

    # Find the inverse of the matrix using Gauss-Jordan elimination
    inverse_matrix = gauss_jordan_inverse(matrix)

    # Calculate the maximum row norm of the original matrix
    original_max_norm = max_row_norm(matrix)

    # Calculate the maximum row norm of the inverse matrix
    inverse_max_norm = max_row_norm(inverse_matrix)

    # Multiply the norms
    product_norms = original_max_norm * inverse_max_norm

    # Printing results with handling for decimal points
    print("Original Matrix:\n", matrix)
    print("Inverse Matrix:\n", np.round(inverse_matrix, decimals=6))
    print("Maximum Row Norm of the Original Matrix:", np.round(original_max_norm, decimals=6))
    print("Maximum Row Norm of the Inverse Matrix:", np.round(inverse_max_norm, decimals=6))
    print("Product of Norms:", np.round(product_norms, decimals=6))

main()
