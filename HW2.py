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

def lu_decomposition(matrix):
    """Perform LU decomposition using Doolittle's method."""
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1  # Diagonal elements of L are set to 1
        for j in range(i, n):
            U[i][j] = matrix[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i+1, n):
            L[j][i] = (matrix[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U

def lu_solve(L, U, b):
    """Solve the system of equations using LU decomposition."""
    n = len(L)
    y = np.zeros(n)
    x = np.zeros(n)

    # Forward substitution to solve Ly = b
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # Backward substitution to solve Ux = y
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x

def main():
    # Example matrix (replace with your desired matrix)
    matrix = np.array([[1, 4, -3],
                       [-2, 1, 5],
                       [3, 2, 1]])

    # Example vector b (replace with your desired vector)
    b = np.array([1, 2, 3])

    # Check if the matrix is invertible (i.e., determinant is not zero)
    if np.linalg.det(matrix) == 0:
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
    print("The vector b:\n", b)
    print("Inverse Matrix using Gauss-Jordan elimination:\n", np.round(inverse_matrix, decimals=6))
    print("Maximum Row Norm of the Original Matrix:", np.round(original_max_norm, decimals=6))
    print("Maximum Row Norm of the Inverse Matrix:", np.round(inverse_max_norm, decimals=6))
    print("Product of Norms:", np.round(product_norms, decimals=6))

    # Perform LU decomposition
    L, U = lu_decomposition(matrix)

    # Print L and U matrices
    print("\nMatrix L (Lower Triangular):\n", np.round(L, decimals=6))
    print("Matrix U (Upper Triangular):\n", np.round(U, decimals=6))

    # Solve the linear system using LU decomposition
    x = lu_solve(L, U, b)

    # Print results
    print("\nSolving Ax = b using LU decomposition:")
    print("Solution x:\n", x)

if __name__ == "__main__":
    main()
