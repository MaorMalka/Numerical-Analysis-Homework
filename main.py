"""
Maor Malka 314944307
Nadav Mozeson 211810874
Topaz Natan 311561567
"""

import numpy as np
import re

def parse_equations(system):
    # Initialize lists for coefficients and variables
    A = []
    b = []

    # Regex pattern to match coefficients and variables
    pattern = r'(-?\d*)\s*([A-Za-z])'

    # Process each equation in the system
    equations = []  # Store equations as strings
    for equation in system:
        # Store the equation string
        equations.append(equation)

        # Initialize coefficients for the current equation
        eq_coeffs = [0, 0, 0]
        # Find all coefficient-variable pairs in the equation
        coeffs_vars = re.findall(pattern, equation)

        # Populate coefficients based on variables found
        for coef_var in coeffs_vars:
            coef = coef_var[0]
            var = coef_var[1]
            if coef == '':
                coef = '1'
            eq_coeffs[ord(var.lower()) - ord('x')] = float(coef)

        # Append coefficients of the current equation to A
        A.append(eq_coeffs)
        # Find the constant term and append to b
        const_term = re.search(r'= (.+)$', equation).group(1)
        b.append(float(const_term))

    return np.array(A), np.array(b), equations

def print_equations(system):
    # Print the system of equations
    print("System of Equations:")
    for eq in system:
        print(eq)

def print_matrix(A):
    # Print the matrix A
    if len(A.shape) == 1:  # Handling for vector
        print("Vector b:")
        print("[", ", ".join(f" {value} " for value in A), "]")
    else:  # Handling for matrix
        print("Matrix A:")
        for row in A:
            print("[", " ".join(f" {value} " for value in row), "]")

def print_vector(b):
    # Print the vector b
    print("Vector b:")
    print("[", ", ".join(f" {value} " for value in b), "]")

def is_diagonally_dominant(A):
    """Check if matrix A is diagonally dominant."""
    rows, cols = A.shape
    for i in range(rows):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        if np.abs(A[i, i]) <= row_sum:
            return False
    return True

def print_diagonal_matrix(A):
    # Print the diagonal elements of matrix A
    n = len(A)
    print("Diagonal Dominant Matrix:")
    for i in range(n):
        print(A[i, i], end=" ")
    print()

def has_diagonally_dominant(A):
    """Check if the matrix A has a diagonally dominant structure."""
    n = len(A)
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        if np.abs(A[i, i]) <= row_sum:
            return False
    return True

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

def jacobi_method(A, b, initial_guess=None, tolerance=0.001, max_iterations=50):
    """Solve the system of equations using the Jacobi method."""
    n = len(A)
    if initial_guess is None:
        x = np.zeros(n)
    else:
        x = initial_guess.copy()
    x_new = np.zeros(n)

    for iteration in range(max_iterations):
        for i in range(n):
            sum_ax = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum_ax) / A[i, i]

        if np.allclose(x, x_new, atol=tolerance):
            print(f'Jacobi method converged in {iteration + 1} iterations.')
            return x_new

        x = x_new.copy()

    print('Jacobi method did not converge within the specified tolerance and maximum iterations.')
    return x_new

def gauss_seidel_method(A, b, initial_guess=None, tolerance=0.001, max_iterations=50):
    """Solve the system of equations using the Gauss-Seidel method."""
    n = len(A)
    if initial_guess is None:
        x = np.zeros(n)
    else:
        x = initial_guess.copy()

    for iteration in range(max_iterations):
        for i in range(n):
            sum_ax = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sum_ax) / A[i, i]

        if np.linalg.norm(np.dot(A, x) - b) < tolerance:
            print(f'Gauss-Seidel method converged in {iteration + 1} iterations.')
            return x

    print('Gauss-Seidel method did not converge within the specified tolerance and maximum iterations.')
    return x

def main():
    # Example system of equations
    system = [
        "2X + 1Y = 2",
        "2X + 8Y + 4Z = 6",
        "4Y + 6Z = 5"
    ]

    # Parse equations into matrix A, vector b, and get equations as strings
    A, b, equations = parse_equations(system)

    # Print the system of equations
    print_equations(equations)

    # Print the matrix A
    print_matrix(A)

    # Print the vector b
    print_vector(b)

    # Check if the matrix is invertible (i.e., determinant is not zero)
    if A.shape[0] != A.shape[1] or np.linalg.det(A) == 0:
        print("The matrix is not invertible.")
        exit()

    # Check if matrix A is diagonally dominant
    if not is_diagonally_dominant(A):
        print("In the given matrix there is no dominant diagonal and therefore we cannot solve it with the methods we learned in class.")
        exit()

    # Print the diagonally dominant matrix A
    print("\nDiagonal Dominant Matrix:")
    print(np.diag(np.diag(A)))

    # Solve the system using Jacobi method
    initial_guess = np.zeros(len(b))
    x_jacobi = jacobi_method(A, b, initial_guess)
    print("\nSolving Ax = b using Jacobi method:")
    print("Solution x:\n", x_jacobi)

    # Solve the system using Gauss-Seidel method
    x_gauss_seidel = gauss_seidel_method(A, b, initial_guess)
    print("\nSolving Ax = b using Gauss-Seidel method:")
    print("Solution x:\n", x_gauss_seidel)

if __name__ == "__main__":
    main()