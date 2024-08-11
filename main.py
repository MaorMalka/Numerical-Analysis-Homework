"""
Maor Malka 314944307
Nadav Mozeson 211810874
Topaz Natan 311561567
"""

import numpy as np
from typing import List, Tuple, Callable

Point = Tuple[float, float]

def interpolate_linear(points: List[Point], x: float) -> float:
    (x0, y0), (x1, y1) = points
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def interpolate_polynomial(points: List[Point], x: float) -> float:
    A = np.vstack([np.array([p[0]**2, p[0], 1]) for p in points])
    B = np.array([p[1] for p in points])
    coef = np.linalg.solve(A, B)
    return np.polyval(coef, x)

def interpolate_lagrange(points: List[Point], x: float) -> float:
    def basis(i: int) -> float:
        return np.prod([(x - points[j][0]) / (points[i][0] - points[j][0])
                        for j in range(len(points)) if j != i])
    return sum(y * basis(i) for i, (_, y) in enumerate(points))

def validate_input(points: List[Point]) -> bool:
    x_values = [p[0] for p in points]
    return len(set(x_values)) == len(points)

def get_points() -> List[Point]:
    while True:
        n = int(input("Type the number of points you want to insert (2 points for linear, 3 points for polynomial/Lagrange): "))
        points = [(float(input(f"x{i}: ")), float(input(f"y{i}: "))) for i in range(n)]
        if validate_input(points):
            return points
        else:
            print("Invalid input: x values must be unique. Please try again.")

def get_interpolation_method() -> Tuple[Callable, int]:
    methods = {
        1: (interpolate_linear, 2, "Linear interpolation"),
        2: (interpolate_polynomial, 3, "Polynomial interpolation"),
        3: (interpolate_lagrange, 3, "Lagrange interpolation")
    }
    choice = int(input("Choose an interpolation method (1-linear, 2-polynomial, 3-Lagrange): "))
    return methods.get(choice, (None, 0, "not legal"))

def main():
    points = get_points()
    x = float(input("Enter the x value for which you want to find the y value: "))
    method, required_points, method_name = get_interpolation_method()

    if method and len(points) == required_points:
        y = method(points, x)
        print(f"The estimated value of y for x = {x} by method {method_name} is {y:.4f}")
    else:
        print("The selection is incorrect or the number of points is wrong.")

if __name__ == "__main__":
    main()
