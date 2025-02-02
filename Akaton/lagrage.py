from colors import bcolors


def lagrange_interpolation(x_data, y_data, x):
    """
    Lagrange Interpolation

    Parameters:
    x_data (list): List of x-values for data points.
    y_data (list): List of y-values for data points.
    x (float): The x-value where you want to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated y-value at the given x.
    """
    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result

if __name__ == '__main__':
    x_data = [0.35, 0.4, 0.55, 0.65, 0.7, 0.85, 0.9]
    y_data = [-213.5991, -204.4416, -194.9375, -185.0256, -174.6711, -163.8656, -152.6271]
    
    x_interpolate = 0.75
    y_interpolate = lagrange_interpolation(x_data, y_data, x_interpolate)

    print(bcolors.OKBLUE, f"\nInterpolated value at x = {x_interpolate} is F(x) = {y_interpolate}", bcolors.ENDC)



