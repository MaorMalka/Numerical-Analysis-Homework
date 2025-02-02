import math

from colors import bcolors


def trapezoidal_rule(f, a, b, n):

    h = (b - a) / n
    T = f(a) + f(b)
    integral = 0.5 * T  

    for i in range(1, n):
        x_i = a + i * h
        integral += f(x_i)

    integral *= h

    return integral


if __name__ == '__main__':
    f = lambda x: math.cos(x**2 + 5*x + 6) / (2 * math.exp(-x))

    n = 10
    a, b = 0, 1  

    print(f"Division into n={n} sections")
    integral = trapezoidal_rule(f, a, b, n)
    print(bcolors.OKBLUE, f"Numerical Integration of definite integral in range [{a},{b}] is {integral}", bcolors.ENDC)