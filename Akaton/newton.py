from colors import bcolors
import math

def newton_raphson(f, df, p0, TOL, N=50):
    print("{:<10} {:<15} {:<15} ".format("Iteration", "po", "p1"))
    for i in range(N):
        if df(p0) == 0:
            print( "Derivative is zero at p0, method cannot continue.")
            return

        p = p0 - f(p0) / df(p0)

        if abs(p - p0) < TOL:
            return p  
        print("{:<10} {:<15.9f} {:<15.9f} ".format(i, p0, p))
        p0 = p
    return p


if __name__ == '__main__':
    f = lambda x: math.sin(2 * math.exp(-2*x)) / (2*x**3 + 5*x**2 - 6)
    
    df = lambda x: (
        (math.cos(2 * math.exp(-2*x)) * (-4 * math.exp(-2*x)) * (2*x**3 + 5*x**2 - 6) - 
        math.sin(2 * math.exp(-2*x)) * (6*x**2 + 10*x)) /
        ((2*x**3 + 5*x**2 - 6) ** 2)
    )
    
    p0 = 0.0  
    TOL = 1e-6
    N = 100
    roots = newton_raphson(f, df, p0, TOL, N)
    
    print("\nThe equation f(x) has an approximate root at x = {:<15.9f} ".format(roots))