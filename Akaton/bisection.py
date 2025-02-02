import math
import numpy as np
from colors import bcolors

"""
Receives 3 parameters:
    1.  a - start value.
    2.  b - end  value. 
    3.  err - value of tolerable error

Returns variables:
    1.  S - The minimum number of iterations required to reach the desired accuracy
"""
def max_steps(a, b, err):
    s = int(np.floor(- np.log2(err / (b - a)) / np.log2(2) - 1))
    return s

"""
Performs Iterative methods for Nonlinear Systems of Equations to determine the roots of the given function f
Receives 4 parameters:
    1. f - continuous function on the interval [a, b], where f (a) and f (b) have opposite signs.
    2. a - start value.
    3. b - end  value. 
    4. tol - the tolerable error , the default value will set as 1e-16

Returns variables:
    1.  c - The approximate root of the function f
"""
def bisection_method(f, a, b, tol=1e-6):
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root")
    c, k = 0, 0
    steps = max_steps(a, b, tol)  

    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))

    while abs(b - a) > tol and k < steps:
        c = a + (b - a) / 2  

        if f(c) == 0 :
            return c  

        if f(c) * f(a) < 0:  
            b = c 
        else:
            a = c  

        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(k, a, b, f(a), f(b), c, f(c)))
        k += 1

    return c  



if __name__ == '__main__':
    f = lambda x: math.cos(x**2 + 5*x + 6) / (2 * math.exp(-x))

    start, end, step = 0.0, 1.0, 0.1
    found_roots = []

    print("\n Searching for roots in sub-intervals:")
    for a in np.arange(start, end, step):
        b = round(a + step, 2)  
        try:
            root = bisection_method(f, a, b)
            if root is not None:
                found_roots.append(root)
                print(bcolors.OKBLUE, f"\n Found root in [{a}, {b}] → x ≈ {root:.9f}", bcolors.ENDC)
        except Exception as e:
            print(f"Skipping interval [{a}, {b}]: {str(e)}")

    if not found_roots:
        print(bcolors.FAIL, "\n No roots found in the given range!", bcolors.ENDC)
    else:
        chosen_root = max(found_roots)  
        print(bcolors.OKGREEN, f"\n All found roots: {found_roots}", bcolors.ENDC)
        print(bcolors.BOLD, f"\n Selected root (largest in range): x ≈ {chosen_root:.9f}", bcolors.ENDC)