import numpy as np
import matplotlib.pyplot as plt
import math

def calculate_complexity(L):
    """
    Calculate D (difficulty) using different equations for a given L value
    """
    D1 = 4.86 + 0.018 * L
    D2 = L / 3000
    if L > 0:
        D3 = 0.0047 + 0.0023 * np.log(L) + 0.000043 * (np.log(L))**2
    else:
        D3 = 0

    D4 = 4.2 + 0.0015 * (L)**(1/3)
    D5 = 0.069 + 0.00156 * L + 0.00000047 * (L)**2
    return D1, D2, D3, D4, D5

def plot_complexity(L):
    """
    Create a bar plot comparing D values from different equations
    """
    D1, D2, D3, D4, D5 = calculate_complexity(L)
    
    print("\nNumerical Results:")
    print(f"{D1:.4f}")
    print(f"{D2:.4f}")
    print(f"{D3:.4f}")
    print(f"{D4:.4f}")
    print(f"{D5:.4f}")

L = int(165.915563 * 60)
plot_complexity(L)