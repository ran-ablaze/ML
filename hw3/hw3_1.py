import numpy as np

def Univariate_gaussian(m, s):
    x = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    #x~N(0,1)->#ax+b~N(b, a^2)
    return s**0.5 * x + m

def Polynomial_basis_linear_model(n, a, w):
    x = np.random.uniform(-1.0, 1.0)
    y = 0.0
    for i in range(n):
        y += w[i] * x**i
    y += Univariate_gaussian(0, a)
    return x, y
