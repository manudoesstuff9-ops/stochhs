import numpy as np

def expectation(x, p=None):
    """E[X]. If p (weights) is None, treat x as samples (Monte Carlo mean)."""

    x = np.asarray(x, dtype=float)

    if p is None:
        return x.mean()

    p = np.asarray(p, dtype=float)
    return np.sum(x * p)


""" so this was the example where we like used the asarray which is used to basically create arrays
also we used the most basic form of like the discrete expectation formula 
"""

def variance(x, p=None):
    """Var(X) = E[X^2] - E[X]^2."""
    x = np.asarray(x, dtype=float)
    mu = expectation(x, p)
    if p is None:
        return np.mean((x - mu) ** 2)
    p = np.asarray(p, dtype=float)
    return np.sum(p * (x - mu) ** 2)
 

""" this just is the most basic example of the variance thing which just uses the simple formula"""

def covariance(x, y):
    """Sample covariance Cov(X, Y) (ddof=0)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.mean((x - x.mean()) * (y - y.mean()))
""" covariance is basically like a method where we like find variance like the realtion between the variances of 2 variables """

def correlation(x, y):
    """Pearson correlation rho = Cov(X,Y) / (sigma_X * sigma_Y)."""
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return covariance(x, y) / (sx * sy)

x = [1,2,3,4,5]
y = [4,5,6,7,8]
print(correlation(x,y))
