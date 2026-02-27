Bivariate Normal Distribution PDF implementation from scratch.
For Formative 3 Part 1 - Probability Distributions.
"""

import numpy as np


def bvn_pdf(x, y, mux, muy, sx, sy, rho):
    """
    Calculate bivariate normal probability density function at point (x, y).
    
    This implements the formula:
    f(x,y) = [1/(2*pi*sx*sy*sqrt(1-rho^2))] * exp(-Q/[2(1-rho^2)])
    
    where Q = [(x-mux)/sx]^2 + [(y-muy)/sy]^2 - 2*rho*[(x-mux)/sx][(y-muy)/sy]
    
    Parameters:
    -----------
    x, y : float
        The point coordinates where PDF is evaluated
    mux, muy : float
        Mean values of the two variables
    sx, sy : float
        Standard deviations (must be positive)
    rho : float
        Correlation coefficient between -1 and 1
    
    Returns:
    --------
    float : Probability density value at (x, y)
    """
    
    # Validate inputs
    if not (-1 < rho < 1):
        raise ValueError("Correlation rho must be between -1 and 1")
    if sx <= 0 or sy <= 0:
        raise ValueError("Standard deviations must be positive")
    
    # Calculate normalization constant: 1 / (2*pi*sx*sy*sqrt(1-rho^2))
    norm = 1.0 / (2.0 * np.pi * sx * sy * np.sqrt(1.0 - rho**2))
    
    # Standardize coordinates (convert to z-scores)
    zx = (x - mux) / sx
    zy = (y - muy) / sy
    
    # Calculate quadratic form Q
    # This measures Mahalanobis distance accounting for correlation
    Q = zx**2 + zy**2 - 2.0 * rho * zx * zy
    
    # Calculate exponent
    exponent = -Q / (2.0 * (1.0 - rho**2))
    
    # Return final PDF value
    return norm * np.exp(exponent)


def get_params(X, Y):
    """
    Calculate bivariate normal distribution parameters from data.
    
    Parameters:
    -----------
    X, Y : array-like
        Data arrays for the two variables
    
    Returns:
    --------
    tuple : (mux, muy, sx, sy, rho)
        Means, standard deviations, and correlation coefficient
    """
    mux = np.mean(X)
    muy = np.mean(Y)
    sx = np.std(X, ddof=0)  # Population standard deviation
    sy = np.std(Y, ddof=0)
    rho = np.corrcoef(X, Y)[0, 1]
    
    return mux, muy, sx, sy, rho