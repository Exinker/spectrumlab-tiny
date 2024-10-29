import numpy as np

from spectrumlab.types import Array, T


def rectangular(x: T | Array[T], x0: T, w: T) -> Array[float]:
    """Rectangular distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        w - width (full width)
    """
    f = np.zeros(x.shape)

    f[(x > x0 - w/2) & (x < x0 + w/2)] = (2/w) / 2
    f[(x == x0 - w/2) | (x == x0 + w/2)] = (2/w) / 4

    return f


def pvoigt(x: T | Array[T], x0: T, w: T, a: float, r: float) -> Array[float]:
    """Pseudo-Voigt distribution with position `x0` and unit intensity.

    Params:
        x0 - position
        w - width (full width at half maximum)
        a - assymetry
        r - ratio (in range [0; 1])

    A simple asymmetric line shape shape for fitting infrared absorption spectra.
    Aaron l. Stancik, Eric B. Brauns
    https://www.sciencedirect.com/science/article/abs/pii/S0924203108000453
    """
    sigma = 2*w / (1 + np.exp(a*(x - x0)))

    gauss = np.sqrt(4*np.log(2)/np.pi) / sigma * np.exp(-4*np.log(2)*((x - x0)/sigma)**2)
    lorentz = 2/np.pi/sigma/(1 + 4*((x - x0)/sigma)**2)
    f = r*lorentz + (1 - r)*gauss

    return f
