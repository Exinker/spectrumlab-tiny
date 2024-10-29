import numpy as np

from spectrumlab.types import Array


# --------        calculate errors        --------
def se(y: float | Array[float], y_hat: Array[float]) -> Array[float]:
    """Calculate squared error (SE) between true values $y$ and predicted values $\hat{y}$."""  # noqa: W605

    return np.square(y - y_hat)


def mse(y: float | Array[float], y_hat: Array[float]) -> float:
    """Calculate mean squared error (MSE) between true values $y$ and predicted values $\hat{y}$."""  # noqa: W605
    n = len(y_hat)

    xi = se(y, y_hat)
    return np.sqrt(np.sum(xi) / n**2)
