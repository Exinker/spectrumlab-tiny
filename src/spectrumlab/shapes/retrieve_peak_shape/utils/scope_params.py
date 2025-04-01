import numpy as np

from spectrumlab.grid import Grid
from spectrumlab.shapes.retrieve_peak_shape.utils.params import (
    AbstractParams,
    Param,
)
from spectrumlab.types import Number


TOLL = 1e-10


class ScopeParams(AbstractParams):

    def __init__(
        self,
        grid: Grid,
        position: Number | None = None,
        intensity: float | None = None,
        background: float | None = None,
    ):
        super().__init__([
            calculate_position(grid, position=position),
            calculate_intensity(grid, intensity=intensity),
            calculate_background(grid, background=background),
        ])

        self.name = 'scope'


def calculate_position(
    grid: Grid,
    position: Number | None = None,
) -> Param:
    initial = position or grid.x[np.argmax(grid.y)]

    if position is None:
        bounds = (initial-2, initial+2)
    else:
        bounds = (initial-TOLL, initial+TOLL)

    return Param('position', initial, bounds, position)


def calculate_intensity(
    grid: Grid,
    intensity: float | None = None,
) -> Param:
    initial = intensity or np.sum(grid.y)*(grid.x[-1] - grid.x[0])/len(grid)

    if intensity is None:
        bounds = (0, +np.inf)
    else:
        bounds = (intensity-TOLL, intensity+TOLL)

    return Param('intensity', initial, bounds, intensity)


def calculate_background(
    grid: Grid,
    background: float | None = None,
) -> Param:
    initial = background or min(grid.y)

    if background is None:
        bounds = (min(grid.y), max(grid.y))
    else:
        bounds = (background-TOLL, background+TOLL)

    return Param('background', initial, bounds, background)
