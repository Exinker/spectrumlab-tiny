import numpy as np

from spectrumlab.approximations.variables import AbstractVariables, Variable
from spectrumlab.grid import Grid
from spectrumlab.types import Number


class ScopeVariables(AbstractVariables):

    def __init__(
        self,
        grid: Grid,
        position: Number | None = None,
        intensity: float | None = None,
        background: float | None = None,
    ):
        super().__init__([
            self._init_position(grid, position=position),
            self._init_intensity(grid, intensity=intensity),
            self._init_background(grid, background=background),
        ])

        self.name = 'scope'

    def _init_position(self, grid: Grid, position: Number | None = None) -> Variable:
        initial = grid.x[np.argmax(grid.y)] if position is None else position
        bounds = (initial-2, initial+2) if position is None else (initial-1e-10, initial+1e-10)
        final = position

        return Variable('position', initial, bounds, final)

    def _init_intensity(self, grid: Grid, intensity: float | None = None) -> Variable:
        initial = np.sum(grid.y)*(grid.x[-1] - grid.x[0])/len(grid) if intensity is None else intensity
        bounds = (0, +np.inf) if intensity is None else (intensity - 1e-10, intensity + 1e-10)
        final = intensity

        return Variable('intensity', initial, bounds, final)

    def _init_background(self, grid: Grid, background: float | None = None) -> Variable:
        initial = min(grid.y) if background is None else background
        # bounds = (min(grid.y), max(grid.y)) if background is None else (background - 1e-10, background + 1e-10)
        bounds = (0, 0) if background is None else (background - 1e-10, background + 1e-10)
        final = background

        return Variable('background', initial, bounds, final)
