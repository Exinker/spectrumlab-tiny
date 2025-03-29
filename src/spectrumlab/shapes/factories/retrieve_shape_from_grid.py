from collections.abc import Sequence
from functools import partial

from scipy import optimize

from spectrumlab.grid import Grid
from spectrumlab.shapes import Shape
from spectrumlab.shapes.factories.utils.full_params import (
    FullParams,
)
from spectrumlab.utils import mse


def retrieve_shape_from_grid(
    grid: Grid,
) -> Shape:

    def _loss(grid: Grid, params: Sequence[float]) -> float:

        shape_params, scope_params = FullParams.parse(
            grid=grid,
            params=params,
        )
        shape = Shape(**shape_params, rx=20, dx=.5)

        return mse(
            y=grid.y,
            y_hat=shape(x=grid.x, **scope_params),
        )

    params = FullParams(grid=grid)
    res = optimize.minimize(
        partial(_loss, grid),
        params.initial,
        # method='SLSQP',
        bounds=params.bounds,
    )
    # assert res['success'], 'Optimization is not succeeded!'

    shape_params, _ = FullParams.parse(
        grid=grid,
        params=res['x'],
    )

    shape = Shape(**shape_params, rx=20, dx=.5)
    return shape
