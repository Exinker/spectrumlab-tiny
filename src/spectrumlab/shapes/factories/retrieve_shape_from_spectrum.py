import contextvars
import logging
import warnings
from collections.abc import Sequence
from functools import partial

import numpy as np
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy import optimize

from spectrumlab.grid import Grid
from spectrumlab.peaks import Peak
from spectrumlab.shapes.factories.retrieve_shape_from_grid import (
    retrieve_shape_from_grid,
)
from spectrumlab.shapes.factories.utils import (
    ScopeParams,
    ShapeParams,
)
from spectrumlab.shapes.shape import Shape
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Number
from spectrumlab.utils import mse


warnings.filterwarnings('ignore')


LOGGER = logging.getLogger('spectrumlab')
FIGURES = contextvars.ContextVar('FIGURES', default=None)

DEFAULT_SHAPE = Shape(width=2, asymmetry=0, ratio=.1)


class RetrieveShapeConfig(BaseSettings):

    default_shape: Shape = Field(None, alias='RETRIEVE_SHAPE_DEFAULT')
    error_max: float = Field(default=.001, alias='RETRIEVE_SHAPE_ERROR_MAX')
    error_mean: float = Field(default=.0001, alias='RETRIEVE_SHAPE_ERROR_MEAN')
    n_peaks_sorted_by_width: int | None = Field(default=None, alias='RETRIEVE_SHAPE_N_PEAKS_SORTED_BY_WIDTH')
    n_peaks_min: int = Field(default=10, alias='RETRIEVE_SHAPE_N_PEAKS_MIN')

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    @field_validator('default_shape', mode='before')
    @classmethod
    def validate_default_shape(cls, data: str | None) -> Shape:

        if data is None:
            return DEFAULT_SHAPE

        try:
            width, asymmetry, ratio = map(float, data.split(';'))
            shape = Shape(width=width, asymmetry=asymmetry, ratio=ratio)
        except Exception:
            shape = DEFAULT_SHAPE
        return shape

    @model_validator(mode='after')
    def validate(self) -> None:
        assert self.n_peaks_sorted_by_width >= self.n_peaks_min


def retrieve_shape_from_spectrum(
    spectrum: Spectrum,
    peaks: Sequence[Peak],
    n: int,
    restore_shape_config: RetrieveShapeConfig | None = None,
) -> Shape:

    restore_shape_config = restore_shape_config or RetrieveShapeConfig()

    # setup figure
    figures = FIGURES.get()
    if figures:
        figure = figures[n]
    else:
        figure = None

    # calculate shape
    n_peaks = len(peaks)
    LOGGER.debug('detector %02d - peaks total: %s', n, n_peaks)

    width = np.zeros(n_peaks)
    offset = np.zeros(n_peaks)
    scale = np.zeros(n_peaks)
    background = np.zeros(n_peaks)
    error = np.zeros(n_peaks)
    mask = np.full(n_peaks, False)
    for i, peak in enumerate(peaks):
        lb, ub = peak.minima

        grid = Grid(spectrum.number[lb:ub], spectrum.intensity[lb:ub], units=Number)
        shape = retrieve_shape_from_grid(
            grid=grid,
        )
        width[i] = shape.width

        scope_params, error[i] = _approximate_grid(
            grid=grid,
            shape=shape,
        )
        offset[i], scale[i], background[i] = scope_params.value

    if restore_shape_config.n_peaks_sorted_by_width:
        index = np.argsort(width)[restore_shape_config.n_peaks_sorted_by_width:]
        mask[index] = True

        LOGGER.debug('detector %02d - peaks stayed: %s', n, n_peaks - sum(mask))

    while True:

        # update shape
        grid = Grid.factory(spectrum=spectrum).create_from_peaks(
            peaks=[peaks[i] for i, is_masked in enumerate(mask) if not is_masked],
            offset=[offset[i] for i, is_masked in enumerate(mask) if not is_masked],
            scale=[scale[i] for i, is_masked in enumerate(mask) if not is_masked],
            background=[background[i] for i, is_masked in enumerate(mask) if not is_masked],
        )
        shape = _retrieve_shape(
            grid=grid,
        )

        # update offset, scale, background
        for i, peak in enumerate(peaks):
            lb, ub = peak.minima
            scope_params, error[i] = _approximate_grid(
                grid=Grid(spectrum.number[lb:ub], spectrum.intensity[lb:ub], units=Number),
                shape=shape,
            )
            offset[i], scale[i], background[i] = scope_params.value

        # breakpoints
        index, = np.where(~mask)

        if np.mean(np.abs(error[index])) <= restore_shape_config.error_mean:
            LOGGER.debug('detector %02d - breakpoint: error_mean (%s peaks)', n, len(index))
            break
        if np.max(np.abs(error[index])) <= restore_shape_config.error_max:
            LOGGER.debug('detector %02d - breakpoint: error_max (%s peaks)', n, len(index))
            break

        if len(index) <= restore_shape_config.n_peaks_min:
            LOGGER.debug('detector %02d - breakpoint: n_peaks_min (%s peaks)', n, len(index))
            break

        # next step
        index, = np.where(~mask)

        step_size = 1
        worst_peaks_index = index[np.argsort(np.abs(error[index]))][-step_size:]
        mask[worst_peaks_index] = True

    LOGGER.debug('detector %02d - peaks stayed: %s', n, n_peaks - sum(mask))
    LOGGER.debug('detector %02d - peaks index: %s', n, error[index])
    LOGGER.debug('detector %02d - peaks offset: %s', n, offset[index])
    LOGGER.debug('detector %02d - peaks scale: %s', n, scale[index])
    LOGGER.debug('detector %02d - peaks background: %s', n, background[index])

    # show figure
    if figure:
        if figure['spectrum']:
            ax = figure['spectrum'].gca()
            ax.clear()

            x = spectrum.wavelength
            y = spectrum.intensity
            ax.step(
                x, y,
                where='mid',
                color='black',
            )

            for i, (peak, is_masked) in enumerate(zip(peaks, mask)):
                x, y = spectrum.wavelength[peak.number], spectrum.intensity[peak.number]
                color = {
                    False: 'red',
                    True: 'grey',
                }[is_masked]

                ax.step(
                    x, y,
                    where='mid',
                    color=color,
                    alpha=1,
                )

                n_points = 5 * (peak.minima[1] - peak.minima[0] + 1)
                x = np.linspace(spectrum.wavelength[peak.minima[0]], spectrum.wavelength[peak.minima[1]], n_points)
                y_hat = shape(
                    x=np.linspace(spectrum.number[peak.minima[0]], spectrum.number[peak.minima[1]], n_points),
                    position=offset[i],
                    intensity=scale[i],
                    background=background[i],
                )
                ax.plot(
                    x, y_hat,
                    color='black', linestyle=':', linewidth=1,
                    alpha=1,
                )

                x, y = spectrum.wavelength[peak.number], spectrum.intensity[peak.number]
                y_hat = shape(spectrum.number[peak.number], offset[i], scale[i], background[i])
                ax.plot(
                    x, y - y_hat,
                    color='black', linestyle='none', marker='s', markersize=0.5,
                    alpha=1,
                )

            ax.set_xlabel(r'$\lambda$ [$nm$]')
            ax.set_ylabel(r'$I$ [$\%$]')
            ax.grid(
                color='grey', linestyle=':',
            )
        if figure['shape']:
            ax = figure['shape'].gca()
            ax.clear()

            grid = Grid.factory(spectrum=spectrum).create_from_peaks(
                peaks=[peaks[i] for i, mask in enumerate(mask) if mask],
                offset=[offset[i] for i, mask in enumerate(mask) if mask],
                scale=[scale[i] for i, mask in enumerate(mask) if mask],
                background=[background[i] for i, mask in enumerate(mask) if mask],
            )
            x, y = grid.x, grid.y
            ax.plot(
                x, y,
                color='grey', linestyle='none', marker='s', markersize=3,
                alpha=.5,
            )

            grid = Grid.factory(spectrum=spectrum).create_from_peaks(
                peaks=[peaks[i] for i, mask in enumerate(mask) if not mask],
                offset=[offset[i] for i, mask in enumerate(mask) if not mask],
                scale=[scale[i] for i, mask in enumerate(mask) if not mask],
                background=[background[i] for i, mask in enumerate(mask) if not mask],
            )
            x, y = grid.x, grid.y
            ax.plot(
                x, y,
                color='red', linestyle='none', marker='s', markersize=3,
                alpha=1,
            )

            x = np.linspace(min(grid.x), max(grid.x), 1000)
            y_hat = shape(x, 0, 1)
            ax.plot(
                x, y_hat,
                color='black', linestyle=':',
            )

            x, y = grid.x, grid.y
            y_hat = shape(x, 0, 1)
            ax.plot(
                x, y - y_hat,
                color='black', linestyle='none', marker='s', markersize=0.5,
            )

            ax.text(
                0.05, 0.95,
                shape.get_info(
                    sep='\n',
                    fields=dict(
                        N=f'{len(peaks)} ({len(index)})',
                    ),
                ),
                transform=ax.transAxes,
                ha='left', va='top',
            )

            ax.set_xlim([-10, +10])
            ax.set_xlabel(r'$number$')
            ax.set_ylabel(r'$I$ [$\%$]')
            ax.grid(color='grey', linestyle=':')

    return shape


def _retrieve_shape(
    grid: Grid,
) -> Shape:
    """Retrieve shape from standardized grid."""

    def _loss(grid: Grid, params: Sequence[float]) -> float:
        shape = Shape(*params)

        return mse(
            y=grid.y,
            y_hat=shape(grid.x, position=0, intensity=1),
        )

    params = ShapeParams()
    res = optimize.minimize(
        partial(_loss, grid),
        params.initial,
        # method='SLSQP',
        bounds=params.bounds,
    )
    # assert res['success'], 'Optimization is not succeeded!'

    shape = Shape(*res['x'])
    return shape


def _approximate_grid(
    grid: Grid,
    shape: Shape,
) -> tuple[ScopeParams, float]:
    """Approximate grid by the shape."""

    def _loss(params: Sequence[float], grid: Grid, shape: Shape) -> float:
        scope_params = ScopeParams(grid, *params)

        y = grid.y
        y_hat = shape(x=grid.x, **scope_params)

        return mse(y, y_hat)

    # variables
    params = ScopeParams(grid=grid)
    res = optimize.minimize(
        partial(_loss, grid=grid, shape=shape),
        params.initial,
        method='SLSQP',
        bounds=params.bounds,
    )
    assert res['success'], 'Optimization is not succeeded!'

    scope_params = ScopeParams(grid, *res['x'])

    y = grid.y
    y_hat = shape(x=grid.x, **scope_params)
    error = mse(y, y_hat) / scope_params['intensity']

    return scope_params, error
