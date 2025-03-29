import contextvars
import logging
import warnings
from collections.abc import Iterable, Sequence
from functools import partial
from operator import itemgetter

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
from spectrumlab.types import Array, Number
from spectrumlab.utils import mse


warnings.filterwarnings('ignore')


LOGGER = logging.getLogger('spectrumlab')
FIGURES = contextvars.ContextVar('FIGURES', default=None)

DEFAULT_SHAPE = Shape(width=2, asymmetry=0, ratio=.1)


class RetrieveShapeConfig(BaseSettings):

    default_shape: Shape = Field(None, alias='RETRIEVE_SHAPE_DEFAULT')
    error_max: float = Field(default=.001, alias='RETRIEVE_SHAPE_ERROR_MAX')
    error_mean: float = Field(default=.0001, alias='RETRIEVE_SHAPE_ERROR_MEAN')
    n_peaks_filtrated_by_width: int | None = Field(default=None, alias='RETRIEVE_SHAPE_N_PEAKS_FILTRATED_BY_WIDTH')
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

        if self.n_peaks_filtrated_by_width:
            assert self.n_peaks_filtrated_by_width >= self.n_peaks_min


def retrieve_shape_from_spectrum(
    spectrum: Spectrum,
    peaks: Sequence[Peak],
    n: int,
    config: RetrieveShapeConfig | None = None,
) -> Shape:

    config = config or RetrieveShapeConfig()

    # setup figure
    figures = FIGURES.get()
    if figures:
        figure = figures[n]
    else:
        figure = None

    # calculate shape
    n_peaks = len(peaks)
    LOGGER.debug('detector %02d - peaks total: %s', n+1, n_peaks)

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

    if (n_peaks > 0) and config.n_peaks_filtrated_by_width:
        index = np.argsort(width)[config.n_peaks_filtrated_by_width:]
        mask[index] = True

        LOGGER.debug('detector %02d - peaks stayed: %s', n+1, n_peaks - sum(mask))

    try:
        while True:

            # update shape
            select = itemgetter(*np.flatnonzero(~mask).tolist())
            grid = Grid.factory(spectrum=spectrum).create_from_peaks(
                peaks=select(peaks),
                offset=select(offset),
                scale=select(scale),
                background=select(background),
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

            if np.mean(np.abs(error[index])) <= config.error_mean:
                LOGGER.debug('detector %02d - breakpoint: error_mean (%s peaks)', n+1, len(index))
                break
            if np.max(np.abs(error[index])) <= config.error_max:
                LOGGER.debug('detector %02d - breakpoint: error_max (%s peaks)', n+1, len(index))
                break

            if len(index) <= config.n_peaks_min:
                LOGGER.debug('detector %02d - breakpoint: n_peaks_min (%s peaks)', n+1, len(index))
                break

            # next step
            index, = np.where(~mask)

            step_size = 1
            worst_peaks_index = index[np.argsort(np.abs(error[index]))][-step_size:]
            mask[worst_peaks_index] = True

    except Exception:
        shape = config.default_shape

    else:
        LOGGER.debug('detector %02d - peaks stayed: %s', n+1, n_peaks - sum(mask))
        LOGGER.debug('detector %02d - peaks index: %s', n+1, error[index])
        LOGGER.debug('detector %02d - peaks offset: %s', n+1, offset[index])
        LOGGER.debug('detector %02d - peaks scale: %s', n+1, scale[index])
        LOGGER.debug('detector %02d - peaks background: %s', n+1, background[index])

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
                color='black', linestyle='-', linewidth=.5,
            )

            for index in split_by_clipped(spectrum.clipped):
                x = spectrum.wavelength[index]
                y = spectrum.intensity[index]
                ax.step(
                    x, y,
                    where='mid',
                    color='red', linestyle='-', linewidth=.5,
                    alpha=1,
                )

            for i, peak in enumerate(peaks):

                x, y = spectrum.wavelength[peak.number], spectrum.intensity[peak.number]
                ax.step(
                    x, y,
                    where='mid',
                    color={False: 'red', True: 'grey'}[mask[i]],
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

            if (n_peaks > 0) and (sum(mask) > 0):
                select = itemgetter(*np.flatnonzero(mask).tolist())
                grid = Grid.factory(spectrum=spectrum).create_from_peaks(
                    peaks=select(peaks),
                    offset=select(offset),
                    scale=select(scale),
                    background=select(background),
                )
                x, y = grid.x, grid.y
                ax.plot(
                    x, y,
                    color='grey', linestyle='none', marker='s', markersize=3,
                    alpha=.5,
                )

            if (n_peaks > 0) and (sum(~mask) > 0):
                select = itemgetter(*np.flatnonzero(~mask).tolist())
                grid = Grid.factory(spectrum=spectrum).create_from_peaks(
                    peaks=select(peaks),
                    offset=select(offset),
                    scale=select(scale),
                    background=select(background),
                )
                x, y = grid.x, grid.y
                ax.plot(
                    x, y,
                    color='red', linestyle='none', marker='s', markersize=3,
                    alpha=1,
                )

                x, y = grid.x, grid.y
                y_hat = shape(x, 0, 1)
                ax.plot(
                    x, y - y_hat,
                    color='black', linestyle='none', marker='s', markersize=0.5,
                )

            x = np.arange(-shape.rx, +shape.rx, shape.dx)
            y_hat = shape(x, 0, 1)
            ax.plot(
                x, y_hat,
                color='black', linestyle=':',
            )

            ax.text(
                0.05, 0.95,
                shape.get_info(
                    sep='\n',
                    fields=dict(
                        N=f'{len(peaks)} ({sum(~mask)})',
                    ),
                ),
                transform=ax.transAxes,
                ha='left', va='top',
            )

            ax.set_xlim([-10, +10])
            ax.set_ylim([-.05, +.55])

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


def split_by_clipped(
    __clipped: Array[bool],
) -> Iterable[Array[Number]]:

    index = []
    for i in np.where(__clipped)[0].tolist():
        index.append(i)

        if (len(index) > 1) and (index[-1] - index[-2] > 1):
            chunk, index = index[:-1], index[-1:]
            yield chunk

    yield index
