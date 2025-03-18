import contextvars
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, overload

import numpy as np
from scipy import interpolate, optimize, signal

from spectrumlab.approximations.scope import ScopeVariables  # noqa: I100
from spectrumlab.approximations.variables import AbstractVariables, Variable
from spectrumlab.emulations.curves import pvoigt, rectangular
from spectrumlab.emulations.noise import Noise
from spectrumlab.grid import Grid
from spectrumlab.peaks.peak import DraftPeakConfig, draft_blinks
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Number, U
from spectrumlab.utils import mse


warnings.filterwarnings('ignore')


LOGGER = logging.getLogger('spectrumlab')
FIGURES = contextvars.ContextVar('FIGURES', default=None)
FIGURE = contextvars.ContextVar('FIGURE', default=None)


# --------        voigt peak shape        --------
class ShapeVariables(AbstractVariables):

    def __init__(
        self,
        width: Number | None = None,
        asymmetry: float | None = None,
        ratio: float | None = None,
    ) -> None:
        super().__init__([
            Variable('width', 2.0, (0.1, 20), width),
            Variable('asymmetry', 0.0, (-0.5, +0.5), asymmetry),
            Variable('ratio', 0.1, (0, 1), ratio),
        ])

        self.name = 'shape'


class AssociatedShapeVariables(AbstractVariables):

    def __init__(
        self,
        grid: Grid,
        *args,
        **kwargs,
    ) -> None:
        super().__init__([
            ShapeVariables(),
            ScopeVariables(grid, *args, **kwargs),
        ])

        self.grid = grid

    @property
    def initial(self) -> tuple[float]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.initial

        return result

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.bounds

        return result

    @property
    def value(self) -> tuple[tuple[float, ...], ...]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.value

        return result

    @classmethod
    def parse_params(
        cls,
        grid: Grid,
        params: Sequence[float],
    ) -> tuple[ShapeVariables, ScopeVariables]:
        assert len(params) == 6

        shape_variables = ShapeVariables(*params[:3])
        scope_variables = ScopeVariables(grid, *params[3:])

        return shape_variables, scope_variables


class Shape:

    def __init__(
        self,
        width: Number,
        asymmetry: float,
        ratio: float,
        rx: Number = 50,
        dx: Number = 1e-2,
    ) -> None:
        """Voigt peak's shape. A convolution of apparatus shape and aperture shape (rectangular) of a detector.

        Params:
            width: Number - apparatus shape's width
            asymmetry: float - apparatus shape's asymmetry
            ratio: float - apparatus shape's ratio

            rx: Number = 50 - range of convolution grid
            dx: Number = 0.01 - step of convolution grid
        """
        super().__init__()

        self.width = width
        self.asymmetry = asymmetry
        self.ratio = ratio
        self.rx = rx  # границы построения интерполяции
        self.dx = dx  # шаг сетки интерполяции

        # grid
        x = np.linspace(-self.rx, +self.rx, 2*int(self.rx/self.dx) + 1)
        y = signal.convolve(
            pvoigt(x, x0=0, w=self.width, a=self.asymmetry, r=self.ratio),
            rectangular(x, x0=0, w=1),
            mode='same',
        ) * self.dx

        self._f = interpolate.interp1d(
            x, y,
            kind='linear',
            bounds_error=False,
            fill_value=0,
        )

    @property
    def f(self) -> Callable[[Array[Number]], Array[U]]:
        return self._f

    def get_info(
        self,
        sep: Literal[r'\n', '; '] = '; ',
        is_signed: bool = True,
        fields: Mapping[str, Any] | None = None,
    ) -> str:
        fields = fields or {}
        sign = {+1: '+'}.get(np.sign(self.asymmetry), '') if is_signed else ''

        return sep.join([
            f'{key}={value}'
            for key, value in fields.items()
        ] + [
            f'width={self.width:.4f}',
            f'asymmetry={sign}{self.asymmetry:.4f}',
            f'ratio={self.ratio:.4f}',
        ])

    @classmethod
    def from_grid(
        cls,
        grid: Grid,
    ) -> 'Shape':

        def _loss(grid: Grid, params: Sequence[float]) -> float:
            shape_variables, scope_variables = AssociatedShapeVariables.parse_params(
                grid=grid,
                params=params,
            )
            shape = Shape(**shape_variables)

            return mse(
                y=grid.y,
                y_hat=shape(x=grid.x, **scope_variables),
            )

        # approx
        variables = AssociatedShapeVariables(grid=grid)

        res = optimize.minimize(
            partial(_loss, grid),
            variables.initial,
            # method='SLSQP',
            bounds=variables.bounds,
        )
        # assert res['success'], 'Optimization is not succeeded!'

        shape_variables, scope_variables = AssociatedShapeVariables.parse_params(
            grid=grid,
            params=res['x'],
        )

        # shape
        shape = cls(**shape_variables)
        return shape

    @overload
    def __call__(self, x: Number, position: Number, intensity: float, background: float = 0) -> U: ...
    @overload
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float = 0) -> Array[U]: ...
    def __call__(self, x, position, intensity, background=0):
        return background + intensity*self.f(x - position)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}({self.get_info()})'


def approx_grid(
    grid: Grid,
    shape: Shape,
) -> tuple[ScopeVariables, float]:
    """Approximate grid by Shape."""

    def _loss(params: Sequence[float], grid: Grid, shape: Shape) -> float:
        scope_variables = ScopeVariables(grid, *params)

        y = grid.y
        y_hat = shape(x=grid.x, **scope_variables)

        return mse(y, y_hat)

    # variables
    variables = ScopeVariables(grid=grid)

    res = optimize.minimize(
        partial(_loss, grid=grid, shape=shape),
        variables.initial,
        method='SLSQP',
        bounds=variables.bounds,
    )
    assert res['success'], 'Optimization is not succeeded!'

    scope_variables = ScopeVariables(grid, *res['x'])

    y = grid.y
    y_hat = shape(x=grid.x, **scope_variables)
    error = mse(y, y_hat) / scope_variables['intensity']

    return scope_variables, error


# --------        restore shape        --------
@dataclass
class RestoreShapeConfig:
    default_shape: Shape = field(default=Shape(2, 0, .1))

    error_max: float = field(default=.01)
    error_mean: float = field(default=.001)
    n_peaks_min: int = field(default=10)


def restore_shape_from_grid(
    grid: Grid,
) -> 'Shape':
    """Restore voigt peaks's shape from standardized grid."""

    def _loss(grid: Grid, params: Sequence[float]) -> float:
        shape = Shape(*params)

        return mse(
            y=grid.y,
            y_hat=shape(grid.x, position=0, intensity=1),
        )

    # variables
    variables = ShapeVariables()

    res = optimize.minimize(
        partial(_loss, grid),
        variables.initial,
        # method='SLSQP',
        bounds=variables.bounds,
    )
    # assert res['success'], 'Optimization is not succeeded!'

    shape = Shape(*res['x'])
    return shape


def restore_shape_from_spectrum(
    spectrum: Spectrum,
    noise: Noise,
    draft_peak_config: DraftPeakConfig | None = None,
    restore_shape_config: RestoreShapeConfig | None = None,
) -> Shape:

    draft_peak_config = draft_peak_config or DraftPeakConfig()
    restore_shape_config = restore_shape_config or RestoreShapeConfig()

    # draft blinks
    blinks = draft_blinks(
        spectrum=spectrum,
        noise=noise,
        config=draft_peak_config,
    )

    # calculate shape
    n_blinks = len(blinks)

    offset = np.zeros(n_blinks)
    scale = np.zeros(n_blinks)
    background = np.zeros(n_blinks)
    error = np.zeros(n_blinks)
    mask = np.full(n_blinks, False)
    for i, blink in enumerate(blinks):
        lb, ub = blink.minima
        grid = Grid(spectrum.number[lb:ub], spectrum.intensity[lb:ub], units=Number)

        scope_variables, error[i] = approx_grid(
            grid=grid,
            shape=Shape.from_grid(
                grid=grid,
            ),
        )
        offset[i], scale[i], background[i] = scope_variables.value

    while True:

        # update shape
        grid = Grid.factory(spectrum=spectrum).create_from_blinks(
            blinks=[blinks[i] for i, is_masked in enumerate(mask) if not is_masked],
            offset=[offset[i] for i, is_masked in enumerate(mask) if not is_masked],
            scale=[scale[i] for i, is_masked in enumerate(mask) if not is_masked],
            background=[background[i] for i, is_masked in enumerate(mask) if not is_masked],
        )
        shape = restore_shape_from_grid(
            grid=grid,
        )

        # update offset, scale, background
        for i, blink in enumerate(blinks):
            lb, ub = blink.minima
            scope_variables, error[i] = approx_grid(
                grid=Grid(spectrum.number[lb:ub], spectrum.intensity[lb:ub], units=Number),
                shape=shape,
            )
            offset[i], scale[i], background[i] = scope_variables.value

        # breakpoints
        index, = np.where(~mask)

        if np.mean(np.abs(error[index])) <= restore_shape_config.error_mean:
            LOGGER.debug('Breakpoint: error_mean (%s peaks)', len(index))
            break
        if np.max(np.abs(error[index])) <= restore_shape_config.error_max:
            LOGGER.debug('Breakpoint: error_max (%s peaks)', len(index))
            break

        if len(index) <= restore_shape_config.n_peaks_min:
            LOGGER.debug('Breakpoint: n_peaks_min (%s peaks)', len(index))
            break

        # next step
        index, = np.where(~mask)

        step_size = 1
        worst_blinks_index = index[np.argsort(np.abs(error[index]))][-step_size:]
        mask[worst_blinks_index] = True

    LOGGER.debug('Peaks index: %s', error[index])
    LOGGER.debug('Peaks offset: %s', offset[index])
    LOGGER.debug('Peaks scale: %s', scale[index])
    LOGGER.debug('Peaks background: %s', background[index])

    # show on figures
    figure = FIGURE.get()
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

            for i, (blink, is_masked) in enumerate(zip(blinks, mask)):
                x, y = spectrum.wavelength[blink.number], spectrum.intensity[blink.number]
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

                n_points = 5 * (blink.minima[1] - blink.minima[0] + 1)
                x = np.linspace(spectrum.wavelength[blink.minima[0]], spectrum.wavelength[blink.minima[1]], n_points)
                y_hat = shape(np.linspace(spectrum.number[blink.minima[0]], spectrum.number[blink.minima[1]], n_points),  offset[i], scale[i], background[i])
                ax.plot(
                    x, y_hat,
                    color='black', linestyle=':', linewidth=1,
                    alpha=1,
                )

                x, y = spectrum.wavelength[blink.number], spectrum.intensity[blink.number]
                y_hat = shape(spectrum.number[blink.number], offset[i], scale[i], background[i])
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

            grid = Grid.factory(spectrum=spectrum).create_from_blinks(
                blinks=[blinks[i] for i, mask in enumerate(mask) if mask],
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

            grid = Grid.factory(spectrum=spectrum).create_from_blinks(
                blinks=[blinks[i] for i, mask in enumerate(mask) if not mask],
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
                        N=f'{len(blinks)} ({len(index)})',
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
