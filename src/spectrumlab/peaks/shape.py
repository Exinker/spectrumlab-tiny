import warnings
from collections.abc import Sequence
from functools import partial
from typing import Callable, Literal, TYPE_CHECKING
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, optimize, signal
from tqdm import tqdm

from spectrumlab.approximations.scope import ScopeVariables  # noqa: I100
from spectrumlab.approximations.variables import AbstractVariables, Variable
from spectrumlab.emulations.curves import pvoigt, rectangular
from spectrumlab.emulations.noise import Noise
from spectrumlab.spectra import Spectrum
from spectrumlab.grid import Grid
from spectrumlab.peaks.peak import DraftPeakConfig, draft_blinks
from spectrumlab.types import Array, MicroMeter, Number, U
from spectrumlab.utils import mse

if TYPE_CHECKING:
    from spectrumlab.peaks.analyte_peak import AnalytePeak


warnings.filterwarnings('ignore')


# --------        voigt peak shape        --------
class ShapeVariables(AbstractVariables):

    def __init__(self, width: Number | None = None, asymmetry: float | None = None, ratio: float | None = None):
        super().__init__([
            Variable('width', 2.0, (0.1, 20), width),
            Variable('asymmetry', 0.0, (-0.5, +0.5), asymmetry),
            Variable('ratio', 0.1, (0, 1), ratio),
        ])

        self.name = 'shape'


class AssociatedShapeVariables(AbstractVariables):

    def __init__(self, grid: Grid, *args, **kwargs):
        super().__init__([
            ShapeVariables(),
            ScopeVariables(grid, *args, **kwargs),
        ])

        #
        self.grid = grid

    @property
    def initial(self) -> tuple[float]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.initial

        return result

    @property
    def bounds(self) -> tuple[tuple[float]]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.bounds

        return result

    @property
    def value(self) -> tuple[tuple[float]]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.value

        return result

    @classmethod
    def parse_params(cls, grid: Grid, params: Sequence[float]) -> tuple[ShapeVariables, ScopeVariables]:
        assert len(params) == 6

        shape_variables = ShapeVariables(*params[:3])
        scope_variables = ScopeVariables(grid, *params[3:])

        return shape_variables, scope_variables


class Shape:

    def __init__(self, width: Number, asymmetry: float, ratio: float, rx: Number = 10, dx: Number = 1e-2) -> None:
        """Voigt peak's shape. A convolution of apparatus shape and aperture shape (rectangular) of a detector.

        Params:
            width: Number - apparatus shape's width
            asymmetry: float - apparatus shape's asymmetry
            ratio: float - apparatus shape's ratio

            rx: Number = 10 - range of convolution grid
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

    def get_content(self, sep: Literal[r'\n', '; '] = '; ', is_signed: bool = True) -> str:
        sign = {+1: '+'}.get(np.sign(self.asymmetry), '') if is_signed else ''

        return sep.join([
            f'width={self.width:.4f}',
            f'asymmetry={sign}{self.asymmetry:.4f}',
            f'ratio={self.ratio:.4f}',
        ])

    # --------        fabric        --------
    @classmethod
    def from_grid(cls, grid: Grid, show: bool = False, scale: MicroMeter = 1) -> 'Shape':

        def _loss(grid: Grid, params: Sequence[float]) -> float:
            shape_variables, scope_variables = AssociatedShapeVariables.parse_params(grid=grid, params=params)
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

        shape_variables, scope_variables = AssociatedShapeVariables.parse_params(grid=grid, params=res['x'])

        # shape
        shape = cls(**shape_variables)

        # show
        if show:
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

            x, y = grid.x, grid.y
            plt.plot(
                scale*x, y,
                color='red', linestyle='none', marker='s', markersize=3,
                alpha=1,
            )

            x = grid.space()
            y_hat = shape(x, **scope_variables)
            plt.plot(
                scale*x, y_hat,
                color='black', linestyle='-', linewidth=1,
                alpha=1,
            )

            x, y = grid.x, grid.y
            y_hat = shape(grid.x, **scope_variables)
            plt.plot(
                scale*x, y - y_hat,
                color='black', linestyle='none', marker='s', markersize=0.5,
                alpha=1,
            )

            content = cls.get_content(shape, sep='\n')
            plt.text(
                0.05, 0.95,
                content,
                transform=ax.transAxes,
                ha='left', va='top',
            )

            xlim = 5 if scale == 1 else 50
            plt.xlim([-xlim, +xlim])
            plt.xlabel(r'$number$' if scale == 1 else r'$x$ [$\mu m$]')
            plt.ylabel(r'$I$ [$\%$]')
            plt.grid(color='grey', linestyle=':')

            plt.show()

        #
        return shape

    @overload
    def __call__(self, x: Number, position: Number, intensity: float, background: float = 0) -> U: ...
    @overload
    def __call__(self, x: Array[Number], position: Number, intensity: float, background: float = 0) -> Array[U]: ...
    def __call__(self, x, position, intensity, background=0):
        return background + intensity*self.f(x - position)

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}({self.get_content()})'


def approx_grid(grid: Grid, shape: Shape, show: bool = False) -> tuple[ScopeVariables, float]:
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

    if show:
        plt.subplots(figsize=(6, 4), tight_layout=True)

        plt.title(f'error: {error:.5f}')

        x, y = grid.x, grid.y
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = np.linspace(min(grid.x), max(grid.x), 1000)
        y_hat = shape(x, **scope_variables)
        plt.plot(
            x, y_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x, y = grid.x, grid.y
        y_hat = shape(x, **scope_variables)
        plt.plot(
            x, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
            alpha=1,
        )

        plt.xlim([-10, +10])
        plt.xlabel(r'$number$')
        plt.ylabel(r'$I$ [$\%$]')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return scope_variables, error


def restore_shape_from_grid(grid: Grid, show: bool = False) -> 'Shape':
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

    if show:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

        x, y = grid.x, grid.y
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = grid.space()
        y_hat = shape(x, 0, 1)
        plt.plot(
            x, y_hat,
            color='black', linestyle='-', linewidth=1,
            alpha=1,
        )

        x, y = grid.x, grid.y
        y_hat = shape(grid.x, 0, 1)
        plt.plot(
            x, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
            alpha=1,
        )

        plt.text(
            0.05, 0.95,
            shape.get_content(sep='\n'),
            transform=ax.transAxes,
            ha='left', va='top',
        )

        plt.xlim([-10, +10])
        plt.xlabel(r'$number$')
        plt.ylabel(r'$I$ [$\%$]')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return shape


def restore_shape_from_spectrum(
    spectrum: Spectrum,
    noise: Noise,
    verbose: bool = False,
    show: bool = False,
) -> Shape:

    # draft blinks
    blinks = draft_blinks(
        spectrum=spectrum,
        noise=noise,
        config=DraftPeakConfig(
            n_counts_min=10,
            n_counts_max=100,

            except_clipped_peak=True,
            except_sloped_peak=True,
            except_edges=False,

            noise_level=10,
        ),
    )

    # calculate shape
    n_blinks = len(blinks)

    offset = np.zeros(n_blinks)
    scale = np.zeros(n_blinks)
    background = np.zeros(n_blinks)
    error = np.zeros(n_blinks)
    mask = np.full(n_blinks, False)
    for i, blink in tqdm(enumerate(blinks), total=n_blinks, desc='Initializing:', unit='blinks', disable=not verbose):
        lb, ub = blink.minima
        grid = Grid(spectrum.number[lb:ub], spectrum.intensity[lb:ub], units=Number)

        scope_variables, error[i] = approx_grid(
            grid=grid,
            shape=Shape.from_grid(
                grid=grid,
            ),
        )
        offset[i], scale[i], background[i] = scope_variables.value

    with tqdm(desc='Filtration:', unit='blinks', disable=not verbose) as pbar:
        while True:
            pbar.update(1)

            # update shape
            grid = Grid.factory(spectrum=spectrum).create_from_blinks(
                blinks=[blinks[i] for i, mask in enumerate(mask) if not mask],
                offset=[offset[i] for i, mask in enumerate(mask) if not mask],
                scale=[scale[i] for i, mask in enumerate(mask) if not mask],
                background=[background[i] for i, mask in enumerate(mask) if not mask],
            )
            shape = restore_shape_from_grid(
                grid=grid,
                show=False,
            )

            # update offset, scale, background
            for i, blink in enumerate(blinks):
                lb, ub = blink.minima
                scope_variables, error[i] = approx_grid(
                    grid=Grid(spectrum.number[lb:ub], spectrum.intensity[lb:ub], units=Number),
                    shape=shape,
                    show=False,
                )
                offset[i], scale[i], background[i] = scope_variables.value

            # udpate mask
            index, = np.where(~mask)
            i = index[np.argmax(np.abs(error[index]))]

            mask[i] = True

            # breakpoints
            if max(np.abs(error[index])) <= .001:
                break

            if len(index) <= 10:
                break

    if show:
        fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, figsize=(15, 5), tight_layout=True)

        plt.sca(ax_left)
        spectrum.show(ax=ax_left)
        for blink in [blinks[i] for i, mask in enumerate(mask) if mask]:
            x, y = spectrum.wavelength[blink.number], spectrum.intensity[blink.number]
            plt.step(
                x, y,
                where='mid',
                color='grey',
                alpha=1,
            )
        for blink in [blinks[i] for i, mask in enumerate(mask) if not mask]:
            x, y = spectrum.wavelength[blink.number], spectrum.intensity[blink.number]
            plt.step(
                x, y,
                where='mid',
                color='red',
            )

        plt.sca(ax_mid)
        grid = Grid.factory(spectrum=spectrum).create_from_blinks(
            blinks=[blinks[i] for i, mask in enumerate(mask) if mask],
            offset=[offset[i] for i, mask in enumerate(mask) if mask],
            scale=[scale[i] for i, mask in enumerate(mask) if mask],
            background=[background[i] for i, mask in enumerate(mask) if mask],
        )
        x, y = grid.x, grid.y
        plt.plot(
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
        plt.plot(
            x, y,
            color='red', linestyle='none', marker='s', markersize=3,
            alpha=1,
        )

        x = np.linspace(min(grid.x), max(grid.x), 1000)
        y_hat = shape(x, 0, 1)
        plt.plot(
            x, y_hat,
            color='black', linestyle=':',
        )

        x, y = grid.x, grid.y
        y_hat = shape(x, 0, 1)
        plt.plot(
            x, y - y_hat,
            color='black', linestyle='none', marker='s', markersize=0.5,
        )

        plt.text(
            0.05, 0.95,
            shape.get_content(sep='\n'),
            transform=ax_mid.transAxes,
            ha='left', va='top',
        )

        plt.xlim([-10, +10])
        plt.xlabel(r'$number$')
        plt.ylabel(r'$I, \%$')
        plt.grid(color='grey', linestyle=':')

        plt.sca(ax_right)

        y = 100*error
        plt.plot(y)
        plt.plot(
            np.arange(n_blinks)[~mask], y[~mask],
            color='red', linestyle='none', marker='.',
        )

        plt.xlabel(r'$index$')
        plt.ylabel(r'$error, \%$')
        plt.grid(color='grey', linestyle=':')

        plt.show()

    return shape
