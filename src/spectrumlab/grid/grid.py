from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, MicroMeter, Number, T, U

if TYPE_CHECKING:
    from spectrumlab.peaks.peak import Peak


class _FactoryBatch:

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    def create_from_peak(self, peak: 'Peak', threshold: float) -> '_Batch':
        lb, ub = peak.minima

        is_clipped = self.spectrum.clipped[lb:ub]
        is_snr_low = np.abs(self.spectrum.intensity[lb:ub]) / self.spectrum.deviation[lb:ub] < threshold
        mask = ~is_clipped & ~is_snr_low

        x = self.spectrum.number[lb:ub][mask]
        y = self.spectrum.intensity[lb:ub][mask]

        return _Batch(x, y)


class _Batch:
    factory = _FactoryBatch

    def __init__(self, x: Array[T], y: Array[U]):
        self.x = x
        self.y = y


class FactoryGrid:

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    def create_from_peaks(
        self,
        peaks: Sequence['Peak'],
        offset: Array[T] | None = None,
        scale: Array[float] | None = None,
        background: Array[float] | None = None,
        threshold: float = 0,
    ) -> 'Grid':
        """Get a grid from sequence of peaks from spectrum."""

        batches = tuple(
            _Batch.factory(spectrum=self.spectrum).create_from_peak(peak=peak, threshold=threshold)
            for peak in peaks
        )

        return self._create(
            batches=batches,
            offset=offset,
            scale=scale,
            background=background,
        )

    def _create(self, batches: Sequence[_Batch], offset: Array[T] | None = None, scale: Array[float] | None = None, background: Array[U] | None = None) -> 'Grid':
        """Get a grid from sequence of batches."""
        n_batches = len(batches)

        if offset is None:
            offset = np.full(n_batches, 0)
        assert len(offset) == n_batches, f'len of `offset` have to be equal of `n_batches`: {n_batches}'

        if scale is None:
            scale = np.full(n_batches, 1)
        assert len(scale) == n_batches, f'len of `scale` have to be equal of `n_batches`: {n_batches}'

        if background is None:
            background = np.full(n_batches, 0)
        assert len(background) == n_batches, f'len of `background` have to be equal of `n_batches`: {n_batches}'

        #
        x, y = [], []
        for t, batch in enumerate(batches):
            x.extend(batch.x - offset[t])
            y.extend((batch.y - background[t]) / scale[t])

        x, y = np.array(x).squeeze(), np.array(y).squeeze()

        index = np.argsort(x)

        #
        return Grid(
            x=x[index],
            y=y[index],
        )


class Grid:
    factory = FactoryGrid

    def __init__(
        self,
        x: Array[T],
        y: Array[float] | None = None,
        units: T | None = None,
    ) -> None:
        assert len(x) == len(y)

        self._x = x
        self._y = y
        self._units = units

        self._interpolate = None

    @property
    def x(self) -> Array[T]:
        return self._x

    @property
    def y(self) -> Array[float]:
        return self._y

    @property
    def units(self) -> T | None:
        return self._units

    @property
    def xlabel(self) -> str:
        return '{label} {units}'.format(
            label={
                Number: r'$number$',
                MicroMeter: r'$x$',
            }.get(self.units, ''),
            units=self.xunits,
        )

    @property
    def xunits(self) -> str:
        return {
            Number: r'',
            MicroMeter: r'[$\mu m$]',
        }.get(self.units, '')

    def __len__(self) -> int:
        return len(self.x)

    def __str__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}({self.units})'
