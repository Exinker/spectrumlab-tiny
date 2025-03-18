from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from spectrumlab.emulations.noise import Noise
from spectrumlab.peaks.utils import find_maxima, find_minima, find_pairs
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Number, U


@dataclass(frozen=False, slots=False)
class Peak:
    """Peak for any secondary application: masking of peaks for background algorithms, masking of overlapping peaks for intensity calculation and etc..."""

    minima: tuple[Number, Number]  # spectrum's internal index of the minima
    maxima: tuple[Number] | tuple[Number, Number] | tuple[Number, ...]  # spectrum's internal index of the maximum (or indices, if line has a self-absorption)
    amplitude: U
    deviation: U

    except_edges: bool = field(default=False)

    @property
    def number(self) -> Array[Number]:
        return np.arange(*self.minima)

    def __repr__(self) -> str:
        cls = self.__class__

        content = '; '.join([
            f'minima: {self.minima}',
            f'amplitude: {self.amplitude:.4f}',
        ])
        return f'{cls.__name__}({content})'


@dataclass(frozen=True, slots=True)
class DraftPeakConfig:
    n_counts_min: int = field(default=1)
    n_counts_max: int = field(default=500)

    except_clipped_peak: bool = field(default=True)
    except_sloped_peak: bool = field(default=True)
    except_edges: bool = field(default=False)

    noise_level: float = field(default=3)
    slope_max: float = field(default=1)  # if the slope is more, this is a tail of a peak


def draft_blinks(
    spectrum: Spectrum,
    noise: Noise,
    config: DraftPeakConfig | None = None,
) -> tuple[Peak]:
    """Draft blinks from the spectrum.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2016.04.09
    """
    config = config or DraftPeakConfig()

    # deviation
    deviation = noise(spectrum.intensity)

    # find pairs of local minima for each maximum
    maxima = find_maxima(spectrum.intensity)
    minima = find_minima(spectrum.intensity)
    pairs = find_pairs(
        maxima=maxima,
        minima=minima,
    )

    # draft peaks
    peaks = []
    for maximum, pair in zip(maxima, pairs):
        left, right = pair  # left and right index of peak

        # correct maxima / TODO: check it!
        if spectrum.clipped[maximum]:
            index = np.arange(left, right+1)
            index = index[spectrum.clipped[index]]

            maxima = np.mean(index).astype(int).item()

        # check n_counts
        n_counts = right - left + 1

        if n_counts < config.n_counts_min:
            continue

        if n_counts > config.n_counts_max:
            continue

        # check blink's amplitude
        _amplitude = spectrum.intensity[maximum] - (spectrum.intensity[left] + spectrum.intensity[right])/2  # от среднего значения на границах до максимума
        _deviation = (deviation[maximum]**2 + .25*deviation[left]**2 + .25*deviation[right]**2)**0.5

        if _amplitude < config.noise_level * _deviation:
            continue

        # check clipped counts
        if config.except_clipped_peak:
            if any(spectrum.clipped[left:right+1]):
                continue

        # check blink's slope
        if config.except_sloped_peak:
            _slope = abs(spectrum.intensity[left] - spectrum.intensity[right]) / _amplitude

            if _slope > config.slope_max:
                continue

        # gather and add a peak
        peak = Peak(
            minima=(left, right),
            maxima=(maximum,),

            amplitude=_amplitude,
            deviation=_deviation,

            except_edges=config.except_edges,
        )

        peaks.append(peak)

    # return peaks
    return tuple(peaks)
