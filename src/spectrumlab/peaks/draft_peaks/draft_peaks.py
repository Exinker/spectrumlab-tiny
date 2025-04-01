from typing import Iterator, Sequence

import numpy as np
from scipy import signal

from spectrumlab.peaks.draft_peaks.config import (
    DRAFT_PEAKS_CONFIG as CONFIG,
)
from spectrumlab.peaks.peak import Peak
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Number, U


def draft_peaks(
    spectrum: Spectrum,
) -> tuple[Peak, ...]:
    """Draft peaks from the spectrum.

    Author: Vaschenko Pavel
     Email: vaschenko@vmk.ru
      Date: 2016.04.09
    """

    # find pairs of local minima for each maximum
    maxima = find_maxima(spectrum.intensity)
    minima = find_minima(spectrum.intensity)
    pairs = find_pairs(
        maxima=maxima,
        minima=minima,
    )

    # find width
    width, *_ = signal.peak_widths(spectrum.intensity, maxima)

    # draft peaks
    peaks = []
    for i, (maximum, pair) in enumerate(zip(maxima, pairs)):
        left, right = pair  # left and right index of peak

        # correct maxima / TODO: check it!
        if spectrum.clipped[maximum]:
            index = np.arange(left, right+1)
            index = index[spectrum.clipped[index]]

            maxima = np.mean(index).astype(int).item()

        # check peaks's width
        if CONFIG.except_wide_peak:
            if width[i] > CONFIG.width_max:
                continue

        # check n_counts
        n_counts = right - left + 1

        if n_counts < CONFIG.n_counts_min:
            continue

        if n_counts > CONFIG.n_counts_max:
            continue

        # check peaks's amplitude
        _amplitude = spectrum.intensity[maximum] - (spectrum.intensity[left] + spectrum.intensity[right])/2  # от среднего значения на границах до максимума
        _deviation = (spectrum.deviation[maximum]**2 + .25*spectrum.deviation[left]**2 + .25*spectrum.deviation[right]**2)**0.5

        if _amplitude < CONFIG.amplitude_min:
            continue

        if _amplitude < CONFIG.noise_level * _deviation:
            continue

        # check clipped counts
        if CONFIG.except_clipped_peak:
            if any(spectrum.clipped[left:right+1]):
                continue

        # check peaks's slope
        if CONFIG.except_sloped_peak:
            _slope = abs(spectrum.intensity[left] - spectrum.intensity[right]) / _amplitude

            if _slope > CONFIG.slope_max:
                continue

        # gather peak
        peak = Peak(
            minima=(left, right),
            maxima=(maximum,),

            amplitude=_amplitude,
            deviation=_deviation,

            except_edges=CONFIG.except_edges,
        )
        peaks.append(peak)

    return tuple(peaks)


def find_minima(values: Sequence[U]) -> tuple[Number, ...]:
    """Find local minima index in a sequence of values."""
    n_values = len(values)
    extrema = []

    # add the first index
    i = 0
    extrema.append(i)

    # add the middle index
    for i in range(1, n_values-1):
        conditions = (
                values[i-1] > values[i] and values[i] <= values[i+1],
                values[i-1] >= values[i] and values[i] < values[i+1],
            )
        if any(conditions):
            extrema.append(i)

    # add the last index
    i = n_values-1
    extrema.append(i)

    return tuple(extrema)


def find_maxima(values: Sequence[U]) -> tuple[Number, ...]:
    """Find local maxima index in a sequence of values."""
    n_values = len(values)
    extrema = []

    # add the first index
    i = 0
    if values[i] > values[i+1]:
        extrema.append(i)

    # add the middle index
    for i in range(1, n_values-1):
        conditions = (
                values[i-1] < values[i] and values[i] >= values[i+1],
                # values[i-1] <= values[i] and values[i] > values[i+1],  # NOT USE IT: в случае, если подряд два отсчета с одинаковыми значениями, то максимум считается дважды!
            )
        if any(conditions):
            extrema.append(i)

    # add the last index
    i = n_values-1
    if values[i-1] < values[i]:
        extrema.append(i)

    return tuple(extrema)


def get_pairwise(values: Sequence[Number]) -> Iterator:
    """Get sequence by pairwise:

    Example:
        a, b, c, d, ... -> ((a, b), (b, c), (c, d), ...)
    """

    for a, b in zip(values, values[1:]):
        yield a, b


def find_pairs(maxima: tuple[Number], minima: tuple[Number]) -> tuple[tuple[Number, Number], ...]:
    """Find pairs (from a sequense of minima) for each of maxima."""
    pairs = get_pairwise(minima)

    edges = []
    for maximum in maxima:

        for left, right in pairs:
            if left <= maximum <= right:
                edge = left, right
                edges.append(edge)

                break
    return tuple(edges)
