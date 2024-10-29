from typing import Iterator, Sequence

from spectrumlab.types import Number, U


def find_minima(values: Sequence[U]) -> tuple[Number]:
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


def find_maxima(values: Sequence[U]) -> tuple[Number]:
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


def find_pairs(maxima: tuple[Number], minima: tuple[Number]) -> list[tuple[Number, Number]]:
    """Find pairs (from a sequense of minima) for each of maxima."""
    pairs = get_pairwise(minima)

    edges = []
    for maximum in maxima:

        for left, right in pairs:
            if left <= maximum <= right:
                edge = left, right
                edges.append(edge)

                break

    return edges
