from dataclasses import dataclass, field

import numpy as np

from spectrumlab.types import Array, Number


@dataclass(frozen=False, slots=False)
class Peak:

    minima: tuple[Number, Number]
    maxima: tuple[Number, ...]

    except_edges: bool = field(default=False)

    @property
    def n_numbers(self) -> int:

        left, right = self.minima
        return right - left + 1

    @property
    def index(self) -> Array[int]:

        if self.except_edges:
            return np.arange(1, self.n_numbers - 1)

        return np.arange(self.n_numbers)

    @property
    def number(self) -> Array[Number]:

        left, right = self.minima
        number = np.arange(left, right + 1)

        return number[self.index]

    def __repr__(self) -> str:
        cls = self.__class__

        content = '; '.join([
            f'minima: {self.minima}',
            f'maxima: {self.maxima}',
        ])
        return f'{cls.__name__}({content})'
