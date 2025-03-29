from dataclasses import dataclass, field

import numpy as np

from spectrumlab.types import Array, Number, U


@dataclass(frozen=False, slots=False)
class Peak:

    minima: tuple[Number, Number]
    maxima: tuple[Number]
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
