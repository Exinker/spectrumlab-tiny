from dataclasses import dataclass

from spectrumlab.types import NanoMeter, Symbol


@dataclass(frozen=True, slots=True)
class Line:

    symbol: Symbol
    wavelength: NanoMeter

    @property
    def nickname(self) -> str:
        return f'{self.symbol} {self.wavelength}'
