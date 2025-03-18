import pickle
from abc import ABC
from collections.abc import Mapping
from typing import Any

import numpy as np

from spectrumlab import VERSION
from spectrumlab.emulations.detectors import Detector
from spectrumlab.types import Array, NanoMeter, Number


class AbstractSpectrum(ABC):
    """Abstract type for any emitted or absorbed spectrum."""

    def __init__(
        self,
        intensity: Array[float],
        wavelength: Array[NanoMeter] | None = None,
        number: Array[Number] | None = None,
        deviation: Array[float] | None = None,
        clipped: Array[bool] | None = None,
        detector: Detector | None = None,
    ) -> None:
        self.intensity = intensity
        self.detector = detector

        self._wavelength = wavelength
        self._number = number
        self._deviation = deviation
        self._clipped = clipped

        assert self.intensity.shape == self.clipped.shape
        assert self.intensity.shape == self.deviation.shape

    @property
    def n_times(self) -> int:
        if self.intensity.ndim == 1:
            return 1
        return self.intensity.shape[0]

    @property
    def time(self) -> Array[int]:
        return np.arange(self.n_times)

    @property
    def n_numbers(self) -> int:
        if self.intensity.ndim == 1:
            return self.intensity.shape[0]
        return self.intensity.shape[1]

    @property
    def index(self) -> Array[int]:
        """internal index of spectrum."""
        return np.arange(self.n_numbers)

    @property
    def shape(self) -> tuple[int, int]:
        return self.intensity.shape

    @property
    def number(self) -> Array[int]:
        """external index of spectrum."""
        if self._number is None:
            self._number = self.index

        return self._number

    @property
    def wavelength(self) -> Array[NanoMeter] | Array[Number]:
        if self._wavelength is None:
            self._wavelength = np.arange(self.n_numbers)

        return self._wavelength

    @property
    def deviation(self) -> Array[float]:
        if self._deviation is None:
            self._deviation = np.full(self.shape, 0)

        return self._deviation

    @property
    def clipped(self) -> Array[bool]:
        if self._clipped is None:
            self._clipped = np.full(self.shape, False)

        return self._clipped

    def __repr__(self) -> str:
        if self.intensity.ndim == 1:
            n_times, n_numbers = 1, self.shape[-1]
        else:
            n_times, n_numbers = self.shape

        cls = self.__class__
        return f'{cls.__name__}(n_times: {n_times}, n_numbers: {n_numbers})'


class Spectrum(AbstractSpectrum):
    """Type for any emitted (or ordinary) spectrum."""

    def __init__(
        self,
        intensity: Array[float],
        wavelength: Array[NanoMeter] | None = None,
        number: Array[Number] | None = None,
        deviation: Array[float] | None = None,
        clipped: Array[bool] | None = None,
        detector: Detector | None = None,
    ) -> None:
        super().__init__(
            intensity=intensity,
            wavelength=wavelength,
            number=number,
            deviation=deviation,
            clipped=clipped,
            detector=detector,
        )

    def dump(self, filepath: str) -> None:

        dat = self.dumps()
        with open(filepath, 'wb') as file:
            pickle.dump(dat, file)

    def dumps(self) -> Mapping[str, Any]:

        dat = dict(
            version=VERSION,
            intensity=pickle.dumps(self.intensity),
            wavelength=pickle.dumps(self.wavelength),
            number=pickle.dumps(self.number),
            deviation=pickle.dumps(self.deviation),
            clipped=pickle.dumps(self.clipped),
            detector=self.detector.name,
        )
        return dat

    @classmethod
    def load(cls, filepath: str) -> 'Spectrum':

        with open(filepath, 'rb') as file:
            dat = pickle.load(file)

        spectrum = cls.loads(dat)
        return spectrum

    @classmethod
    def loads(cls, dat: Mapping[str, Any]) -> 'Spectrum':

        spectrum = Spectrum(
            intensity=pickle.loads(dat['intensity']),
            wavelength=pickle.loads(dat['wavelength']),
            number=pickle.loads(dat['number']),
            deviation=pickle.loads(dat['deviation']),
            clipped=pickle.loads(dat['clipped']),
            detector=Detector[dat['detector']],
        )
        return spectrum
