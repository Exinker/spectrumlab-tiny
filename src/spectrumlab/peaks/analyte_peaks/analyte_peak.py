from dataclasses import dataclass, field
from typing import Self

import numpy as np

from spectrumlab.line import Line
from spectrumlab.peaks.blink_peaks import DraftBlinksConfig, draft_blinks
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Number


@dataclass(frozen=True, slots=True)
class AnalytePeak:

    minima: tuple[Number, Number]
    maxima: tuple[Number] | tuple[Number, Number] | tuple[Number, ...]
    spectrum: Spectrum
    mask: Array

    except_edges: bool = field(default=False)

    @classmethod
    def create(
        cls,
        line: Line,
        spectrum: Spectrum,
    ) -> Self:
        assert spectrum.n_times == 1, 'time resolved spectra are not supported yet!'

        cursor = abs(spectrum.wavelength - line.wavelength).argmin()
        minima = [0, spectrum.n_numbers-1]
        maxima = []

        # mask
        blinks = draft_blinks(
            spectrum=spectrum,
            config=DraftBlinksConfig(
                except_clipped_peak=False,
                except_sloped_peak=False,
                except_edges=False,

                noise_level=5,
            ),
        )

        is_found = False
        mask = np.full(spectrum.shape, True)
        for blink in blinks:
            left, right = blink.minima

            is_far = (np.abs(left - cursor) > 2) and (np.abs(right - cursor) > 2)  # расстояние от cursor до blink больше 2х отсчетов  # noqa: E501
            if is_far:
                if blink.include(cursor):  # cursor находится внутри этого blink (линия без самопоглощения)
                    is_found = True
                    maxima.append(blink.maxima[0])

                else:
                    mask[blink.number] = False

            else:
                if blink.amplitude > 100 * blink.deviation:  # этот blink достаточную амплитуду, чтобы считаться частью линии (линия с самопоглощением)  # noqa: E501
                    is_found = True
                    maxima.append(blink.maxima[0])

                else:
                    mask[blink.number] = False

        if not is_found:
            mask = np.full(spectrum.shape, True)

        # mask clipped counts
        mask[spectrum.clipped] = False

        # gather peak
        peak = AnalytePeak(
            minima=tuple(minima),
            maxima=tuple(maxima),
            spectrum=spectrum,
            mask=mask,

            except_edges=False,
        )
        return peak

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
        number = np.arange(left, right+1)

        return number[self.index]

    @property
    def intensity(self):
        return self.spectrum.intensity[self.index]

    def __repr__(self) -> str:
        cls = self.__class__

        content = '; '.join([
            f'minima: {self.minima}',
            f'maxima: {self.maxima}',
        ])
        return f'{cls.__name__}({content})'
