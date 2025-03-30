from dataclasses import dataclass, field
from enum import Enum

from spectrumlab.types import Electron, MicroMeter


@dataclass(frozen=True)
class DetectorConfig:
    """Detector's confg."""
    name: str
    capacity: Electron
    read_noise: Electron
    n_pixels: int
    width: MicroMeter
    height: MicroMeter
    description: str = field(default='')

    def __str__(self) -> str:
        cls = self.__class__
        name = self.name

        return f'{cls.__name__}: {name}'


class Detector(Enum):
    """Enums with detectors."""

    BLPP369M1 = DetectorConfig(
        name='БЛПП-369М1',
        capacity=2_000_000,
        read_noise=120,
        n_pixels=2612,
        width=12.5,
        height=1000,
    )
    BLPP2000 = DetectorConfig(
        name='БЛПП-2000',
        capacity=200_000,
        read_noise=25,
        n_pixels=2048,
        width=14,
        height=1000,
    )
    BLPP4000 = DetectorConfig(
        name='БЛПП-4000',
        capacity=80_000,
        read_noise=16,
        n_pixels=4096,
        width=7,
        height=200,
    )
    S8377_256Q = DetectorConfig(
        name='S8377-256Q',
        capacity=80_000_000,
        read_noise=4000,
        n_pixels=256,
        width=50,
        height=500,
        description=r"""See more info at [hamamatsu](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/s8377-128q_etc_kmpd1066e.pdf).""",
    )

    @property
    def config(self) -> DetectorConfig:
        """Config of the detector."""
        return self.value

    def __str__(self) -> str:
        cls = self.__class__
        name = self.config.name

        return f'{cls.__name__}: {name}'


if __name__ == '__main__':
    for detector in Detector:
        print(detector)
