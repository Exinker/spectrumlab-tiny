from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Param:

    name: str
    initial: float
    bounds: tuple[float, float]
    value: float | None


class AbstractParams(Mapping):
    """Abstract variables type."""

    def __init__(self, __items: Sequence[Param]) -> None:
        self._items = {
            item.name: item
            for item in __items
        }

    @property
    def initial(self) -> tuple[float]:
        return tuple(self._items[key].initial for key in self.keys())

    @property
    def bounds(self) -> tuple[tuple[float, float]]:
        return tuple(self._items[key].bounds for key in self.keys())

    @property
    def value(self) -> tuple[float] | tuple[None]:
        return tuple(self._items[key].value for key in self.keys())

    def __repr__(self) -> str:
        cls = self.__class__

        content = '\n\t'.join([
            f'{self._items[key]},'
            for key in self.keys()
        ])
        return f'{cls.__name__}(\n\t{content}\n)'

    def __getitem__(self, __key: str) -> float | None:
        item = self._items[__key]
        return item.value

    def __iter__(self) -> Iterator:
        return iter(key for key in self._items.keys())

    def __len__(self) -> int:
        return len(self._items)
