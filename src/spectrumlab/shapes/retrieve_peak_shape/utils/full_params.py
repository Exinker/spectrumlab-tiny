from collections.abc import Sequence

from spectrumlab.grid import Grid
from spectrumlab.shapes.retrieve_peak_shape.utils.params import AbstractParams
from spectrumlab.shapes.retrieve_peak_shape.utils.scope_params import ScopeParams
from spectrumlab.shapes.retrieve_peak_shape.utils.shape_params import ShapeParams


class FullParams(AbstractParams):

    def __init__(
        self,
        grid: Grid,
        *args,
        **kwargs,
    ) -> None:
        super().__init__([
            ShapeParams(),
            ScopeParams(grid, *args, **kwargs),
        ])

        self.grid = grid

    @property
    def initial(self) -> tuple[float]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.initial

        return result

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.bounds

        return result

    @property
    def value(self) -> tuple[tuple[float, ...], ...]:

        result = ()
        for key in self._items.keys():
            item = self._items[key]
            result += item.value

        return result

    @classmethod
    def parse(
        cls,
        grid: Grid,
        params: Sequence[float],
    ) -> tuple[ShapeParams, ScopeParams]:
        assert len(params) == 6

        shape_variables = ShapeParams(*params[:3])
        scope_variables = ScopeParams(grid, *params[3:])

        return shape_variables, scope_variables
