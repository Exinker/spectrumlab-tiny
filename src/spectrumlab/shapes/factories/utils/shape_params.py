from spectrumlab.shapes.factories.utils.params import (
    AbstractParams,
    Param,
)
from spectrumlab.types import Number


class ShapeParams(AbstractParams):

    def __init__(
        self,
        width: Number | None = None,
        asymmetry: float | None = None,
        ratio: float | None = None,
    ) -> None:
        super().__init__([
            Param('width', 2.0, (0.1, 20), width),
            Param('asymmetry', 0.0, (-0.5, +0.5), asymmetry),
            Param('ratio', 0.1, (0, 1), ratio),
        ])

        self.name = 'shape'
