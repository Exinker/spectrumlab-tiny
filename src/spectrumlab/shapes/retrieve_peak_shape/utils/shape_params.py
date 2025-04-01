from spectrumlab.shapes.retrieve_peak_shape.config import (
    RETRIEVE_SHAPE_CONFIG as CONFIG,
)
from spectrumlab.shapes.retrieve_peak_shape.utils.params import (
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
            Param('width', 2.0, (CONFIG.min_width, CONFIG.max_width), width),
            Param('asymmetry', 0.0, (-CONFIG.max_asymmetry, +CONFIG.max_asymmetry), asymmetry),
            Param('ratio', 0.1, (0, 1), ratio),
        ])

        self.name = 'shape'
