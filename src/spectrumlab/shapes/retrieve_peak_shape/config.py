from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from spectrumlab.shapes.shape import Shape


DEFAULT_SHAPE = Shape(width=2, asymmetry=0, ratio=.1)


class RetrieveShapeConfig(BaseSettings):

    default_shape: Shape = Field(None, alias='RETRIEVE_SHAPE_DEFAULT')
    min_width: float = Field(1, ge=1e-2, le=1e+2, alias='RETRIEVE_SHAPE_MIN_WIDTH')
    max_width: float = Field(7, ge=1e-2, le=1e+2, alias='RETRIEVE_SHAPE_MAX_WIDTH')
    max_asymmetry: Shape = Field(.5, ge=0, le=.5, alias='RETRIEVE_SHAPE_MAX_ASYMMETRY')
    error_max: float = Field(default=1e-3, alias='RETRIEVE_SHAPE_ERROR_MAX')
    error_mean: float = Field(default=1e-4, alias='RETRIEVE_SHAPE_ERROR_MEAN')
    n_peaks_filtrate_by_width: int | None = Field(default=None, alias='RETRIEVE_SHAPE_N_PEAKS_FILTRATE_BY_WIDTH')
    n_peaks_min: int = Field(default=10, alias='RETRIEVE_SHAPE_N_PEAKS_MIN')

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    @field_validator('default_shape', mode='before')
    @classmethod
    def validate_default_shape(cls, data: str | None) -> Shape:

        if data is None:
            return DEFAULT_SHAPE

        try:
            width, asymmetry, ratio = map(float, data.split(';'))
            shape = Shape(width=width, asymmetry=asymmetry, ratio=ratio)
        except Exception:
            shape = DEFAULT_SHAPE
        return shape

    @model_validator(mode='after')
    def validate(self) -> None:

        if self.n_peaks_filtrate_by_width:
            assert self.n_peaks_filtrate_by_width >= self.n_peaks_min

        assert self.min_width < self.max_width


RETRIEVE_SHAPE_CONFIG = RetrieveShapeConfig()
