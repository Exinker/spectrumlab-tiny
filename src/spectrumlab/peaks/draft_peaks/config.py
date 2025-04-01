from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DraftPeaksConfig(BaseSettings):

    n_counts_min: int = Field(10, ge=1, le=50, alias='DRAFT_PEAK_N_COUNTS_MIN')
    n_counts_max: int = Field(100, ge=1, le=500, alias='DRAFT_PEAK_N_COUNTS_MAX')

    except_clipped_peak: bool = Field(True, alias='DRAFT_PEAK_EXCEPT_CLIPPED_PEAK')
    except_wide_peak: bool = Field(False, alias='DRAFT_PEAK_EXCEPT_WIDE_PEAK')
    except_sloped_peak: bool = Field(True, alias='DRAFT_PEAK_EXCEPT_SLOPED_PEAK')
    except_edges: bool = Field(False, alias='DRAFT_PEAK_EXCEPT_EDGES')

    amplitude_min: float = Field(0, ge=0, le=1e+3, alias='DRAFT_PEAK_AMPLITUDE_MIN')
    width_max: float = Field(3.5, ge=1, le=10, alias='DRAFT_PEAK_WIDTH_MAX')
    slope_max: float = Field(.25, ge=0, le=1, alias='DRAFT_PEAK_SLOPE_MAX')

    noise_level: int = Field(10, ge=1, le=100, alias='DRAFT_PEAK_NOISE_LEVEL')

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    @model_validator(mode='after')
    def validate(self) -> Self:
        assert self.n_counts_min < self.n_counts_max

        return self


DRAFT_PEAKS_CONFIG = DraftPeaksConfig()
