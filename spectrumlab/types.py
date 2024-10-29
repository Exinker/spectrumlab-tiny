from pathlib import Path
from typing import NewType, TypeAlias, TypeVar

import pandas as pd
from numpy.typing import NDArray  # noqa: I100


# --------        paths        --------
DirPath: TypeAlias = str | Path
FilePath: TypeAlias = str | Path


# --------        structures        --------
Array: TypeAlias = NDArray

Index: TypeAlias = pd.Index | pd.MultiIndex
Series: TypeAlias = pd.Series
Frame: TypeAlias = pd.DataFrame


# --------        spacial units        --------
MicroMeter = NewType('MicroMeter', float)
NanoMeter = NewType('NanoMeter', float)

Number = NewType('Number', float)

T = TypeVar('T', Number, MicroMeter)

# --------        value units        --------
Digit = NewType('Digit', float)
Electron = NewType('Electron', float)
Percent = NewType('Percent', float)

U = TypeVar('U', Digit, Electron, Percent)
