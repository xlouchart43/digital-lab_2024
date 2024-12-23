from re import sub as re_sub

import numpy as np

from ..config.constants import NB_VEGETATION_INDEXES


class VegetationIndice:
    """Class to define how to compute a vegetation index from Sentinel2 bands
    in Google Earth Engine API"""

    def __init__(
        self, name: str, percentile1: float, percentile99: float, formula: str
    ) -> None:
        self.name = name
        self.percentile1 = percentile1
        self.percentile99 = percentile99
        self.formula = formula
        # Change all occurence of "B" followed by 1 or two digit or letter to 'b("found_string")'
        self.formula = re_sub(r"B(\d{1,2}[A-Z]?)", r'b("B\1")', self.formula)
        self.formula = f"min(255, max(0, {self.formula}))"

    def computation_expression_ee(self) -> np.ndarray:
        """Return the string corresponding to the computation of the vegetation index
        in Google Earth Engine API, based on its formula."""
        stretch_coef = 255 / (self.percentile99 - self.percentile1)
        shift_coef = -self.percentile1 * stretch_coef
        return f"""{stretch_coef} * ({self.formula}) + {shift_coef}"""


# Define the vegetation indices
VEGETATION_INDICES_FORMULA = {
    "NDWI": "(B8A - B11) / (B8A + B11 + 1e-6)",
    "DWSI": "(B8 + B3) / (B4 + B11 + 1e-6)",
    "NGRDI": "(B3 - B4) / (B3 + B4 + 1e-6)",
    "RDI": "B12 / (B8A + 1e-6)",
    "GLI": "(2 * B3 - B2 - B4) / (2 * B3 + B2 + B4 + 1e-6)",
    "NDRE2": "(B7 - B5) / (B7 + B5 + 1e-6)",
    "PBI": "B8 / (B3 + 1e-6)",
    "NDVI": "(B8A - B4) / (B8A + B4 + 1e-6)",
    "GNDVI": "(B8A - B3) / (B8A + B3 + 1e-6)",
    "CIG": "(B8A / (B3 + 1e-6) - 1)",
    "CVI": "(B8A * B5) / (B3**2 + 1e-6)",
    "NDRE3": "(B8A - B7) / (B8A + B7 + 1e-6)",
    "DRS": "sqrt((B4**2) + (B12**2))",
}

VEGETATION_INDICES_NAMES = [
    "NDWI",
    "DWSI",
    "NGRDI",
    "RDI",
    "GLI",
    "NDRE2",
    "PBI",
    "NDVI",
    "GNDVI",
    "CIG",
    "CVI",
    "NDRE3",
    "DRS",
]

VEGETATION_INDICES_PERCENTILE1 = [-1, 0, -1, 0, -1, -1, 0, -1, -1, 0, 0, -1, 100]

VEGETATION_INDICES_PERCENTILE99 = [1, 5, 1, 2, 1, 1, 20, 1, 1, 20, 50, 1, 3300]

VEGETATION_INDICES = []

for i, indice_name in enumerate(VEGETATION_INDICES_NAMES):
    VEGETATION_INDICES.append(
        VegetationIndice(
            indice_name,
            VEGETATION_INDICES_PERCENTILE1[i],
            VEGETATION_INDICES_PERCENTILE99[i],
            VEGETATION_INDICES_FORMULA[indice_name],
        )
    )

if len(VEGETATION_INDICES) != NB_VEGETATION_INDEXES:
    raise ValueError(
        f"The number of vegetation indices {len(VEGETATION_INDICES)} does not match the expected number {NB_VEGETATION_INDEXES}, please check the configuration."
    )
