import numpy as np

from ..config import NB_S2_BANDS


class Sentinel2Band:
    """Class to define the normalization of a Sentinel2 band to improve contrast"""

    def __init__(self, name, percentile1, percentile99) -> None:
        self.name = name
        self.percentile1 = percentile1
        self.percentile99 = percentile99

    def normalize(self, band: np.array) -> np.ndarray:
        """Apply normalization to the band from init_scale to 0-255"""
        return np.min(
            255,
            np.max(
                0,
                255
                * (band - self.percentile1)
                / (self.percentile99 - self.percentile1),
            ),
        )

    def normalize_expression_ee(self) -> str:
        """Return a string representing the normalization expression to
        be processed by Google Earth Engine API while loading Sentinel2 data."""
        return f"""max(0, min(255, 255 * (b("{self.name}") - {self.percentile1}) / {self.percentile99 - self.percentile1}))"""

    def denormalize(self, band: np.array) -> np.ndarray:
        """Apply reverse normalization to the band in 0-255 scale"""
        band = np.array(band).astype(np.int32)
        return band * (self.percentile99 - self.percentile1) + self.percentile1


# Define the bands and their percentiles
BAND_NAMES = [
    "B1",  # Coastal aerosol
    "B2",  # Blue
    "B3",  # Green
    "B4",  # Red
    "B5",  # Vegetation red edge
    "B6",  # Vegetation red edge
    "B7",  # Vegetation red edge
    "B8",  # NIR
    "B8A",  # Vegetation red edge
    "B11",  # SWIR 1
    "B12",  # SWIR 2
]

BANDS_PERCENTILES_1 = [0, 0, 0, 0, 200, 700, 900, 800, 1000, 300, 100]
BANDS_PERCENTILES_99 = [
    1200,
    1400,
    1600,
    1600,
    2000,
    3600,
    4400,
    4500,
    4700,
    3500,
    2700,
]

# Create variables for each band
BANDS = []

for i, band_name in enumerate(BAND_NAMES):
    BANDS.append(
        Sentinel2Band(band_name, BANDS_PERCENTILES_1[i], BANDS_PERCENTILES_99[i])
    )

if len(BANDS) != NB_S2_BANDS:
    raise ValueError(
        f"Number of bands {len(BANDS)} is different from the number of bands expected {NB_S2_BANDS}, please check the configuration."
    )
