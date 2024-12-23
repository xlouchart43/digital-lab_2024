from logging import Logger

from ee import Geometry as eeGeometry
from shapely import geometry as shapelyGeometry
from shapely import get_coordinates
from tqdm import tqdm

from ..config.constants import TILE_OVERLAP_KM, TILE_SIZE_KM
from ..logger_utils import write_message
from .file_operations import load_shp


def format_ee_geometry(
    geometry: shapelyGeometry, custom_buffer: int = None, logger: Logger = None
) -> eeGeometry:
    """
    Format geometry to match Earth Engine formating.
    You can add a buffer to your geometry by specifying the custom_buffer parameter.
    To get a whole tile from a point, the buffer should be set to 1000 * (TILE_SIZE_KM - TILE_OVERLAP_KM) / 2.
    """
    coordinates = get_coordinates(geometry).tolist()
    if type(geometry) == shapelyGeometry.polygon.Polygon:
        if custom_buffer:
            return eeGeometry.Polygon(coordinates).buffer(custom_buffer)
        return eeGeometry.Polygon(coordinates)
    elif type(geometry) == shapelyGeometry.point.Point:
        coordinates = coordinates[0]
        if custom_buffer:
            return eeGeometry.Point(coordinates).buffer(custom_buffer).bounds()
        return (
            eeGeometry.Point(coordinates)
            .buffer(1000 * (TILE_SIZE_KM - TILE_OVERLAP_KM) / 2)
            .bounds()
        )

    else:
        error_message = f"Unsupported geometry type: {geometry.type}"
        write_message(error_message, logger, "error")
        raise Exception(error_message)


def extract_aoi_from_shp(
    shp_file: str, custom_buffer: int = None, logger: Logger = None
) -> list[tuple[str, eeGeometry]]:
    """Format the geometry of the shapefile."""
    try:
        gdf = load_shp(shp_file, logger)
        aoi_list = []
        for i, row in tqdm(enumerate(gdf.itertuples()), "Formatting geometry"):
            ee_geometry = format_ee_geometry(
                row.geometry, custom_buffer=custom_buffer, logger=logger
            )
            if "Name" in gdf.columns:
                aoi_list.append((row.Name, ee_geometry))
            else:
                aoi_list.append((f"aoi_{i}", ee_geometry))
        write_message(
            f"Formatted geometry for area of interest sucessfully", logger, "success"
        )
        return aoi_list
    except Exception as e:
        write_message(
            f"An error occurred while formatting the geometry: {e}", logger, "error"
        )
        return None
