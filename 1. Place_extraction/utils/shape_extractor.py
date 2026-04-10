import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


cm = 1/2.54
palette = ["#3F3517",  '#CE2E31', '#C96F6B', '#CCA464', '#F8D768', '#F0DCD4']



def to_metric(gdf: gpd.GeoDataFrame, crs: int | str = 3577) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to a metric CRS (default: EPSG:3577 Australian Albers).
    Returns a *new* GeoDataFrame.
    """
    return gdf.to_crs(crs)


def all_vertices(geom: Polygon | MultiPolygon) -> List[Tuple[float, float]]:
    """
    Return all vertices (outer + inner rings) of a Polygon/MultiPolygon as (x, y) tuples.
    """
    if geom.is_empty:
        return []

    def _poly_vertices(p: Polygon) -> List[Tuple[float, float]]:
        vs = list(p.exterior.coords)
        for ring in p.interiors:
            vs.extend(list(ring.coords))
        return vs

    if geom.geom_type == "Polygon":
        return _poly_vertices(geom)
    if geom.geom_type == "MultiPolygon":
        out: List[Tuple[float, float]] = []
        for p in geom.geoms:
            out.extend(_poly_vertices(p))
        return out

    return []



def furthest_vertex_from_centroid(
    geom: Polygon | MultiPolygon
) -> Tuple[float, Optional[Point], Point]:
    """
    Compute the distance from centroid to the furthest vertex.
    Returns max_distance.
    Assumes geometry is in a metric CRS (units = meters).
    """
    if geom is None or geom.is_empty:
        return np.nan

    c = geom.centroid
    verts = all_vertices(geom)
    if not verts:
        return 0.0

    dists = [Point(x, y).distance(c) for x, y in verts]
    i = int(np.argmax(dists))
    return dists[i]



def final_gdf_creator(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    gdf['centroid'] = gdf.geometry.centroid

    gdf_m = to_metric(gdf, crs=3577)
    gdf_m["furthest_dist_m"] = gdf_m.geometry.apply(furthest_vertex_from_centroid)
    gdf["furthest_dist_m"] = gdf_m["furthest_dist_m"]

    return gdf, gdf_m