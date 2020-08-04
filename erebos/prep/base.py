import cartopy.crs as ccrs
import numpy as np


def translate_calipso_locations(calipso_ds, goes_ds):
    """Tranform the CALIPSO latitude/longitude values into the 
    GOES Geostationary projection"""
    goes_cloud_pos = goes_ds.erebos.crs.transform_points(
        ccrs.Geodetic(),
        calipso_ds.erebos.Longitude[:, 0].values,
        calipso_ds.erebos.Latitude[:, 0].values,
    )
    return goes_cloud_pos[:, :2]


def _convert_attrs(attrs):
    # h5netcdf/h5py does not like string arrays for attrs
    out = attrs.copy()
    for k, v in attrs.items():
        if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "U"):
            out[k] = list(v)
    return out
