"""
Make combined file with a grid of GOES data around each CALIPSO point.
Use to train CNN
"""

import logging


import numpy as np
import xarray as xr


from erebos import __version__
from .base import _convert_attrs


logger = logging.getLogger(__name__)


def make_combined_dataset(calipso_file, goes_file, calipso_vars, buffer_=25):
    """
    buffer_ : int
        Number of points on either side of central CALIPSO location
    """
    calipso_ds = xr.open_dataset(calipso_file, engine="h5netcdf")
    goes_ds = xr.open_dataset(goes_file, engine="h5netcdf")
    lats = calipso_ds.erebos.Latitude[:, 0]
    lons = calipso_ds.erebos.Longitude[:, 0]
    ix, iy = goes_ds.erebos.find_nearest_xy(lons, lats)
    pts_overlap_edge = (
        (iy - buffer_ <= 0)
        | (iy + buffer_ >= goes_ds.dims["y"])
        | (ix - buffer_ <= 0)
        | (ix + buffer_ >= goes_ds.dims["x"])
    )
    buffer_range = np.arange(int(-buffer_), int(buffer_) + 1, 1)
    xs = xr.DataArray(
        np.atleast_2d(ix[~pts_overlap_edge]).T + buffer_range, dims=["rec", "gx"]
    )
    ys = xr.DataArray(
        np.atleast_2d(iy[~pts_overlap_edge]).T + buffer_range, dims=["rec", "gy"]
    )
    limited_goes = goes_ds.erebos.isel(x=xs, y=ys)
    limited_calipso = calipso_ds.erebos.sel(record=~pts_overlap_edge)

    vars_ = {}
    for v in calipso_vars:
        var = limited_calipso.variables[v]
        # only the selected level of calipso file is kept
        # may want to average levels or something in future
        vals = var[:, 0].values.astype("float32")
        da = xr.DataArray(vals, dims=("rec"), attrs=_convert_attrs(var.attrs))
        da.encoding = {"zlib": True, "complevel": 1, "shuffle": True}
        vars_[v] = da

    for name, var in limited_goes.variables.items():
        if name.startswith("DQF") or "gx" not in var.dims or "gy" not in var.dims:
            continue
        cmi = var.values
        # anything != 0 is considered questionable quality
        # not quite true for COD, but ok for now
        dqf = limited_goes.variables[f"DQF_{name}"].values.astype(bool)
        da = xr.DataArray(np.ma.array(cmi, mask=dqf), dims=var.dims, attrs=var.attrs)
        da.encoding = var.encoding
        vars_[name] = da

    vars_["goes_imager_projection"] = goes_ds.goes_imager_projection
    coords = {"x": limited_goes.x, "y": limited_goes.y}
    out = xr.Dataset(
        vars_,
        coords=coords,
        attrs={
            "goes_time": str(goes_ds.erebos.t.values),
            "goes_file": str(goes_file.name),
            "calipso_file": str(calipso_file.name),
            "erebos_version": __version__,
        },
    )
    for k in out.coords.keys():
        if k.startswith("erebos_"):
            del out[k]
    calipso_ds.close()
    goes_ds.close()
    return out
