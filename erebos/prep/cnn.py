"""
Make combined file with a grid of GOES data around each CALIPSO point.
Use to train CNN
"""

import logging


import numpy as np
import xarray as xr


from erebos import __version__, utils
from .base import _convert_attrs


logger = logging.getLogger(__name__)


def translate_calipso_locations_to_apparent_position(
    calipso_ds, goes_ds, fill_na=True, level=0
):
    # nan means no cloud
    cloud_heights = calipso_ds.erebos.cloud_top_altitude[:, level].values
    if fill_na:
        cloud_heights = np.ma.fix_invalid(cloud_heights).filled(0)
    terrain_height = calipso_ds.erebos.surface_elevation[:, 0].values
    cloud_locations = utils.RotatedECRPosition.from_geodetic(
        calipso_ds.erebos.Latitude[:, 0].values,
        calipso_ds.erebos.Longitude[:, 0].values,
        0.0,
    )
    actual_cloud_pos = utils.find_actual_cloud_position(
        calipso_ds.erebos.spacecraft_location, cloud_locations, cloud_heights
    )
    apparent_cloud_pos = utils.find_apparent_cloud_position(
        goes_ds.erebos.spacecraft_location, actual_cloud_pos, terrain_height
    )
    alat, alon = apparent_cloud_pos.to_geodetic()
    return alat, alon


def make_combined_dataset(
    calipso_file, goes_file, calipso_vars, size, seed=8927, label_jitter=32
):
    calipso_ds = xr.open_dataset(calipso_file, engine="h5netcdf")
    goes_ds = xr.open_dataset(goes_file, engine="h5netcdf")
    lats = calipso_ds.erebos.Latitude[:, 0]
    lons = calipso_ds.erebos.Longitude[:, 0]
    ix, iy = goes_ds.erebos.find_nearest_xy(lons, lats)
    alat, alon = translate_calipso_locations_to_apparent_position(calipso_ds, goes_ds)
    nx, ny = goes_ds.erebos.find_nearest_xy(alon, alat)
    buffer_ = size // 2
    rng = np.random.default_rng(seed)
    rnd_pos = rng.integers(-label_jitter, label_jitter, [2, len(ix)])
    xpos = ix + rnd_pos[0]
    ypos = iy + rnd_pos[1]
    pts_overlap_edge = (
        (ypos - buffer_ <= 0)
        | (ypos + buffer_ >= goes_ds.dims["y"])
        | (xpos - buffer_ <= 0)
        | (xpos + buffer_ >= goes_ds.dims["x"])
    )
    dx = (nx - ix)[~pts_overlap_edge]
    dy = (ny - iy)[~pts_overlap_edge]
    buffer_range = np.arange(int(-buffer_), int(buffer_), 1)
    xs = xr.DataArray(
        np.atleast_2d(xpos[~pts_overlap_edge]).T + buffer_range, dims=["rec", "gx"]
    )
    ys = xr.DataArray(
        np.atleast_2d(ypos[~pts_overlap_edge]).T + buffer_range, dims=["rec", "gy"]
    )
    mask = xr.DataArray(
        np.zeros((xs.shape[0], ys.shape[1], xs.shape[1], 2), dtype=bool),
        dims=["rec", "gy", "gx", "adjusted"],
    )
    mx = xr.DataArray(buffer_ - rnd_pos[0][~pts_overlap_edge], dims="rec")
    my = xr.DataArray(buffer_ - rnd_pos[1][~pts_overlap_edge], dims="rec")
    mask[dict(gy=my, gx=mx, adjusted=0)] = True
    mask[dict(gy=my + dy, gx=mx + dx, adjusted=1)] = True
    mask.encoding = {"zlib": True, "complevel": 4, "shuffle": True}

    limited_goes = goes_ds.erebos.isel(x=xs, y=ys)
    limited_calipso = calipso_ds.erebos.sel(record=~pts_overlap_edge)

    vars_ = {"label_mask": mask}
    for v in calipso_vars:
        var = limited_calipso.variables[v]
        # only the selected level of calipso file is kept
        # may want to average levels or something in future
        vals = var[:, 0].values.astype("float32")
        da = xr.DataArray(vals, dims=("rec"), attrs=_convert_attrs(var.attrs))
        da.encoding = {"zlib": True, "complevel": 4, "shuffle": True}
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
        da.encoding["complevel"] = 4
        vars_[name] = da

    vars_["goes_imager_projection"] = goes_ds.goes_imager_projection
    xda = limited_goes.x.copy()
    xenc = xda.encoding
    xenc["_FillValue"] = -9999
    xenc["zlib"] = True
    xda.encoding = xenc
    yda = limited_goes.y.copy()
    yenc = yda.encoding
    yenc["_FillValue"] = -9999
    yenc["zlib"] = True
    yda.encoding = yenc

    coords = {"x": xda, "y": yda}
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
