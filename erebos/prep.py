import logging


import cartopy.crs as ccrs
from numba import types, generated_jit
import numpy as np
import pandas as pd
import xarray as xr


from erebos import utils, __version__
from erebos.adapters.goes import GOESFilename


def match_goes_file(calipso_file, goes_files, max_diff="6min"):
    gt = np.asarray([f.start for f in goes_files])
    with xr.open_dataset(calipso_file, engine="pynio") as cds:
        ctime = pd.Timestamp(cds.erebos.time.data.min(), tz="UTC")
    diff = np.abs(gt - np.array(ctime))
    if diff.min() < pd.Timedelta(max_diff):
        return goes_files[np.argmin(diff)].filename
    else:
        return None


def translate_calipso_locations_to_apparent_position(
        calipso_ds, goes_ds, fill_na=True, level=0):
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
    goes_cloud_pos = goes_ds.erebos.crs.transform_points(ccrs.Geodetic(), alon, alat)
    return goes_cloud_pos[:, :2]


def translate_calipso_locations(calipso_ds, goes_ds):
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
    goes_cloud_pos = goes_ds.erebos.crs.transform_points(ccrs.Geodetic(), alon, alat)
    return goes_cloud_pos[:, :2]


def _mapit(vals, index, first):
    ovals = []
    inds = []
    cts = []
    for i, val in enumerate(vals):
        ind = index[i]
        if ind not in inds:
            inds.append(ind)
            ovals.append(val)
            cts.append(1.0)
        else:
            # basically do a nanmean
            idx = inds.index(ind)
            if np.isnan(val):
                continue
            elif np.isnan(ovals[idx]):
                ovals[idx] = val
                cts[idx] = 1.0
            elif not first:
                ovals[idx] += val
                cts[idx] += 1.0
    out = np.zeros(len(ovals), vals.dtype)
    for i in range(len(ovals)):
        out[i] = ovals[i] / cts[i]
    ind_out = np.array(inds, index.dtype)
    j = np.argsort(ind_out)
    return out[j], ind_out[j]


@generated_jit(nopython=True)
def map_values_to_index_num(vals, index, first):
    if isinstance(vals, types.Array):
        return _mapit
    else:
        raise TypeError("vals must be a numpy array")


def calipso_indices(calipso_ds, goes_ds, level=0, k=1, dub=1.3e3):
    cloud_pts = translate_calipso_locations(calipso_ds, goes_ds, level=level)
    dist, inds = goes_ds.erebos.kdtree.query(
        cloud_pts.astype("float32"), k=k, distance_upper_bound=dub
    )
    return inds


def make_combined_dataset(
    calipso_file, goes_file, calipso_mean_vars, calipso_first_vars, level=0
):
    calipso_ds = xr.open_dataset(calipso_file, engine="pynio")
    goes_ds = xr.open_dataset(goes_file, engine="netcdf4")
    # include the nearest 4 points in the goes file
    inds = calipso_indices(calipso_ds, goes_ds, k=4, dub=2e3)
    over_ind = goes_ds.dims["x"] * goes_ds.dims["y"]
    _, uniq = np.unique(inds[:, 0], return_index=True)
    do_not_include = np.logical_or.reduce(inds[uniq] == over_ind, axis=1)
    # basically, do not include points within 3 km of border
    adj_inds = inds[uniq][~do_not_include]
    vars_ = {}
    for v in calipso_mean_vars:
        var = calipso_ds.erebos.variables[v]
        vals = var[:, level].values.astype("float32")
        avg, ninds = map_values_to_index_num(vals, inds[:, 0], False)
        assert (ninds == inds[uniq, 0]).all()
        da = xr.DataArray(avg[~do_not_include], dims=("rec"), attrs=var.attrs)
        da.encoding = {"zlib": True, "complevel": 1, "shuffle": True}
        vars_[v] = da

    for v in calipso_first_vars:
        var = calipso_ds.erebos.variables[v]
        vals = var[:, 0].values
        avg, ninds = map_values_to_index_num(vals, inds[:, 0], True)
        assert (ninds == inds[uniq, 0]).all()
        da = xr.DataArray(avg[~do_not_include], dims=("rec"), attrs=var.attrs)
        da.encoding = {"zlib": True, "complevel": 1, "shuffle": True}
        vars_[v] = da

    for name, var in goes_ds.erebos.variables.items():
        if name.startswith("DQF") or "x" not in var.dims or "y" not in var.dims:
            continue
        cmi = var.values.reshape(-1)[adj_inds]
        # anything != 0 is considered questionable quality
        # not quite true for COD, but ok for now
        dqf = (
            goes_ds.erebos.variables[f"DQF_{name}"]
            .values.reshape(-1)[adj_inds]
            .astype(bool)
        )
        da = xr.DataArray(
            np.ma.array(cmi, mask=dqf), dims=("rec", "near"), attrs=var.attrs
        )
        da.encoding = var.encoding
        vars_[name] = da
    vars_["goes_imager_projection"] = goes_ds.goes_imager_projection
    iy, ix = np.unravel_index(adj_inds, (goes_ds.dims["y"], goes_ds.dims["x"]))
    coords = {
        "x": (("rec", "near"), goes_ds.erebos.x[ix.ravel()].values.reshape(ix.shape)),
        "y": (("rec", "near"), goes_ds.erebos.y[iy.ravel()].values.reshape(iy.shape)),
    }
    out = xr.Dataset(
        vars_,
        coords=coords,
        attrs={
            "goes_time": str(goes_ds.erebos.t.values),
            "goes_file": str(goes_file),
            "calipso_file": str(calipso_file),
            "erebos_version": __version__,
        },
    )
    calipso_ds.close()
    goes_ds.close()
    return out


def combine_calipso_goes_files(
    calipso_dir, goes_dir, save_dir, goes_glob, calipso_glob, limits=(0, None)
):
    calipso_files = list(calipso_dir.glob(calipso_glob))[slice(*limits)]
    goes_files = [
        GOESFilename(f, start=pd.Timestamp(f.name.split("_")[0], tz="UTC"))
        for f in goes_dir.glob(goes_glob)
    ]
    for cfile in calipso_files:
        logging.info("Processing %s", cfile)
        gfile = match_goes_file(cfile, goes_files)
        if gfile is None:
            logging.warning("No matching GOES file for %s", cfile)
            continue

        filename = save_dir / gfile.name

        if filename.exists():
            logging.info("File already exists at %s", filename)
            continue

        ds = make_combined_dataset(
            cfile,
            gfile,
            [
                "cloud_top_altitude",
                "cloud_thickness",
                "cloud_base_altitude",
                "cloud_layers",
                "solar_azimuth",
                "solar_zenith",
            ],
            ["cloud_type", "day_night_flag", "surface_elevation"],
        )
        logging.info("Saving file to %s", filename)
        ds.to_netcdf(filename, engine="netcdf4")
        ds.close()
