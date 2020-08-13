from concurrent.futures import ProcessPoolExecutor
import logging


import numpy as np
import pandas as pd
import xarray as xr


from erebos import __version__
from erebos.adapters.goes import GOESFilename


from . import pointwise
from . import cnn


logger = logging.getLogger(__name__)


def match_goes_file(calipso_file, goes_files, max_diff="6min"):
    gt = np.asarray([f.start for f in goes_files])
    with xr.open_dataset(calipso_file, engine="h5netcdf") as cds:
        ctime = pd.Timestamp(cds.erebos.time.data.min(), tz="UTC")
    diff = np.abs(gt - np.array(ctime))
    if diff.min() < pd.Timedelta(max_diff):
        gfile = goes_files[np.argmin(diff)]
        return gfile.start, gfile.filename
    else:
        return None, None


def combine_calipso_goes_files(
    calipso_dir,
    goes_dir,
    save_dir,
    goes_glob,
    calipso_glob,
    limits=(0, None),
    fnc="pointwise",
):
    calipso_files = list(calipso_dir.glob(calipso_glob))[slice(*limits)]
    goes_files = [
        GOESFilename(f, start=pd.Timestamp(f.stem.split("_")[-1], tz="UTC"))
        for f in goes_dir.glob(goes_glob)
    ]
    for cfile in calipso_files:
        logger.info("Processing %s", cfile)
        gtime, gfile = match_goes_file(cfile, goes_files)
        if gfile is None:
            logger.warning("No matching GOES file for %s", cfile)
            continue

        filename = (
            save_dir / f'colocated_calipso_goes_{gtime.strftime("%Y%m%dT%H%M%SZ")}.nc'
        )

        if filename.exists():
            logger.info("File already exists at %s", filename)
            continue

        mean_vars = [
            "cloud_top_altitude",
            "cloud_thickness",
            "cloud_base_altitude",
            "cloud_layers",
            "solar_azimuth",
            "solar_zenith",
        ]
        first_vars = ["cloud_type", "day_night_flag", "surface_elevation"]

        if fnc == "pointwise":
            ds = pointwise.make_combined_dataset(cfile, gfile, mean_vars, first_vars)
        else:
            try:
                ds = cnn.make_combined_dataset(
                    cfile, gfile, mean_vars + first_vars, size=100
                )
            except ValueError as e:
                logger.error("Failed to combine %s, %s: %s", cfile, gfile, str(e))
                continue
        logger.info("Saving file to %s", filename)
        ds.to_netcdf(filename, engine="h5netcdf")
        ds.close()


def dataset_properties(filename):
    with xr.open_dataset(filename, engine="h5netcdf") as ds:
        nanvars = [f"CMI_C{i:02d}" for i in range(1, 17)] + [
            "solar_zenith",
            "solar_azimuth",
            "x",
            "y",
            "goes_time",
        ]
        nanrec = (
            ds[nanvars].isnull().any(dim=("gy", "gx")).to_array().any(dim="variable")
        )

        nans = nanrec.values.nonzero()[0]
        totalrecs = ds.dims["rec"]
        availrecs = totalrecs - len(nans)
        ftime = pd.Timestamp(ds.goes_time.isel(rec=0).item(), tz="UTC")
    return str(filename), totalrecs, availrecs, nans, ftime


def load_combined_files(combined_dir, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as exc:
        futs = exc.map(dataset_properties, combined_dir.glob("*.nc"))
    df = pd.DataFrame(
        futs,
        columns=[
            "filename",
            "total_records",
            "not_nan_records",
            "nan_locs",
            "file_time",
        ],
    )
    return df.set_index("filename").sort_index().reset_index()


def filter_times(combined_df, daytime_only):
    if not daytime_only:
        return combined_df
    else:
        return (
            combined_df.reset_index()
            .set_index("file_time", drop=False)
            .tz_convert("America/Phoenix")
            .between_time("10:00", "15:00")
            .set_index("index")
        )


def split_data(combined_df, train_pct, test_pct, seed):
    len_ = len(combined_df)
    train, test, val = np.split(
        combined_df.sample(frac=1, random_state=seed),
        [int(train_pct / 100 * len_), int((train_pct + test_pct) / 100 * len_)],
    )
    tot = combined_df.not_nan_records.sum()
    logger.info(
        "Split with %0.1f%% in training, %0.1f%% in test, %0.1f%% in validation",
        train.not_nan_records.sum() / tot * 100,
        test.not_nan_records.sum() / tot * 100,
        val.not_nan_records.sum() / tot * 100,
    )
    return train, test, val


def concat_datasets(datasets, outpath):
    logger.info("Saving data to %s", outpath)
    first = True
    xr.set_options(file_cache_maxsize=10)
    tmppath = outpath.with_suffix(".nc")
    for dataset in datasets:
        logger.debug("Processing %s", dataset)
        ds = xr.open_dataset(dataset, engine="h5netcdf")
        nanvars = [f"CMI_C{i:02d}" for i in range(1, 17)] + [
            "solar_zenith",
            "solar_azimuth",
            "x",
            "y",
            "goes_time",
        ]
        nanrec = (
            ds[nanvars].isnull().any(dim=("gy", "gx")).to_array().any(dim="variable")
        )
        ds = ds.sel(rec=~nanrec).chunk(dict(rec=500))
        if first:
            nds = ds
        else:
            logger.debug("Concatenating...")
            nds = xr.open_dataset(tmppath, engine="h5netcdf")
            nds = xr.concat(
                [nds, ds],
                dim="rec",
                join="left",
                combine_attrs="drop",
                compat="override",
                coords="minimal",
            )
        nds.to_netcdf(tmppath, engine="h5netcdf", mode="w")
    nds.attrs = {
        "erebos_version": __version__,
        "combined_calipso_files": list(datasets),
    }
    nds.to_zarr(outpath, mode="w")
