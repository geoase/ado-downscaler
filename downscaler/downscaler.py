import xarray as xr
import xesmf as xe
import numpy as np
import os

import geopandas as gp
import pyproj

# import multiprocessing.popen_spawn_posix
import dask

from statsmodels.distributions.empirical_distribution import ECDF

# TODO: leap day

class Downscaler(object):
    """The :class:`Downscaler` class provides ...
    """

    def __init__(self, xds_obs, xds_mod, **kwargs):
        """
        Parameters
        ----------
        cds_product : string
            the cds product string
        cds_filter : dict
            the cds filter dictionary

        """
        # UERRA
        # Extract projection variable
        self.obs_proj = xds_obs["Lambert_Conformal"]
        xds_obs = xds_obs.drop(["Lambert_Conformal"])

        # Extract variable attributes
        self.obs_var_attrs = {k:v.attrs for k,v in xds_obs.variables.items()}

        xds_obs = xds_obs.sel(time=slice("1979-01-01","2018-12-31"))

        #xds_mod = xds_mod.drop(["number","surface"])
        xds_mod = xds_mod.reindex_like(xds_obs)

        self.regridder = xe.Regridder(xds_mod, xds_obs, 'bilinear')

        # Regrid model data and set x and y accordingly
        xds_mod = self.regridder(xds_mod)
        xds_mod["x"] = xds_obs.x
        xds_mod["y"] = xds_obs.y

        self.xds_obs = xds_obs
        self.xds_mod = xds_mod

        logging.info('New downscaler object initialized')


    def downscale(self, xds_sce):
        # Regrid scenario data and set x and y accordingly
        xds_sce = self.regridder(xds_sce)
        xds_sce["x"] = self.xds_obs.x
        xds_sce["y"] = self.xds_obs.y

        xds_sce = assign_doy_coord(xds_sce)

        xds_mod = assign_doy_coord(self.xds_mod)
        xds_mod = xds_mod.rolling(time=30, center=True).construct("window")

        xds_obs = assign_doy_coord(self.xds_obs)
        xds_obs = xds_obs.rolling(time=30, center=True).construct("window")

        # Loop over days of year in xds_sce
        for idx, sce in xds_sce.groupby("grouping_zip"):
            # Extract days of year, stack temporal and spatial dimensions and rechunk
            mod = xds_mod.isel(time=[group == idx for group in xds_mod.grouping_zip.data])
            mod = mod.stack(windowed_time=["time","window"], spatial_dim=["x","y"])
            mod = mod.chunk({"windowed_time":-1,"spatial_dim":966})

            # Extract days of year, stack temporal and spatial dimensions and rechunk
            obs = xds_obs.isel(time=[group == idx for group in xds_obs.grouping_zip.data])
            obs = obs.stack(windowed_time=["time","window"], spatial_dim=["x","y"])
            obs = obs.chunk({"windowed_time":-1,"spatial_dim":966})

            sce = sce.stack(spatial_dim=["x","y"])
            sce = sce.chunk({"time":-1,"spatial_dim":966})

            # Apply vectorized quantile_mapping method
            xds_qm_tp = xr.apply_ufunc(
                quantile_mapping,
                mod,
                obs,
                sce,
                input_core_dims=[["windowed_time"], ["windowed_time"], ["time"]],
                output_core_dims=[["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[sce.tp.dtype]
            )
            # Unstack and transpose dims
            xds_qm_tp = xds_qm_tp.unstack().transpose("time","y","x")
            # Variable attributes from uerra dataset
            for k,v in xds_qm_tp.variables.items():
                v.attrs.update(self.obs_var_attrs[k])
            # Projection from uerra dataset
            xds_qm_tp["Lambert_Conformal"] = self.obs_proj

            xds_qm_tp = xds_qm_tp.sel(
                x=slice(*eusalp_bounds[0::2]),
                y=slice(*eusalp_bounds[1::2])
            )

            # Same time units in all files
            xds_qm_tp.time.encoding["units"] = "days since 1979-01-01 00:00:00"

            # Demonstration only stop after one iteration
            xds_qm_tp = xds_qm_tp.compute(); break;

            # Write downscaled data for doy to disk
            xds_qm_tp.to_netcdf(f"../data/qm_test/{idx[0]:02d}_{idx[1]:02d}_qm_era5.nc")


def assign_doy_coord(ds):
    """
    Assing new coord 'grouping_zip' for days of year (doy)
    doy are generated via grouping by month and day
    """
    ds = ds.assign_coords(month=ds.time.dt.month, day=ds.time.dt.day)
    ds = _assign_multicoords(ds, ["month","day"])
    return ds

def _assign_multicoords(ds, fields):
    """
    Assing combined coord 'grouping_zip' for list of fields
    Adapted from:
    stackoverflow.com/questions/60380993/groupby-multiple-coords-along-a-single-dimension-in-xarray
    """
    common_dim = ds.coords[fields[0]].dims[0]
    tups_arr = np.empty(len(ds[common_dim]), dtype=object)
    tups_arr[:] = list(zip(*(ds[f].values for f in fields)))
    return ds.assign_coords(grouping_zip=xr.DataArray(tups_arr, dims=common_dim))


def quantile_mapping(mod, obs, downscale, *args, **kwargs):
    """
    Quantile Mapping using empirical cumulative distribution function
    """
    mod_ecdf = ECDF(mod)
    p = mod_ecdf(downscale) * 100
    corr = np.percentile(obs[~np.isnan(obs)], p) - \
           np.percentile(mod[~np.isnan(mod)], p)
    return downscale + corr
