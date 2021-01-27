import xarray as xr
import xesmf as xe
import numpy as np

import pathlib
import os
import logging

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

        xds_mod = xds_mod.drop(["crs"])
        xds_mod = xds_mod.reindex_like(xds_obs)

        self.regridder = xe.Regridder(xds_mod, xds_obs, 'bilinear')

        # Regrid model data and set x and y accordingly
        xds_mod = self.regridder(xds_mod)
        xds_mod["x"] = xds_obs.x
        xds_mod["y"] = xds_obs.y

        self.xds_obs = xds_obs
        self.xds_mod = xds_mod

        logging.info('New downscaler object initialized')


    @classmethod
    def from_filepaths(cls, obs_filepath, mod_filepath, **kwargs):
        """
        Create Downscaler from filepaths

        Parameters
        ----------
        obs_filepath: string
            filepath to observation data
        mod_filepath: string
            filepath to model data
        """
        try:
            xds_obs = xr.open_mfdataset(
                obs_filepath,
                parallel=True,
                decode_cf=True
            )

            xds_mod = xr.open_mfdataset(
                mod_filepath,
                parallel=True,
                decode_cf=True
            )

            downscaler = cls(xds_obs, xds_mod)

            return downscaler
        except Exception as e:
            print(e.args)
            raise


    def downscale_era5(self, sce_filepath, storage_path):
        xds_sce = xr.open_dataset(
            sce_filepath,
            engine="cfgrib"
        )
        xds_sce = self.prepare_era5(xds_sce)
        self.downscale(xds_sce, storage_path)

    def downscale(self, xds_sce, tmp_storage_path):
        # Regrid scenario data and set x and y accordingly
        xds_sce = self.regridder(xds_sce)
        xds_sce["x"] = self.xds_obs.x
        xds_sce["y"] = self.xds_obs.y

        xds_sce = self.assign_doy_coord(xds_sce)

        xds_mod = self.assign_doy_coord(self.xds_mod)
        xds_mod = xds_mod.rolling(time=30, center=True).construct("window")

        xds_obs = self.assign_doy_coord(self.xds_obs)
        xds_obs = xds_obs.rolling(time=30, center=True).construct("window")

        # Create storage directory if not existing
        pathlib.Path(tmp_storage_path).mkdir(parents=True, exist_ok=True)

        # Loop over days of year in xds_sce
        for idx, sce in xds_sce.groupby("grouping_zip"):
            # Extract days of year
            mod = xds_mod.isel(time=[group == idx for group in xds_mod.grouping_zip.data])
            obs = xds_obs.isel(time=[group == idx for group in xds_obs.grouping_zip.data])

            # Leap day: use 28th February from no leap years
            if idx == (2, 29):
                mod_noyear = xds_mod.isel(time=[group == (2,28) for group in xds_mod.grouping_zip.data])
                mod_noyear = mod_noyear.sel(time=~mod_noyear.time.dt.year.isin(mod.time.dt.year))
                mod = xr.concat([mod, mod_noyear], "time")
                obs_noyear = xds_obs.isel(time=[group == (2,28) for group in xds_obs.grouping_zip.data])
                obs_noyear = obs_noyear.sel(time=~obs_noyear.time.dt.year.isin(obs.time.dt.year))
                obs = xr.concat([obs, obs_noyear], "time")

            # Stack temporal and spatial dimensions
            mod = mod.stack(windowed_time=["time","window"], spatial_dim=["x","y"])
            obs = obs.stack(windowed_time=["time","window"], spatial_dim=["x","y"])
            sce = sce.stack(spatial_dim=["x","y"])
            # Rechunk
            mod = mod.chunk({"windowed_time":-1,"spatial_dim":966})
            obs = obs.chunk({"windowed_time":-1,"spatial_dim":966})
            sce = sce.chunk({"time":-1,"spatial_dim":966})

            var_key = list(sce.keys())[0]

            # Apply vectorized quantile_mapping method
            xds_qm = xr.apply_ufunc(
                self.quantile_mapping,
                mod,
                obs,
                sce,
                input_core_dims=[["windowed_time"], ["windowed_time"], ["time"]],
                output_core_dims=[["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[sce[var_key].dtype]
            )
            # Unstack and transpose dims
            xds_qm = xds_qm.unstack().transpose("time","y","x")
            # Variable attributes from uerra dataset
            for k,v in xds_qm.variables.items():
                if k in self.obs_var_attrs:
                    v.attrs.update(self.obs_var_attrs[k])
            # Projection from uerra dataset
            xds_qm["Lambert_Conformal"] = self.obs_proj

            #xds_qm = xds_qm.sel(
            #    x=slice(*eusalp_bounds[0::2]),
            #    y=slice(*eusalp_bounds[1::2])
            #)

            # Encoding
            # xds_qm.tp.encoding["_FillValue"] = None
            # Same time units in all files
            xds_qm.time.encoding["units"] = "days since 1979-01-01 00:00:00"
            xds_qm.time.encoding["dtype"] = "double"

            xds_qm.attrs.update({
                "title":"Quantile Mapped UERRA - ERA5",
                "institution":"Zentralanstalt fuer Meteorologie und Geodynamik",
                "source":"UERRA - ERA5",
                "comment":"Bilinear interpolated ERA5 data, empirically bias corrected with quantile mapping and UERRA (1979/01-2018/12)",
                "Conventions":"CF-1.7"
            })


            # Write downscaled data for doy to disk
            xds_qm.to_netcdf(f"{tmp_storage_path}/{idx[0]:02d}_{idx[1]:02d}_qm_era5.nc")

    @staticmethod
    def prepare_era5(xds):
        """
        TODO
        Parameters
        ----------
        cds_product : string
            the cds product string

        """
        xr.set_options(keep_attrs=True)

        if len(list(xds)) != 1:
            raise("Only one variable allowed for downscaling")

        var_key = list(xds)[0]

        xds = xds.stack({"time_new":xds.valid_time.dims})
        xds["time_new"] = xds.valid_time
        xds = xds.rename({"latitude":"lat", "longitude":"lon", "time_new":"time"})

        if "tp" in var_key:
            # Adapt units
            xds[var_key] = xds[var_key]*1000
            xds[var_key].attrs["units"] = "kg m**-2"

            # Define list of indices with full days
            # Special Case: Timestamp 06:00, accumulation of preceding 24h
            idx_full_time = xds.time.resample(time="24H", base=7, loffset="-1H").count().idxmax("time")
            xds = xds.resample(time="24H", base=7, loffset="-1H").sum()

        elif "ssr" in var_key:
            # Define list of indices with full days
            idx_full_time = xds.time.resample(time="24H").count().idxmax("time")
            xds = xds.resample(time="24H").sum()

        else:
            idx_full_time = xds.time.resample(time="24H").count().idxmax("time")
            xds = xds.resample(time="24H").mean()

        # Select only full days
        xds = xds.sel(time=idx_full_time)
        # Invert latitude coordinate
        xds = xds.reindex(lat=xds.lat[::-1])

        return xds


    @staticmethod
    def assign_doy_coord(ds):
        """
        Assing new coord 'grouping_zip' for days of year (doy)
        doy are generated via grouping by month and day
        """
        # Expand time to dim if not present
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")

        ds = ds.assign_coords(month=ds.time.dt.month, day=ds.time.dt.day)
        fields = ["month","day"]

        common_dim = ds.coords[fields[0]].dims[0]
        tups_arr = np.empty(len(ds[common_dim]), dtype=object)
        tups_arr[:] = list(zip(*(ds[f].values for f in fields)))

        return ds.assign_coords(grouping_zip=xr.DataArray(tups_arr, dims=common_dim))


    @staticmethod
    def quantile_mapping(mod, obs, downscale, *args, **kwargs):
        """
        Quantile Mapping using empirical cumulative distribution function
        """
        mod_ecdf = ECDF(mod)
        p = mod_ecdf(downscale) * 100
        corr = np.percentile(obs[~np.isnan(obs)], p) - \
               np.percentile(mod[~np.isnan(mod)], p)
        return downscale + corr
