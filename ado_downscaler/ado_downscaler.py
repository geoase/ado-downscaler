import logging
import os
import pathlib
import glob

import xarray as xr
import xesmf as xe
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF


class Downscaler(object):
    """The :class:`Downscaler` class provides ...
    """

    def __init__(self, xds_obs, xds_mod, **kwargs):
        """
        Parameters
        ----------
        xds_obs : :obj:`xarray.Dataset`
            The observation dataset containing the observation history of a single variable
        xds_mod : :obj:`xarray.Dataset`
            The model dataset containing the model history of a single variable
        """
        # UERRA
        # Extract projection variable
        self.obs_proj = xds_obs["Lambert_Conformal"]
        xds_obs = xds_obs.drop_vars(["Lambert_Conformal"])

        # Extract variable attributes
        self.obs_var_attrs = {k:v.attrs for k,v in xds_obs.variables.items()}

        xds_obs = xds_obs.sel(time=slice("1979-01-01","2018-12-31"))
        xds_mod = xds_mod.sel(time=slice("1979-01-01","2018-12-31"))

#        xds_mod = xds_mod.drop_vars(["crs"])
#        xds_mod = xds_mod.reindex_like(xds_obs)

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
        lst_paths = self.downscale(xds_sce, storage_path)
        return lst_paths

    def downscale(self, xds_sce, tmp_storage_path, file_stem=None, spatial_chunk=3864):
        if not file_stem:
            file_stem = pathlib.Path(xds_sce.encoding.get("source")).stem

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

        lst_paths = []
        # Loop over days of year in xds_sce
        for idx, sce in xds_sce.groupby("grouping_zip"):
            # Extract days of year
            mod = xds_mod.isel(time=[group == idx for group in xds_mod.grouping_zip.data])
            obs = xds_obs.isel(time=[group == idx for group in xds_obs.grouping_zip.data])
            # Leap day: use 28th February from no leap years
            if idx == (2, 29):
                mod_noyear = xds_mod.isel(time=[group == (2,28) for group in xds_mod.grouping_zip.data])
                mod_noyear = mod_noyear.sel(time=~mod_noyear.time.dt.year.isin(mod.time.dt.year))
                mod = xr.concat([mod.load(), mod_noyear], "time")
                obs_noyear = xds_obs.isel(time=[group == (2,28) for group in xds_obs.grouping_zip.data])
                obs_noyear = obs_noyear.sel(time=~obs_noyear.time.dt.year.isin(obs.time.dt.year))
                obs = xr.concat([obs.load(), obs_noyear], "time")

            # Stack temporal and spatial dimensions
            mod = mod.stack(windowed_time=["time","window"], spatial_dim=["x","y"])
            obs = obs.stack(windowed_time=["time","window"], spatial_dim=["x","y"])
            sce = sce.stack(spatial_dim=["x","y"])
            # Rechunk
            mod = mod.chunk({"windowed_time":-1,"spatial_dim":spatial_chunk})
            obs = obs.chunk({"windowed_time":-1,"spatial_dim":spatial_chunk})
            sce = sce.chunk({"time":-1,"spatial_dim":spatial_chunk})

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
            xds_qm["Lambert_Conformal"] = None
            xds_qm["Lambert_Conformal"].attrs = self.obs_proj.attrs

            # Encoding
            # Same time units in all files
#            xds_qm.time.encoding["units"] = "days since 1979-01-01 00:00:00"
#            xds_qm.time.encoding["dtype"] = "double"

            xds_qm.attrs.update({
                "title":"Quantile Mapped UERRA - ERA5",
                "institution":"Zentralanstalt fuer Meteorologie und Geodynamik",
                "source":"UERRA - ERA5",
                "comment":"Bilinear interpolated ERA5 data, empirically bias corrected with quantile mapping and UERRA (1979/01-2018/12)",
                "Conventions":"CF-1.7"
            })
            file_path = os.path.join(tmp_storage_path,f"{file_stem}_{idx[0]:02d}_{idx[1]:02d}_qm.nc")
            lst_paths.append(file_path)
            # Write downscaled data for doy to disk
            xds_qm.to_netcdf(file_path)

        return lst_paths


    @staticmethod
    def prepare_era5(xds, only_full_days=True):
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

            # Special Case: Timestamp 06:00, accumulation of preceding 24h
            args_resample = {"time":"24H", "base":7, "loffset":"-1H"}
            func_resample = np.nansum

        elif any(ele in var_key for ele in ["ssr","str"]):
            args_resample={"time":"24H"}
            func_resample = np.nansum
        else:
            args_resample={"time":"24H"}
            func_resample = np.mean
        
        if only_full_days:
            # Define list of indices with full days
            time_res_count = xds.time.resample(**args_resample).count()
            idx_full_time = time_res_count.time.where(time_res_count == time_res_count.max(), drop=True)
    
            xds = xds.resample(**args_resample).reduce(func_resample)
    
            # Select only full days
            xds = xds.sel(time=idx_full_time)
        else:
            xds = xds.dropna(dim="time").resample(**args_resample).reduce(func_resample)

        return xds


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
    def merge_doy_files(input_path, *args, **kwargs):
        """
        Merge and sort day of year files
        """
        file_paths = glob.glob(input_path)
        if len(file_paths) == 0:
            raise("No files found under provided path")

        lst_xds = []
        for p in file_paths:
            lst_xds.append(xr.open_dataset(p, decode_cf=True, chunks=-1))

        xds_qm_sorted = xr.concat(lst_xds, dim="time").sortby("time")

        return xds_qm_sorted

    @staticmethod
    def cut_eusalp_domain(xds):
        eusalp_bounds = (
                2591436.355343933, 2146519.5114292144,
                3628337.31222056, 3012927.872261234
        )

        xds = xds.sel(
            x=slice(*eusalp_bounds[0::2]),
            y=slice(*eusalp_bounds[1::2])
        )

        return xds

    @staticmethod
    def add_metadata(xds):
        """
        Add additional metadata for total precipitation and potential evapotranspiration
        """
        if "tp" in xds:
            xds.attrs = {
                "title":"Quantile Mapped Daily Precipitation sum from ERA5 data (downscaled using UERRA MESCAN-Surfex data)",
                "institution":"Zentralanstalt fuer Meteorologie und Geodynamik",
                "description":"Total precipitation is the amount of precipitation falling onto the ground/water surface. "\
                "It includes all kind of precipitation forms as convective precipitation, large scale precipitation, "\
                "liquid and solid precipitation. The amount is valid for a grid box, whereas values at timestamp represent "\
                "the sum of the preceding 24 hours.",
                "license":"Creative Commons Zero (CC0)",
                "keywords":"PRECIPITATION, UERRA, ADO",
                "providers":"Producer: Météo-France; Processor: ZAMG Austria",
                "links":["https://datastore.copernicus-climate.eu/documents/uerra/D322_Lot1.4.1.2_User_guides_v3.3.pdf"],
                "lineage":"Quantile Mapped ERA5 data is used in order to calculate the daily sum of precipitation.",
                "comment":"Daily Precipitation sum quantile mapped ERA5 data.",
                "source":"ERA5; UERRA MESCAN-Surfex",
                "Conventions":"CF-1.7",
            }

            xds.tp.attrs= { 
                "long_name":"Total Precipitation",
                "units":"kg m**-2",
                "grid_mapping":"Lambert_Conformal",
                "standard_name":"precipitation_amount"
            }

        elif "pet" in xds:
            xds.attrs = {
                "title":"Quantile Mapped Potential Evapotranspiration following Penman-Monteith using UERRA MESCAN-Surfex data",
                "institution":"Zentralanstalt fuer Meteorologie und Geodynamik",
                "description":"The term evapotranspiration (ET) is commonly used to describe two processes "\
                "of water loss from land surface to atmosphere, evaporation and transpiration. Evaporation "\
                "is the process where liquid water is converted to water vapor (vaporization) and removed "\
                "from sources such as the soil surface, wet vegetation, pavement, water bodies, etc. "\
                "Transpiration consists of the vaporization of liquid water within a plant and subsequent "\
                "loss of water as vapor through leaf stomata.",
                "license":"Creative Commons Zero (CC0)",
                "keywords":["PET", "UERRA", "ADO"],
                "providers":"Producer: Météo-France; Processor: ZAMG Austria",
                "links":["https://edis.ifas.ufl.edu/ae459","https://datastore.copernicus-climate.eu/documents/uerra/D322_Lot1.4.1.2_User_guides_v3.3.pdf"],
                "lineage":"Quantile Mapped ERA5 data (mapped with UERRA MESCAN-Surfex data) is used in order to calculate PET following Penman-Monteith FAO-56 method",
                "comment":"Quantile Mapped Potential Evapotranspiration calculated following Penman-Monteith FAO-56 method "\
                "described in AE459. Daily averages/sums are used, whereas wind direction and speed are "\
                "converted to u and v wind components before averaging.",
                "source":"ERA5, UERRA MESCAN-Surfex",
                "Conventions":"CF-1.7",
            }

            xds.pet.attrs = {
                "long_name":"Potential Evapotranspiration",
                "units":"mm day**-1",
                "grid_mapping":"Lambert_Conformal"
            }

        return xds
