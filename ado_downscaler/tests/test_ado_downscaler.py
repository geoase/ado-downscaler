import os
import pytest

from ado_downscaler import Downscaler

import xarray as xr

@pytest.fixture
def era5_downscaler():
    return Downscaler.from_filepaths(
        os.path.join("data","test_pr_uerra_1961-2018.nc"),
        os.path.join("data","test_pr_era5_1979-2018.nc"),
    )

def test_downscale_era5(era5_downscaler, tmp_path):
    lst_paths = era5_downscaler.downscale_era5(
        os.path.join("data", "total_precipitation_2021_1_21_reanalysis-era5-single-levels.grib"),
        tmp_path
    )
    assert len(lst_paths) == 1
