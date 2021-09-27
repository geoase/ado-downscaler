# ADO Downscaler

[![Build Status](https://dev.azure.com/gseyerl/gseyerl/_apis/build/status/geoase.ado-downscaler?branchName=master)](https://dev.azure.com/gseyerl/gseyerl/_build/latest?definitionId=1&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/ado-downscaler/badge/?version=latest)](https://ado-downscaler.readthedocs.io/en/latest/?badge=latest)

This downloader package is part of the EU Interreg Project: Alpine Drought Observation


## Documentation

Package documentation is available at [readthedocs](https://ado-downscaler.readthedocs.io/en/latest/).



## Installation

ado-downscaler can be installed using the following command:

    python setup.py install

For detailed installation instructions, especially how to install dependencies,
please refer to the
[INSTALL](https://ado-downscaler.readthedocs.io/en/latest/install.html) section
of the documentation.



## Usage

- Downscaling with xarray.Dataset object
```
from ado_downscaler import Downscaler
import xarray as xr

xds_sce = xr.open_dataset("ado_downscaler/tests/data/test_pr_era5_1979-2018.nc")
downscaler = Downscaler.from_filepaths(
    "ado_downscaler/tests/data/test_pr_uerra_1961-2018.nc", 
    "ado_downscaler/tests/data/test_pr_era5_1979-2018.nc"
)

downscaler.downscale(xds_sce,"tmp")
```

- Downscale ERA5 from gribfile
```
from ado_downscaler import Downscaler

downscaler = Downscaler.from_filepaths(
    "ado_downscaler/tests/data/test_pr_uerra_1961-2018.nc", 
    "ado_downscaler/tests/data/test_pr_era5_1979-2018.nc"
)

downscaler.downscale_era5(
    "ado_downscaler/tests/data/total_precipitation_2020_2_29_reanalysis-era5-single-levels.grib",
    "tmp"
)
```
