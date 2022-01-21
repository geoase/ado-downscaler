import xarray as xr
import os
import numpy as np
import dask
from ado_downscaler import Downscaler

def calculate_pet(dct_paths, storage_path):
    """
    calculate potential evapotranspiration

    Parameters
    ----------
    dct_paths : dict
        dictionary with input paths for each variable 
        (2m_dewpoint_temperature, 2m_temperature, 
        surface_net_solar_radiation, surface_net_thermal_radiation, 
        10m_u_component_of_wind, 10m_v_component_of_wind,
        total_precipitation)
    storage_path : string
        storage path of data collection as string
    """
        
    xr.set_options(keep_attrs=True)
    
    # Prepare Data ==========================================
    # Wind ----------------------------------------------------
    # http://weatherclasses.com/uploads/3/6/2/3/36231461/computing_wind_direction_and_speed_from_u_and_v.pdf
    xds_u = xr.open_dataset(dct_paths["10m_u_component_of_wind"], decode_coords="all", decode_cf=True) 
    #chunks={"time":-1, "x":30,"y":30}
    xds_v = xr.open_dataset(dct_paths["10m_v_component_of_wind"], decode_coords="all", decode_cf=True)
    
    xda_ws = xr.ufuncs.sqrt(xds_u.u10**2+xds_v.v10**2)
    xda_ws2 = xda_ws*(4.87/(xr.ufuncs.log(67.8*10-5.42)))

    # Dewpoint Temperature ------------------------------------
    # https://journals.ametsoc.org/view/journals/bams/86/2/bams-86-2-225.xml?tab_body=pdf
    xds_dewpoint = xr.open_dataset(dct_paths["2m_dewpoint_temperature"], decode_coords="all", decode_cf=True)
    xds_temp = xr.open_dataset(dct_paths["2m_temperature"], decode_coords="all", decode_cf=True)
    
    # Convert to degree Centigrade
    xda_temp = xds_temp.t2m-273.15
    A1 = 17.625
    B1 = 243.04
    C1 = 610.94
    xda_dewpoint = xds_dewpoint.d2m-273.15
    
    # Actual vapour pressure ----------------------------------
    # http://www.fao.org/3/x0490e/x0490e07.htm#calculation%20procedures
    xda_ea = 0.6108*xr.ufuncs.exp((17.27*xda_dewpoint)/(xda_dewpoint+237.3))
    
    # Slope of saturation vapour pressure curve
    xda_delta = (4098*(0.6108*xr.ufuncs.exp((17.27*xda_temp)/(xda_temp+237.3))))/((xda_temp+237.3)**2)
    
    
    # Mean saturation vapour pressure -------------------------
    xda_es = 0.6108*xr.ufuncs.exp((17.27*xda_temp)/(xda_temp+237.3))
    
    # Atmospheric Pressure and Psychrometric Constant ---------
    xds_oro = xr.open_dataset(dct_paths["orography_reanalysis"], decode_coords="all", decode_cf=True)
    xds_oro = Downscaler.cut_eusalp_domain(xds_oro)
    
    xda_pres = 101.3*((293-0.0065*xds_oro.orog)/293)**5.26
    xda_gamma = 0.665*10**-3*xda_pres[0].drop("time")
    
    # Radiation -----------------------------------------------
    xds_rns = xr.open_dataset(dct_paths["surface_net_solar_radiation"], decode_coords="all", decode_cf=True)
    xds_rnl = xr.open_dataset(dct_paths["surface_net_thermal_radiation"], decode_coords="all", decode_cf=True)
    
    # Correct Units of radiation
    xda_rns = xds_rns.ssr/(10**6)
    xda_rnl = xds_rnl.str/(10**6)
    
    # Longwave Radiation
    xda_rn = xda_rns + xda_rnl
    xda_rng = 0.408*xda_rn
    
    # Calculation Terms ---------------------------------------
    xda_DT = xda_delta / (xda_delta + xda_gamma*(1+0.34*xda_ws2))
    xda_PT = xda_gamma / (xda_delta + xda_gamma*(1+0.34*xda_ws2))
    xda_TT = (900/(xda_temp+273.15))*xda_ws2
    
    ET_rad = xda_DT*xda_rng
    ET_wind = xda_PT * xda_TT * (xda_es - xda_ea)
    
    ET_0 = ET_wind + ET_rad
    ET_0 = ET_0.compute()
    
    xds_pet = xds_temp.drop("t2m")
    xds_pet["pet"] = ET_0
    
    xds_pet["pet"].attrs = {
        "long_name":"Potential Evapotranspiration",
        "units":"mm day**-1",
        "grid_mapping":"Lambert_Conformal"
    }
    
    # DataSet Attributes
    xds_pet.attrs = {
        "title": "Quantile Mapped Potential Evapotranspiration following Penman-Monteith using UERRA MESCAN-Surfex data",
        "institution": "Zentralanstalt fuer Meteorologie und Geodynamik",
        "description": "The term evapotranspiration (ET) is commonly used to describe two processes "\
            "of water loss from land surface to atmosphere, evaporation and transpiration. Evaporation "\
            "is the process where liquid water is converted to water vapor (vaporization) and removed "\
            "from sources such as the soil surface, wet vegetation, pavement, water bodies, etc. "\
            "Transpiration consists of the vaporization of liquid water within a plant and subsequent "\
            "loss of water as vapor through leaf stomata.",
        "license": "Creative Commons Zero (CC0)",
        "keywords": ["PET", "UERRA", "ADO"],
        "providers": "Producer: Météo-France; Processor: ZAMG Austria",
        "links": ["https://edis.ifas.ufl.edu/ae459","https://datastore.copernicus-climate.eu/documents/uerra/D322_Lot1.4.1.2_User_guides_v3.3.pdf"],
        "lineage": "Quantile Mapped ERA5 data (mapped with UERRA MESCAN-Surfex data) is used in order to calculate PET following Penman-Monteith FAO-56 method",
        "comment": "Quantile Mapped Potential Evapotranspiration calculated following Penman-Monteith FAO-56 method "\
            "described in AE459. Daily averages/sums are used, whereas wind direction and speed are "\
            "converted to u and v wind components before averaging.",
        "source": "ERA5, UERRA MESCAN-Surfex",
        "Conventions": "CF-1.7",
    }
    
    # COARDS order of dimensions T,Y,X
    xds_pet = xds_pet.transpose("time","y","x")
    
    file_suffix = dct_paths["2m_temperature"][len("2m_temperature"):]
    xds_pet.to_netcdf(os.path.join(storage_path, f"potential_evapotranspiration_{file_suffix}"))
    
