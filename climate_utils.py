"""
climate_utils.py

Reusable utility functions for climate data processing.
These functions are intended to be shared across notebooks
in this repository.
"""

def kelvin_to_celsius(da):
    """
    Convert temperature DataArray from Kelvin to degrees Celsius.

    Parameters
    ----------
    da : xarray.DataArray
        Temperature variable in Kelvin.

    Returns
    -------
    xarray.DataArray
        Temperature in degrees Celsius.
    """
    da_c = da - 273.15
    da_c.attrs = da.attrs.copy()
    da_c.attrs["units"] = "°C"
    return da_c
