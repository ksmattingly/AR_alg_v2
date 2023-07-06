"""
Miscellaneous utility functions for AR algorithm.
"""

import numpy as np

def rename_coords(ds):
    """
    Make sure latitude and longitude in xarray dataset are named "lat" and "lon",
    so that these coordinate names can be used to subset data.
    
    (Files obtained from ERA5 have coords named "latitude" and "longitude";
    MERRA-2 files have "lat" and "lon". This may need to be adapted for any
    other data sources.)
    
    This function also changes very small (but nonzero) values of lat/lon to
    a value of 0 (using the helper function _fix_zero_values).
    """
    
    if ('lat' in ds.coords) and ('lon' in ds.coords):
        pass
    else:
        if ('latitude' in ds.coords) and ('longitude' in ds.coords):
            ds = ds.rename({'latitude':'lat','longitude':'lon'})
        else:
            raise Exception('Unknown lat/lon coordinate names')
    
    ds = _fix_zero_values(ds)
    
    return ds


def _fix_zero_values(ds):
    """
    Change very small (but nonzero) values of lat/lon to a value of 0.
    """
    
    lats = ds.lat.data
    lons = ds.lon.data
    
    lats[np.where(np.abs(lats) < 0.0001)] = 0
    lons[np.where(np.abs(lons) < 0.0001)] = 0
    
    ds['lat'] = lats
    ds['lon'] = lons
    
    return ds


def rename_IVT_components(ds):
    """
    Make sure u- and v-IVT components in xarray dataset are named "uIVT" and "vIVT".
    
    (ERA5 pre-calculated uIVT and vIVT are named "p71.162" and "p72.162". Most
    other datasets will have variables named "uIVT" and "vIVT" as needed for
    AR ID script, because these variables are calculated "in house" by calc_IVT.py
    rather than being provided in the original dataset.)
    """
    
    if ('p71.162') and ('p72.162') in ds.variables:
        ds = ds.rename({'p71.162':'uIVT', 'p72.162':'vIVT'})
    elif ('IVTx') and ('IVTy') in ds.variables:
        ds = ds.rename({'IVTx':'uIVT', 'IVTy':'vIVT'})
        
    return ds