"""
Miscellaneous utility functions for AR algorithm.
"""

def rename_coords(ds):
    """
    Make sure latitude and longitude in xarray dataset are named "lat" and "lon",
    so that these coordinate names can be used to subset data.
    
    (Files obtained from ERA5 have coords named "latitude" and "longitude";
    MERRA-2 files have "lat" and "lon". This may need to be adapted for any
    other data sources.)
    """
    
    if ('lat' in ds.coords) and ('lon' in ds.coords):
        pass
    else:
        if ('latitude' in ds.coords) and ('longitude' in ds.coords):
            ds = ds.rename({'latitude':'lat','longitude':'lon'})
        else:
            raise Exception('Unknown lat/lon coordinate names')
    
    return ds


def rename_IVT_components(ds):
    """
    Make sure u- and v-IVT components in xarray dataset are named "uIVT" and "vIVT".
    
    (ERA5 pre-calculated uIVT and vIVT are named "" and "". Most other datasets
    will have variables named "uIVT" and "vIVT" as needed for AR ID script,
    because these variables are calculated "in house" by calc_IVT.py rather than
    being provided in the original dataset.)
    """
    
    if ('p71.162') and ('p72.162') in ds.variables:
        ds = ds.rename({'p71.162':'uIVT', 'p72.162':'vIVT'})
    elif ('IVTx') and ('IVTy') in ds.variables:
        ds = ds.rename({'IVTx':'uIVT', 'IVTy':'vIVT'})
        
    return ds
