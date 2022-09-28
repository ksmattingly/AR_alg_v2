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