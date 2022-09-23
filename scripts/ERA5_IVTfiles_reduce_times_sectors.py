import xarray as xr
import os
import numpy as np
import glob
import pandas as pd
import datetime as dt

full_files_dir = '/doppler/data7/kmattingly/ERA5_IVT_global_1hr_from-model-levs/'
full_fpaths = sorted(glob.glob(full_files_dir+'*.nc'))

hemis_lats = {'NH':np.arange(10,90.25,0.25), 'SH':np.arange(-90,-9.75,0.25)}
sector_lons = {'1':np.arange(0,120.25,0.25), '2':np.arange(120.25,240.25,0.25), '3':np.arange(240.25,360,0.25)}

for hemi in sorted(hemis_lats.keys()):
    for sector in sorted(sector_lons.keys()):
        output_dir = f'/doppler/data7/kmattingly/ERA5_IVT_reduced_{hemi}_sector{sector}/'
        
        for fpath in full_fpaths:
            ds = xr.open_dataset(fpath)
            
            times = pd.date_range(pd.to_datetime(ds.time[0].data),
                                  pd.to_datetime(ds.time[-1].data),
                                  freq='3H')
            
            IVT_subset = ds.IVT.sel(time=times, 
                                    latitude=hemis_lats[hemi], 
                                    longitude=sector_lons[sector])
            
            IVT_subset.to_netcdf(output_dir+os.path.basename(fpath))
            
            print('Wrote '+hemi+', '+sector+' '+os.path.basename(fpath)+' at '+\
                  str(dt.datetime.now()))