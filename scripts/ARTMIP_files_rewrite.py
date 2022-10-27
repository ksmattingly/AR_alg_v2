"""
One-time script to make changes to ARTMIP netCDF files:
- encode time information that can be read by xarray
- discard unused variables: surface pressure (PS) and integrated water vapor (IWV)
- write to monthly instead of yearly files
- change file name to match format expected by calc_IVT_at_pctiles.py and ARs_ID.py

Write out new netCDF files (with time as a dimension) to a "test" directory, to
avoid overwriting the existing ARTMIP data. Then manually copy over the new files
and delete the old files after confirming the new files are correct.
"""

import xarray as xr
from netCDF4 import Dataset
import datetime as dt
from datetime import timedelta
import glob
import os
import pandas as pd
import calendar

IVT_dir = '/run/media/kmattingly/kyle_8tb/ARTMIP_tier1/MERRA2_IVT_source_data/'
IVT_fpaths = sorted(glob.glob(IVT_dir+'*.nc'))

output_dir = '/run/media/kmattingly/kyle_8tb/ARTMIP_tier1/MERRA2_IVT_source_data_rewrite_test/'

for i,p in enumerate(IVT_fpaths):
    input_fname = os.path.basename(p)
    
    # ARTMIP data for 2018 is partitioned into 2 files (Jan-May and June-Dec);
    # other years have a single yearly file
    if len(input_fname) < 24:
        yr = int(input_fname[-7:-3])
        begin_month = 1
    else:
        begin_t_str = input_fname.split('ARTMIP_MERRA_2D_')[1][:6]
        end_t_str = input_fname.split('ARTMIP_MERRA_2D_')[1][7:13]
        yr = int(begin_t_str[:4])
        begin_month = int(begin_t_str[4:])
    
    base_dt = dt.datetime(yr, begin_month, 1, 0)

    with Dataset(p, 'r') as nc:
        times = [base_dt + timedelta(hours=3*i) for i in range(0,nc.dimensions['record'].size,1)]
        lats = nc.variables['lat'][:]
        lons = nc.variables['lon'][:]
        IVT = nc.variables['IVT'][:]
        uIVT = nc.variables['uIVT'][:]
        vIVT = nc.variables['vIVT'][:]
        
        # Copy attributes from original dataset
        attrs = {}
        for attr in nc.ncattrs():
            attrs[attr] = nc.getncattr(attr)
    
    new_large_ds = xr.Dataset(
            {
              'IVT':(('time','lat','lon'), IVT.astype('float32')),
              'uIVT':(('time','lat','lon'), uIVT.astype('float32')),
              'vIVT':(('time','lat','lon'), vIVT.astype('float32'))
            },
            coords={
                'time':times,
                'lat':lats.astype('float32'),
                'lon':lons.astype('float32')
            }
        )
    
    for attr in attrs:
        new_large_ds.attrs[attr] = attrs[attr]

    months_in_ds = set([t.month for t in pd.to_datetime(new_large_ds.time.data[:])])
    for month in months_in_ds:
        month_end_day = calendar.monthrange(yr, month)[1]
        month_times = pd.date_range(dt.datetime(yr, month, 1, 0),
                                    dt.datetime(yr, month, month_end_day, 21),
                                    freq='3H')
        ds_month = new_large_ds.sel(time=month_times)
        
        file_begin_t = month_times[0].strftime('%Y%m%d%H%M')
        file_end_t = month_times[-1].strftime('%Y%m%d%H%M')
        output_fname = f'ARTMIP_MERRA_2D_{file_begin_t}_{file_end_t}.nc'
        output_fpath = output_dir+output_fname
        ds_month.to_netcdf(output_fpath)
        
        t_now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Finished writing {output_fpath} at {t_now}')