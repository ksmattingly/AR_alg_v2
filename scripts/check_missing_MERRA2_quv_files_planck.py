# Check for missing MERRA-2 model level q/u/v/delp files after initial batch download 
# of 3-hourly, full globe data for 1980-2021.
# - Batch download was performed using the list of links obtained from earthdata
#   and their wget instructions

import glob
import datetime as dt
from datetime import timedelta
import pandas as pd

quv_dir = '/doppler/data7/kmattingly/MERRA2_qv-u-v-delp_global_3hr_model-levs-41-72/'
quv_flist = sorted(glob.glob(quv_dir+'MERRA2*.nc'))

start_dt = t = dt.datetime(1980,1,1,0)
end_dt = dt.datetime(2021,12,31,0)

missing_files = []
while t <= end_dt:
    if (t in pd.date_range(dt.datetime(2020,9,1,0), dt.datetime(2020,9,30,21), freq='1D')) or \
        (t in pd.date_range(dt.datetime(2021,6,1,0), dt.datetime(2021,9,30,21), freq='1D')):
        collection_number = '401'
    else:
        if t in pd.date_range(dt.datetime(1980,1,1,0), dt.datetime(1991,12,31,21), freq='1D'):
            collection_number = '100'
        if t in pd.date_range(dt.datetime(1992,1,1,0), dt.datetime(2000,12,31,21), freq='1D'):
            collection_number = '200'
        if t in pd.date_range(dt.datetime(2001,1,1,0), dt.datetime(2010,12,31,21), freq='1D'):
            collection_number = '300'
        if t in pd.date_range(dt.datetime(2011,1,1,0), dt.datetime(2021,12,31,21), freq='1D'):
            collection_number = '400'    
    
    t_str = t.strftime('%Y%m%d')
    fname = quv_dir+f'MERRA2_{collection_number}.inst3_3d_asm_Nv.{t_str}.SUB.nc'
    if fname not in quv_flist:
        missing_files.append(fname)
    
    t += timedelta(days=1)