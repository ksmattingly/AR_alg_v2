# Check for missing MERRA-2 IVT files after full-POR (1980-2021), 3-hourly, full globe,
#  model-level calculations were performed.

import glob
import numpy as np
import datetime as dt
from calendar import monthrange

IVT_dir = '/doppler/data7/kmattingly/MERRA2_IVT_global_3hr_model_levs_from_quv_files/'
IVT_flist = sorted(glob.glob(IVT_dir+'IVT_MERRA2*.nc'))

years = np.arange(1980,2022,1)
months = np.arange(1,13,1)

missing_files = []
for year in years:
    for month in months:
        monthdays = monthrange(year,month)
        file_start_t_str = dt.datetime(year,month,1,0).strftime('%Y%m%d%H%M')
        file_end_t_str = dt.datetime(year,month,monthdays[1],21).strftime('%Y%m%d%H%M')
        fname = f'IVT_MERRA2_NH_subset_3hr_{file_start_t_str}_{file_end_t_str}.nc'
        
        if IVT_dir+fname not in IVT_flist:
            missing_files.append(fname)