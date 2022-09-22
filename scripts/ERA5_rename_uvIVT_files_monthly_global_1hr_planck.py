# Rename ERA5 u-/v-IVT files on planck after IVT has been added as a variable to files
# - Original format: ERA5_uvIVT_[timestep]hr_[domain]_[year]_[month].nc
#   - Example: ERA5_uvIVT_1hr_global_2021_01.nc
# - Updated format (to match MERRA-2): IVT_ERA5_[domain]_[timestep]hr_[startyear][startmonth][startday][starthour][startminute]_[endyear][endmonth][endday][endhour][endminute].nc
#   - Example: IVT_ERA5_global_1hr_202101010000_202101312300.nc

import os
import datetime as dt
import calendar

ERA5_IVT_dir = '/doppler/data7/kmattingly/ERA5_IVT_global_1hr_from-model-levs/'

orig_fnames = os.listdir(ERA5_IVT_dir)
for orig_fname in orig_fnames:
    year_str = orig_fname.split('global_')[1][0:4]
    month_str = orig_fname.split('global_')[1][5:7]
    
    start_dt = dt.datetime(int(year_str), int(month_str), 1, 0)
    end_day_of_month = calendar.monthrange(start_dt.year, start_dt.month)[1]
    end_dt = dt.datetime(int(year_str), int(month_str), end_day_of_month, 23)
    
    start_t_str = start_dt.strftime('%Y%m%d%H%M')
    end_t_str = end_dt.strftime('%Y%m%d%H%M')
    
    new_fname = f'IVT_ERA5_global_1hr_{start_t_str}_{end_t_str}.nc'
    os.rename(ERA5_IVT_dir+orig_fname, ERA5_IVT_dir+new_fname)