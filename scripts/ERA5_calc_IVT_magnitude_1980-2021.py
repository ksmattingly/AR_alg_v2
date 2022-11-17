"""
Driver script for ERA5 IVT calculations (full-POR (1980-2021), 1-hourly, full globe)
The calculation adds IVT magnitude to files with pre-calculated u/v-IVT data.
"""

import os
import datetime as dt
import pandas as pd

begin_t = dt.datetime(1980,1,1,0)
end_t = dt.datetime(2021,12,31,23)
timestep_hrs = '1'

output_file_times = pd.date_range(begin_t, end_t, freq='1M')

for file_t in output_file_times:
    file_begin_t = dt.datetime(file_t.year, file_t.month, 1, 0, 0).strftime('%Y-%m-%d_%H%M')
    file_end_t = dt.datetime(file_t.year, file_t.month, file_t.day, 23, 0).strftime('%Y-%m-%d_%H%M')
    
    os.system('python ../AR_alg_v2/calc_IVT_ERA5.py {file_begin_t} {file_end_t} {timestep_hrs}')
    
    print('Finished writing '+str(file_t)+' at '+str(dt.datetime.now()))