"""
Driver script for MERRA-2 IVT calculations (1980-2021, 3-hourly, full globe,
model-level, using input MERRA-2 data q/u/v/delp data stored locally)
- Note that this driver script could also be used to calculate IVT using MERRA-2
  data stored remotely, using OPeNDAP. No modifications would be needed to this
  script, rather the option would be changed in AR_ID_config.hjson.

Output IVT files in the following format:
- Monthly
- 3 hourly timestep
- Covering entire globe
"""

import os
import datetime as dt
import pandas as pd

config_fpath = '/home/kmattingly/projects/AR_alg_v2/config_files/AR_ID_config_MERRA2_global_IVT_calc.hjson'

begin_t = dt.datetime(1980,1,1,0)
end_t = dt.datetime(2021,12,31,21)
timestep_hrs = '3'

output_file_times = pd.date_range(begin_t, end_t, freq='1M')

for file_t in output_file_times:
    file_begin_t = dt.datetime(file_t.year, file_t.month, 1, 0, 0).strftime('%Y-%m-%d_%H%M')
    file_end_t = dt.datetime(file_t.year, file_t.month, file_t.day, 21, 0).strftime('%Y-%m-%d_%H%M')
    
    os.system(f'python ../AR_alg_v2/calc_IVT.py {file_begin_t} {file_end_t} {timestep_hrs} {config_fpath}')