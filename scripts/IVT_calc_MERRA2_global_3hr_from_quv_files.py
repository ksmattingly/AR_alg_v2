# Driver script for MERRA-2 IVT calculations (full-POR (1980-2021), 3-hourly, full globe,
# model-level, using input MERRA-2 data q/u/v/delp data stored locally)
# - Script was originally run on planck
# - Note that this driver script could also be used to calculate IVT using MERRA-2
#   data stored remotely, using OPeNDAP. No modifications would be needed to this
#   script, rather the option would be changed in AR_ID_config.hjson.
#   - On planck, it proved to be faster to download the input MERRA-2 quv files first,
#     then run IVT calculation on these locally stored files (rather than using
#     OPeNDAP).

# Output IVT files in the following format:
# - Monthly
# - 3 hourly timestep
# - Covering entire globe

import os
import datetime as dt
import pandas as pd

begin_t = dt.datetime(1980,1,1,0)
end_t = dt.datetime(2021,12,31,21)
timestep_hrs = 3

output_file_times = pd.date_range(begin_t, end_t, freq='1M')

for file_t in output_file_times:
    file_begin_t = dt.datetime(file_t.year, file_t.month, 1, 0, 0)
    file_end_t = dt.datetime(file_t.year, file_t.month, file_t.day, 21, 0)
    
    os.system('python ../AR_alg_v2/calc_IVT.py '+\
              file_begin_t.strftime('%Y-%m-%d_%H%M')+' '+\
              file_end_t.strftime('%Y-%m-%d_%H%M')+' '+\
              str(timestep_hrs))