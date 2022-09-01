# Output IVT files in the following format:
# - Monthly
# - 3 hourly
# - Covering entire Northern Hemisphere from 10 degrees latitude to the pole

import os
import datetime as dt
import pandas as pd

# Plan is to run 5 scripts in parallel, so partition the 1980-2021 analysis 
# period into 5 chunks:

# Chunk 1: 1980-01-01 00:00 - 1988-12-31 21:00
# Chunk 2: 1989-01-01 00:00 - 1997-12-31 21:00
# Chunk 3: 1998-01-01 00:00 - 2005-12-31 21:00
# Chunk 4: 2006-01-01 00:00 - 2013-12-31 21:00
# Chunk 5: 2014-01-01 00:00 - 2021-12-31 21:00

# (Only the driver script for chunk 1 is committed to git an as example)

begin_t = dt.datetime(1980,1,1,0)
end_t = dt.datetime(1988,12,31,21)
timestep_hrs = 3

output_file_times = pd.date_range(begin_t, end_t, freq='1M')

for file_t in output_file_times:
    file_begin_t = dt.datetime(file_t.year, file_t.month, 1, 0, 0)
    file_end_t = dt.datetime(file_t.year, file_t.month, file_t.day, 21, 0)
    
    os.system('python ../AR_alg_v2/calc_IVT.py '+\
              file_begin_t.strftime('%Y-%m-%d_%H%M')+' '+\
              file_end_t.strftime('%Y-%m-%d_%H%M')+' '+\
              str(timestep_hrs))