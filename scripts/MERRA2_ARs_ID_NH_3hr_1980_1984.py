# Driver script for MERRA-2 AR ID (NH, 3-hourly) for 1980-1984
# - Output data are divided into files spanning 1 month each
# - Script was run on planck
# - Only script for the "chunk" from 1980-1984 is committed to git as an example;
#   the rest of the 1980-2021 period of record was processed by writing other
#   scripts with different 5-year chunk begin/end dates

import os
import datetime as dt
import pandas as pd
import calendar

chunk_begin_t = dt.datetime(1980,1,1,0)
chunk_end_t = dt.datetime(1984,12,31,21)
timestep_hrs = 3

times_monthly = pd.date_range(chunk_begin_t, chunk_end_t, freq='1M')

for t_monthly in times_monthly:
    t_begin = dt.datetime(t_monthly.year, t_monthly.month, 1, 0)
    month_end_day = calendar.monthrange(t_monthly.year, t_monthly.month)[1]
    t_end = dt.datetime(t_monthly.year, t_monthly.month, month_end_day, 21)

    t_begin_str = t_begin.strftime('%Y-%m-%d_%H%M')
    t_end_str = t_end.strftime('%Y-%m-%d_%H%M')
    
    os.system(f'python ../AR_alg_v2/ARs_ID.py {t_begin_str} {t_end_str} 3')