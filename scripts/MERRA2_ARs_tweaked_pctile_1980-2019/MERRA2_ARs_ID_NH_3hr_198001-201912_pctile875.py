"""
Driver script for MERRA-2 AR ID (3-hourly, NH) for 1980–2019, 87.5th percentile
- Output data are divided into files spanning 1 month each
- Spatial parameters (including NH or SH domain) are specified in AR_ID_config.hjson
"""

import os
import datetime as dt
import pandas as pd
import calendar

config_fpath = '/home/kmattingly/projects/AR_alg_v2/scripts/MERRA2_ARs_tweaked_pctile_1980-2019/AR_ID_config_MERRA2_NH_87.5_pctile_1980-2019.hjson'

chunk_begin_t = dt.datetime(1980,1,1,0)
chunk_end_t = dt.datetime(2019,12,31,21)
timestep_hrs = '3'

times_monthly = pd.date_range(chunk_begin_t, chunk_end_t, freq='1M')

for t_monthly in times_monthly:
    t_begin = dt.datetime(t_monthly.year, t_monthly.month, 1, 0)
    month_end_day = calendar.monthrange(t_monthly.year, t_monthly.month)[1]
    t_end = dt.datetime(t_monthly.year, t_monthly.month, month_end_day, 21)

    t_begin_str = t_begin.strftime('%Y-%m-%d_%H%M')
    t_end_str = t_end.strftime('%Y-%m-%d_%H%M')
    
    os.system(f'python /home/kmattingly/projects/AR_alg_v2/scripts/MERRA2_ARs_tweaked_pctile_1980-2019/ARs_ID_tweaked_pctile.py {t_begin_str} {t_end_str} {timestep_hrs} {config_fpath}')