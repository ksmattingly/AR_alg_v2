"""
Driver script for CESM-LE AR ID
- Output data are divided into files spanning 1 month each
- Spatial parameters (including NH or SH domain) are specified in AR_ID_config.hjson
"""

import os
import datetime as dt
import pandas as pd
import calendar

ens_member = '1031'
hemi = 'SH'

config_fpath = f'/home/kmattingly/projects/AR_alg_v2/scripts/CESM2-LE_ARTMIP_polar/AR_ID_config_CESM2-LE_AR_calc_{hemi}_1990-2009.hjson'

chunk_begin_t = dt.datetime(1990,1,1,0)
chunk_end_t = dt.datetime(2009,12,31,18)
timestep_hrs = '6'

times_monthly = pd.date_range(chunk_begin_t, chunk_end_t, freq='1M')

for t_monthly in times_monthly:
    t_begin = dt.datetime(t_monthly.year, t_monthly.month, 1, 0)
    month_end_day = calendar.monthrange(t_monthly.year, t_monthly.month)[1]
    t_end = dt.datetime(t_monthly.year, t_monthly.month, month_end_day, 18)

    t_begin_str = t_begin.strftime('%Y-%m-%d_%H%M')
    t_end_str = t_end.strftime('%Y-%m-%d_%H%M')
    
    os.system(
        f'python /home/kmattingly/projects/AR_alg_v2/scripts/CESM2-LE_ARTMIP_polar/'+\
        f'ARs_ID_CESM2-LE_{hemi}.py {t_begin_str} {t_end_str} {timestep_hrs} {config_fpath} {ens_member}')