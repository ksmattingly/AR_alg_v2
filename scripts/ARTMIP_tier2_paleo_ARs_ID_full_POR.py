"""
Driver script to identify ARs for the full period of record (POR) of an
ARTMIP Tier 2 paleo dataset.

This script is an example for identifying Northern Hemisphere ARs in the
Pre-Industrial dataset. It can be adapted to other ARTMIP Tier 2 paleo datasets
and/or to the Southern Hemisphere by using a different configuration file. 
"""

import os
import datetime as dt
import pandas as pd
import calendar

config_fpath = '/home/kmattingly/projects/AR_alg_v2/config_files/AR_ID_config_ARTMIP_tier2_paleo_PreIndust_NH.hjson'

chunk_begin_t = dt.datetime(1970,1,1,3)
chunk_end_t = dt.datetime(1999,12,31,21)
timestep_hrs = '6'

times_monthly = pd.date_range(chunk_begin_t, chunk_end_t, freq='1M')

for t_monthly in times_monthly:
    t_begin = dt.datetime(t_monthly.year, t_monthly.month, 1, 3)
    month_end_day = calendar.monthrange(t_monthly.year, t_monthly.month)[1]
    t_end = dt.datetime(t_monthly.year, t_monthly.month, month_end_day, 21)

    t_begin_str = t_begin.strftime('%Y-%m-%d_%H%M')
    t_end_str = t_end.strftime('%Y-%m-%d_%H%M')
    
    os.system(f'python ../AR_alg_v2/ARs_ID_ARTMIP_tier2_paleo.py {t_begin_str} {t_end_str} {timestep_hrs} {config_fpath}')