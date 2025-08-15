"""
Compress yearly ar_tag files using ncks.
"""

import os
import glob
import numpy as np
import datetime as dt

time_pds = ['hist','future']
ens_members = ['1011','1031','1051']

for time_pd in time_pds:
    input_dir_temp = f'/doppler/data7/kmattingly/AR_data_v2/CESM2-LE/CESM2-LE_ARs_global_temp/{time_pd}/'
    output_dir_comp = f'/doppler/data7/kmattingly/AR_data_v2/CESM2-LE/CESM2-LE_ARs_global_compressed/{time_pd}/'

    if time_pd == 'hist':
        years = np.arange(1990,2010,1)
    elif time_pd == 'future':
        years = np.arange(2080,2100,1)

    for ens_member in ens_members:
        for year in years:

            temp_yearly_fpath = glob.glob(os.path.join(
                input_dir_temp,
                f'*LE2-{ens_member}*6hr.{year}*.nc'
            ))[0]

            fname = os.path.basename(temp_yearly_fpath)
            comp_yearly_fpath = os.path.join(
                output_dir_comp,
                f'{fname}4'
            )
            
            os.system(f'ncks -4 -L 1 {temp_yearly_fpath} {comp_yearly_fpath}')

            now = dt.datetime.now()
            print(f'Finished compressing {year}, time period: {time_pd}, member: {ens_member} at {now}')