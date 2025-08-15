"""
Compress yearly ar_tag files using ncks.
"""

import os
import numpy as np
import datetime as dt

pctiles = ['85','875','98']

for pctile in pctiles:

    input_dir_temp = f'/doppler/data7/kmattingly/AR_data_v2/MERRA2_ARTMIP_polar_tweaked_pctiles/MERRA2_ARs_pctile{pctile}_global_temp/'
    output_dir_comp = f'/doppler/data7/kmattingly/AR_data_v2/MERRA2_ARTMIP_polar_tweaked_pctiles/MERRA2_ARs_pctile{pctile}_global_compressed/'

    years = np.arange(1980,2020,1)

    for year in years:
        year_str = str(year)
        
        temp_yearly_fpath = f'{input_dir_temp}MERRA2.ar_tag.Mattingly_v2_pct{pctile}.3hourly.{year}0101_{year}1231.nc4'
        comp_yearly_fpath = f'{output_dir_comp}MERRA2.ar_tag.Mattingly_v2_pct{pctile}.3hourly.{year}0101-{year}1231.nc4'

        os.system(f'ncks -4 -L 1 {temp_yearly_fpath} {comp_yearly_fpath}')

        now = dt.datetime.now()
        print(f'Finished compressing {year}, pctile: {pctile} at {now}')