"""
Compress ARTMIP Tier 2 paleo yearly ar_tag files using ncks.
"""

import os
import numpy as np
import datetime as dt

paleo_datasets = ['PI_21ka-CO2','10ka-Orbital','PreIndust']

years_orig = {'PI_21ka-CO2':np.arange(61,91,1),
              '10ka-Orbital':np.arange(271,301,1),
              'PreIndust':np.arange(121,151,1)}

for paleo_dataset in paleo_datasets:    
    input_dir_temp = f'/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier2_paleo/ARs_{paleo_dataset}_temp/'
    output_dir_comp = f'/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier2_paleo/ARs_{paleo_dataset}_compressed/'
        
    for year_orig in years_orig[paleo_dataset]:
        year_orig_str = '{:04d}'.format(year_orig)
        
        temp_yearly_fpath = f'{input_dir_temp}{paleo_dataset}.ar_tag.Mattingly_v2.6hr.{year_orig_str}.nc4'
        comp_yearly_fpath = f'{output_dir_comp}{paleo_dataset}.ar_tag.Mattingly_v2.6hr.{year_orig_str}.nc4'
    
        os.system(f'ncks -4 -L 1 {temp_yearly_fpath} {comp_yearly_fpath}')
    
        now = dt.datetime.now()
        print(f'Finished compressing {paleo_dataset}, year {year_orig_str} at {now}')