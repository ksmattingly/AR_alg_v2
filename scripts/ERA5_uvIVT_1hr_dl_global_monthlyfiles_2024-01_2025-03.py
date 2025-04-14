"""
Download ERA5 u/v-IVT data, 1-hourly resolution, global, for 2024-01 through
2025-03.
"""

import cdsapi
import numpy as np
import os
import datetime as dt

client = cdsapi.Client()

outdir = os.path.join(
    '/doppler', 'data7', 'kmattingly', 'AR_data_v2',
    'ERA5_uvIVT_global_1hr_from-model-levs')

dataset = 'reanalysis-era5-single-levels'

years = [2024,2025]
months = np.arange(1,13,1)
# months = [2]

for year in years:
    for month in months:
        if (year == 2025) and (month > 2):
            continue

        yrmth_dt = dt.datetime(year,month,1,0)
        yrmth_str = yrmth_dt.strftime('%Y_%m.nc')

        fname = f'ERA5_uvIVT_1hr_global_{yrmth_str}'
        out_fpath = os.path.join(outdir, fname)

        print(out_fpath)

        year_str = yrmth_dt.strftime('%Y')
        month_str = yrmth_dt.strftime('%m')
        
        request = {
            'product_type': ['reanalysis'],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
            'variable': [
                'vertical_integral_of_eastward_water_vapour_flux', 
                'vertical_integral_of_northward_water_vapour_flux',
            ],
            'year': [year_str],
            'month': [month_str],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ]
        }
    
        client.retrieve(
            # dataset, request, target=out_fpath).download()
            dataset, request, target=out_fpath)
        
        now = dt.datetime.now()
        print(f'Finished {fname} at {now}')