"""
Download ERA5 u/v-IVT data, 1-hourly resolution, global, for 1980-2022.
"""

import cdsapi
import numpy as np
import datetime as dt

c = cdsapi.Client()

outdir = '/doppler/data7/kmattingly/ERA5_uvIVT_global_1hr_from-model-levs/'

years = np.arange(1980,2022,1)
months = np.arange(1,13,1)

for year in years:
    for month in months:
        yrmth_dt = dt.datetime(year,month,1,0)
        year_str = yrmth_dt.strftime('%Y')
        month_str = yrmth_dt.strftime('%m')
        
        fname = 'ERA5_uvIVT_1hr_global_'+yrmth_dt.strftime('%Y_%m.nc')
        
        c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'vertical_integral_of_eastward_water_vapour_flux', 'vertical_integral_of_northward_water_vapour_flux',
        ],
        'month': month_str,
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
        ],
        'year': year_str,
    },
    outdir+fname)
        
    print('Finished '+fname+' at '+str(dt.datetime.now()))