"""
Take the ERA5 IVT at percentiles files for the 6 spatial sectors and "stitch" into a 
single file for each hemisphere (NH and SH), containing only the 85th IVT percentile
array (to reduce file size).
"""

import xarray as xr
import numpy as np

data_dir = '/run/media/kmattingly/kyle_8tb/AR_data_v2/ERA5_IVT_at_pctiles/'

hemis = ['NH','SH']
sectors = ['1','2','3']

hemi_lat_bounds = {'NH':'10_90',
                   'SH':'-90_-10'}

sector_lon_bounds = {'1':'0_120',
                     '2':'120.25_240',
                     '3':'240.25_359.75'}

for hemi in hemis:
    lats_str = hemi_lat_bounds[hemi]

    sector_arrays = {}
    sector_lons = {}
    
    for sector in sectors:
        lons_str = sector_lon_bounds[sector]
        fname = f'IVT_at_pctiles_ERA5_lat_{lats_str}_lon_{lons_str}_3hr_doys_1_365_1980_2020_climo.nc'
        
        ds = xr.open_dataset(data_dir+fname)
        sector_arrays[sector] = ds.IVT_pctile_85.to_numpy()
        sector_lons[sector] = ds.lon.to_numpy()
        lats = ds.lat.to_numpy()
        doys = ds.doy.to_numpy()
    
    full_hemi_IVT_pctile_85 = np.concatenate((sector_arrays['1'],
                                              sector_arrays['2'],
                                              sector_arrays['3']),
                                             axis=2)
    full_hemi_lons = np.concatenate((sector_lons['1'],
                                     sector_lons['2'],
                                     sector_lons['3']))
    
    stitched_ds = xr.Dataset(
        {
         'IVT_pctile_85':(('doy','lat','lon'), full_hemi_IVT_pctile_85.astype('float32'))
        },
        coords={
            'doy':doys.astype('uint16'),
            'lat':lats.astype('float32'),
            'lon':full_hemi_lons.astype('float32')
            }
    )
    
    stitched_ds.IVT_pctile_85.attrs['units'] = 'kg/m/s'
    stitched_ds.lat.attrs['units'] = 'degrees_north'
    stitched_ds.lon.attrs['units'] = 'degrees_east'  
    
    stitched_ds.attrs['data_source'] = ds.attrs['data_source']
    stitched_ds.attrs['IVT_data_origin'] = ds.attrs['IVT_data_origin']
    stitched_ds.attrs['IVT_vert_coord'] = ds.attrs['IVT_vert_coord']
    stitched_ds.attrs['IVT_climatology_start_year'] = ds.attrs['IVT_climatology_start_year']
    stitched_ds.attrs['IVT_climatology_end_year'] = ds.attrs['IVT_climatology_end_year']
    stitched_ds.attrs['IVT_climatology_timestep_hrs'] = ds.attrs['IVT_climatology_timestep_hrs']
    stitched_ds.attrs['IVT_calc_model_levels'] = ds.attrs['IVT_calc_model_levels']
    
    output_fname = f'IVT_at_pctiles_ERA5_{hemi}_3hr_1980_2020_climo.nc'
    stitched_ds.to_netcdf(data_dir+output_fname)