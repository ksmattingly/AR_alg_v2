"""
Functions specific to the ERA5 reanalysis to calculate integrated water vapor
transport (IVT) vector magnitude from the u/v-IVT components provided as an
analysis variable by ERA5.

The code assumes pre-existing files containing both the u/v-IVT components, and
adds IVT magnitude as a new variable to these files.
"""

import argparse
import hjson
import xarray as xr
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import datetime as dt
import glob
import warnings
import os

warnings.simplefilter('ignore', category=RuntimeWarning)


def calc_IVT_from_uvIVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str):
    """
    Calculate IVT magnitude from ERA5 data, which provides u- and v-IVT components as 
    pre-calculated variables.
    """
    
    begin_t = dt.datetime.strptime(begin_t_str, '%Y-%m-%d_%H%M')
    end_t = dt.datetime.strptime(end_t_str, '%Y-%m-%d_%H%M')
    timestep_hrs = int(timestep_hrs_str)
    times = pd.date_range(begin_t, end_t, freq=str(timestep_hrs)+'H')
    file_t = pd.date_range(begin_t, end_t, freq=AR_config['uvIVT_precalc_files_timestep'])
    
    uvIVT_fpath = glob.glob(AR_config['uvIVT_precalc_dir']+\
                            AR_config['uvIVT_precalc_fname_prefix']+\
                            '*'+file_t[0].strftime(AR_config['uvIVT_precalc_fname_date_format'])+'*.nc')[0]
    uvIVT_ds = xr.open_dataset(uvIVT_fpath, mask_and_scale=True)
    lats = uvIVT_ds.latitude.data
    lons = uvIVT_ds.longitude.data
    
    uIVT = np.empty((len(times), uvIVT_ds.latitude.shape[0], uvIVT_ds.longitude.shape[0]), dtype='float32')
    vIVT = np.empty((len(times), uvIVT_ds.latitude.shape[0], uvIVT_ds.longitude.shape[0]), dtype='float32')
    IVT = np.empty((len(times), uvIVT_ds.latitude.shape[0], uvIVT_ds.longitude.shape[0]), dtype='float32')
    
    for i, t in enumerate(times):
        # Assumes we are calculating over the full lat/lon grid of the original file
        uvIVT_ds_ts = uvIVT_ds.sel(time=t)
        uIVT_ts = uvIVT_ds_ts['p71.162']
        vIVT_ts = uvIVT_ds_ts['p72.162']
        IVT_ts = np.sqrt((uIVT_ts**2) + (vIVT_ts**2))
        
        uIVT[i,:,:] = uIVT_ts
        vIVT[i,:,:] = vIVT_ts
        IVT[i,:,:] = IVT_ts
    uvIVT_ds.close()
    
    return uvIVT_fpath, times, uIVT, vIVT, IVT, lats, lons


def write_IVT_output_file(AR_config, timestep_hrs_str, uvIVT_fpath, times,
                          uIVT, vIVT, IVT, lats, lons):
    """
    Write files including IVT magnitude to the same directory as the input ERA5
    uvIVT files.
    """
    output_fname = os.path.basename(uvIVT_fpath)
    output_fpath = AR_config['IVT_dir']+output_fname
    
    IVT_ds = xr.Dataset(
        {
         'uIVT':(('time','lat','lon'), uIVT),
         'vIVT':(('time','lat','lon'), vIVT),
         'IVT':(('time','lat','lon'), IVT)
        },
        coords={
            'time':times,
            'lat':lats,
            'lon':lons
        }
    )
    
    encoding = {
        'time':{'units':'hours since 1900-01-01', 'calendar': 'gregorian'},
    }

    IVT_ds.uIVT.attrs['units'] = 'kg/m/s'
    IVT_ds.vIVT.attrs['units'] = 'kg/m/s'
    IVT_ds.IVT.attrs['units'] = 'kg/m/s'
    IVT_ds.lat.attrs['units'] = 'degrees_north'
    IVT_ds.lon.attrs['units'] = 'degrees_east'  
    
    IVT_ds.attrs['data_source'] = AR_config['data_source']
    IVT_ds.attrs['IVT_data_origin'] = AR_config['IVT_data_origin']
    IVT_ds.attrs['IVT_vert_coord'] = AR_config['IVT_vert_coord']
    
    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        IVT_ds.attrs['IVT_calc_pressure_levels'] = str(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        IVT_ds.attrs['IVT_calc_model_levels'] = str(AR_config['IVT_calc_mlevs'])

    IVT_ds.to_netcdf(output_fpath, encoding=encoding)

    # with Dataset(uvIVT_fpath, 'r+') as nc:
    #     IVT_out = nc.createVariable('IVT', 'float32', ('time','latitude','longitude',))
    #     IVT_out.units = 'kg m**-1 s**-1'
    #     IVT_out.long_name = 'Magnitude of vertically integrated water vapor transport'
    #     IVT_out[:] = IVT
        
    #     nc.setncattr('data_source', AR_config['data_source'])
    #     nc.setncattr('IVT_data_origin', AR_config['IVT_data_origin'])
    #     nc.setncattr('IVT_vert_coord', AR_config['IVT_vert_coord'])
        

def parse_args():
    """
    Parse arguments passed to script at runtime.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('begin_time', help='Begin time in the format YYYY-MM-DD_HHMM')
    parser.add_argument('end_time', help='End time in the format YYYY-MM-DD_HHMM')
    parser.add_argument('timestep', help='Timestep as integer number of hours (e.g. 3)')
    parser.add_argument('AR_ID_config_path', help='Path to AR ID configuration file')
    args = parser.parse_args()
    
    return args.begin_time, args.end_time, args.timestep, args.AR_ID_config_path


def main():
    begin_t_str, end_t_str, timestep_hrs_str, AR_ID_config_path = parse_args()
    
    with open(AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())
    
    # uvIVT_fname, times, IVT = calc_IVT_from_uvIVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str)
    uvIVT_fpath, times, uIVT, vIVT, IVT, lats, lons = \
        calc_IVT_from_uvIVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str)
    # write_IVT_output_file(AR_config, timestep_hrs_str, uvIVT_fname, times, IVT)
    write_IVT_output_file(AR_config, timestep_hrs_str, uvIVT_fpath, times,
                          uIVT, vIVT, IVT, lats, lons)


if __name__ == '__main__':
    main()