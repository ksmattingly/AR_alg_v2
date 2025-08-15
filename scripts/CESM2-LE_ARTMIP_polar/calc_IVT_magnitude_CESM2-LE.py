"""
Functions specific to CESM2-LE data to calculate integrated water vapor
transport (IVT) vector magnitude from the u/v-IVT components provided as an
analysis variable in source files.

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


def calc_IVT_from_uvIVT(AR_config, ens_member):
    """
    Calculate IVT magnitude from u- and v-IVT components.
    """
    
    # begin_t = dt.datetime.strptime(begin_t_str, '%Y-%m-%d_%H%M')
    # end_t = dt.datetime.strptime(end_t_str, '%Y-%m-%d_%H%M')
    # timestep_hrs = int(timestep_hrs_str)
    # times = pd.date_range(begin_t, end_t, freq=str(timestep_hrs)+'H')
    # file_t = pd.date_range(begin_t, end_t, freq=AR_config['uvIVT_precalc_files_timestep'])
    
    prefix = AR_config['uvIVT_precalc_fname_prefix']
    uIVT_fpaths = sorted(glob.glob(os.path.join(
        AR_config['uvIVT_precalc_dir'], f'{prefix}{ens_member}*uIVT*.nc')))
    uIVT_ds = xr.open_mfdataset(uIVT_fpaths)
    uIVT_ds['time'] = uIVT_ds.indexes['time'].to_datetimeindex()

    vIVT_fpaths = sorted(glob.glob(os.path.join(
        AR_config['uvIVT_precalc_dir'], f'{prefix}{ens_member}*vIVT*.nc')))
    vIVT_ds = xr.open_mfdataset(vIVT_fpaths)
    vIVT_ds['time'] = vIVT_ds.indexes['time'].to_datetimeindex()
    
        # AR_config['uvIVT_precalc_dir']+\
        # AR_config['uvIVT_precalc_fname_prefix']+\
        # '*'+file_t[0].strftime(AR_config['uvIVT_precalc_fname_date_format'])+'*.nc')[0]
    # uvIVT_fpath = glob.glob(AR_config['uvIVT_precalc_dir']+\
    #                         AR_config['uvIVT_precalc_fname_prefix']+\
    #                         '*'+file_t[0].strftime(AR_config['uvIVT_precalc_fname_date_format'])+'*.nc')[0]
    # uvIVT_ds = xr.open_dataset(uvIVT_fpath, mask_and_scale=True)
    lats = uIVT_ds.lat.data
    lons = uIVT_ds.lon.data
    
    times = pd.to_datetime(vIVT_ds['time'].data)

    # Write out a separate file for each month
    t1_all = times[0]
    t2_all = times[-1]

    times_monthly_all = pd.date_range(t1_all, t2_all, freq='1M')

    for t_monthly in times_monthly_all:
        t1_month = dt.datetime(t_monthly.year, t_monthly.month, 1, 0)
        t2_month = dt.datetime(
            t_monthly.year, t_monthly.month, t_monthly.day, 18)
        
        # Filter out leap days because they're not included in CESM2-LE files
        if (t2_month.month == 2) and (t2_month.day == 29):
            t2_month = dt.datetime(t_monthly.year, t_monthly.month, 28, 18)

        times_monthly = pd.date_range(t1_month, t2_month, freq='6h')

        uIVT = np.empty(
            (len(times_monthly), uIVT_ds.lat.shape[0], uIVT_ds.lon.shape[0]),
            dtype='float32')
        vIVT = np.empty(
            (len(times_monthly), uIVT_ds.lat.shape[0], uIVT_ds.lon.shape[0]),
            dtype='float32')
        IVT = np.empty(
            (len(times_monthly), uIVT_ds.lat.shape[0], uIVT_ds.lon.shape[0]),
            dtype='float32')

        for i, t in enumerate(times_monthly):
            # Assumes we are calculating over the full lat/lon grid of the
            # original file
            uIVT_ds_ts = uIVT_ds.sel(time=t)
            vIVT_ds_ts = vIVT_ds.sel(time=t)
            uIVT_ts = uIVT_ds_ts['uIVT']
            vIVT_ts = vIVT_ds_ts['vIVT']

            IVT_ts = np.sqrt((uIVT_ts**2) + (vIVT_ts**2))
            
            uIVT[i,:,:] = uIVT_ts
            vIVT[i,:,:] = vIVT_ts
            IVT[i,:,:] = IVT_ts
        
        # Call write_IVT_output_file function to write output file
        out_tstr = t1_month.strftime('%Y%m%d%H%M')+'_'+\
            t2_month.strftime('%Y%m%d%H%M')
        output_fname = f'IVT_CESM2-LE-{ens_member}_global_6hr_{out_tstr}.nc'
        write_IVT_output_file(
            AR_config, output_fname, times_monthly, uIVT, vIVT, IVT,
            lats, lons, ens_member)
        
        now = dt.datetime.now()
        tstr_month = t1_month.strftime('%Y-%m')
        print(f'Finished {tstr_month} at {now}')
    
    # return uvIVT_fpath, times, uIVT, vIVT, IVT, lats, lons
    # return times, uIVT, vIVT, IVT, lats, lons


def write_IVT_output_file(AR_config, output_fname, times_monthly, uIVT, vIVT,
                          IVT, lats, lons, ens_member):
    """
    Write files including IVT magnitude to the same directory as the input
    uvIVT files.
    """
    # output_fname = os.path.basename(uvIVT_fpath)
    output_fpath = os.path.join(AR_config['IVT_dir'], output_fname)
    
    IVT_ds = xr.Dataset(
        {
         'uIVT':(('time','lat','lon'), uIVT),
         'vIVT':(('time','lat','lon'), vIVT),
         'IVT':(('time','lat','lon'), IVT)
        },
        coords={
            'time':times_monthly,
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
    IVT_ds.attrs['ens_member'] = ens_member
    
    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        IVT_ds.attrs['IVT_calc_pressure_levels'] = str(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        IVT_ds.attrs['IVT_calc_model_levels'] = str(AR_config['IVT_calc_mlevs'])

    IVT_ds.to_netcdf(output_fpath, encoding=encoding)
        

def parse_args():
    """
    Parse arguments passed to script at runtime.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('ens_member', help='CESM2-LE member number')
    parser.add_argument('AR_ID_config_path', help='Path to AR ID configuration file')
    args = parser.parse_args()
    
    return args.ens_member, args.AR_ID_config_path


def main():
    ens_member, AR_ID_config_path = parse_args()
    
    with open(AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())
    
    # uvIVT_fname, times, IVT = calc_IVT_from_uvIVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str)
    # uvIVT_fpath, times, uIVT, vIVT, IVT, lats, lons = \
    # times, uIVT, vIVT, IVT, lats, lons = \
    #     calc_IVT_from_uvIVT(AR_config, ens_member)
    # write_IVT_output_file(AR_config, timestep_hrs_str, uvIVT_fname, times, IVT)
    # write_IVT_output_file(
    #     AR_config, timestep_hrs_str, uvIVT_fpath, times, uIVT, vIVT, IVT, lats,
    #     lons, ens_member)

    calc_IVT_from_uvIVT(AR_config, ens_member)

if __name__ == '__main__':
    main()