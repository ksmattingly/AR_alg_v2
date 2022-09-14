import os
import sys
import glob
import hjson
import datetime as dt
from datetime import timedelta
import xarray as xr
import numpy as np
import pandas as pd


def calc_IVT_at_pctiles(AR_config, start_year, end_year, timestep_hrs_str):
    """
    Calculate IVT values at a range of percentiles for each julian day of the year.
    """
    
    IVT_dir = AR_config['IVT_dir']
    IVT_paths = glob.glob(IVT_dir+'*.nc')
    IVT_ds_full = xr.open_mfdataset(IVT_paths)
    IVT_ds_full['time'] = pd.to_datetime(IVT_ds_full.time.data)
    
    begin_t = dt.datetime(int(start_year), 1, 1, 0)
    end_t = dt.datetime(int(end_year), 12, 31, 21)
    timestep_hrs = int(timestep_hrs_str)
    times = pd.date_range(begin_t, end_t, freq=str(timestep_hrs)+'H')
    
    lats_subset = np.arange(AR_config['min_lat'], AR_config['max_lat']+AR_config['lat_res'], AR_config['lat_res'])
    lons_subset = np.arange(AR_config['min_lon'], AR_config['max_lon']+AR_config['lon_res'], AR_config['lon_res'])
    IVT_ds = IVT_ds_full.sel(time=times, lat=lats_subset, lon=lons_subset)
    
    # For leap years, subtract 1 from all doys after the leap day to get corrected doy
    leap_years = np.arange(1900,2100,4)
    doys_fixed = []
    for t in pd.to_datetime(IVT_ds.time):
        doy = t.timetuple().tm_yday
        if (t.year in leap_years) and (doy >= 60):
            doys_fixed.append((doy - 1))
        else:
            doys_fixed.append(doy)
    IVT_ds['doy'] = xr.DataArray(doys_fixed, coords=[IVT_ds.time], dims=['time'])
    
    doys = np.arange(1,366,1)
    pctiles = np.array([1,5,10,15,20,25,50,75,80,85,90,95,99])
    IVT_pctiles_out_data = np.empty((len(doys), len(pctiles), IVT_ds.lat.shape[0], IVT_ds.lon.shape[0]))
    
    for i, doy in enumerate(doys):
        # "Dummy" time will only be used to get doy of window start and end
        # - For dummy time, use a year where neither the year before or after is a leap year
        t_dummy = dt.datetime(1902,1,1,0) + timedelta(days = int(doy-1))
        window_start_doy = (t_dummy - timedelta(days=15)).timetuple().tm_yday
        window_end_doy = (t_dummy + timedelta(days=15)).timetuple().tm_yday
        
        # Handle the case where window starts at the end of one year and continues into the next
        if window_start_doy > window_end_doy:
            window_doys = np.concatenate((np.arange(window_start_doy, 366, 1),
                                          np.arange(1, window_end_doy + 1, 1)))
        else:
            window_doys = np.arange(window_start_doy, window_end_doy + 1, 1)
            
        IVT_ds_window = IVT_ds.sel(time=IVT_ds.time.dt.dayofyear.isin(window_doys))
        IVT_at_pctiles_doy = IVT_ds_window.IVT.chunk(dict(time=-1)).quantile(pctiles/100, dim='time', skipna=True)
        IVT_pctiles_out_data[i,:,:,:] = IVT_at_pctiles_doy.to_numpy()
        
        print(f'Finished doy: {doy} at '+str(dt.datetime.now()))

    return doys, pctiles, IVT_ds.lat.data, IVT_ds.lon.data, IVT_pctiles_out_data


def write_IVT_at_pctiles_output_file(AR_config, start_year, end_year, timestep_hrs_str,
                                     doys, pctiles, lats, lons, IVT_pctiles_out_data):
    """
    Write IVT at percentiles output file with metadata supplied by the AR ID configuration.
    """
    
    IVT_at_pctiles_ds = xr.Dataset(
        data_vars={},
        coords=dict(
            doy=doys,
            lat=lats,
            lon=lons
            )
    )
    
    # Write each percentile as a separate output variable
    for i, pctile in enumerate(pctiles):
        IVT_at_pctiles_ds['IVT_pctile_'+str(pctile)] = (['doy','lat','lon'], IVT_pctiles_out_data[:,i,:,:])
        IVT_at_pctiles_ds['IVT_pctile_'+str(pctile)].attrs['units'] = 'kg/m/s'
    
    IVT_at_pctiles_ds.attrs['data_source'] = AR_config['data_source']
    IVT_at_pctiles_ds.attrs['IVT_vert_coord'] = AR_config['IVT_vert_coord']
    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        IVT_at_pctiles_ds.attrs['IVT_calc_pressure_levels'] = str(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        IVT_at_pctiles_ds.attrs['IVT_calc_model_levels'] = str(AR_config['IVT_calc_mlevs'])

    if (AR_config['min_lat'] == -90) and (AR_config['max_lat'] == 90) and (AR_config['min_lon'] == -180) and (AR_config['max_lon'] == 179.375):
        fname = 'IVT_at_pctiles_'+AR_config['data_source']+'_global_'+timestep_hrs_str+'hr_'+start_year+'_'+end_year+'_climo.nc'    
    elif (AR_config['min_lat'] == 10) and (AR_config['max_lat'] == 90) and (AR_config['min_lon'] == -180) and (AR_config['max_lon'] == 179.375):
        fname = 'IVT_at_pctiles_'+AR_config['data_source']+'_NH_'+timestep_hrs_str+'hr_'+start_year+'_'+end_year+'_climo.nc'
    elif (AR_config['min_lat'] == -90) and (AR_config['max_lat'] == -10) and (AR_config['min_lon'] == -180) and (AR_config['max_lon'] == 179.375):
        fname = 'IVT_at_pctiles_'+AR_config['data_source']+'_SH_'+timestep_hrs_str+'hr_'+start_year+'_'+end_year+'_climo.nc'
    else:
        fname = 'IVT_at_pctiles_'+AR_config['data_source']+'_subset_'+timestep_hrs_str+'hr_'+start_year+'_'+end_year+'_climo.nc'
    
    IVT_at_pctiles_ds.to_netcdf(AR_config['IVT_PR_dir']+fname)
        

if __name__ == '__main__':
    # ** Change this to use argparse
    if len(sys.argv) != 4:
        print('Usage: python calc_IVT_PR.py <start_year> <end_year> <timestep>')
        print('<timestep> must be an integer number of hours (e.g. 3)')
        sys.exit()
    
    start_year = sys.argv[1]
    end_year = sys.argv[2]
    timestep_hrs_str = sys.argv[3]
        
    _code_dir = os.path.dirname(os.path.realpath(__file__))
    AR_ID_config_path = _code_dir+'/AR_ID_config.hjson'
    with open(AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())
    
    doys, pctiles, lats, lons, IVT_pctiles_out_data = calc_IVT_at_pctiles(AR_config,
                                                                          start_year,
                                                                          end_year,
                                                                          timestep_hrs_str)
    write_IVT_at_pctiles_output_file(AR_config, start_year, end_year, timestep_hrs_str,
                                     doys, pctiles, lats, lons, IVT_pctiles_out_data)