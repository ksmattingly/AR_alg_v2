"""
Calculate IVT at the climatological percentile rank values specified in the
configuration file.

The percentile ranks are calculated from the distribution of all values in a
31-day centered window at each analysis grid cell for the times specified by
the IVT_climatology_start_year, IVT_climatology_end_year, and
IVT_climatology_timestep_hrs in the configuration file.
"""

import argparse
import glob
import hjson
import datetime as dt
from datetime import timedelta
import xarray as xr
import numpy as np
import pandas as pd

from AR_alg_v2.misc_utils import rename_coords


def calc_IVT_at_pctiles(AR_config, start_year, end_year, timestep_hrs, start_doy, end_doy):
    """
    Calculate IVT values at a range of percentiles for each julian day of the year.
    """
    
    IVT_dir = AR_config['IVT_dir']
    IVT_paths = sorted(glob.glob(IVT_dir+'*.nc'))
    years_str = [str(year) for year in np.arange(int(start_year), int(end_year)+1, 1)]
    IVT_paths_filtered = [p for p in IVT_paths if p[-15:-11] in years_str]
    IVT_ds_full = rename_coords(xr.open_mfdataset(IVT_paths_filtered))
    IVT_ds_full['time'] = pd.to_datetime(IVT_ds_full.time.data)
        
    begin_dt = dt.datetime(int(start_year), 1, 1, 0)
    end_dt = dt.datetime(int(end_year), 12, 31, 21)
    times = pd.date_range(begin_dt, end_dt, freq=f'{timestep_hrs}H')
    
    lats_subset = np.arange(
        AR_config['min_lat'],
        AR_config['max_lat']+AR_config['lat_res'],
        AR_config['lat_res']
        )
    lons = IVT_ds_full.lon.data[:]
        
    IVT_ds = IVT_ds_full.IVT.sel(time=times, lat=lats_subset)

    # For leap years, subtract 1 from all doys after the leap day to get corrected doy
    leap_years = np.arange(1900, int(end_year)+1, 4)
    doys_fixed = []
    for t in pd.to_datetime(IVT_ds.time):
        doy = t.timetuple().tm_yday
        if (t.year in leap_years) and (doy >= 60):
            doys_fixed.append((doy - 1))
        else:
            doys_fixed.append(doy)
    IVT_ds['doy'] = xr.DataArray(doys_fixed, coords=[IVT_ds.time], dims=['time'])
    
    doys = np.arange(int(start_doy),int(end_doy)+1,1)
    pctiles = np.array(AR_config['IVT_percentiles_to_calc'])
    IVT_pctiles_out_data = np.empty(
        (len(doys), len(pctiles), IVT_ds.lat.shape[0], IVT_ds.lon.shape[0])
        )
    
    for i, doy in enumerate(doys):
        # "Dummy" time will only be used to get doy of window start and end
        # - For dummy time, use a year where neither the year before or after
        #   is a leap year
        t_dummy = dt.datetime(1902,1,1,0) + timedelta(days = int(doy-1))
        window_start_doy = (t_dummy - timedelta(days=15)).timetuple().tm_yday
        window_end_doy = (t_dummy + timedelta(days=15)).timetuple().tm_yday
        
        # Handle the case where window starts at the end of one year and
        # continues into the next
        if window_start_doy > window_end_doy:
            window_doys = np.concatenate((np.arange(window_start_doy, 366, 1),
                                          np.arange(1, window_end_doy + 1, 1)))
        else:
            window_doys = np.arange(window_start_doy, window_end_doy + 1, 1)
            
        IVT_ds_window = IVT_ds.sel(time=IVT_ds.time.dt.dayofyear.isin(window_doys))
        IVT_at_pctiles_doy = IVT_ds_window.chunk(dict(time=-1)).quantile(
            pctiles/100, dim='time', skipna=True
            )
        IVT_pctiles_out_data[i,:,:,:] = IVT_at_pctiles_doy.to_numpy()
        
        print(f'Finished doy: {doy} at '+str(dt.datetime.now()))

    return doys, pctiles, IVT_ds.lat.data[:], lons, IVT_pctiles_out_data


def write_IVT_at_pctiles_output_file(AR_config, start_year, end_year, timestep_hrs,
                                     doys, pctiles, lats, lons, IVT_pctiles_out_data):
    """
    Write IVT at percentiles output file with metadata supplied by the AR ID
    configuration.
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
        IVT_at_pctiles_ds['IVT_pctile_'+str(pctile)] = \
            (['doy','lat','lon'], IVT_pctiles_out_data[:,i,:,:])
        IVT_at_pctiles_ds['IVT_pctile_'+str(pctile)].attrs['units'] = 'kg/m/s'
    
    IVT_at_pctiles_ds.lat.attrs['units'] = 'degrees_north'
    IVT_at_pctiles_ds.lon.attrs['units'] = 'degrees_east'  
    
    IVT_at_pctiles_ds.attrs['data_source'] = AR_config['data_source']
    IVT_at_pctiles_ds.attrs['IVT_data_origin'] = AR_config['IVT_data_origin']
    IVT_at_pctiles_ds.attrs['IVT_vert_coord'] = AR_config['IVT_vert_coord']
    IVT_at_pctiles_ds.attrs['IVT_climatology_start_year'] = start_year
    IVT_at_pctiles_ds.attrs['IVT_climatology_end_year'] = end_year
    IVT_at_pctiles_ds.attrs['IVT_climatology_timestep_hrs'] = timestep_hrs

    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        IVT_at_pctiles_ds.attrs['IVT_calc_pressure_levels'] = \
            str(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        IVT_at_pctiles_ds.attrs['IVT_calc_model_levels'] = \
            str(AR_config['IVT_calc_mlevs'])
        
    minlat = AR_config['min_lat']
    maxlat = AR_config['max_lat']
    minlon = AR_config['min_lon']
    maxlon = AR_config['max_lon']
    lonres = AR_config['lon_res']
    start_doy = str(doys[0])
    end_doy = str(doys[-1])

    if ((minlat == -90) and (maxlat == 90) and ((maxlon+lonres) - minlon == 360)) and\
        (start_doy == '1' and end_doy == '365'):
        fname = 'IVT_at_pctiles_'+AR_config['data_source']+\
            f'_global_{timestep_hrs}hr_{start_year}_{end_year}_climo.nc'    
    elif ((minlat == 10) and (maxlat == 90) and ((maxlon+lonres) - minlon == 360)) and\
        (start_doy == '1' and end_doy == '365'):
        fname = 'IVT_at_pctiles_'+AR_config['data_source']+\
            f'_NH_{timestep_hrs}hr_{start_year}_{end_year}_climo.nc'  
    elif ((minlat == -90) and (maxlat == -10) and ((maxlon+lonres) - minlon == 360)) and\
        (start_doy == '1' and end_doy == '365'):
        fname = 'IVT_at_pctiles_'+AR_config['data_source']+\
            f'_SH_{timestep_hrs}hr_{start_year}_{end_year}_climo.nc'  
    else:
        fname = 'IVT_at_pctiles_'+AR_config['data_source']+\
            f'_lat_{minlat}_{maxlat}_lon_{minlon}_{maxlon}_{timestep_hrs}hr_doys_{start_doy}_{end_doy}_{start_year}_{end_year}_climo.nc' 
    
    IVT_at_pctiles_ds.to_netcdf(AR_config['IVT_at_pctiles_dir']+fname)


def parse_args():
    """
    Parse arguments passed to script at runtime.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'start_doy', 
        help='Day of year (1-365) at which to start calculations'
        )
    parser.add_argument(
        'end_doy',
        help='Day of year (1-365) at which to end calculations'
        )
    parser.add_argument(
        'AR_ID_config_path',
        help='Path to AR ID configuration file'
        )
    args = parser.parse_args()
    
    return args.start_doy, args.end_doy, args.AR_ID_config_path


def main():
    start_doy, end_doy, AR_ID_config_path = parse_args()
        
    with open(AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())
    
    start_year = AR_config['IVT_climatology_start_year']
    end_year = AR_config['IVT_climatology_end_year']
    timestep_hrs = AR_config['IVT_climatology_timestep_hrs']
    
    doys, pctiles, lats, lons, IVT_pctiles_out_data = calc_IVT_at_pctiles(
        AR_config, start_year, end_year, timestep_hrs, start_doy, end_doy
        )
    
    write_IVT_at_pctiles_output_file(
        AR_config, start_year, end_year, timestep_hrs, doys, pctiles,
        lats, lons, IVT_pctiles_out_data
        )


if __name__ == '__main__':
    main()