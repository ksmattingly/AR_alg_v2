import os
import hjson
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import glob
import sys
import warnings

warnings.simplefilter('ignore', category=RuntimeWarning)

g = 9.80665

def calc_IVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str, calc_ll_mean_wind=False):
        
    begin_t = dt.datetime.strptime(begin_t_str, '%Y-%m-%d_%H%M')
    end_t = dt.datetime.strptime(end_t_str, '%Y-%m-%d_%H%M')
    timestep_hrs = int(timestep_hrs_str)
    
    times = pd.date_range(begin_t, end_t, freq=str(timestep_hrs)+'H')
    
    # Narrow the list of input quv files to only the approximate time span of files
    # needed to calculate IVT for the time range covered by (begin_t, end_t)
    # - The purpose of this step is to shorten the list of files read in as a mfdataset
    #   by xarray, which will hopefully save processing time
    # - This assumes that input files follow CF conventions and can be read as a mfdataset
    quv_file_times = pd.date_range(begin_t, end_t, freq=AR_config['quv_files_timestep'])
    quv_flist = []
    for t in quv_file_times:
        quv_fname = glob.glob(AR_config['quv_dir']+\
                              AR_config['quv_fname_prefix']+\
                              '*'+t.strftime(AR_config['quv_fname_date_format'])+'*.nc')[0]
        quv_flist.append(quv_fname)
    
    quv_ds = xr.open_mfdataset(quv_flist, mask_and_scale=True)
    lats = np.array(quv_ds.coords['lat'])
    lons = np.array(quv_ds.coords['lon'])
    
    # Make sure number of vertical levels (model or pressure levels) matches 
    # the number given in the AR configuration
    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        nlevs_expected = len(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels': 
        nlevs_expected = AR_config['num_model_levels']
    if nlevs_expected != len(quv_ds.lev):
        raise Exception('Number of vertical levels in input data does not match configuration')
    
    uIVT = np.empty((len(times), quv_ds.dims['lat'], quv_ds.dims['lon']))
    vIVT = np.empty((len(times), quv_ds.dims['lat'], quv_ds.dims['lon']))
    IVT = np.empty((len(times), quv_ds.dims['lat'], quv_ds.dims['lon']))
    if calc_ll_mean_wind:
        ll_mean_u_wind = np.empty((len(times), quv_ds.dims['lat'], quv_ds.dims['lon']))
        ll_mean_v_wind = np.empty((len(times), quv_ds.dims['lat'], quv_ds.dims['lon']))
    
    for i, t in enumerate(times):
        quv_ds_ts = quv_ds.sel(time=t)
        if calc_ll_mean_wind:
            uIVT[i,:,:], vIVT[i,:,:], IVT[i,:,:], ll_mean_u_wind[i,:,:], ll_mean_v_wind[i,:,:] = \
                _IVT_ll_mean_wind_calc_pres_levs(quv_ds_ts)
        else:
            uIVT[i,:,:], vIVT[i,:,:], IVT[i,:,:] = _IVT_calc_model_levs(quv_ds_ts)
        print('Finished calculating time '+str(t)+' at '+str(dt.datetime.now()))
    
    quv_ds.close()
    
    if calc_ll_mean_wind:
        return times, lats, lons, uIVT, vIVT, IVT, ll_mean_u_wind, ll_mean_v_wind
    else:
        return times, lats, lons, uIVT, vIVT, IVT

def _IVT_calc_model_levs(quv_ds_ts):
    # ** This method (including variable names and pre-existing delta-p) works for MERRA-2 only.
    # Will need adapting for any other datasets where IVT is calculated on model levels.
    # - Possibly add a dictionary with variable names in the AR_ID_config.hjson
    uIVT_ts = np.nansum(quv_ds_ts['U']*quv_ds_ts['QV']*quv_ds_ts['DELP'], axis=0)/g
    vIVT_ts = np.nansum(quv_ds_ts['V']*quv_ds_ts['QV']*quv_ds_ts['DELP'], axis=0)/g
    
    IVT_ts = np.sqrt((uIVT_ts**2) + (vIVT_ts**2))
    
    return uIVT_ts, vIVT_ts, IVT_ts
    
def _IVT_ll_mean_wind_calc_pres_levs(quv_ds_ts):
    # ** This method (including variable names and calculation of delta-p in Pa) works for MERRA-2 only.
    # Will need adapting for any other datasets where IVT is calculated on model levels.
    # - Possibly add a dictionary with variable names in the AR_ID_config.hjson
    plevs = list(quv_ds_ts.lev.data)
    delp = [(plevs[i+1] - plevs[i])*-100 for i in range(len(plevs) - 1)]
    delp.append(5000.0)
        
    uIVT_ts = np.nansum(quv_ds_ts['U']*quv_ds_ts['QV']*np.array(delp)[:,np.newaxis,np.newaxis], axis=0)/g
    vIVT_ts = np.nansum(quv_ds_ts['V']*quv_ds_ts['QV']*np.array(delp)[:,np.newaxis,np.newaxis], axis=0)/g
    
    IVT_ts = np.sqrt((uIVT_ts**2) + (vIVT_ts**2))
    
    # ** Assumes pressure levels are incremented every 25 hPa from 1000 to 700 hPa
    ll_mean_u_wind_ts = np.nanmean(quv_ds_ts['U'][:13], axis=0)
    ll_mean_v_wind_ts = np.nanmean(quv_ds_ts['V'][:13], axis=0)
    
    return uIVT_ts, vIVT_ts, IVT_ts, ll_mean_u_wind_ts, ll_mean_v_wind_ts

def write_IVT_output_file(AR_config, timestep_hrs_str, times, lats, lons, uIVT, vIVT, IVT):
        
    IVT_ds = xr.Dataset(
        {
         'uIVT':(('time','lat','lon'), uIVT),
         'vIVT':(('time','lat','lon'), vIVT),
         'IVT':(('time','lat','lon'), IVT),
        },
        coords={
            'time':times,
            'lat':lats,
            'lon':lons
        }
    )

    encoding = {
        'time':{'units':'hours since 1900-01-01'},
    }
    
    IVT_ds.uIVT.attrs['units'] = 'kg/m/s'
    IVT_ds.vIVT.attrs['units'] = 'kg/m/s'
    IVT_ds.IVT.attrs['units'] = 'kg/m/s'
    IVT_ds.lat.attrs['units'] = 'degrees_north'
    IVT_ds.lon.attrs['units'] = 'degrees_east'  
    
    IVT_ds.attrs['data_source'] = AR_config['data_source']
    IVT_ds.attrs['IVT_vert_coord'] = AR_config['IVT_vert_coord']
    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        IVT_ds.attrs['IVT_calc_plevs'] = str(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        IVT_ds.attrs['num_model_levels'] = str(AR_config['num_model_levels'])
        
    t_begin_str = times[0].strftime('%Y%m%d%H%M')
    t_end_str = times[-1].strftime('%Y%m%d%H%M')

    # Include "_subset" in file name if area given by AR_config is not an entire hemisphere poleward of 10 degrees latitude
    # Example file name: IVT_MERRA2_SH[_subset]_3hr_2021010100_2021013121.nc
    if (AR_config['min_lat'] == 10) and (AR_config['max_lat'] == 90) and (AR_config['min_lon'] == -180) and (AR_config['max_lon'] == 180):
        fname = 'IVT_'+AR_config['data_source']+'_'+AR_config['hemisphere']+'H_'+timestep_hrs_str+'hr_'+t_begin_str+'_'+t_end_str+'.nc'
    else:
        fname = 'IVT_'+AR_config['data_source']+'_'+AR_config['hemisphere']+'H_subset_'+timestep_hrs_str+'hr_'+t_begin_str+'_'+t_end_str+'.nc'
        
    IVT_ds.to_netcdf(AR_config['IVT_dir']+fname, encoding=encoding)
    
    
def write_ll_mean_wind_output_file(AR_config, timestep_hrs_str, times, lats, lons, ll_mean_u_wind, ll_mean_v_wind):
    
    ll_mean_wind_ds = xr.Dataset(
        {
         'u':(('time','lat','lon'), ll_mean_u_wind),
         'v':(('time','lat','lon'), ll_mean_v_wind),
        },
        coords={
            'time':times,
            'lat':lats,
            'lon':lons
        }
    )
    
    encoding = {
        'time':{'units':'hours since 1900-01-01'},
    }

    ll_mean_wind_ds.attrs['data_source'] = AR_config['data_source']
    ll_mean_wind_ds.u.attrs['units'] = 'm/s'
    ll_mean_wind_ds.v.attrs['units'] = 'm/s'
    ll_mean_wind_ds.lat.attrs['units'] = 'degrees_north'
    ll_mean_wind_ds.lon.attrs['units'] = 'degrees_east'
    
    ll_mean_wind_ds.u.attrs['longname'] = '1000-700 hPa mean u-wind'
    ll_mean_wind_ds.v.attrs['longname'] = '1000-700 hPa mean v-wind'
    
    t_begin_str = times[0].strftime('%Y%m%d%H%M')
    t_end_str = times[-1].strftime('%Y%m%d%H%M')
    
    # Include "_subset" in file name if area given by AR_config is not an entire hemisphere poleward of 10 degrees latitude
    # Example file name: IVT_MERRA2_SH[_subset]_3hr_2021010100_2021013121.nc
    if (AR_config['min_lat'] == 10) and (AR_config['max_lat'] == 90) and (AR_config['min_lon'] == -180) and (AR_config['max_lon'] == 180):
        fname = 'mean_wind_1000_700_hPa_'+AR_config['data_source']+'_'+AR_config['hemisphere']+'H_'+timestep_hrs_str+'hr_'+t_begin_str+'_'+t_end_str+'.nc'
    else:
        fname = 'mean_wind_1000_700_hPa_'+AR_config['data_source']+'_'+AR_config['hemisphere']+'H_subset_'+timestep_hrs_str+'hr_'+t_begin_str+'_'+t_end_str+'.nc'
        
    ll_mean_wind_ds.to_netcdf(AR_config['wind_1000_700_mean_dir']+fname, encoding=encoding)


if __name__ == '__main__':    
    if len(sys.argv) != 4:
        print('Usage: python calc_IVT.py <begin_time> <end_time> <timestep>')
        print('<begin_time> and <end_time> must be in the format YYYY-MM-DD_HHMM')
        print('<timestep> must be an integer number of hours (e.g. 3)')
        sys.exit()
    
    begin_t_str = sys.argv[1]
    end_t_str = sys.argv[2]
    timestep_hrs_str = sys.argv[3]

    _code_dir = os.path.dirname(os.path.realpath(__file__))
    AR_ID_config_path = _code_dir+'/AR_ID_config.hjson'
    with open(AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())
        
    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        times, lats, lons, uIVT, vIVT, IVT, ll_mean_u_wind, ll_mean_v_wind = \
            calc_IVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str,
                     calc_ll_mean_wind=True)
        write_ll_mean_wind_output_file(AR_config, timestep_hrs_str, times, lats, lons, ll_mean_u_wind, ll_mean_v_wind)
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        times, lats, lons, uIVT, vIVT, IVT = \
            calc_IVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str)

    write_IVT_output_file(AR_config, timestep_hrs_str, times, lats, lons, uIVT, vIVT, IVT)