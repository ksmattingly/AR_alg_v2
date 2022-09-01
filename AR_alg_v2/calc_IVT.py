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

def calc_IVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str, use_opendap=False,
             calc_ll_mean_wind=False, netrc_path='/home/kmattingly/.netrc'):
    """
    Calculate arrays of IVT and its u- and v-components over the time range
    given by the begin time, end time, and timestep.
    
    Currently only MERRA-2 IVT calculation is supported. MERRA-2 IVT can be
    calculated on pressure levels or on model levels, using data stored locally
    or remotely (using OPeNDAP). Note that the proper configuration files are
    needed to access data remotely - see comments below.
    
    If pressure level data are used to calculate IVT and calc_ll_mean_wind is True,
    also calculate 1000-700 hPa mean wind.
    
    Return geolocation (lat, lon) and time arrays in addition to the IVT arrays.
    """
    
    begin_t = dt.datetime.strptime(begin_t_str, '%Y-%m-%d_%H%M')
    end_t = dt.datetime.strptime(end_t_str, '%Y-%m-%d_%H%M')
    timestep_hrs = int(timestep_hrs_str)
    times = pd.date_range(begin_t, end_t, freq=str(timestep_hrs)+'H')
    quv_file_times = pd.date_range(begin_t, end_t, freq=AR_config['quv_files_timestep'])
    
    lats = np.arange(AR_config['min_lat'], AR_config['max_lat']+AR_config['lat_res'], AR_config['lat_res'])
    lons = np.arange(AR_config['min_lon'], AR_config['max_lon']+AR_config['lon_res'], AR_config['lon_res'])
    
    # Output arrays for all timesteps that will be filled by the loop below
    uIVT = np.empty((len(times), len(lats), len(lons)))
    vIVT = np.empty((len(times), len(lats), len(lons)))
    IVT = np.empty((len(times), len(lats), len(lons)))
    if calc_ll_mean_wind:
        ll_mean_u_wind = np.empty((len(times), len(lats), len(lons)))
        ll_mean_v_wind = np.empty((len(times), len(lats), len(lons)))
    
    # Must have opendap configuration set up (.netrc, .urs_cookies, .dodsrc) 
    # to open earthdata dataset with xarray
    # - See https://daac.gsfc.nasa.gov/information/howto?title=How%20to%20Access%20MERRA-2%20Data%20using%20OPeNDAP%20with%20Python3%20and%20Calculate%20Daily%2FWeekly%2FMonthly%20Statistics%20from%20Hourly%20Data%20
    if use_opendap:
        i = 0
        # For remotely stored data, open each daily file in sequence and loop
        # timesteps in that file
        for file_t in quv_file_times:
            # Assumes we are calculating over the full longitude span for either the NH or SH
            url = _get_MERRA2_opendap_url(AR_config, file_t)
            quv_ds_ts_full = xr.open_dataset(url)
            # Assumes MERRA-2 data is organized into daily files
            times_in_ds = times[times.to_period('1D') == file_t.strftime('%Y%m%d')]
            
            for t in times_in_ds:
                if calc_ll_mean_wind:
                    quv_ds_ts = quv_ds_ts_full.sel(time=t,
                                                   lev=AR_config['IVT_calc_plevs'],
                                                   lat=lats)
                    uIVT[i,:,:], vIVT[i,:,:], IVT[i,:,:], ll_mean_u_wind[i,:,:], ll_mean_v_wind[i,:,:] = \
                        _IVT_ll_mean_wind_calc_pres_levs(quv_ds_ts)
                else:
                    quv_ds_ts = quv_ds_ts_full.sel(time=t,
                                                   lev=AR_config['IVT_calc_mlevs'],
                                                   lat=lats)
                    uIVT[i,:,:], vIVT[i,:,:], IVT[i,:,:] = _IVT_calc_model_levs(quv_ds_ts)
                i += 1
                print('Finished calculating time '+str(t)+' at '+str(dt.datetime.now()))
                quv_ds_ts.close()
                
            quv_ds_ts_full.close()
    else:
        quv_flist = []
        for t in quv_file_times:
            quv_fname = glob.glob(AR_config['quv_dir']+\
                                  AR_config['quv_fname_prefix']+\
                                  '*'+t.strftime(AR_config['quv_fname_date_format'])+'*.nc')[0]
            quv_flist.append(quv_fname)
        
        # For locally stored data, open all files simultaneously as a multi-file dataset
        quv_ds = xr.open_mfdataset(quv_flist, mask_and_scale=True)
        for i, t in enumerate(times):
            # Assumes we are calculating over the full longitude span for either the NH or SH
            if calc_ll_mean_wind:
                quv_ds_ts = quv_ds.sel(time=t,
                                       lev=AR_config['IVT_calc_plevs'],
                                       lat=lats)
                uIVT[i,:,:], vIVT[i,:,:], IVT[i,:,:], ll_mean_u_wind[i,:,:], ll_mean_v_wind[i,:,:] = \
                    _IVT_ll_mean_wind_calc_pres_levs(quv_ds_ts)
            else:
                quv_ds_ts = quv_ds.sel(time=t,
                                       lev=AR_config['IVT_calc_mlevs'],
                                       lat=lats)
                uIVT[i,:,:], vIVT[i,:,:], IVT[i,:,:] = _IVT_calc_model_levs(quv_ds_ts)
            print('Finished calculating time '+str(t)+' at '+str(dt.datetime.now()))
            
        quv_ds.close()
    
    if calc_ll_mean_wind:
        return times, lats, lons, uIVT, vIVT, IVT, ll_mean_u_wind, ll_mean_v_wind
    else:
        return times, lats, lons, uIVT, vIVT, IVT

def _IVT_calc_model_levs(quv_ds_ts):
    """
    Perform IVT calculation on model levels for a single timestep.
    
    This method (including variable names and calculation of delta-p) works for MERRA-2 only.
    Will need adapting for any other datasets where IVT is calculated on pressure levels.
    - Possibly add a dictionary with variable names in the AR_ID_config.hjson   
    """
    
    uIVT_ts = np.nansum(quv_ds_ts['U']*quv_ds_ts['QV']*quv_ds_ts['DELP'], axis=0)/g
    vIVT_ts = np.nansum(quv_ds_ts['V']*quv_ds_ts['QV']*quv_ds_ts['DELP'], axis=0)/g
    
    IVT_ts = np.sqrt((uIVT_ts**2) + (vIVT_ts**2))
    
    return uIVT_ts, vIVT_ts, IVT_ts
    
def _IVT_ll_mean_wind_calc_pres_levs(quv_ds_ts):
    """
    Perform IVT and 1000-700 hPa mean wind calculation on pressure levels for
    a single timestep.
    
    This method (including variable names and calculation of delta-p) works for MERRA-2 only.
    Will need adapting for any other datasets where IVT is calculated on pressure levels.
    - Possibly add a dictionary with variable names in the AR_ID_config.hjson   
    """
    
    # Assumes the cutoff pressure level for calculation is 150 hPa and the next
    # level above this one is 100 hPa
    plevs = list(quv_ds_ts.lev.data)
    delp = [(plevs[i+1] - plevs[i])*-100 for i in range(len(plevs) - 1)]
    delp.append(5000.0)

    uIVT_ts = np.nansum(quv_ds_ts['U']*quv_ds_ts['QV']*np.array(delp)[:,np.newaxis,np.newaxis], axis=0)/g
    vIVT_ts = np.nansum(quv_ds_ts['V']*quv_ds_ts['QV']*np.array(delp)[:,np.newaxis,np.newaxis], axis=0)/g
    
    # Assumes pressure levels are incremented every 25 hPa from 1000 to 700 hPa
    ll_mean_u_wind_ts = np.nanmean(quv_ds_ts['U'][:13], axis=0)
    ll_mean_v_wind_ts = np.nanmean(quv_ds_ts['V'][:13], axis=0)
    
    IVT_ts = np.sqrt((uIVT_ts**2) + (vIVT_ts**2))
    
    return uIVT_ts, vIVT_ts, IVT_ts, ll_mean_u_wind_ts, ll_mean_v_wind_ts

def _get_MERRA2_opendap_url(AR_config, t):
    """
    Get the OPeNDAP URL for a given MERRA-2 file given the file time.
    The URL points to either a model level or pressure level file, depending on
    the "IVT_vert_coord" parameter in the AR ID configuration.
    """
    
    opendap_url_base = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2/'
    if AR_config['IVT_vert_coord'] == 'model_levels':
        collection_shortname = 'M2I3NVASM'
        collection_longname = 'inst3_3d_asm_Nv'
    elif AR_config['IVT_vert_coord'] == 'pres_levels':
        collection_shortname = 'M2I3NPASM'
        collection_longname = 'inst3_3d_asm_Np'
    version = '5.12.4'
    
    # Info on reprocessed data collection number changes to 401:
    # https://disc.gsfc.nasa.gov/information/documents?title=Records%20of%20MERRA-2%20Data%20Reprocessing%20and%20Service%20Changes
    if (t in pd.date_range(dt.datetime(2020,9,1,0), dt.datetime(2020,9,30,21), freq=AR_config['quv_files_timestep'])) or \
        (t in pd.date_range(dt.datetime(2021,6,1,0), dt.datetime(2021,12,31,21), freq=AR_config['quv_files_timestep'])):
        collection_number = '401'
    else:
        if t in pd.date_range(dt.datetime(1980,1,1,0), dt.datetime(1991,12,31,21), freq=AR_config['quv_files_timestep']):
            collection_number = '100'
        if t in pd.date_range(dt.datetime(1992,1,1,0), dt.datetime(2000,12,31,21), freq=AR_config['quv_files_timestep']):
            collection_number = '200'
        if t in pd.date_range(dt.datetime(2001,1,1,0), dt.datetime(2010,12,31,21), freq=AR_config['quv_files_timestep']):
            collection_number = '300'
        if t in pd.date_range(dt.datetime(2011,1,1,0), dt.datetime(2021,12,31,21), freq=AR_config['quv_files_timestep']):
            collection_number = '400'    
    
    yr = t.strftime('%Y')
    mth = t.strftime('%m')
    day = t.strftime('%d')
    url = f'{opendap_url_base}{collection_shortname}.'+\
          f'{version}/{yr}/{mth}/MERRA2_{collection_number}.{collection_longname}.{yr}{mth}{day}.nc4'
    
    return url


def write_IVT_output_file(AR_config, timestep_hrs_str, times, lats, lons, uIVT, vIVT, IVT):
    """
    Write IVT output file with metadata supplied by the AR ID configuration.
    """
        
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
        IVT_ds.attrs['IVT_calc_pressure_levels'] = str(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        IVT_ds.attrs['IVT_calc_model_levels'] = str(AR_config['IVT_calc_mlevs'])
        
    t_begin_str = times[0].strftime('%Y%m%d%H%M')
    t_end_str = times[-1].strftime('%Y%m%d%H%M')

    # Include "_subset" in file name if area given by AR_config is not an entire hemisphere poleward of 10 degrees latitude
    # Example file name: IVT_MERRA2_SH[_subset]_3hr_2021010100_2021013121.nc
    if (AR_config['min_lat'] == 10) and (AR_config['max_lat'] == 90) and (AR_config['min_lon'] == -180) and (AR_config['max_lon'] == 179.375):
        fname = 'IVT_'+AR_config['data_source']+'_NH_'+timestep_hrs_str+'hr_'+t_begin_str+'_'+t_end_str+'.nc'
    elif (AR_config['min_lat'] == -90) and (AR_config['max_lat'] == -10) and (AR_config['min_lon'] == -180) and (AR_config['max_lon'] == 179.375):
        fname = 'IVT_'+AR_config['data_source']+'_SH_'+timestep_hrs_str+'hr_'+t_begin_str+'_'+t_end_str+'.nc'
    else:
        fname = 'IVT_'+AR_config['data_source']+'_'+AR_config['hemisphere']+'H_subset_'+timestep_hrs_str+'hr_'+t_begin_str+'_'+t_end_str+'.nc'
        
    IVT_ds.to_netcdf(AR_config['IVT_dir']+fname, encoding=encoding)
    
    
def write_ll_mean_wind_output_file(AR_config, timestep_hrs_str, times, lats, lons, ll_mean_u_wind, ll_mean_v_wind):
    """
    Write 1000-700 hPa mean wind output file with metadata supplied by the
    AR ID configuration.
    """

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
                    
    if AR_config['use_opendap'] == 'True':
        use_opendap = True
    elif AR_config['use_opendap'] == 'False':
        use_opendap = False
        
    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        times, lats, lons, uIVT, vIVT, IVT, ll_mean_u_wind, ll_mean_v_wind = \
            calc_IVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str, use_opendap=use_opendap,
                     calc_ll_mean_wind=True)
        write_ll_mean_wind_output_file(AR_config, timestep_hrs_str, times, lats, lons, ll_mean_u_wind, ll_mean_v_wind)
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        times, lats, lons, uIVT, vIVT, IVT = \
            calc_IVT(AR_config, begin_t_str, end_t_str, timestep_hrs_str, use_opendap=use_opendap)

    write_IVT_output_file(AR_config, timestep_hrs_str, times, lats, lons, uIVT, vIVT, IVT)