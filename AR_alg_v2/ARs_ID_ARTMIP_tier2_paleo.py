"""
Identify AR outlines by applying AR criteria to fields of IVT magnitude and
direction, and IVT at climatological percentile rank threshold.

In this "ARTMIP Tier 2 paleo" version of the AR ID code, several of the core AR
ID functions do not require any changes from the main version of the code. These
functions are imported into this script from the primary "ARs_ID.py" script.
"""

import argparse
import hjson
import glob
import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
from pandas import DataFrame
from geopy.distance import great_circle
from scipy import ndimage
from calendar import isleap

from AR_alg_v2.misc_utils import rename_coords, rename_IVT_components
from AR_alg_v2.ARs_ID import check_input_zonal_extent,\
    filter_duplicate_features, apply_AR_criteria


def ARs_ID(AR_config, begin_time, end_time, timestep_hrs):
    """
    Main function to orchestrate AR identification for all times in the window
    defined by the begin and end times passed to the script.
    """
    
    # Verify that input data covers the entire zonal width of the earth
    check_input_zonal_extent(AR_config)
        
    # Create list of output data times
    # - ** Note that all paleo datasets are treated as though their 30-year
    #   period extends from 1970-01-01 03:00 to 1999-12-31 21:00:00 (6-hourly)
    begin_dt = dt.datetime.strptime(begin_time, '%Y-%m-%d_%H%M')
    end_dt = dt.datetime.strptime(end_time, '%Y-%m-%d_%H%M')
    times_all = pd.date_range(begin_dt, end_dt, freq=timestep_hrs+'H')
    times = times_all[~((times_all.month == 2) & (times_all.day == 29))]
    
    # Create placeholder datetimes that pretend the 30-year paleo dataset extends from
    # 1970-01-01 03:00 to 1999-12-31 21:00, with no leap days
    por_begin = dt.datetime(1970,1,1,3)
    por_end = dt.datetime(1999,12,31,21)
    times_por_withleapdays = pd.date_range(por_begin, por_end, freq=timestep_hrs+'H')
    times_por = times_por_withleapdays[~((times_por_withleapdays.month == 2) & (times_por_withleapdays.day == 29))]

    # Build indexing data frame with input file names, times, and time indices
    ix_df = build_input_file_indexing_df(AR_config, times_por)

    # Load IVT percentile rank dataset; also use this dataset to get lats, lons,
    # and IVT climatology info (start year, end year, timestep in hrs)
    IVT_at_pctiles_fpath = glob.glob(AR_config['IVT_at_pctiles_dir']+\
                                     'IVT_at_pctiles_'+\
                                     AR_config['data_source']+'_'+\
                                     AR_config['hemisphere']+'*.nc')[0]
    # IVT_at_pctiles_ds = rename_coords(xr.open_dataset(IVT_at_pctiles_fpath)).sel(lat=lats_subset)
    IVT_at_pctiles_ds = xr.open_dataset(IVT_at_pctiles_fpath)
    lats = IVT_at_pctiles_ds.lat.data
    lons = IVT_at_pctiles_ds.lon.data
    climo_start_year = str(IVT_at_pctiles_ds.attrs['IVT_climatology_start_year'])
    climo_end_year = str(IVT_at_pctiles_ds.attrs['IVT_climatology_end_year'])
    climo_timestep_hrs = str(IVT_at_pctiles_ds.attrs['IVT_climatology_timestep_hrs'])

    grid_cell_area_df = calc_grid_cell_areas_by_lat(AR_config, lats)
        
    # Loop through all output times and add AR labels to output array
    # leap_years = np.arange(1900,2100,4)
    AR_labels = np.empty((len(times), lats.shape[0], lons.shape[0]))
    for i,t in enumerate(times):
        t_str = t.strftime('%Y-%m-%d %H:%M:%S')
        ix_df_t = ix_df.loc[t]
        
        # For leap years, subtract 1 from all doys after the leap day to get corrected doy
        # (for selecting IVT pctile rank climatology by julian day)
        doy = t.timetuple().tm_yday
        if isleap(t.year) and (doy >= 60):
            doy = doy - 1
        IVT_at_pctiles_ds_doy = IVT_at_pctiles_ds.sel(doy=doy)
        
        # Build "wrap" arrays that span the width of the globe twice
        label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap = \
            build_wrap_arrays(AR_config, ix_df, ix_df_t, IVT_at_pctiles_ds_doy)

        # Filter potential AR features to a set of unique features that are not
        # duplicated across both "halves" of the "wrap" arrays
        label_array, feature_props_df = filter_duplicate_features(AR_config,
                                                                  label_array_prelim, IVT_wrap,
                                                                  lons_wrap, lats)
        
        # Apply AR screening criteria to determine which features qualify as ARs,
        # and add to AR labels output array
        AR_labels_timestep, AR_count_timestep = \
            apply_AR_criteria(AR_config, feature_props_df, grid_cell_area_df,
                              label_array, u_wrap, v_wrap,
                              lats, lons, lons_wrap)
        # Renumber AR feature labels
        # - Either start at 1 and number features sequentially, or label all
        #   AR features as 1 (depending on setting in AR config)
        for j, label in enumerate(np.unique(AR_labels_timestep)[1:]):
            if AR_config['AR_labels'] == 'same_value':
                AR_labels_timestep[np.where(AR_labels_timestep == label)] = 1
            elif AR_config['AR_labels'] == 'unique':
                AR_labels_timestep[np.where(AR_labels_timestep == label)] = j + 1
        AR_labels[i::] = AR_labels_timestep
        
        now_str = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Processed {t_str} at {now_str} ({AR_count_timestep} ARs)')
    
    return AR_labels, times, lats, lons, \
        climo_start_year, climo_end_year, climo_timestep_hrs


def calc_grid_cell_areas_by_lat(AR_config, lats):
    """
    Create data frame to look up grid cell areas based on latitude.
    """
    
    lat_res = AR_config['lat_res']
    lon_res = AR_config['lon_res']
    
    grid_cell_areas_km2 = []
    
    for lat in lats:
        if AR_config['hemisphere'] == 'NH':
            grid_cell_min_lat = lat - (lat_res/2)
            if lat == 90:
                grid_cell_max_lat = lat
            else:
                grid_cell_max_lat = lat + (lat_res/2)
        elif AR_config['hemisphere'] == 'SH':
            grid_cell_max_lat = lat + (lat_res/2)
            if lat == -90:
                grid_cell_min_lat = lat
            else:
                grid_cell_min_lat = lat - (lat_res/2)
        
        # Calculate areas of sample "measurement" grid cells along the prime meridian
        # (grid cells areas don't vary zonally, only meridionally)
        grid_cell_min_lon = -(lon_res / 2)
        grid_cell_max_lon = (lon_res / 2)
        
        width_meas_pt_w = (lat, grid_cell_min_lon)
        width_meas_pt_e = (lat, grid_cell_max_lon)
        hgt_meas_pt_s = (grid_cell_min_lat, 0)
        hgt_meas_pt_n = (grid_cell_max_lat, 0)
        
        approx_grid_cell_width_m = great_circle(width_meas_pt_w, width_meas_pt_e).meters
        approx_grid_cell_hgt_m = great_circle(hgt_meas_pt_s, hgt_meas_pt_n).meters
        approx_grid_cell_area_m2 = approx_grid_cell_width_m*approx_grid_cell_hgt_m
        
        approx_grid_cell_area_km2 = approx_grid_cell_area_m2*1e-6
    
        grid_cell_areas_km2.append(approx_grid_cell_area_km2)
    
    grid_cell_area_df = DataFrame({'lat':lats, 'grid_cell_area_km2':grid_cell_areas_km2})

    return grid_cell_area_df


def build_input_file_indexing_df(AR_config, times_por):
    """
    Build data frame with file path, time, and time index for each timestep of
    the input IVT files.
    
    If ll_mean_wind is True, then the indexing information for the input low-level
    mean wind files is also included in the data frame. Note that the time span
    and time indexing of the low-level wind files must be *exactly* the same as
    the IVT files.
    """
    
    IVT_fpaths = glob.glob(AR_config['IVT_dir']+'*.nc')
    
    IVT_fpaths_alltimes = []
    time_ixs = []
    for fpath in IVT_fpaths:
        ds = xr.open_dataset(fpath)
        IVT_fpaths_alltimes.extend([fpath for i in range(len(ds.time))])
        time_ixs.extend(list(np.arange(0,len(ds.time),1)))
                
    ix_df = DataFrame({'IVT_fpath':IVT_fpaths_alltimes, 'time_ix':time_ixs, 't':times_por},
                      index=times_por)
    
    return ix_df


def build_wrap_arrays(AR_config, ix_df, ix_df_t, IVT_at_pctiles_ds_doy):
    """
    Create "wrap arrays" that encircle the entire zonal width of the globe *twice*,
    so that features that cross the antimeridian and/or a pole can be handled
    as contiguous features by the image processing functions.
    
    Arrays created are:
    - label arrays of unique potential AR ID "features"
    - u/v arrays used for filtering final AR objects according to AR ID criteria
      (either u/v-IVT or low-level mean u/v-wind)
    """
    
    IVT_ds_orig = rename_IVT_components(rename_coords(xr.open_dataset(ix_df_t['IVT_fpath'])))
    IVT_ds_alltimes = IVT_ds_orig.assign_coords(time=('time',
                                                      ix_df[ix_df.IVT_fpath == ix_df_t.IVT_fpath].t))
    IVT_ds_full = IVT_ds_alltimes.sel(time=ix_df_t.t)

    min_lat = AR_config['min_lat']
    max_lat = AR_config['max_lat']
    mask_lats = (IVT_ds_full.lat >= min_lat) & (IVT_ds_full.lat <= max_lat)
    IVT_ds = IVT_ds_full.where(mask_lats, drop=True)

    lons_wrap = np.concatenate((IVT_ds.lon, IVT_ds.lon))
    # In ARTMIP IVT source files, reading in lat/lon values results in very small
    # decimal values for 0 degrees lat/lon. Change these lon values to 0.
    # - Not an issue for lats because AR ID domain is set to poleward of 10N / -10S
    lons_wrap[np.where(np.abs(lons_wrap) < 0.0001)] = 0
    IVT = IVT_ds.IVT
    IVT_wrap = np.concatenate((IVT, IVT), axis=1)

    IVT_at_thresh = IVT_at_pctiles_ds_doy['IVT_pctile_'+str(AR_config['IVT_PR_thresh'])]
    IVT_at_thresh_wrap = np.concatenate((IVT_at_thresh, IVT_at_thresh), axis=1)

    # Array containing information on whether the basic potential AR threshold
    # values are met (values are 1 if IVT > IVT threshold and IVT PR > IVT PR threshold,
    # 0 if not)
    thresh_array_wrap = np.where(np.logical_and(IVT_wrap >= AR_config['IVT_thresh'], \
                                                IVT_wrap >= IVT_at_thresh_wrap), \
                                 1, 0)

    # Assign a unique label to each contiguous area where the threshold is met
    # - These contiguous areas are referred to as "features" throughout this script
    label_array_prelim, num_labels = ndimage.label(thresh_array_wrap)

    u_wrap = np.concatenate((IVT_ds.uIVT, IVT_ds.uIVT), axis=1)
    v_wrap = np.concatenate((IVT_ds.vIVT, IVT_ds.vIVT), axis=1)

    return label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap
        

def write_AR_labels_file(AR_config, begin_t, end_t, timestep_hrs,
                         AR_labels, times, lats, lons,
                         climo_start_year, climo_end_year, climo_timestep_hrs):
    """
    Write AR labels output file, with detailed metadata supplied by AR_ID_config.hjson.
    """
    
    # Change data types of AR labels, lats, and lons to save disk space
    ARs_ds = xr.Dataset(
        {
         'AR_labels':(('time','lat','lon'), AR_labels.astype('uint16'))
        },
        coords={
            'time':times,
            'lat':lats.astype('float32'),
            'lon':lons.astype('float32')
        }
    )

    ARs_ds.lat.attrs['units'] = 'degrees_north'
    ARs_ds.lon.attrs['units'] = 'degrees_east'
    
    ARs_ds.attrs['data_source'] = AR_config['data_source']
    ARs_ds.attrs['IVT_data_origin'] = AR_config['IVT_data_origin']
    ARs_ds.attrs['IVT_vert_coordinate'] = AR_config['IVT_vert_coord']
    
    ARs_ds.attrs['IVT_31day_centered_climatology_start_year'] = int(climo_start_year)
    ARs_ds.attrs['IVT_31day_centered_climatology_end_year'] = int(climo_end_year)
    ARs_ds.attrs['IVT_31day_centered_climatology_timestep_hrs'] = int(climo_timestep_hrs)

    ARs_ds.attrs['AR_IVT_minimum_threshold'] = AR_config['IVT_thresh']
    ARs_ds.attrs['AR_IVT_minimum_percentile_rank'] = AR_config['IVT_PR_thresh']
    ARs_ds.attrs['AR_min_length'] = AR_config['min_length']
    ARs_ds.attrs['AR_min_length_width_ratio'] = AR_config['min_length_width_ratio']
    ARs_ds.attrs['AR_direction_filter_type'] = AR_config['direction_filter_type']
    ARs_ds.attrs['AR_poleward_transport_requirement_cutoff_lat'] = AR_config['v_poleward_cutoff_lat']
    ARs_ds.attrs['AR_subtropical_westerly_transport_requirement_cutoff_lat'] = AR_config['subtrop_bound_lat']
    if AR_config['direction_filter_type'] == 'IVT':
        ARs_ds.attrs['AR_poleward_transport_threshold_value_kg_m-1_s-1'] = AR_config['v_thresh']
        ARs_ds.attrs['AR_subtropical_westerly_transport_requirement_value_kg_m-1_s-1'] = AR_config['subtrop_u_thresh']
    elif AR_config['direction_filter_type'] == 'mean_wind_1000_700_hPa':
        ARs_ds.attrs['AR_poleward_transport_threshold_value_m_s-1'] = AR_config['v_thresh']
        ARs_ds.attrs['AR_subtropical_westerly_transport_requirement_value_m_s-1'] = AR_config['subtrop_u_thresh']

    t_begin_str = times[0].strftime('%Y%m%d%H%M')
    t_end_str = times[-1].strftime('%Y%m%d%H%M')

    minlat = AR_config['min_lat']
    maxlat = AR_config['max_lat']
    data_source = AR_config['data_source']
    
    # If output data grid covers the 10-90 degree band in the given hemisphere,
    # then label output file as "NH" or "SH". Otherwise, note the latitude band of
    # AR data in the file name.
    if (minlat == 10) and (maxlat == 90):
        fname = f'ARs_{data_source}_NH_{timestep_hrs}hr_{t_begin_str}_{t_end_str}.nc'
    elif (minlat == -90) and (maxlat == -10):
        fname = f'ARs_{data_source}_SH_{timestep_hrs}hr_{t_begin_str}_{t_end_str}.nc'  
    else:
        fname = f'ARs_{data_source}_lat_{minlat}_{maxlat}_{timestep_hrs}hr_{t_begin_str}_{t_end_str}.nc'  
    
    ARs_ds.to_netcdf(AR_config['AR_output_dir']+fname)


def parse_args():
    """
    Parse arguments passed to script at runtime.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('begin_time', help='Begin time in the format YYYY-MM-DD_HHMM')
    parser.add_argument('end_time', help='End time in the format YYYY-MM-DD_HHMM')
    parser.add_argument('timestep_hrs', help='Timestep as integer number of hours (e.g. 3)')
    parser.add_argument('AR_ID_config_path', help='Path to AR ID configuration file')
    args = parser.parse_args()
    
    return args.begin_time, args.end_time, args.timestep_hrs, args.AR_ID_config_path


def main():
    """
    Main block to control reading inputs from command line, ingesting AR ID
    configuration, calculating ARs, and writing AR output file.
    """

    begin_time, end_time, timestep_hrs, AR_ID_config_path = parse_args()
    
    with open(AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())

    AR_labels, times, lats, lons, climo_start_year, climo_end_year, climo_timestep_hrs = \
        ARs_ID(AR_config, begin_time, end_time, timestep_hrs)
    write_AR_labels_file(AR_config, begin_time, end_time, timestep_hrs,
                         AR_labels, times, lats, lons,
                         climo_start_year, climo_end_year, climo_timestep_hrs)


if __name__ == '__main__':
    main()