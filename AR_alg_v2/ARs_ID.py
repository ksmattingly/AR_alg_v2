"""
Identify AR outlines by applying AR criteria to fields of IVT magnitude and
direction, and IVT at climatological percentile rank threshold.

Optionally, lower-tropospheric mean wind may be used to determine transport
direction instead of the IVT vector components.
"""

import os
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
from skimage.measure import regionprops
import math
import itertools

from AR_alg_v2.misc_utils import rename_coords, rename_IVT_components


def ARs_ID(AR_config, begin_time, end_time, timestep_hrs):
    """
    Main function to orchestrate AR identification for all times in the window
    defined by the begin and end times passed to the script.
    """
    
    # Verify that input data covers the entire zonal width of the earth
    check_input_zonal_extent(AR_config)
    
    # Calculate area of grid cells at each latitude of grid
    grid_cell_area_df = calc_grid_cell_areas_by_lat(AR_config)
    
    # Create latitude array for subset domain
    lats_subset = np.arange(AR_config['min_lat'], 
                            AR_config['max_lat']+AR_config['lat_res'], 
                            AR_config['lat_res'])
    
    # Create list of output data times
    begin_dt = dt.datetime.strptime(begin_time, '%Y-%m-%d_%H%M')
    end_dt = dt.datetime.strptime(end_time, '%Y-%m-%d_%H%M')
    times = pd.date_range(begin_dt, end_dt, freq=timestep_hrs+'H')

    # Build indexing data frame with input file names, times, and time indices
    ix_df = build_input_file_indexing_df(AR_config, times, ll_mean_wind=False)

    # Load IVT percentile rank dataset; also use this dataset to get lats, lons,
    # and IVT climatology info (start year, end year, timestep in hrs)
    IVT_at_pctiles_fpath = glob.glob(AR_config['IVT_at_pctiles_dir']+\
                                     'IVT_at_pctiles_'+\
                                     AR_config['data_source']+'_'+\
                                     AR_config['hemisphere']+'*.nc')[0]
    IVT_at_pctiles_ds = rename_coords(xr.open_dataset(IVT_at_pctiles_fpath)).sel(lat=lats_subset)
    lats = IVT_at_pctiles_ds.lat.data
    lons = IVT_at_pctiles_ds.lon.data
    climo_start_year = str(IVT_at_pctiles_ds.attrs['IVT_climatology_start_year'])
    climo_end_year = str(IVT_at_pctiles_ds.attrs['IVT_climatology_end_year'])
    climo_timestep_hrs = str(IVT_at_pctiles_ds.attrs['IVT_climatology_timestep_hrs'])
        
    # Loop through all output times and add AR labels to output array
    leap_years = np.arange(1900,2100,4)
    AR_labels = np.empty((len(times), lats.shape[0], lons.shape[0]))
    for i,t in enumerate(times):
        t_str = t.strftime('%Y-%m-%d %H:%M:%S')
        ix_df_t = ix_df.loc[t]
        
        # For leap years, subtract 1 from all doys after the leap day to get corrected doy
        # (for selecting IVT pctile rank climatology by julian day)
        doy = t.timetuple().tm_yday
        if (t.year in leap_years) and (doy >= 60):
            doy = doy - 1
        IVT_at_pctiles_ds_doy = IVT_at_pctiles_ds.sel(doy=doy)
        
        # Build "wrap" arrays that span the width of the globe twice
        if AR_config['direction_filter_type'] == 'mean_wind_1000_700_hPa':
            ll_mean_wind = True
        else:
            ll_mean_wind = False
        label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap = \
            build_wrap_arrays(AR_config, lats_subset, ix_df_t, IVT_at_pctiles_ds_doy, ll_mean_wind)

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


def check_input_zonal_extent(AR_config):
    """
    Verify that input data covers the entire zonal width of the earth.
    
    AR identification in narrower domains is currently not supported due
    to the "double wrapping" method used to handle antimeridian-crossing
    features.
    """
    
    # MERRA-2: for global data, min lon = -180 and max lon = 179.375
    # ERA5: for global data, min lon = -180 and max lon = 179.75
    # ** This would need to be adapted for coordinates not structured this way
    #   (e.g. if the first longitude is -180 and the last longitude is a
    #   "cyclic point" of 180)
    if ((AR_config['max_lon'] - AR_config['min_lon']) + AR_config['lon_res']) != 360:
        raise Exception('Input data must span the entire globe zonally')


def calc_grid_cell_areas_by_lat(AR_config):
    """
    Create data frame to look up grid cell areas based on latitude.
    """
    
    lat_res = AR_config['lat_res']
    lon_res = AR_config['lon_res']
    
    lats = []
    grid_cell_areas_km2 = []
    
    for lat in np.arange(AR_config['min_lat'], AR_config['max_lat']+lat_res, lat_res):
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
    
        lats.append(lat)
        grid_cell_areas_km2.append(approx_grid_cell_area_km2)
    
    grid_cell_area_df = DataFrame({'lat':lats, 'grid_cell_area_km2':grid_cell_areas_km2})

    return grid_cell_area_df


def build_input_file_indexing_df(AR_config, times, ll_mean_wind=False):
    """
    Build data frame with file path, time, and time index for each timestep of
    the input IVT files.
    
    If ll_mean_wind is True, then the indexing information for the input low-level
    mean wind files is also included in the data frame. Note that the time span
    and time indexing of the low-level wind files must be *exactly* the same as
    the IVT files.
    """
    
    IVT_fpaths = _sift_fpaths(AR_config, AR_config['IVT_dir'],
                              times[0], times[-1])
    IVT_fpaths_alltimes = []
    time_ixs = []
    file_times = []
    
    if ll_mean_wind:
        ll_mean_wind_fpaths_alltimes = []
    
    for fpath in IVT_fpaths:
        ds = xr.open_dataset(fpath)
        
        IVT_fpaths_alltimes.extend([fpath for i in range(len(ds.time))])
        time_ixs.extend(list(np.arange(0,len(ds.time),1)))
        file_times.extend(list(pd.to_datetime(ds.time)))
        
        if ll_mean_wind:
            ll_mean_wind_fpath = AR_config['wind_1000_700_mean_dir'] + \
                'mean_wind_1000_700_hPa' + \
                os.path.basename(fpath).split('IVT')[1]
            ll_mean_wind_fpaths_alltimes.extend([ll_mean_wind_fpath for i in range(len(ds.time))])
        
    ix_df = DataFrame({'IVT_fpath':IVT_fpaths_alltimes, 'time_ix':time_ixs, 't':file_times},
                      index=file_times)
    if ll_mean_wind:
        ix_df['ll_mean_wind_fpath'] = ll_mean_wind_fpaths_alltimes
    
    return ix_df
        
def _sift_fpaths(AR_config, data_dir, begin_dt, end_dt):
    """
    Helper to build_input_file_indexing_df.
    
    Reduce input file paths to only those from the years of the window defined by
    the begin and end time passed to the script. This makes creating the input 
    file indexing data frame quicker. 
    - (If the range from begin_dt to end_dt doesn't cover an exact full year, there
       will still be some timesteps included in the data frame that aren't
       actually processed for the AR output file.)
    
    File names must end with "...[startdate]_[enddate].nc", in the format given in the
    AR config (default used for MERRA-2 and ERA5 files is '%Y%m%d%H%M').
    """
    
    analysis_yrs = np.arange(begin_dt.year, end_dt.year + 1, 1)
    fpaths_all = glob.glob(data_dir+'*.nc')
    fpaths = []
    for fpath in fpaths_all:
        start_yr = dt.datetime.strptime(os.path.basename(fpath).split('_')[-2], 
                                        AR_config['IVT_fname_date_format']).year
        end_yr = dt.datetime.strptime(os.path.basename(fpath).split('_')[-1], 
                                      AR_config['IVT_fname_date_format']+'.nc').year
        if (start_yr in analysis_yrs) or (end_yr in analysis_yrs):
            fpaths.append(fpath)
    
    return sorted(fpaths)


def build_wrap_arrays(AR_config, lats_subset, ix_df_t, IVT_at_pctiles_ds_doy, ll_mean_wind):
    """
    Create "wrap arrays" that encircle the entire zonal width of the globe *twice*,
    so that features that cross the antimeridian and/or a pole can be handled
    as contiguous features by the image processing functions.
    
    Arrays created are:
    - label arrays of unique potential AR ID "features"
    - u/v arrays used for filtering final AR objects according to AR ID criteria
      (either u/v-IVT or low-level mean u/v-wind)
    """
    
    IVT_ds = rename_IVT_components(rename_coords(xr.open_dataset(ix_df_t['IVT_fpath'])))\
                                   .sel(time=ix_df_t.t, lat=lats_subset)
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

    if ll_mean_wind == True:
        ll_mean_wind_ds = rename_coords(xr.open_dataset(ix_df_t['ll_mean_wind_fpath']))\
                                        .sel(time=ix_df_t.t, lat=lats_subset)
        u_wrap = np.concatenate((ll_mean_wind_ds.u, ll_mean_wind_ds.u), axis=1)
        v_wrap = np.concatenate((ll_mean_wind_ds.v, ll_mean_wind_ds.v), axis=1)
    else:
        u_wrap = np.concatenate((IVT_ds.uIVT, IVT_ds.uIVT), axis=1)
        v_wrap = np.concatenate((IVT_ds.vIVT, IVT_ds.vIVT), axis=1)

    return label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap
    

def filter_duplicate_features(AR_config, label_array_prelim, IVT_wrap, lons_wrap, lats):
    """
    Apply two tests to ensure that potential AR features in "wrap array" are not
    duplicated across the two globe-encircling "halves" of the "wrap array":
    (1) Check if any of the labeled features wrap zonally around the entire hemisphere
        (common near the North Pole; not sure about South Pole). If so, change the
        label array values to 0 for the second "wrap" of the array so that the entire
        720-degree feature in the "double wrapped" label array is not labeled as
        one feature.
    (2) Check if mean IVT is exactly the same within any two features. (Indicating
        that the same feature is present in both "halves" of the "wrap" array spanning
        the hemisphere twice.)
        
    Return:
    - label_array with potential AR features that wrap zonally around the entire
      hemisphere filtered out
    - A data frame with attributes of potential AR features that are not
      duplicated across both "halves" of the "wrap array", and haven't yet been
      filtered into the *final* AR features by the direction, size, and shape criteria.
    """
    
    label_array, labels_prelim = _full_zonal_wraps_filter(AR_config,
                                                          label_array_prelim, lons_wrap, lats)
    feature_props_IVT = regionprops(label_array, intensity_image=IVT_wrap)
    # Create data frame containing info on each feature:
    # - feature label
    # - the feature's image processing "regionprops" object
    # - mean IVT within feature
    # - feature number of grid cells
    feature_props_df = DataFrame({
        'label':labels_prelim[1:],
        'feature_props_IVT':feature_props_IVT,
        'feature_mean_IVT':[feature.mean_intensity for feature in feature_props_IVT],
        'feature_num_grid_cells':[len(feature.coords) for feature in feature_props_IVT]
        })
    
    feature_props_df_IVT_filtered = _identical_IVT_filter(feature_props_df)    

    return label_array, feature_props_df_IVT_filtered

def _full_zonal_wraps_filter(AR_config,
                             label_array_prelim, lons_wrap, lats):
    """
    Helper to filter_duplicate_features.
    
    Check if any of the labeled features wrap zonally around the entire hemisphere
    (common near the North Pole; not sure about South Pole). If so, change the
    label array values to 0 for the second "wrap" of the array so that the entire
    720-degree feature in the "double wrapped" label array is not labeled as
    one feature.
    """
    
    labels_prelim = np.unique(label_array_prelim)
    for label in labels_prelim:
        binary_array_label = np.where((label_array_prelim == label), 1, 0)

        # Only perform check on large features; small features will be filtered out by
        # minimum area check in apply_AR_criteria
        if np.sum(binary_array_label) >= AR_config['min_num_grid_pts']:
            for lat_ix in range(lats.shape[0]):
                if np.sum(binary_array_label[lat_ix,:]) == lons_wrap.shape[0]:
                    # Make label array in second wrap equal to 0
                    label_array_replace_ixs_full = np.where(label_array_prelim == label)
                    label_array_second_wrap_ixs = \
                        np.where(label_array_replace_ixs_full[1] >= int(lons_wrap.shape[0]/2))
                    label_array_replace_ixs_second_wrap = \
                        (label_array_replace_ixs_full[0][label_array_second_wrap_ixs], \
                         label_array_replace_ixs_full[1][label_array_second_wrap_ixs])
                    label_array_prelim[label_array_replace_ixs_second_wrap] = 0
                    break

    # Updated label array with any duplicated hemisphere-encircling features removed
    label_array = label_array_prelim
    
    return label_array, labels_prelim

def _identical_IVT_filter(feature_props_df):
    """
    Helper to filter_duplicate_features.
    
    Check if mean IVT is exactly the same within any two features.
    
    If there are two or more features with exactly the same mean IVT value,
    then remove all but the *first* feature from the data frame of potential
    AR features.
    """
    
    labels_to_remove = []
    for unique_IVT_value in feature_props_df.feature_mean_IVT.unique():        
        df_at_value = feature_props_df[feature_props_df.feature_mean_IVT == unique_IVT_value]
        if len(df_at_value) > 0:
            # Remove all but the first duplicated feature from the data frame
            min_label_at_value = np.min(df_at_value.label)
            labels_to_remove_at_value = list(df_at_value.label)
            del labels_to_remove_at_value[labels_to_remove_at_value.index(min_label_at_value)]
            labels_to_remove.extend(labels_to_remove_at_value)
    
    unique_labels_to_remove = list(set(labels_to_remove))
    feature_props_df_IVT_filtered = feature_props_df.drop(\
        feature_props_df[feature_props_df.label.isin(unique_labels_to_remove)].index)

    return feature_props_df_IVT_filtered


def apply_AR_criteria(AR_config, feature_props_df, grid_cell_area_df,
                      label_array, u_wrap, v_wrap,
                      lats, lons, lons_wrap):
    """
    Apply final AR criteria to potential AR features, returning an array containing
    the unique labels for each AR.
    - Initial filter to remove very small features that are well below AR area threshold
    - Direction filters (v transport poleward if centroid not in polar region;
      u transport from west to east if centroid in tropics / subtropics)
    - Length and length-to-width ratio filters
    """
    
    # Output AR labels array. The features are translated from the "wrap" array to this
    # array (which only spans the globe zonally once) by manually placing AR labels
    # at the lat/lon coordinates found to be part of actual ARs in the "wrap" array.
    AR_labels_timestep = np.zeros(shape=(u_wrap.shape[0], int(u_wrap.shape[1]/2)))
    
    # Array keeping track of features that span the central longitude
    # discontinuity in the center of the "double wrap" array and are filtered
    # out by either the "direction" flags or the "length and shape" flags
    filtered_features = np.zeros(
        shape=(u_wrap.shape[0], int(u_wrap.shape[1]/2)))
    # Indices of the longitudes that span the discontinuity at the center of
    # the "double wrap" arrays ("left center" and "right center")
    lon_ix_lc = int(u_wrap.shape[1]/2) - 1
    lon_ix_rc = int(u_wrap.shape[1]/2) + 1

    # Loop through all potential AR features and determine if they meet AR criteria
    for row in feature_props_df.iterrows():
        label = row[1]['label']
        feature_props = row[1]['feature_props_IVT']
        feature_area = row[1]['feature_num_grid_cells']
        
        # Initial filter of very small features (to avoid unnecessary processing time)
        # - For this and subsequent tests, continue to the next feature in the loop
        #   if the current feature "fails" the test
        if feature_area < AR_config['min_num_grid_pts']:
            continue
        
        # Calculate feature centroid
        centroid_lat = _calc_centroid_lat(feature_props, lats)
        
        feature_array = np.where(label_array == label, 1, 0)
        # Test if feature is transporting moisture poleward (if not centered in
        # polar latitudes), and is not an east-to-west tropical/subtropical
        # moisture transport plume (if centered in tropical/subtropical latitudes)
        direction_flags = _check_direction(AR_config, centroid_lat, feature_array, u_wrap, v_wrap)
        if direction_flags > 0:
            feat_lon_ixs = feature_props.coords[:,1]
            if (lon_ix_lc in feat_lon_ixs) and (lon_ix_rc in feat_lon_ixs):
                for ixs in feature_props.coords:
                    lat_ix = ixs[0]
                    lon_wrap = lons_wrap[ixs[1]]
                    lon_ix = np.where(lons==lon_wrap)

                    filtered_features[lat_ix,lon_ix] = 1
                
            continue
        
        # Test if feature meets length and length-to-width ratio criteria
        length_and_shape_flags = _check_length_and_shape(AR_config, grid_cell_area_df,
                                                         feature_array, feature_props,
                                                         lats, lons_wrap)
        if length_and_shape_flags > 0:
            feat_lon_ixs = feature_props.coords[:,1]
            if (lon_ix_lc in feat_lon_ixs) and (lon_ix_rc in feat_lon_ixs):
                for ixs in feature_props.coords:
                    lat_ix = ixs[0]
                    lon_wrap = lons_wrap[ixs[1]]
                    lon_ix = np.where(lons==lon_wrap)

                    filtered_features[lat_ix,lon_ix] = 1

            continue
        
        # If the feature passes all the tests, check to see if it overlaps with 
        # (a) a pre-existing AR feature in the AR output array, or (b) a
        # feature along the longitude discontinuity in the center of the
        # "double wrap" array that was previously filtered out.
        # - If there is no pre-existing AR feature in the same location, add the
        #   feature's labeled points to the AR output array.
        # - If there is a pre-existing AR feature in the same location, retain
        #   the label of whichever feature is larger (either the pre-existing
        #   feature or the current feature being inspected). This has the effect
        #   of discarding features that are on the left or right edge of the "wrap"
        #   array, and retains features that cross the antimeridian or prime meridian
        #   in the center of the "wrap" array as a single feature with the same label.
        # - Previously-filtered features are removed after all potential ARs
        #   are processed for the given timestep.
        for ixs in feature_props.coords:
            lat_ix = ixs[0]
            lon_wrap = lons_wrap[ixs[1]]
            lon_ix = np.where(lons==lon_wrap)
                        
            if AR_labels_timestep[lat_ix,lon_ix] != 0:
                other_feature_label = AR_labels_timestep[lat_ix,lon_ix]
                other_feature_area = list(feature_props_df[feature_props_df.label == int(other_feature_label)].\
                                          feature_num_grid_cells)[0]
                if feature_area > other_feature_area:
                    AR_labels_timestep[lat_ix,lon_ix] = label
                else:
                    break
            else:
                AR_labels_timestep[lat_ix,lon_ix] = label
    
    # Remove non-AR features that span the longitude discontinuity in the
    # center of the "double wrap" array
    AR_labels_timestep[np.where(filtered_features > 0)] = 0

    # Get the number of final ARs for this timestep
    AR_count_timestep = len(np.unique(AR_labels_timestep)) - 1

    return AR_labels_timestep, AR_count_timestep

def _calc_centroid_lat(feature_props, lats):
    """
    Helper to apply_AR_criteria.
    
    Calculate feature centroid lat/lon.
    """
    
    cent_ixs_float = feature_props.centroid
    lat_ix = \
        math.ceil(cent_ixs_float[0]) if ((math.ceil(cent_ixs_float[0]) - cent_ixs_float[0]) <= 0.5) \
        else math.floor(cent_ixs_float[0])
    centroid_lat = lats[lat_ix]
    
    return centroid_lat

def _check_direction(AR_config, centroid_lat, feature_array, u_wrap, v_wrap):
    """
    Helper to apply_AR_criteria.
    
    Filter out features with equatorward v-winds (if equatorward of v_poleward_cutoff_lat)
    and tropical/subtropical east-to-west directed moisture plumes.
    """
    
    feature_ixs = np.where(feature_array == 1)
    
    vwind_equatorward_flag = 0
    v_mean = np.nanmean(v_wrap[feature_ixs])
    if AR_config['hemisphere'] == 'NH':
        if np.logical_and((v_mean < AR_config['v_thresh']), 
                          (np.abs(centroid_lat) < np.abs(AR_config['v_poleward_cutoff_lat']))):
            vwind_equatorward_flag += 1
    elif AR_config['hemisphere'] == 'SH': 
        if np.logical_and((v_mean > AR_config['v_thresh']), 
                          (np.abs(centroid_lat) > np.abs(AR_config['v_poleward_cutoff_lat']))):
            vwind_equatorward_flag += 1

    trop_subtrop_east_west_plume_flag = 0
    if np.abs(centroid_lat) < np.abs(AR_config['subtrop_bound_lat']):
        u_mean = np.nanmean(u_wrap[feature_ixs])
        if u_mean < AR_config['subtrop_u_thresh']:
            trop_subtrop_east_west_plume_flag += 1        
    
    direction_flags = trop_subtrop_east_west_plume_flag + vwind_equatorward_flag
    
    return direction_flags

def _check_length_and_shape(AR_config, grid_cell_area_df, feature_array, feature_props, lats, lons_wrap):
    """
    Helper to apply_AR_criteria.
    
    Filter out features that don't meet the length and length-to-width ratio
    criteria.
    """
        
    # Calculate feature area
    feature_grid_cell_areas = []
    for ixs in feature_props.coords:
        lat = lats[ixs[0]]
        area_df_at_lat = grid_cell_area_df[grid_cell_area_df.lat == lat]
        feature_grid_cell_areas.append(list(area_df_at_lat['grid_cell_area_km2'])[0])
    feature_area_km2 = np.nansum(feature_grid_cell_areas)
    
    # Calculate feature perimeter
    feature_array_eroded = ndimage.binary_erosion(feature_array, border_value=0)
    feature_perim_array = feature_array - feature_array_eroded
    feature_perim_ixs = np.where(feature_perim_array == 1)
    perim_pts = []
    for (lat_ix, lon_ix) in zip(feature_perim_ixs[0], feature_perim_ixs[1]):
        perim_pts.append((lats[lat_ix], lons_wrap[lon_ix]))

    # Get the 2 perimeter points with the maximum great circle distance from one
    # another and use these to calculate maximum perimeter great circle distance
    # (proxy for length) in km
    perim_pt1s = []
    perim_pt2s = []
    perim_great_circle_distances = []
    for perim_pt1, perim_pt2 in itertools.combinations(perim_pts, 2):
        perim_pt1s.append(perim_pt1)
        perim_pt2s.append(perim_pt2)
        
        perim_great_circle_distance = great_circle(perim_pt1, perim_pt2).kilometers
        perim_great_circle_distances.append(perim_great_circle_distance)
    
    feature_max_perim_great_circle_distance = np.max(perim_great_circle_distances)
    # "Width" = the effective earth surface width of the feature               
    feature_effective_width_km = feature_area_km2 / feature_max_perim_great_circle_distance
    feature_length_width_ratio = feature_max_perim_great_circle_distance / feature_effective_width_km
                            
    if feature_max_perim_great_circle_distance < AR_config['min_length']:
        length_flag = 1
    else:
        length_flag = 0
        
    if feature_length_width_ratio < AR_config['min_length_width_ratio']:
        length_width_ratio_flag = 1
    else:
        length_width_ratio_flag = 0
    
    length_and_shape_flags = length_flag + length_width_ratio_flag
    
    return length_and_shape_flags
    

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

    if AR_config['IVT_vert_coord'] == 'pressure_levels':
        ARs_ds.attrs['IVT_calc_pressure_levels'] = str(AR_config['IVT_calc_plevs'])
    elif AR_config['IVT_vert_coord'] == 'model_levels':
        ARs_ds.attrs['IVT_calc_model_levels'] = str(AR_config['IVT_calc_mlevs'])

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