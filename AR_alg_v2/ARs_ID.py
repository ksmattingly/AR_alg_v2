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

from AR_alg_v2.misc_utils import rename_coords


def ARs_ID(AR_config, begin_time, end_time, timestep_hrs):
    """
    Main function to orchestrate AR identification for all times in the window
    defined by the begin and end times passed to the script.
    """
    
    # Verify that input data covers the entire zonal width of the earth
    check_input_zonal_extent(AR_config)
    
    # Calculate area of grid cells at each latitude of grid
    grid_cell_area_df = calc_grid_cell_areas_by_lat(AR_config)
    
    # Create list of output data times
    begin_dt = dt.datetime.strptime(begin_time, '%Y-%m-%d_%H%M')
    end_dt = dt.datetime.strptime(end_time, '%Y-%m-%d_%H%M')
    times = pd.date_range(begin_dt, end_dt, freq=timestep_hrs+'H')
    
    # Load input datasets and subset to latitude range of interest
    lats_subset = np.arange(AR_config['min_lat'], 
                            AR_config['max_lat']+AR_config['lat_res'], 
                            AR_config['lat_res'])
    if AR_config['direction_filter_type'] == 'mean_wind_1000_700_hPa':
        IVT_ds_full, IVT_at_pctiles_ds_full, climo_start_year, climo_end_year, climo_timestep_hrs, \
            ll_mean_wind_ds_full = \
            load_input_datasets(AR_config, begin_dt, end_dt, ll_mean_wind=True)
        ll_mean_wind_ds_domain = ll_mean_wind_ds_full.sel(lat=lats_subset)
    elif AR_config['direction_filter_type'] == 'IVT':
        IVT_ds_full, IVT_at_pctiles_ds_full, climo_start_year, climo_end_year, climo_timestep_hrs = \
            load_input_datasets(AR_config, begin_dt, end_dt)
    IVT_ds_domain = IVT_ds_full.sel(lat=lats_subset)
    IVT_at_pctiles_ds_domain = IVT_at_pctiles_ds_full.sel(lat=lats_subset)
    
    # Loop through all output times and add AR labels to output array
    AR_labels = np.empty((len(times), IVT_ds_domain.lat.shape[0], IVT_ds_domain.lon.shape[0]))
    for i,t in enumerate(times):
        doy = t.timetuple().tm_yday
        
        # Build "wrap" arrays that encircle the entire zonal width of the earth *twice*
        # - label_array_prelim may contain potential AR features near the poles that wrap
        #   around the entire globe *twice*; these features are removed in the
        #   label_array returned from filter_duplicate_features
        if AR_config['direction_filter_type'] == 'mean_wind_1000_700_hPa':
            label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap = \
                 build_wrap_arrays(AR_config,
                                   IVT_ds_domain.sel(time=t),
                                   IVT_at_pctiles_ds_domain.sel(doy=doy),
                                   ll_mean_wind_ds=ll_mean_wind_ds_domain.sel(time=t))

        else:
            label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap = \
                build_wrap_arrays(AR_config,
                                  IVT_ds_domain.sel(time=t),
                                  IVT_at_pctiles_ds_domain.sel(doy=doy))

        # Filter potential AR features to a set of unique features that are not
        # duplicated across both "halves" of the "wrap" array, and apply AR screening
        # criteria to determine which are ARs
        label_array, feature_props_df = filter_duplicate_features(label_array_prelim, IVT_wrap,
                                                                  lons_wrap, IVT_ds_domain.lat.data)
        AR_labels_timestep, AR_count_timestep = \
            apply_AR_criteria(AR_config, feature_props_df, grid_cell_area_df,
                              label_array, u_wrap, v_wrap, 
                              IVT_ds_domain.lat.data, IVT_ds_domain.lon.data, lons_wrap)
        
        AR_labels[i::] = AR_labels_timestep
        
        t_str = t.strftime('%Y-%m-%d %H:%M:%S')
        now_str = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'Processed {t_str} at {now_str} ({AR_count_timestep} ARs)')
    
    return AR_labels, times, IVT_ds_domain.lat.data, IVT_ds_domain.lon.data, \
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
        grid_cell_min_lat = lat - (lat_res/2)
        if lat == 90:    
            grid_cell_max_lat = lat
        else:
            grid_cell_max_lat = lat + (lat_res/2)
        
        # Calculate areas of sample "measurement" grid cells along the prime meridian
        # (grid cells areas don't vary zonally, only meridionally)
        grid_cell_min_lon = -(lon_res / 2)
        grid_cell_max_lon = (lon_res / 2)
        
        grid_cell_width_meas_pt_w = (lat, grid_cell_min_lon)
        grid_cell_width_meas_pt_e = (lat, grid_cell_max_lon)
        grid_cell_hgt_meas_pt_s = (grid_cell_min_lat, 0)
        grid_cell_hgt_meas_pt_n = (grid_cell_max_lat, 0)
        
        approx_grid_cell_width_m = great_circle(grid_cell_width_meas_pt_w, grid_cell_width_meas_pt_e).meters
        approx_grid_cell_hgt_m = great_circle(grid_cell_hgt_meas_pt_s, grid_cell_hgt_meas_pt_n).meters
        approx_grid_cell_area_m2 = approx_grid_cell_width_m*approx_grid_cell_hgt_m
        
        approx_grid_cell_area_km2 = approx_grid_cell_area_m2*1e-6
    
        lats.append(lat)
        grid_cell_areas_km2.append(approx_grid_cell_area_km2)
    
    grid_cell_area_df = DataFrame({'lat':lats, 'grid_cell_area_km2':grid_cell_areas_km2})

    return grid_cell_area_df


def load_input_datasets(AR_config, begin_dt, end_dt, ll_mean_wind=False):
    """
    Load input files (IVT, IVT at percentiles, [and low-level mean wind if needed])
    as xarray datasets.
    
    Only load input files for the year(s) covered by the range from begin_dt to
    end_dt to save processing time.
    
    If lat/lon coordinates in input files are named "latitude" and "longitude",
    rename to "lat" and "lon".
    """
    
    IVT_fpaths = _sift_fpaths(AR_config, AR_config['IVT_dir'], begin_dt, end_dt)
    IVT_ds_full = rename_coords(xr.open_mfdataset(IVT_fpaths))
    
    IVT_at_pctiles_fpath = glob.glob(AR_config['IVT_PR_dir']+'IVT_at_pctiles_'+\
                                     AR_config['data_source']+'_'+\
                                     AR_config['hemisphere']+'*.nc')[0]
    IVT_at_pctiles_ds_full = rename_coords(xr.open_dataset(IVT_at_pctiles_fpath))
    
    # Get IVT at percentiles climatology parameters to be stored in AR output file.
    climo_start_year = str(IVT_at_pctiles_ds_full.attrs['IVT_climatology_start_year'])
    climo_end_year = str(IVT_at_pctiles_ds_full.attrs['IVT_climatology_end_year'])
    climo_timestep = str(IVT_at_pctiles_ds_full.attrs['IVT_climatology_timestep_hrs'])
    
    if ll_mean_wind:
        ll_mean_wind_fpaths = _sift_fpaths(AR_config, AR_config['wind_1000_700_mean_dir'], begin_dt, end_dt)
        ll_mean_wind_ds_full = rename_coords(xr.open_mfdataset(ll_mean_wind_fpaths))
        return IVT_ds_full, IVT_at_pctiles_ds_full, climo_start_year, climo_end_year, climo_timestep, \
            ll_mean_wind_ds_full
    else:
        return IVT_ds_full, IVT_at_pctiles_ds_full, climo_start_year, climo_end_year, climo_timestep
    
def _sift_fpaths(AR_config, data_dir, begin_dt, end_dt):
    """
    Helper to load_input_datasets.
    
    Reduce input file paths to only those from the years of the window defined by
    the begin and end time passed to the script. This makes reading in the xarray
    multi-file dataset quicker. 
    - (If the range from begin_dt to end_dt doesn't cover an exact full year, there
       will still be some timesteps included in the multi-file dataset that aren't
       actually included in AR output file.)
    
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
    
    return fpaths


def build_wrap_arrays(AR_config, IVT_ds, IVT_at_pctiles_ds,
                      ll_mean_wind_ds=None):
    """
    Create "wrap arrays" that encircle the entire zonal width of the globe *twice*,
    so that features that cross the antimeridian and/or a pole can be handled
    as contiguous features by the image processing functions.
    
    Arrays created are:
    - label arrays of unique potential AR ID "features"
    - u/v arrays used for filtering final AR objects according to AR ID criteria
        - either u/v-IVT or low-level mean u/v-wind
    
    After creating the label arrays, filter out features that wrap zonally around
    the entire hemisphere.
    """
    
    lons_wrap = np.concatenate((IVT_ds.lon, IVT_ds.lon))
    
    IVT = IVT_ds.IVT
    IVT_wrap = np.concatenate((IVT, IVT), axis=1)
    
    IVT_at_thresh = IVT_at_pctiles_ds['IVT_pctile_'+str(AR_config['IVT_PR_thresh'])]
    IVT_at_thresh_wrap = np.concatenate((IVT_at_thresh, IVT_at_thresh), axis=1)
    
    # Array containing information on whether the basic potential AR threshold
    # values are met (values are 1 if IVT > IVT threshold and IVT PR > IVT PR threshold,
    # 0 if not)
    thresh_array_wrap = np.where(np.logical_and(IVT_wrap >= AR_config['IVT_thresh'], \
                                                IVT_wrap >= IVT_at_thresh_wrap), \
                                 1, 0)
    
    # Assign a unique label to each contiguous area where the threshold is met
    # - These contiguous areas are referred to as "features" throughout this script
    label_array_prelim, num_labels = ndimage.measurements.label(thresh_array_wrap)

    if ll_mean_wind_ds:
        u_wrap = np.concatenate((ll_mean_wind_ds.u, ll_mean_wind_ds.u), axis=1)
        v_wrap = np.concatenate((ll_mean_wind_ds.v, ll_mean_wind_ds.v), axis=1)
    else:
        u_wrap = np.concatenate((IVT_ds.uIVT, IVT_ds.uIVT), axis=1)
        v_wrap = np.concatenate((IVT_ds.vIVT, IVT_ds.vIVT), axis=1)

    return label_array_prelim, IVT_wrap, IVT_at_thresh_wrap, u_wrap, v_wrap, lons_wrap


def filter_duplicate_features(label_array_prelim, IVT_wrap, lons_wrap, lats):
    """
    Apply three tests to ensure that potential AR features in "wrap array" are not
    duplicated across the two globe-encircling "halves" of the "wrap array":
    (1) Check if any of the labeled features wrap zonally around the entire hemisphere
        (common near the North Pole; not sure about South Pole). If so, change the
        label array values to 0 for the second "wrap" of the array so that the entire
        720-degree feature in the "double wrapped" label array is not labeled as
        one feature.
    (2) Check if mean IVT is exactly the same within any two features. (Indicating
        that the same feature is present in both "halves" of the "wrap" array spanning
        the hemisphere twice.)
    (3) Check if any lat/lon points within each feature are found within any other
        feature. (Indicating an antimeridian-spanning feature located along the 
        "left or right edge" of the "wrap" array, which should be deleted in favor
        of processing the feature that spans the antimeridian in the "middle" of
        the "wrap" array. This is accomplished by retaining the largest feature
        when features overlap.)
        
    Return:
    - label_array with potential AR features that wrap zonally around the entire
      hemisphere filtered out
    - A data frame with attributes of potential AR features that are not
      duplicated across both "halves" of the "wrap array", and haven't yet been
      filtered into the *final* AR features by the direction, size, and shape criteria.
    """
    
    label_array, labels_prelim = _full_zonal_wraps_filter(label_array_prelim, lons_wrap, lats)
    
    feature_props_IVT = regionprops(label_array, intensity_image=IVT_wrap)
    # Create pandas data frame containing:
    # (1) the image processing "regionprops" object of each feature
    # (2) the mean IVT within each feature (derived from regionprops)
    feature_props_df_all = DataFrame({
        'label':labels_prelim[1:],
        'feature_props_IVT':feature_props_IVT,
        'feature_mean_IVT':[feature.mean_intensity for feature in feature_props_IVT]
        })
    
    feature_props_df_IVT_filtered = _identical_IVT_filter(feature_props_df_all)    
    feature_props_df = _spatial_overlap_filter(feature_props_df_IVT_filtered, lons_wrap, lats)

    return label_array, feature_props_df

def _full_zonal_wraps_filter(label_array_prelim, lons_wrap, lats):
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
            else:
                pass
    # Updated label array with any duplicated hemisphere-encircling features removed
    label_array = label_array_prelim
    
    return label_array, labels_prelim

def _identical_IVT_filter(feature_props_df_all):
    """
    Helper to filter_duplicate_features.
    
    Check if mean IVT is exactly the same within any two features.
    
    If there are two or more features with exactly the same mean IVT value,
    then remove all but the *first* feature from the data frame of potential
    AR features.
    """
    
    labels_to_remove = []
    for unique_IVT_value in feature_props_df_all.feature_mean_IVT.unique():
        df_at_value = feature_props_df_all[feature_props_df_all.feature_mean_IVT == unique_IVT_value]
        if len(df_at_value) > 0:
            # Remove all but the first duplicated feature from the data frame
            min_label_at_value = np.min(df_at_value.label)
            labels_to_remove_at_value = list(df_at_value.label)
            del labels_to_remove_at_value[labels_to_remove_at_value.index(min_label_at_value)]
            labels_to_remove.extend(labels_to_remove_at_value)
    
    unique_labels_to_remove = list(set(labels_to_remove))
    feature_props_df_IVT_filtered = feature_props_df_all.drop(\
        feature_props_df_all[feature_props_df_all.label.isin(unique_labels_to_remove)].index)

    return feature_props_df_IVT_filtered

def _spatial_overlap_filter(feature_props_df_IVT_filtered, lons_wrap, lats):
    """
    Helper to filter_duplicate_features.
    
    Check if any lat/lon points within each feature are found within any other
    feature to filter out antimeridian-spanning features located along the 
    "left or right edge" of the "wrap" array.
    
    When overlapping features are found, retain the largest feature to ensure that
    it is the one that spans the antimeridian in the "middle" of the "wrap" array.
    """
    
    pts_all_features = []
    labels = []
    num_grid_cells = []
    overlap_flag = []
    
    for (label, feature_props) in zip(feature_props_df_IVT_filtered.label, 
                                      feature_props_df_IVT_filtered.feature_props_IVT):
        feature_num_grid_cells = len(feature_props.coords)
        for ixs in feature_props.coords:
            pt = (lats[ixs[0]], lons_wrap[ixs[1]])
            if pt in pts_all_features:
                overlap_flag.append(1)
            else:
                overlap_flag.append(0)
            
            pts_all_features.append(pt)
            labels.append(label)
            num_grid_cells.append(feature_num_grid_cells)
    
    # Data frame of *all* features, including information on whether they overlap
    # with another feature
    feature_pts_labels_df = DataFrame({'pt':pts_all_features,
                                       'label':labels,
                                       'num_grid_cells':num_grid_cells,
                                       'overlap_flag':overlap_flag})
    
    # Determine feature labels to remove and return data frame with these features
    # excluded
    labels_to_remove = []
    for row in feature_pts_labels_df.iterrows():
        if row[1]['overlap_flag'] != 0:
            df_at_pt = feature_pts_labels_df[feature_pts_labels_df.pt == row[1]['pt']]
            if len(df_at_pt) > 1:
                feature_sizes = df_at_pt.num_grid_cells.unique()
                if all(size == feature_sizes[0] for size in feature_sizes):
                    min_label = np.min(df_at_pt.label)
                    labels_to_remove_pt = list(df_at_pt.label)
                    del labels_to_remove_pt[labels_to_remove_pt.index(min_label)]
                else:
                    max_size = np.max(df_at_pt.num_grid_cells)
                    df_at_pt_notmax = df_at_pt[df_at_pt.num_grid_cells != max_size]
                    labels_to_remove_pt = df_at_pt_notmax.label.unique()
                labels_to_remove.extend(labels_to_remove_pt)
    
    feature_props_df = feature_props_df_IVT_filtered.drop(\
        feature_props_df_IVT_filtered[\
            feature_props_df_IVT_filtered.label.isin(list(set(labels_to_remove)))].index)
    
    return feature_props_df


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
    # array (which only spans the globe zonally once) by manually placing AR
    # labels at the lat/lon coordinates found to be part of actual ARs in the "wrap"
    # array.
    AR_labels_timestep = np.zeros(shape=(u_wrap.shape[0], int(u_wrap.shape[1]/2)))
    
    AR_count_timestep = 0
    # Loop through all potential AR features and determine if they meet AR criteria
    for row in feature_props_df.iterrows():
        label = row[1]['label']
        feature_props = row[1]['feature_props_IVT']
        
        # Initial filter of very small features (to avoid unnecessary processing time)
        # - For this and subsequent tests, continue to the next feature in the loop
        #   if the current feature "fails" the test
        if feature_props.area < AR_config['min_num_grid_pts']:
            continue
        
        # Calculate feature centroid
        centroid_lat = _calc_centroid_lat(feature_props, lats)
        
        feature_array = np.where(label_array == label, 1, 0)
        # Test if feature is transporting moisture poleward (if not centered in
        # polar latitudes), and is not an east-to-west tropical/subtropical
        # moisture transport plume (if centered in tropical/subtropical latitudes)
        direction_flags = _check_direction(AR_config, centroid_lat, feature_array, u_wrap, v_wrap)
        if direction_flags > 0:
            continue
        
        # Test if feature meets length and length-to-width ratio criteria
        length_and_shape_flags = _check_length_and_shape(AR_config, grid_cell_area_df,
                                                         feature_array, feature_props,
                                                         lats, lons_wrap)
        if length_and_shape_flags > 0:
            continue
        
        # If the feature passes all the tests, add its labeled points to AR output array            
        for ixs in feature_props.coords:
            lat_ix = ixs[0]
            lon_wrap = lons_wrap[ixs[1]]
            lon_ix = np.where(lons==lon_wrap)
            AR_labels_timestep[lat_ix,lon_ix] = label

        AR_count_timestep += 1

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
    args = parser.parse_args()
    
    return args.begin_time, args.end_time, args.timestep_hrs

    
if __name__ == '__main__':
    """
    Main block to control reading inputs from command line, ingesting AR ID
    configuration, calculating ARs, and writing AR output file.
    """

    begin_time, end_time, timestep_hrs = parse_args()
    
    _code_dir = os.path.dirname(os.path.realpath(__file__))
    AR_ID_config_path = _code_dir+'/AR_ID_config.hjson'
    with open(AR_ID_config_path) as f:
        AR_config = hjson.loads(f.read())

    AR_labels, times, lats, lons, climo_start_year, climo_end_year, climo_timestep_hrs = \
        ARs_ID(AR_config, begin_time, end_time, timestep_hrs)
    write_AR_labels_file(AR_config, begin_time, end_time, timestep_hrs,
                         AR_labels, times, lats, lons,
                         climo_start_year, climo_end_year, climo_timestep_hrs)