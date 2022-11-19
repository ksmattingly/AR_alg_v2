"""
Script to reformat MERRA-2 AR output files to the requested data format for
ARTMIP Tier 1 MERRA-2 submissions.

The requested data format is described in the "Requested Data Format for ARTMIP
Tier 1 MERRA-2 Submissions" google doc. Details of the requested format include:
- Yearly netCDF files, 3-hourly temporal resolution
- File name: <Reanalysis dataset>.ar_tag.<Algorithm>.<hourly/3hourly/6hourly/daily>.<Date(s)>.nc4
  - The date is either a single day in the form YYYYMMDD or a range of dates
    in the form YYYYMMDD-YYYYMMDD
- Regional data put onto MERRA2 global grid, with 0’s for locations you do not
  detect ARs and provide a mask that shows where your ARDT operates
  - AR mask file name: <Reanalysis dataset>.ar_mask.<Algorithm>.nc4
  - The value of ar binary mask should be binary (either a 1 or 0). Grid cells
    that are covered by the detection scheme should be given a value of 1, and
    those that are not covered should be given a value of 0.
- The time, lat, and lon variables should be inherited directly from the MERRA2
  dataset.
- The ar binary tag variable should be of type byte (an 8-bit integer) so as to
  limit the size of each submission. This variable should be 1 if an atmospheric
  river is detected at this grid point / time and 0 if no detection occurred.
- The ar binary tag variable should have attributes: description, scheme,
  version.

Files must also be compressed. Compression is handled by NCO (not in this script):
    ncks -4 -L 1 file.nc compressed_file.nc
"""

import xarray as xr
import numpy as np
import pandas as pd
import glob
import datetime as dt
from datetime import timedelta


def add_attrs(ds, time_units=None):
    """
    Add variable attributes to output dataset (either AR domain mask or AR
    tag dataset).
    """
    
    ds.lat.attrs = {
        'standard_name':'latitude',
        'long_name':'latitude',
        'units':'degrees_north',
        'axis':'Y'
        }
    ds.lon.attrs = {
        'standard_name':'longitude',
        'long_name':'longitude',
        'units':'degrees_east',
        'axis':'X'
        }
    if 'ar_binary_mask' in ds:
        ds.ar_binary_mask.attrs = {
            'description':'atmospheric river regional coverage mask',
            'scheme':'Mattingly',
            'version':'2.0'
            }
    elif 'ar_binary_tag' in ds:
        ds.ar_binary_tag.attrs = {
            'description':'binary indicator of atmospheric river',
            'scheme':'Mattingly',
            'version':'2.0'
            }
        ds.time.attrs = {
            'standard_name':'time',
            'long_name':'time',
            'units':time_units,
            'calendar':'standard'
            }

    return ds


def main(write_mask_file=False):

    artmip_input_dir = '/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier1/MERRA2_IVT_source_data/'
    ars_nh_dir = '/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier1/MERRA2_ARs_NH_3hr/'
    ars_sh_dir = '/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier1/MERRA2_ARs_SH_3hr/'
    
    output_dir_comp = '/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier1/MERRA2_ARs_global_compressed/'
    output_dir_temp = '/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier1/MERRA2_ARs_global_temp/'
    
    # Get lats/lons from an example ARTMIP input file, output mask file
    geoloc_ds = xr.open_dataset(f'{artmip_input_dir}ARTMIP_MERRA_2D_198001010000_198001312100.nc')
    lats_global = geoloc_ds.lat.data[:]
    lons_global = geoloc_ds.lon.data[:]
    
    if write_mask_file:
        lons_mesh, lats_mesh = np.meshgrid(lons_global, lats_global)
        ar_binary_mask = np.where(np.abs(lats_mesh) < 10, 0, 1)
        
        mask_ds = xr.Dataset(
                {
                   'ar_binary_mask':(('lat','lon'), ar_binary_mask.astype('byte'))
                },
                coords={
                    'lat':lats_global.astype('double'),
                    'lon':lons_global.astype('double')
                }
            )
        mask_ds = add_attrs(mask_ds)
        mask_ds.to_netcdf(f'{output_dir_comp}MERRA2.ar_mask.Mattingly.nc4',
                          encoding = {'lat':{'_FillValue':None}, 'lon':{'_FillValue':None}})
    
    # Loop through all monthly ARTMIP IVT input and AR output files, output 
    # monthly files
    years = np.arange(1980,2020,1)
    
    for year in years:
        ar_fpaths_nh = sorted(glob.glob(f'{ars_nh_dir}ARs_MERRA2_NH_3hr_{year}*.nc'))
        ar_fpaths_sh = sorted(glob.glob(f'{ars_sh_dir}ARs_MERRA2_SH_3hr_{year}*.nc'))
        
        # Use multi-file dataset to get number of output times for the given year
        ar_nh_mfdataset = xr.open_mfdataset(ar_fpaths_nh)
        ntimes_year = len(ar_nh_mfdataset.time)
        ar_nh_mfdataset.close()
        
        ar_binary_tag = np.zeros(shape=(ntimes_year,
                                        lats_global.shape[0],
                                        lons_global.shape[0]),
                                 dtype='byte')
        
        time_units = f'minutes since {year}-01-01 00:00:00'
        times = []
        base_dt = dt.datetime(year,1,1,0)
        time_ix_counter = 0
        
        for (ar_fpath_nh, ar_fpath_sh) in zip(ar_fpaths_nh, ar_fpaths_sh):            
            with xr.open_dataset(ar_fpath_nh) as ds:
                ar_tags_nh = ds.AR_labels.data
                times_monthly = pd.to_datetime(ds.time)
            with xr.open_dataset(ar_fpath_sh) as ds:
                ar_tags_sh = ds.AR_labels.data
            
            time_ixs = np.arange(time_ix_counter,
                                 time_ix_counter+ar_tags_nh.shape[0],
                                 1)
            ar_binary_tag[time_ixs, -ar_tags_nh.shape[1]:, :] = ar_tags_nh
            ar_binary_tag[time_ixs, :ar_tags_nh.shape[1], :] = ar_tags_sh
            
            output_timedeltas_monthly = [int((t - base_dt) / timedelta(minutes=1)) for t in times_monthly]
        
            times.extend(output_timedeltas_monthly)
            time_ix_counter += ar_tags_nh.shape[0]
            
        ar_ds = xr.Dataset(
                {
                   'ar_binary_tag':(('time','lat','lon'), ar_binary_tag.astype('byte'))
                },
                coords={
                    'time':times,
                    'lat':lats_global.astype('double'),
                    'lon':lons_global.astype('double')
                }
            )
        ar_ds = add_attrs(ar_ds, time_units=time_units)
        
        # Write yearly output file
        output_fname = f'MERRA2.ar_tag.Mattingly_v2.3hourly.{year}0101_{year}1231.nc4'
        ar_ds.to_netcdf(f'{output_dir_temp}{output_fname}',
                        encoding = {'lat':{'_FillValue':None}, 'lon':{'_FillValue':None}})
        
        now = dt.datetime.now()
        print(f'Finished reformatting {year} at {now}')
    
if __name__ == '__main__':
    main()