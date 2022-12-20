"""
Script to reformat ARTMIP Tier 2 paleo AR output files to the requested data
format for ARTMIP Tier 2 paleo submissions.

The requested data format is described in the "ARTMIP-DataFormat_for_T2_paleo"
google doc. Details of the requested format include:
- Yearly netCDF files, 6-hourly temporal resolution
- File name: <PaleoID>.ar_tag.<Algorithm>.<timefreq>hr.<Year>.nc4
  - where <PaleoID> is one of "PreIndust, 10ka-Orbital, PI_21ka-CO2"
    (indicating the experiment)
- Regional data put onto global grid, with 0’s for locations you do not
  detect ARs and provide a mask that shows where your ARDT operates
  - AR mask file name: <PaleoID>.ar_mask.<Algorithm>.nc4
  - The value of ar binary mask should be binary (either a 1 or 0). Grid cells
    that are covered by the detection scheme should be given a value of 1, and
    those that are not covered should be given a value of 0.
- The time, lat, and lon variables should be inherited directly from the original
  datasets.
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
import glob
import datetime as dt

years_orig = {'PI_21ka-CO2':np.arange(61,91,1),
              '10ka-Orbital':np.arange(271,301,1),
              'PreIndust':np.arange(121,151,1)}

years_dummy = np.arange(1970,2000,1)


def add_attrs(ds, time_units=None):
    """
    Add variable attributes to output dataset (either AR domain mask or AR
    tag dataset).
    """
    
    ds.lat.attrs = {
        'standard_name':'latitude',
        'long_name':'latitude',
        'units':'degrees_north',
        }
    ds.lon.attrs = {
        'standard_name':'longitude',
        'long_name':'longitude',
        'units':'degrees_east',
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


def main(write_mask_file=True):    

    for paleo_dataset in years_orig.keys():    
        artmip_input_dir = f'/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier2_paleo/{paleo_dataset}/'
        ars_dir = f'/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier2_paleo/ARs_{paleo_dataset}/'
        
        output_dir_comp = f'/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier2_paleo/ARs_{paleo_dataset}_compressed/'
        output_dir_temp = f'/run/media/kmattingly/kyle_8tb/AR_data_v2/ARTMIP_tier2_paleo/ARs_{paleo_dataset}_temp/'
        
        # Get lats/lons from an example ARTMIP input file, output mask file
        geoloc_ds = xr.open_dataset(glob.glob(f'{artmip_input_dir}*.nc')[0])
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
            mask_ds.to_netcdf(f'{output_dir_comp}{paleo_dataset}.ar_mask.Mattingly_v2.nc4',
                              encoding = {'lat':{'_FillValue':None}, 'lon':{'_FillValue':None}})
        
        # Loop through all monthly ARTMIP IVT input and AR output files, output 
        # monthly files
        years = years_orig[paleo_dataset]
        
        for year_orig, year_dummy in zip(years, years_dummy):
            year_orig_str = '{:04d}'.format(year_orig)

            ar_fpaths_nh = sorted(glob.glob(f'{ars_dir}ARs_{paleo_dataset}_NH_6hr_{year_dummy}*.nc'))
            ar_fpaths_sh = sorted(glob.glob(f'{ars_dir}ARs_{paleo_dataset}_SH_6hr_{year_dummy}*.nc'))
            
            # Use multi-file dataset to get number of output times for the given year
            ar_nh_mfdataset = xr.open_mfdataset(ar_fpaths_nh)
            ntimes_year = len(ar_nh_mfdataset.time)
            ar_nh_mfdataset.close()
            
            ar_binary_tag = np.zeros(shape=(ntimes_year,
                                            lats_global.shape[0],
                                            lons_global.shape[0]),
                                     dtype='byte')
            
            time_units = f'days since {year_orig_str}-01-01 00:00:00'
            time_ixs_yr = []
            time_ix_counter = 0
            
            for (ar_fpath_nh, ar_fpath_sh) in zip(ar_fpaths_nh, ar_fpaths_sh):            
                with xr.open_dataset(ar_fpath_nh) as ds:
                    ar_tags_nh = ds.AR_labels.data
                with xr.open_dataset(ar_fpath_sh) as ds:
                    ar_tags_sh = ds.AR_labels.data
                
                time_ixs = np.arange(time_ix_counter,
                                     time_ix_counter+ar_tags_nh.shape[0],
                                     1)
                ar_binary_tag[time_ixs, -ar_tags_nh.shape[1]:, :] = ar_tags_nh
                ar_binary_tag[time_ixs, :ar_tags_nh.shape[1], :] = ar_tags_sh
                
                time_ixs_yr.extend(time_ixs)
                time_ix_counter += ar_tags_nh.shape[0]
            
            # Times in tier 2 ARTMIP original IVT datasets are fractional days
            # since some start day (the start of the century, excluding leap days??),
            # with days in each year starting at day=0.25.
            # So encode time as fractional days since the start of the given
            # year, with the first timestep at day=0.25.
            times_days_since_yr_start = (np.array(time_ixs_yr) + 1) / 4
            
            ar_ds = xr.Dataset(
                    {
                       'ar_binary_tag':(('time','lat','lon'), ar_binary_tag.astype('byte'))
                    },
                    coords={
                        'time':times_days_since_yr_start.astype('double'),
                        'lat':lats_global.astype('double'),
                        'lon':lons_global.astype('double')
                    }
                )
            ar_ds = add_attrs(ar_ds, time_units=time_units)
            
            # Write yearly output file
            output_fname = f'{paleo_dataset}.ar_tag.Mattingly_v2.6hr.{year_orig_str}.nc4'
            ar_ds.to_netcdf(f'{output_dir_temp}{output_fname}',
                            encoding = {'lat':{'_FillValue':None}, 'lon':{'_FillValue':None}})
            
            now = dt.datetime.now()
            print(f'Finished reformatting {paleo_dataset}, year {year_orig} at {now}')
    
if __name__ == '__main__':
    main()