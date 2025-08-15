"""
Script to reformat CESM2-LE AR output files to the requested data format for
ARTMIP Tier 2 polar submissions.

The requested data format is described in the "Requested Data Format for ARTMIP
Polar Future Experiment Submissions " google doc.
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import datetime as dt
from datetime import timedelta

time_pds = ['hist','future']
ens_members = ['1011','1031','1051']

ens_nums = {
    '1011':'001',
    '1031':'002',
    '1051':'003'
}

time_pd_yrs = {
    'hist':'1990-2009',
    'future':'2080-2099'
}


def add_attrs(ds, time_units=None, time_pd=None, ens_member=None):
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
        time_pd_yrs_str = time_pd_yrs[time_pd]
        ds.attrs = {
            'relative_time_period_info':\
                'IVT percentile is calculated using '+\
                f'{time_pd_yrs_str} climatology, specific to ensemble '
                f'member: {ens_member}'
        }

    return ds


def main(time_pd, ens_member, write_mask_file=False):

    artmip_input_dir_hist = os.path.join(
        '/doppler/data7/kmattingly/AR_data_v2/CESM2-LE/CESM2-LE_src_data/',
        'hist')
    ars_nh_dir = os.path.join(
        '/doppler/data7/kmattingly/AR_data_v2/CESM2-LE/CESM2-LE_ARs_NH/',
        time_pd)
    ars_sh_dir = os.path.join(
        '/doppler/data7/kmattingly/AR_data_v2/CESM2-LE/CESM2-LE_ARs_SH/',
        time_pd)
    
    output_dir_comp = os.path.join(
        '/doppler/data7/kmattingly/AR_data_v2/CESM2-LE/',
        'CESM2-LE_ARs_global_compressed', time_pd)
    output_dir_temp = os.path.join(
        '/doppler/data7/kmattingly/AR_data_v2/CESM2-LE/',
        'CESM2-LE_ARs_global_temp', time_pd)
    
    # Get lats/lons from an example ARTMIP input file, then write out mask file
    geoloc_ds = xr.open_dataset(os.path.join(
        artmip_input_dir_hist,
        'b.e21.BHISTsmbb.f09_g17.LE2-1051.003.cam.h2.uIVT.1990010100-2000010100.nc'))
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
        mask_ds.to_netcdf(
            os.path.join(
                output_dir_comp,
                'b.e21.BHISTsmbb.f09_g17.LE2-1051.003.ar_mask.Mattingly_v2.nc4'
            ),
            encoding = {'lat':{'_FillValue':None}, 'lon':{'_FillValue':None}}
        )
    
    # Loop through all monthly ARTMIP IVT input and AR output files, output 
    # monthly files
    if time_pd == 'hist':
        years = np.arange(1990,2010,1)
    elif time_pd == 'future':
        years = np.arange(2080,2100,1)
    
    for year in years:
        ar_fpaths_nh = sorted(glob.glob(
            os.path.join(
                ars_nh_dir,
                f'ARs_CESM2-LE_{ens_member}_NH_6hr_{year}*.nc')
            ))
        ar_fpaths_sh = sorted(glob.glob(
            os.path.join(
                ars_sh_dir,
                f'ARs_CESM2-LE_{ens_member}_SH_6hr_{year}*.nc')
            ))
        
        # Use multi-file dataset to get number of output times for the given
        # year
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
                ar_tags_nh_prelim = ds.AR_labels.data
                times_monthly = pd.to_datetime(ds.time)
            with xr.open_dataset(ar_fpath_sh) as ds:
                ar_tags_sh_prelim = ds.AR_labels.data
            
            # ** Change AR tags to be binary (1 or 0), rather than a unique
            # value for each AR
            ar_tags_nh = np.where(ar_tags_nh_prelim != 0, 1, 0)
            ar_tags_sh = np.where(ar_tags_sh_prelim != 0, 1, 0)
            
            time_ixs = np.arange(time_ix_counter,
                                 time_ix_counter+ar_tags_nh.shape[0],
                                 1)
            ar_binary_tag[time_ixs, -ar_tags_nh.shape[1]:, :] = ar_tags_nh
            ar_binary_tag[time_ixs, :ar_tags_nh.shape[1], :] = ar_tags_sh
            
            output_timedeltas_monthly = [
                int((t - base_dt) / timedelta(minutes=1)) for t in \
                    times_monthly]
        
            times.extend(output_timedeltas_monthly)
            time_ix_counter += ar_tags_nh.shape[0]
            
        ar_ds = xr.Dataset(
                {
                   'ar_binary_tag':(
                       ('time','lat','lon'),
                        ar_binary_tag.astype('byte'))
                },
                coords={
                    'time':np.array(times).astype('double'),
                    'lat':lats_global.astype('double'),
                    'lon':lons_global.astype('double')
                }
            )
        ar_ds = add_attrs(
            ar_ds, time_units=time_units, time_pd=time_pd,
            ens_member=ens_member)
        
        # Write yearly output file
        out_tstr = f'{year}010100-{year}123118'
        ens_num = ens_nums[ens_member]
        if time_pd == 'hist':
            output_fname = f'b.e21.BHISTsmbb.f09_g17.LE2-{ens_member}.{ens_num}.ar_tag.Mattingly_v2.6hr.{out_tstr}.nc'
        elif time_pd == 'future':
            output_fname = f'b.e21.BSSP370smbb.f09_g17.LE2-{ens_member}.{ens_num}.ar_tag.Mattingly_v2.6hr.{out_tstr}.nc'

        ar_ds.to_netcdf(
            os.path.join(output_dir_temp, output_fname),
            encoding = {'lat':{'_FillValue':None}, 'lon':{'_FillValue':None}}
        )
        
        now = dt.datetime.now()
        print(
            f'Finished reformatting {year} for time period: {time_pd}, '+\
            f'ensemble member: {ens_member} at {now}')
    
if __name__ == '__main__':
    for time_pd in time_pds:
        for ens_member in ens_members:
            # Only need to write the mask file once
            if (time_pd == 'hist') and (ens_member == '1051'):
                main(time_pd, ens_member, write_mask_file=True)
            else:
                main(time_pd, ens_member)