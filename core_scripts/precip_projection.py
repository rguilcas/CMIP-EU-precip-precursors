from dask.distributed import Client, LocalCluster
import argparse
import xarray as xr
import os
import numpy as np
import pint_xarray
from pint_xarray import unit_registry as ureg
import sys

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Aggregate variable over ERA5 region masks.")

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='name of model simulation to load as input.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='name of simulation experiment, e.g. historical, ssp370')

    # Optional arguments
    parser.add_argument('--member', type=str, default='',
                        help='Which ensemble member, if any.')
    
    parser.add_argument('--overwrite',action='store_true',
                        help='Overwrite indices if they already exist.')
    
    parser.add_argument('--seasons', nargs='+', type=str, default=['DJF', 'MAM','JJA','SON'],
                    help='Which seasons to compute indices for.')
    
    parser.add_argument('--regions', nargs='+', type=str, default=None,
                help='Which regions to compute indices for. Default is to use all regions in maskfile.')

    parser.add_argument('--inputdir',type=str,default='/Data/gfi/share/ModData/CMIP_EU_Precip_Precursors/',
                        help='Directory in which to look for field data.')
    
    parser.add_argument('--auxdir',type=str,default='/Data/gfi/share/ModData/CMIP_EU_Precip_Precursors/aux/',
                    help='Directory in which to look for mask file.')
    
    parser.add_argument('--savedir',type=str,default='/Data/gfi/share/ModData/CMIP_EU_Precip_Precursors/indices/',
                    help='Directory in which to save output.')

    parser.add_argument('--maskname',type=str,default='ERA5_rainfall_regions.nc',
                        help='filename of mask netcdf.')

    return parser.parse_args(arg_list)

def get_save_path(args):
    savedir=f'{args.savedir}{args.model}/{args.variable}/{args.experiment}/'
    os.makedirs(savedir,exist_ok=True)

    if args.member!='':
        savedir=savedir+f'member_{args.member}_'
    
    return savedir

def load_input_field(args):
    indir=f'{args.inputdir}raw/{args.model}/{args.variable}/{args.experiment}/'
    filenames=[file for file in os.listdir(indir) if file.endswith('.nc')]
    if args.member=='': #single model case
        if len(filenames)>1:
            raise(IOError(f'Expected single file in input directory {indir} when --member flag is absent. Found {len(filenames)}.'))

    else: #ensemble case
        filenames=[file for file in filenames if file.find(f'_{args.member}_')!=-1]
        if len(filenames)>1:
            raise(IOError(f'Expected to find only one file in input directory {indir} containing string _{args.member}_. Found {len(filenames)}.'))
    file_path = indir+filenames[0]
    data=xr.open_dataset(file_path,
        chunks=dict(time=-1,lat=-1,lon=-1))[args.variable].chunk('auto')
    # data=xr.open_dataset(file_path,
    #     chunks=dict(time=365*10,lat=30,lon=30))[args.variable]
    return data

def apply_region_masking_and_average(mask,field,regions,lon='lon',lat='lat'):

    weights=np.cos(np.deg2rad(field[lat]))
    indices=[]

    for r in regions:
        M=((mask==r).astype(int).interp_like(field)>0)
        masked_field=field.where(M==1)
        collapsed_da=masked_field.mean(lon).weighted(weights).mean(lat)
        indices.append(collapsed_da.assign_coords(mask_id=r))

    return indices

def split_and_save_indices(indices,outdir,regions,args):

    for index,r in zip(indices,regions):
        for s in args.seasons:
            in_season=index['time.season']==s
            ix=index.isel(time=in_season)
            savepath=outdir+f'{s}_region{r}.nc'
            if os.path.isfile(savepath) and not args.overwrite:
                print(f'{savepath} exists and --overwrite not set. Passing.')
            else:
                ix.to_netcdf(savepath)
    return

def ensure_lon_180(da):
    try:
        lon=da.lon
    except:
        raise(ValueError('This script assumes the model data has a "lon" coordinate.'))
    
    #this does nothing to a da which already is in -180 to 180 degree longitude
    da=da.assign_coords(lon = (lon + 180) % 360 - 180)
    da=da.sortby("lon")
    return da

def to_mm_day(da):
    """
    Load and convert a precipitation DataArray to mm/day.
    Handles mass flux (e.g., kg/m^2/s) via water density, then strips units.
    """
    
    # Some unit fixes to make pint work
    # if da.attrs['units'] == 'kg m-2 s-1':
    #     da.attrs['units'] = 'kg/m^2/s'
    # for coord in ['lat', 'lon']:
    #     if 'units' in da.coords[coord].attrs:
    #         da.coords[coord].attrs['units'] = 'degrees'
    # da.attrs['units'] = 'kg / m 2/ s'
    da = da.pint.quantify()
    try:
        # Try direct conversion (e.g. mm/day, m/s)
        da = da.pint.to("mm/day")

    except:

        # Handle mass flux: convert kg/m^2/s → m/s by dividing by water density
        water_density = 1000 * ureg("kg / m^3")
        da = (da / water_density).pint.to("mm/day")

    # Strip units which can mess up sums, return plain DataArray
    return da.pint.dequantify()

if __name__=='__main__':

    #use multi-core for speed
    cluster = LocalCluster(n_workers=4, memory_limit='8GiB')
    client = Client(cluster)
    print('Access dask dashboard: ', client.dashboard_link)

    args = parse_args()
    args.variable='pr' #hardcoding this for now.

    #work out the full save directory
    outdir=get_save_path(args)

    #load mask with region ids encoded
    mask_path=args.auxdir+args.maskname
    mask=xr.open_dataarray(mask_path).load()

    #use all regions in the mask if none specified
    if args.regions is None:
        regions=np.unique(mask.values)
        regions=[int(r) for r in regions if not np.isnan(r)]
    else:
        regions=args.regions

    #load model precip data and interpolate it onto the mask grid
    targ_field=load_input_field(args)
    targ_field=ensure_lon_180(targ_field)
    
    #This is an efficient way to do this, subselecting to the region around
    #the mask and then interpolating to the higher res.
    targ_field = targ_field.load()


    interpolated_targ_field=targ_field.sel(
            lat=slice(float(mask.lat.min()-1), float(mask.lat.max()+1)),
            lon=slice(float(mask.lon.min())-1, float(mask.lon.max()+1))
        ).interp_like(mask)

    interpolated_targ_field=to_mm_day(interpolated_targ_field)    
    sys.exit()
    interpolated_targ_field = interpolated_targ_field.load()
    

    #do the area averaging and save
    targ_indices = apply_region_masking_and_average(mask,interpolated_targ_field,regions)
    split_and_save_indices(targ_indices,outdir,regions,args)
    sys.exit()
