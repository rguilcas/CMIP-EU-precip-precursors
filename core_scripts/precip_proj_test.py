import xarray as xr 
import numpy as np
import os
import flox.xarray
from dask.distributed import Client, LocalCluster



def main():
    ds = xr.open_dataset('/Data/gfi/share/ModData/CMIP_EU_Precip_Precursors/raw/IPSL-CM6A-LR/pr/historical/pr_day_IPSL-CM6A-LR_historical_r1i1p1f1_gn_18500101-20141231.nc', chunks = dict(time=365, lon=10, lat=10))['pr']
    ds = ensure_lon_180(ds)
    mask = xr.open_dataset('/Data/gfi/share/ModData/CMIP_EU_Precip_Precursors/aux/ERA5_rainfall_regions.nc').tp
    ds_masked = ds.sel(lat=slice(float(mask.lat.min()-1), float(mask.lat.max()+1)),lon=slice(float(mask.lon.min())-1, float(mask.lon.max()+1))).isel(time=0)
    ones = xr.ones_like(ds_masked)
    weights_on_mask_grid = xr.ones_like(mask)*np.cos(np.deg2rad(mask.lat))

    lon_bin_size = ones.lon.diff('lon').isel(lon=0).values/2
    lon_bins = list((ones.lon - lon_bin_size).values) + [ones.lon.isel(lon=-1).values+lon_bin_size]
    lat_bin_size = ones.lat.diff('lat').isel(lat=0).values/2
    lat_bins = list((ones.lat - lat_bin_size).values) + [ones.lat.isel(lat=-1).values+lat_bin_size]



    mask_reduced = flox.xarray.xarray_reduce(
        # variable qu'on veut aggréger 
        weights_on_mask_grid,
        # Variables selon lesquelles on veut grouer : ici il y a deux variables mais on peut en rajouter autant qu'on veut
        mask.lat, mask.lon, mask,
        # Fonction à appliquer à chacun des groupes pour aggrégation
        func="sum",
        # Groupes attendus pour chacune des variables : soit on met les bins, soit on met None quand on veut grouper pour toutes les valeurs possibles
        expected_groups=(lat_bins,
                        lon_bins,
                        np.unique(mask),),
        # Déterminer si les groupes données correspondent à des bins ou non
        isbin=[True, True, False,],
        # Méthode de calcul parallèl : flox propose plein de méthodes différentes pour optimiser les calculs en fonction de la forme des données
        method="map-reduce",
    )
    mask_reduced = mask_reduced.rename(lon_bins='lon', lat_bins='lat').load()
    mask_reduced = mask_reduced.assign_coords(lon=[k.mid for k in mask_reduced.lon.values], 
                            lat=[k.mid for k in mask_reduced.lat.values])
    mask_reduced = mask_reduced.isel(tp=slice(None, -1))
    ds = ds.sel(lon=mask_reduced.lon, lat=ones.lat.values)
    ds['lat'] = mask_reduced.lat
    timeseries = ds.weighted(mask_reduced.fillna(0)).mean(['lon','lat']).load()
    print(timeseries.sel(tp=14))


def ensure_lon_180(da):
    try:
        lon=da.lon
    except:
        raise(ValueError('This script assumes the model data has a "lon" coordinate.'))
    
    #this does nothing to a da which already is in -180 to 180 degree longitude
    da=da.assign_coords(lon = (lon + 180) % 360 - 180)
    da=da.sortby("lon")
    return da

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=16, memory_limit='4GiB')
    client = Client(cluster)
    print('Access dask dashboard: ', client.dashboard_link)
    main()