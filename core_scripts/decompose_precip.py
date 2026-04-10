import os
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.decomposition import PCA
import argparse


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Use precursor indices and precipitation indices to decompose model bias and trends."
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Name of model to decompose.')

    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of model experiment.')

    parser.add_argument('--time_period', type=int, nargs=2, required=True,
                        help='The time period to use, given as 2 years (inclusive), e.g. 1979 2015.')

    # Paths
    parser.add_argument('--inputdir', type=str, default='/Data/skd/projects/global/cmip6_precursors/outputs/indices/',
                    help='Directory in which to look for indices.')

    parser.add_argument('--savedir', type=str, required=True,
                        help='Base directory for output decompositions.')
    
    parser.add_argument('--auxdir', type=str, required=True,
                        help='Directory to store auxiliary artifacts like PCA patterns and bin edges.')

    # Data configuration
    parser.add_argument('--variables', nargs='+', type=str, default=['z500','u850','v850'],
                        help='List of precursor variable names.')
    
    parser.add_argument('--hazardvariable', type=str, default='pr',
                        help='Name of the hazard variable to decompose.')

    # Ensemble configuration
    parser.add_argument('--members', nargs='+', type=str, default='r1i1p1f1',
                        help='Which ensemble member or members to use. Defaults to r1i1p1f1. If multiple members are passed, output folder will be specified by "--ensname"')

    parser.add_argument('--ensname',type=str,default='ens',help='specifies name of member directory where the ensemble will be stored.')

    # Reference / precomputed parameters
    parser.add_argument('--ref', action='store_true',
                        help='If set, treat current run as reference (no experiment or member suffix).')
    parser.add_argument('--ref_pca', type=str, default=None,
                        help='Path to precomputed PCA patterns (optional).')
    parser.add_argument('--ref_bins', type=str, default=None,
                        help='Path to precomputed bin edges (optional).')
    parser.add_argument('--ref_thresh', type=str, default=None,
                        help='Path to precomputed categorical threshold (optional).')

    # Decomposition parameters
    parser.add_argument('--nprecursorbins', type=int, default=10,
                        help='Number of bins for precursor decomposition.')
    parser.add_argument('--eventthreshold', type=float, default=0.95,
                        help='Threshold for converting hazard to categorical event.')

    # Execution control
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files.')

    # Filtering
    parser.add_argument('--regions', nargs='+', type=int, default=None,
                        help='Which regions to compute metrics for. Default is to use all regions.')
    parser.add_argument('--seasons', nargs='+', type=str, default=['DJF', 'MAM', 'JJA', 'SON'],
                        help='Seasons to process.')

    return parser.parse_args(arg_list)


def get_decomp_output_path(args, suff='csv'):
    model=args.model

    if args.ref:
        path= f'{args.savedir}/decompositions/{model}/decomp.{suff}'
        return path
    
    exp=args.experiment
    members=np.atleast_1d(args.members)
    if len(members)>1:
        mem_lab=args.ensname
    else:
        mem_lab=members[0]
        
    path = f'{args.savedir}/decompositions/{model}/{exp}/{mem_lab}/decomp.{suff}'

    return path

def get_index_and_hazard_data(args):

    bp=f'{args.inputdir}{args.model}/'
    vars=list(np.atleast_1d(args.variables)) + [args.hazardvariable]
    ds=[]
    mems=np.atleast_1d(args.members)
    
    for v in vars:
        vdir=f'{bp}{v}/{args.experiment}/'
        os.makedirs(vdir,exist_ok=True)
        a0=[]
        for mem in mems:
            if args.ref:
                dir = vdir
            else:
                dir = vdir + f'member_{mem}_'
            os.makedirs(dir,exist_ok=True)
            a1=[]
            for r in args.regions:
                a2=[]
                for s in args.seasons:
                    da=xr.open_dataarray(f'{dir}{s}_region{r}.nc')
                    a2.append(da.assign_coords(season=s))
                a1.append(xr.concat(a2, 'season', coords='different', compat='equals', join='outer').assign_coords(region_id=r))

            a1=xr.concat(a1, 'region_id', coords='different', compat='equals', join='outer')
            if not args.ref: a1=a1.assign_coords(member=mem)
            a0.append(a1)
        a0=xr.concat(a0, 'member', coords='different', compat='equals', join='outer')#.rename(v)
        ds.append(a0)

    ds=xr.merge(ds, compat='override', join='outer')
    y0,y1=args.time_period
    ds = ds.sel(time=slice(str(y0), str(y1)))
    try:
        ds=ds.rename(tp='pr')
    except:
        ds.pr #fails if pr does not exist on ds.
    ds=ds.dropna('time',how='all')
    return ds

def compute_pca_pattern(ds,condition_var,p_dvs):

    if not condition_var.startswith('PC'):
        return None
    
    N = int(condition_var[2:])
    solver = PCA(n_components=N, whiten=True).fit(
        ds[p_dvs].squeeze().to_array('feature').T)
    PCA_pattern=solver.components_
    return PCA_pattern # a 2d numpy array

def handle_PCA_projection(ds,condition_var,p_dvs,PCA_pattern):

    if not condition_var.startswith('PC'):
        return ds
    
    X=ds[p_dvs].to_array('feature').T
    N = int(condition_var[2:])

    PC=(X.values @ PCA_pattern.T)[:,N-1][:,0]
    ds[condition_var]=xr.DataArray(data=PC,coords=dict(time=ds.time))
    return ds

def compute_bin_edges(ds, condition_var, bin_num):
    #our bins are based on the reference ds. We've modified this to include two unobserved bins, at the high and low end. 
    bins=np.array([-1000,*ds[condition_var].quantile(np.linspace(0,1,bin_num+1)),1000])
    bins[1]=bins[1]-0.01 # so min ref value is not included in unobserved first bin
    return bins #a list of floats

def compute_ref_thresh(ds, hazard_var, quantile):
    quantile_thresh=ds[hazard_var].quantile(quantile)
    return quantile_thresh #just a float

def handle_categorical_target(ds, h_var, is_cat, thresh):
    if not is_cat:
        return ds, h_var
    
    ds[h_var+'_cat']=(ds[h_var]>thresh).astype(int)
    h_var=h_var+'_cat'

    return ds, h_var

def apply_decomposition(ds, h_var, s_var, bins):
    Ph_s=ds.groupby_bins(s_var,bins=bins).mean()[h_var].fillna(0) #average value of hazard in bin. Is a probability for binary data. Bins with no hazard risk get a 0
    P_s=ds.groupby_bins(s_var,bins=bins).count()[s_var].fillna(0)/ds[s_var].time.size #occurence prob of synoptic bin. Bins that don't occur get a 0.

    return Ph_s, P_s

def decomp_to_df(Ph_s,P_s,model,season,region_id):
    
    data=dict(
        dyn= (P_s,'dynamical'),
        conv= (Ph_s,'conversion')
    )
    decomp_rows=[]
    for name, (vals, s1) in data.items():
        for b, v in enumerate(vals, 1):
            decomp_rows.append({"model": model, "season": season,
                        "region_id":region_id, "bin": b,
                        "source": s1, "value": np.atleast_1d(v.values)[0]})
                        
    return pd.DataFrame(decomp_rows)

def postprocess_decomp_dfs(df):
    return pd.concat(df,ignore_index=True)

def save_decomp_df(df,output_path):
    os.makedirs('/'.join(output_path.split('/')[:-1]),exist_ok=True)
    print('saving:')
    print(output_path)
    df.to_csv(output_path)
    return

def save_ensemble_documentation(output_path, members):
    if len(np.atleast_1d(members))<=1:
        return
    dirs=output_path.split('/')
    filename=dirs.pop(-1)

    output_dir='/'.join(dirs)
    doc_path=output_dir+f'members_included_in_{filename}'
    
    df=pd.DataFrame(data=list(members),columns=['member'])
    df.to_csv(doc_path)
    return

def save_pca(pca_dict,auxdir,model, experiment, p_dvs):
    filename=f'{model}_{experiment}_'
    for v in p_dvs:
        filename+=f'{v}_'
    filename+='PCA_patterns.npz'
    np.savez(auxdir+filename,**pca_dict)
    return
"""
def load_pca_pattern(auxdir,model,experiment,p_dvs):
    filename=f'{model}_{experiment}_'
    for v in p_dvs:
        filename+=f'{v}_'
    filename+='PCA_patterns.npz'
    return dict(np.load(auxdir+filename))

def load_bin_edges(auxdir, model, experiment, condition_var, binnum):
    filename=f'{model}_{experiment}_{condition_var}_n{binnum}_bin_edges.npz'
    return dict(np.load(auxdir+filename))

def load_cat_thresh(auxdir, model, experiment,hazardvar,eventthreshold):
    filename=f'{model}_{experiment}_{hazardvar}_p{eventthreshold}_thresholds.npz'
    return dict(np.load(auxdir+filename))

"""
def save_bin_edges(bin_dict, auxdir, model, experiment, condition_var, binnum):
    filename=f'{model}_{experiment}_{condition_var}_n{binnum}_bin_edges.npz'
    np.savez(auxdir+filename,**bin_dict)
    return

def load_param_data(path):
    return dict(np.load(path))

def save_cat_thresh(thresh_dict, auxdir, model, experiment,hazardvar,eventthreshold):

    filename=f'{model}_{experiment}_{hazardvar}_p{eventthreshold}_thresholds.npz'
    np.savez(auxdir+filename,**thresh_dict)
    return


def validate_sel_data(ds,s,r):

    if s=='JJA' and r==25: # a known missing case.
        return 1
    
    nan_vals=ds.isnull().sum().to_array('feature')
    for dv,c in zip(nan_vals.feature.values,nan_vals.values):
        if c!=0: 
            N=ds[dv].size
            print(f'Warning: {c} nan values out of {N} for variable {dv} in data {s} region {r}.')
    return 0

def main(args):

    decomp_output_path=get_decomp_output_path(args)
    if os.path.isfile(decomp_output_path) and not args.overwrite:
        print(f'Decomp. file {decomp_output_path} exists, and --overwrite flag not set. Exiting.')
        exit()


    condition_var = args.variables[0] if len(args.variables) == 1 else 'PC1'
    p_dvs = [v.split('_detrend')[0] + '_lag0_index_val1' for v in args.variables]
    make_h_var_cat = True

    if args.regions is None:
        args.regions = [1, *np.arange(3, 40)]


    #these are each region and season specific
    ref_pca_proj= (args.ref_pca is not None)
    ref_bin_edges= (args.ref_bins is not None)
    ref_cat_thresh = (args.ref_thresh is not None)

    pca_patterns=None
    bins=None
    threshs=None

    pca_dict={}
    bin_dict={}
    thresh_dict={}

    #If we've predefined the PCs, bins or thresh, we load them now.
    if ref_pca_proj:
        
        """pca_dict=load_pca_pattern(args.auxdir, args.model, args.experiment, p_dvs)"""
        pca_dict=load_param_data(args.ref_pca)
    if ref_bin_edges:
        """bin_dict = load_bin_edges(
            args.auxdir,
            args.model,
            args.experiment,
            condition_var,
            args.nprecursorbins,
        )"""
        bin_dict=load_param_data(args.ref_bins)

    if ref_cat_thresh:
        """thresh_dict = load_cat_thresh(
            args.auxdir,
            args.model,
            args.experiment,
            args.hazardvariable,
            args.eventthreshold,
        )"""
        thresh_dict=load_param_data(args.ref_thresh)

    data=get_index_and_hazard_data(args)
    decomp_dfs=[]


    for season in args.seasons:
        for region_id in args.regions:

            key = f"{season}_{region_id}"

            sel_data=data.sel(season=season,region_id=region_id).dropna('time',how='all')
            data_val=validate_sel_data(sel_data,season,region_id)
            if data_val==1:
                continue
            #if we did not predefine PCs, bins or thresh, we compute them now
            if not ref_pca_proj:
                pca_pattern=compute_pca_pattern(sel_data,condition_var,p_dvs)
                pca_dict[key]=pca_pattern

            #either way, the param dicts are now populated:
            sel_data=handle_PCA_projection(sel_data,condition_var,p_dvs,pca_dict[key])

            if not ref_bin_edges:
                bins=compute_bin_edges(sel_data,condition_var,args.nprecursorbins)
                bin_dict[key]=bins

            h_var = args.hazardvariable

            if not ref_cat_thresh:
                thresh=compute_ref_thresh(sel_data, h_var, args.eventthreshold)
                thresh_dict[key]=thresh

            sel_data,h_var=handle_categorical_target(sel_data, h_var, make_h_var_cat,
                                                     thresh_dict[key])
            
            Ph_s, P_s = apply_decomposition(sel_data,h_var, condition_var, bin_dict[key])
            decomp_dfs.append(
                decomp_to_df(Ph_s, P_s, args.model, season, region_id)
            )

    decomp_df=postprocess_decomp_dfs(decomp_dfs)
    save_decomp_df(decomp_df,decomp_output_path)
    save_ensemble_documentation(decomp_output_path,args.members)
    if not ref_pca_proj:
        save_pca(pca_dict,args.auxdir,args.model, args.experiment, p_dvs)

    if not ref_bin_edges:
        save_bin_edges(
            bin_dict, args.auxdir,
            args.model, args.experiment,
            condition_var,args.nprecursorbins,
        )
    if not ref_cat_thresh:
        save_cat_thresh(
            thresh_dict, args.auxdir,
            args.model, args.experiment,
            args.hazardvariable, args.eventthreshold,
        )
if __name__=='__main__':

    args = parse_args()
    main(args)


