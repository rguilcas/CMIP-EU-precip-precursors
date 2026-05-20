import os
import numpy as np
import pandas as pd
import argparse
def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description='Compute terms from precomputed decomposition CSVs.')

    parser.add_argument('--savedir', type=str, required=True,
                        help='Base directory for output decompositions.')
    
    parser.add_argument('--hist_model', type=str, required=True,
                        help='name of model decomp.')
    
    parser.add_argument('--hist_member', type=str, required=True,
                help='name of historical member/ensemble.')
    
    parser.add_argument('--future_model', type=str, default=None,
                        help='name of future decomp. if any')

    parser.add_argument('--future_experiment', type=str, default=None,
                    help='name of future experiment, must be provided if --future_model set.')
    
    parser.add_argument('--future_member', type=str, default=None,
                    help='name of future member/ensemble, must be provided if --future_model set.')

    parser.add_argument('--hist_experiment', type=str, default='historical',
                    help='name of historical experiment.')

    parser.add_argument('--ref_model', type=str, default='ERA5',
                        help='name of reference decomp.')
    
    parser.add_argument('--ref_experiment', type=str, default='',
                    help='name of reference experiment, if any.')
    
    parser.add_argument('--ref_member', type=str, default='',
                help='name of reference member/ensemble, if any.')

    # Execution control
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files.')

    # Filtering
    parser.add_argument('--regions', nargs='+', type=int, default=None,
                        help='Which regions to compute metrics for. Default is to use all regions.')
    parser.add_argument('--seasons', nargs='+', type=str, default=['DJF', 'MAM', 'JJA', 'SON'],
                        help='Seasons to process.')

    return parser.parse_args(arg_list)


def get_decomp_paths(args):

    ref_path=f'{args.savedir}/decompositions/{args.ref_model}/{args.ref_experiment}/{args.ref_member}/decomp.csv'
    hist_path=f'{args.savedir}/decompositions/{args.hist_model}/{args.hist_experiment}/{args.hist_member}/decomp.csv'

    if args.future_model:
        future_path=f'{args.savedir}/decompositions/{args.future_model}/{args.future_experiment}/{args.future_member}/decomp.csv'
    else:
        future_path=None # no future experiment indicated by a None path.
    return ref_path, hist_path, future_path

def get_terms_savepath(args, suff='csv'):

    if (args.future_model is None) or (args.hist_model==args.future_model):
        model_lab=args.hist_model
        
    else:
        model_lab='_'.join([args.hist_model,args.future_model])

    if args.future_model is None:
        mem_lab='_'.join([args.hist_experiment,args.hist_member])
    
    else:
        mem_lab='_'.join([args.hist_experiment,args.hist_member,
                          args.future_experiment,args.future_member])


    output_path=f'{args.savedir}/terms/{model_lab}/{mem_lab}/'
    os.makedirs(output_path,exist_ok=True)
    output_path+=f'{model_lab}_{mem_lab}_paper_terms_df.{suff}'
    return output_path , model_lab, mem_lab   

def load_and_collate_decomp_data(ref_path,hist_path,future_path):

    ref_df=pd.read_csv(ref_path,index_col=0)
    ref_df['period']='ref'

    hist_df=pd.read_csv(hist_path,index_col=0)
    hist_df['period']='hist'

    df_arr=[ref_df,hist_df]

    if future_path:
        future_df=pd.read_csv(future_path,index_col=0)
        future_df['period']='future'
        df_arr.append(future_df)

    df=pd.concat(df_arr,ignore_index=True)
    return df

def prepare_decomposed_data(region_id,season,decomp_df):
    case_df=decomp_df.query(
        f"season == '{season}' and region_id == {region_id}"
    )
    
    P_s_0=np.array(case_df.query("period == 'ref' and source == 'dynamical'"
                ).sort_values(by='bin',ascending=True)['value'])

    Ph_s_0=np.array(case_df.query("period == 'ref' and source == 'conversion'"
                ).sort_values(by='bin',ascending=True)['value'])

    P_s_h=np.array(case_df.query("period == 'hist' and source == 'dynamical'"
                ).sort_values(by='bin',ascending=True)['value'])

    Ph_s_h=np.array(case_df.query("period == 'hist' and source == 'conversion'"
                ).sort_values(by='bin',ascending=True)['value'])

    P_s_f=np.array(case_df.query("period == 'future' and source == 'dynamical'"
                ).sort_values(by='bin',ascending=True)['value'])

    Ph_s_f=np.array(case_df.query("period == 'future' and source == 'conversion'"
                ).sort_values(by='bin',ascending=True)['value'])

    if len(P_s_f)==0:
        P_s_f=None
        Ph_s_f=None

    return [P_s_0,Ph_s_0,P_s_h,Ph_s_h,P_s_f,Ph_s_f]

def terms_from_decomp(decomp):

    def blending_function(Ph_s_h, Ph_s_0, pow, mu):
        x = Ph_s_h / Ph_s_0
        return x**pow / (x**pow + mu**pow)

    P_s_0,Ph_s_0,P_s_h,Ph_s_h,P_s_f,Ph_s_f = decomp

    epsilon = 1e-12
    Ph_s_0 = Ph_s_0 + epsilon
    Ph_s_h = Ph_s_h + epsilon

    
    xi = (Ph_s_h / Ph_s_0) - 1
    delta_P_s = P_s_h - P_s_0


    data = dict(
        dyn_bias=(delta_P_s * Ph_s_0, 'dynamical', 'bias'),
        cnv_bias=(P_s_0 * xi * Ph_s_0, 'conversion', 'bias'),
        nlr_bias=(delta_P_s * xi * Ph_s_0, 'nonlinear', 'bias'),
    )

    if P_s_f is not None:
        blending_pow = 4
        blending_param = 0.1
        w = blending_function(Ph_s_h, Ph_s_0, blending_pow, blending_param)

        Delta_P_s = P_s_f - P_s_h

        alpha = (1 - w) * Ph_s_f / Ph_s_0 + w * ((Ph_s_f / Ph_s_h) - 1)
        raw_alpha = (Ph_s_f / Ph_s_h) - 1

        trend_data = dict(
            dyn_caltrend=(Delta_P_s * Ph_s_0, 'dynamical', 'change'),
            cnv_caltrend=(P_s_0 * alpha * Ph_s_0, 'conversion', 'change'),
            nlr_caltrend=(Delta_P_s * alpha * Ph_s_0, 'nonlinear', 'change'),
            dyn_rawtrend=(Delta_P_s * Ph_s_h, 'dynamical', 'uncalibrated_change'),
            cnv_rawtrend=(P_s_h * raw_alpha * Ph_s_h, 'conversion', 'uncalibrated_change'),
            nlr_rawtrend=(Delta_P_s * raw_alpha * Ph_s_h, 'nonlinear', 'uncalibrated_change'),
        )
        data = data | trend_data

    term_rows = []
    for vals, source, term in data.values():
        for b, value in enumerate(vals, 1):
            term_rows.append(
                {
                    'bin': b,
                    'source': source,
                    'term': term,
                    'value': value,
                }
            )

    return pd.DataFrame(term_rows)


def postprocess_term_dfs(df_arr):
    df= pd.concat(df_arr,ignore_index=True)
    #sum the terms over all bins, prior to saving
    summed_terms_df=df.groupby(
        ["season","region_id", "source", "term"], 
        as_index=False
    )["value"].sum()
    return summed_terms_df

def save_term_df(df,path):
    print('saving:')
    print(path)
    df.to_csv(path)

def main(args):
                         #if future_path is None, then we do a bias only analysis.
    ref_path, hist_path, future_path = get_decomp_paths(args)
    term_output_path, model_lab, mem_lab = get_terms_savepath(args)

    if os.path.isfile(term_output_path) and not args.overwrite:
        print(f'Term file {term_output_path} exists, and --overwrite flag not set. Continuing.')
        exit()

    decomp_df=load_and_collate_decomp_data(ref_path, hist_path, future_path)

    if args.regions is None:
        args.regions = [1, *np.arange(3, 40)]

    terms_dataframes=[]
    for season in args.seasons:
        for region_id in args.regions:

            decomposed_hazard=prepare_decomposed_data(region_id,season,
                decomp_df)

            terms_df = terms_from_decomp(decomposed_hazard)

            terms_df['region_id']=region_id
            terms_df['season']=season
            terms_dataframes.append(terms_df)
            
    terms_dataframe=postprocess_term_dfs(terms_dataframes)
    terms_dataframe['model']=model_lab# removing these two lines
    terms_dataframe['member']=mem_lab #can halve the file size. Consider!!

    save_term_df(terms_dataframe,term_output_path)

if __name__=='__main__':

    args = parse_args()
    main(args)


