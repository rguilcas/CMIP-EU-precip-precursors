
import pandas as pd
import numpy as np
import glob 
import os 


def postprocess_csv(model_path):
    members = os.listdir(model_path)


if __name__ == "__main__":
    # Define the path to the directory containing the CSV files
    all_data = []
    model_path = "/Data/skd/projects/global/cmip6_precursors/outputs/decompositions/ACCESS-CM2/"
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        for region in [1] + [k for k in range(3,40)]:
            try:
                df = pd.read_csv(f"/Data/skd/projects/global/cmip6_precursors/outputs/decompositions/ACCESS-CM2//terms_{season}_region{region}.csv", index_col=0)
                all_data.append(df)
            except FileNotFoundError:
                print(f"File not found for region {region}, season {season}")
                all_data.append(pd.DataFrame())  # Append an empty DataFrame if the file is not found
                pd.DataFrame([
                    ['NorESM2-MM',season,region,'conversion','bias',0,],
                    ['NorESM2-MM',season,region,'conversion','change',0,],
                    ['NorESM2-MM',season,region,'conversion','uncalibrated_change',0,],
                    ['NorESM2-MM',season,region,'dynamical','bias',0,],
                    ['NorESM2-MM',season,region,'dynamical','change',0,],
                    ['NorESM2-MM',season,region,'dynamical','uncalibrated_change',0,],
                    ['NorESM2-MM',season,region,'nonlinear','bias',0,],
                    ['NorESM2-MM',season,region,'nonlinear','change',0,],
                    ['NorESM2-MM',season,region,'nonlinear','uncalibrated_change',0,],
                ], columns = ['model','season','region_id','source','term','value'])
    df_out = pd.concat(all_data, ignore_index=True)
    df_out_members = df_out.copy()
    df_out_members.insert(loc = 5,column = 'member',value = 'r1i1p1f1')
    df_out_members.to_csv('/home/rogui7909/code/CMIP_precursors/interactive_plot/cmip6-interactive/data/results/ensemble/ACCESS-CM2_paper_terms_ens_df.csv')
    df_out.to_csv('/home/rogui7909/code/CMIP_precursors/interactive_plot/cmip6-interactive/data/results/ensemble_mean/ACCESS-CM2_paper_terms_df.csv')
    # Get a list of all CSV files in the directory
    postprocess_csv(model_path)
