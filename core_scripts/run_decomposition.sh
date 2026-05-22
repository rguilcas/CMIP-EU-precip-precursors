#!/bin/bash
set -e

##USAGE:
# bash run_decomposition.sh --model MODEL_NAME --members "MEMBER1 MEMBER2 ..."
# Optional arguments:
# --future_members "FUTURE_MEMBER1 FUTURE_MEMBER2 ..." (defaults to same as --members if not provided)
# --histexp HISTORICAL_EXPERIMENT_NAME (default: historical)
# --futureexp FUTURE_EXPERIMENT_NAME (default: ssp370)
# --histperiod HISTORICAL_START_YEAR HISTORICAL_END_YEAR (default: 1979 2014)
# --futureperiod FUTURE_START_YEAR FUTURE_END_YEAR (default: 2060 2100)     

# Initialize variables with default values
MODEL="" #required, as --model
MEMBERS="" # required as --members
FUTMEMBERS="" # optional as --future_members, default to same as --members if not provided
HISTEXP="historical" #optional as --histexp
FUTUREEXP="ssp370" #optional as --futureexp, if "none" is passed, will skip future decomposition and just compute historical terms
HISTPERIOD=(1979 2014) # optional as --histperiod
FUTUREPERIOD=(2060 2100) # optional as --futureperiod
ENSNAME="ens"
INPUTDIR="/Data/skd/projects/global/cmip6_precursors/outputs/indices/"
SAVEDIR="/Data/skd/projects/global/cmip6_precursors/outputs/"
AUXDIR="/Data/skd/projects/global/cmip6_precursors/aux/decomp_parameter_files/"
REFPCA="${AUXDIR}/ERA5_historical_z500_lag0_index_val1_u850_lag0_index_val1_v850_lag0_index_val1_PCA_patterns.npz"
REFBINS="${AUXDIR}/ERA5_historical_PC1_n10_bin_edges.npz"
REFTHRESH="${AUXDIR}/ERA5_historical_pr_p0.95_thresholds.npz"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --members) MEMBERS=($2); shift ;;
        --future_members) FUTMEMBERS=($2); shift ;;
        --histexp) HISTEXP="$2"; shift ;;
        --futureexp) FUTUREEXP="$2"; shift ;;
        --histperiod) HISTPERIOD=($2); shift ;;
        --futureperiod) FUTUREPERIOD=($2); shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

#if FUTMEMBERS is still the default of "", set it to the same as MEMBERS:
if [ -z "$FUTMEMBERS" ]; then
  FUTMEMBERS=${MEMBERS[@]}
fi

##We assume the ERA5 data is in place, but this can be uncommented to re-run / extend as needed. 
##Note that the reference PCA, bins, and thresholds will need to be re-computed if the historical period is changed, 
##or if the input data is changed in a way that would impact the PCA patterns or index distributions (e.g. using a different variable, or changing the lag).
python decompose_precip.py --model ERA5 --experiment historical --time_period ${HISTPERIOD[@]} \
 --inputdir $INPUTDIR \
 --savedir $SAVEDIR \
 --auxdir $AUXDIR \
 --ref 

python decompose_precip.py --model $MODEL --experiment $HISTEXP --time_period ${HISTPERIOD[@]} \
 --members ${MEMBERS[@]} \
 --inputdir $INPUTDIR \
 --savedir $SAVEDIR \
 --auxdir $AUXDIR \
 --ref_pca $REFPCA \
 --ref_bins $REFBINS \
 --ref_thresh $REFTHRESH \
 #--overwrite 

#not if futureexp is "none":
if [ "$FUTUREEXP" != "none" ]; then 
    python decompose_precip.py --model $MODEL --experiment $FUTUREEXP --time_period ${FUTUREPERIOD[@]} \
    --members ${FUTMEMBERS[@]} \
    --inputdir $INPUTDIR \
    --savedir $SAVEDIR \
    --auxdir $AUXDIR \
    --ref_pca $REFPCA \
    --ref_bins $REFBINS \
    --ref_thresh $REFTHRESH \
    #--overwrite 

    #if len of MEMBERS is greater than 1, set  members to ENSNAME:
    if [ ${#MEMBERS[@]} -gt 1 ]; then
        MEMBERS=($ENSNAME)
    fi
    #if len of FUTMEMBERS is greater than 1, set  members to ENSNAME:
    if [ ${#FUTMEMBERS[@]} -gt 1 ]; then
        FUTMEMBERS=($ENSNAME)
    fi

    python compute_terms.py \
    --ref_model ERA5 --hist_model $MODEL --future_model $MODEL \
    --hist_experiment $HISTEXP --future_experiment $FUTUREEXP \
    --hist_member ${MEMBERS[@]} --future_member ${FUTMEMBERS[@]} \
    --savedir $SAVEDIR --overwrite

else
    #if len of MEMBERS is greater than 1, set  members to ENSNAME:
    if [ ${#MEMBERS[@]} -gt 1 ]; then
        MEMBERS=($ENSNAME)
    fi

    python compute_terms.py \
    --ref_model ERA5 --hist_model $MODEL  \
    --hist_experiment $HISTEXP \
    --hist_member ${MEMBERS[@]} \
    --savedir $SAVEDIR --overwrite
fi
