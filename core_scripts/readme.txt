These scripts provide end-to-end projection and decomposition of precursor indices in climate model datasets.

The pipeline is as follows:


precip_projection.py: computes scalar time series of precipitation aggregated over specified regions.
z500_detrend.py: applies a variable specific detrending step, to remove area-mean, dynamically-irrelevant thickening trend.
precursor_projection.py: computes precursor indices for specified variables.
decompose_precip.py: Decomposes heavy precipitation occurrence into dynamical and conversion components, with statistics estimated using 1 or more ensemble members.
compute_terms.py: Uses a ref, historical and [optional] future decomposition to compute final decomposition terms for plotting and analysis.


The run_decomposition.sh script wraps decompose_precip.py and compute_terms.py. Example usage:

#Apply decomposition to a single member:
bash run_decomposition.sh --model NorESM2-LM --members "r1i1p1f1"

#Apply decomposition to an ensemble:
bash run_decomposition.sh --model NorESM2-LM --members "r1i1p1f1 r2i1p1f1 r3i1p1f1"

#Compute biases using an ensemble, and then use those when computing future changes for a single member:
bash run_decomposition.sh --model NorESM2-LM --members "r1i1p1f1 r2i1p1f1 r3i1p1f1" --future_members "r1i1p1f1"

