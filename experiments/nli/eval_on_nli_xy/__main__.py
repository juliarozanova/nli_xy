#%%
# %load_ext autoreload
# %autoreload 2
import os
import sys
import torch
from pathlib import Path
import pandas as pd

#%%
nli_xy_root = Path(__file__).parent.parent.parent.parent
os.chdir(nli_xy_root)
sys.path.append('.')
from nli_xy.encoding import parse_encode_config
from nli_xy.analysis import eval_on_nli_datasets

#%%
ENCODE_CONFIG_FILE = './experiments/nli/eval_on_nli_xy/encode_configs.json'
encode_configs = parse_encode_config(ENCODE_CONFIG_FILE)
eval_outputs = eval_on_nli_datasets(encode_configs, from_nli_xy_datasets=True)

# %%
# Error Breakdowns
meta_dfs = eval_outputs['meta_dfs']
results = eval_outputs['results']
#%%
# for rep_name, encode_config in encode_configs["representations"].items():
# 	print(rep_name)
# 	meta_df = meta_dfs[rep_name]

#%%
rep_name = 'roberta-large-mnli'
meta_df = meta_dfs[rep_name]

bn_data = meta_df[['insertion_rel', 'context_monotonicity', 'y_pred']].copy()

#%%
# from pgmpy.models import BayesianModel

# model = BayesianModel([('insertion_rel', 'y_pred'), ('context_monotonicity', 'y_pred')])  # fruit -> tasty <- size

# #%%
# from pgmpy.estimators import ParameterEstimator
# pe = ParameterEstimator(model, bn_data)
# print("\n", pe.state_counts('context_monotonicity'))  # unconditional
# print("\n", pe.state_counts('y_pred'))  # conditional on fruit and size


# #%%

# from pgmpy.estimators import MaximumLikelihoodEstimator
# mle = MaximumLikelihoodEstimator(model, bn_data)
# print(mle.estimate_cpd('context_monotonicity'))  # unconditional
# with open('results.txt', 'w+') as f:
#     print(mle.estimate_cpd('y_pred'), file=f)  # conditional

# #%% #%%

# up_df = meta_df.loc[meta_df.context_monotonicity=='up']
# up_leq_df = up_df.loc[meta_df.insertion_rel=='leq']
# heat = up_leq_df.pivot_table(values='correct', index='context', 
# 								columns='insertion_pair')

# import plotly.graph_objects as go
# fig = go.Figure()
# fig.add_trace(go.Heatmap(
#     z=heat,
# 	colorscale='Viridis',
#     colorbar=dict(
#         titleside="top",
#         tickmode="array",
#         tickvals=[0, 1],
#         ticktext=["Incorrect", "Correct"],
#         ticks="outside"
#     )
# ))

# fig.update_layout(
#     title=f"Decomposed Error Heatmap ({rep_name})",
# 	title_x=0.5,
# 	legend=dict(
# 		orientation='h',
# 	),
#     xaxis_title="Insertion Pair (Forward Inclusion)",
#     yaxis_title="Context (Upward Monotone)",
# )
# fig.show()

# #%%
# down_df = meta_df.loc[meta_df.context_monotonicity=='down']
# down_geq_df = down_df.loc[meta_df.insertion_rel=='geq']
# heat = down_geq_df.pivot_table(values='correct', index='context', 
# 								columns='insertion_pair')
# fig = go.Figure()
# fig.add_trace(go.Heatmap(
#     z=heat,
# 	colorscale='Viridis',
#     colorbar=dict(
#         titleside="top",
#         tickmode="array",
#         tickvals=[0, 1],
#         ticktext=["Incorrect", "Correct"],
#         ticks="outside"
#     )
# ))
# fig.update_layout(
#     title=f"Decomposed Error Heatmap ({rep_name})",
# 	title_x=0.5,
# 	legend=dict(
# 		orientation='h',
		
# 	),
#     xaxis_title="Insertion Pair (Reverse Inclusion)",
#     yaxis_title="Context (Downward Monotone)",
# )
# fig.show()

# #%%
# from nli_xy.visualization import plot_all_probing_results