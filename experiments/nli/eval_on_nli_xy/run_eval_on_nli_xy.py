#%%
%load_ext autoreload
%autoreload 2
import sys
from prefect import Flow
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
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
eval_outputs = eval_on_nli_datasets.run(encode_configs, from_nli_xy_datasets=True)

# %%
# Error Breakdowns
meta_dfs = eval_outputs['meta_dfs']
results = eval_outputs['results']

#%%
# for rep_name, encode_config in encode_configs["representations"].items():
# 	print(rep_name)
# 	meta_df = meta_dfs[rep_name]

#%%
rep_name = 'roberta-large-mnli-double-finetuning'
meta_df = meta_dfs[rep_name]
#%%
#%%

#%%
# grouped = meta_df.groupby(by=['context'])
# heat = grouped.correct.apply(lambda x: pd.Series(x.values)).unstack()
# heat = heat.dropna(axis=1)

#%%
# new_grouped = meta_df.groupby(by=['context', 'insertion_pair'])
# df =new_grouped.correct.apply(lambda x: pd.Series(x.values)).unstack()
up_df = meta_df.loc[meta_df.context_monotonicity=='up']
up_leq_df = up_df.loc[meta_df.insertion_rel=='leq']
heat = up_leq_df.pivot_table(values='correct', index='context', 
								columns='insertion_pair')

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=heat,
	colorscale='Viridis',
    colorbar=dict(
        titleside="top",
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Incorrect", "Correct"],
        ticks="outside"
    )
))

fig.update_layout(
    title=f"Decomposed Error Heatmap ({rep_name})",
	title_x=0.5,
	legend=dict(
		orientation='h',
		
	),
    xaxis_title="Insertion Pair (Forward Inclusion)",
    yaxis_title="Context (Upward Monotone)",
)
fig.show()

#%%
down_df = meta_df.loc[meta_df.context_monotonicity=='down']
down_geq_df = down_df.loc[meta_df.insertion_rel=='geq']
heat = down_geq_df.pivot_table(values='correct', index='context', 
								columns='insertion_pair')
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=heat,
	colorscale='Viridis',
    colorbar=dict(
        titleside="top",
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Incorrect", "Correct"],
        ticks="outside"
    )
))

fig.update_layout(
    title=f"Decomposed Error Heatmap ({rep_name})",
	title_x=0.5,
	legend=dict(
		orientation='h',
		
	),
    xaxis_title="Insertion Pair (Reverse Inclusion)",
    yaxis_title="Context (Downward Monotone)",
)
fig.show()

#%%
from nli_xy.visualization import plot_all_probing_results

# %%
