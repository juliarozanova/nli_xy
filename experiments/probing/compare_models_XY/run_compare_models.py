#%%
import os
os.chdir('../../')

#%%
%load_ext autoreload
%autoreload 2

#%%
import sys
import pandas as pd
import numpy as np
from prefect import Flow
from nli_xy.encoding import parse_encode_config, encode_from_config
from nli_xy.probing import parse_probe_config, prep_task_data_for_probeably

PROBE_ABLY_DIR = '/data/Code/PhD/Probe-Ably/'
sys.path.append(PROBE_ABLY_DIR)
DATA_DIR = './data/nlixy_small/'
ENCODE_CONFIG_FILE = './experiments/compare_models_XY/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/compare_models_XY/probe_config.json'

#%%
from probe_ably.core.tasks.probing import TrainProbingTask
from probe_ably.core.tasks.metric_task import ProcessMetricTask
train_probing_task = TrainProbingTask()
process_metric_task = ProcessMetricTask()

#%%
#with Flow("Compare_Models") as flow:
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
all_data_encodings = encode_from_config.run(encode_configs)

#%%
prepared_data = prep_task_data_for_probeably.run(all_data_encodings, 
											task_name=encode_configs["shared_config"]["task_label"])
probe_config = parse_probe_config.run(PROBE_CONFIG_FILE)
train_results = train_probing_task.run(prepared_data, probe_config)
processed_results = process_metric_task.run(
	train_results, probe_config
)

#%%
from nli_xy.analysis import error_analysis

for rep_name, encode_config in encode_configs["representations"].items():
	print(rep_name)
	meta_df = error_analysis(rep_name, encode_config)
#%%
rep_name = 'roberta-large-mnli-help'
meta_df = error_analysis(rep_name, encode_config)

#%%
meta_df = meta_df.loc[meta_df.context_monotonicity=='down']
meta_df = meta_df.loc[meta_df.insertion_rel=='leq']
import seaborn as sns
heat_df = meta_df.pivot_table(values='correct', index='context', 
								columns='insertion_pair')
sns.heatmap(heat_df)
#%%
grouped = meta_df.groupby(by=['context'])
heat = grouped.correct.apply(lambda x: pd.Series(x.values)).unstack()
heat = heat.dropna(axis=1)

#%%
new_grouped = meta_df.groupby(by=['context', 'insertion_pair'])
df =new_grouped.correct.apply(lambda x: pd.Series(x.values)).unstack()

df.pivot_table(index='context', columns='insertion_pair')
#%%
from nli_xy.visualization import plot_all_probing_results

#%%
flow.run()
