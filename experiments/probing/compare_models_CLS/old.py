#%%
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from prefect import Flow
import pickle
#%%
nli_xy_root = Path(__file__).parent.parent.parent.parent
os.chdir(nli_xy_root)
sys.path.append('./')

from nli_xy.encoding import parse_encode_config, encode_from_config, load_encoder_model
from nli_xy.probing import parse_probe_config, prep_data_for_probeably
from nli_xy.visualization import plot_all_probing_results, plot_results

PROBE_ABLY_DIR = '/media/julia/Data/Code/PhD/Probe-Ably/'
sys.path.append(PROBE_ABLY_DIR)
ENCODE_CONFIG_FILE = './experiments/probing/compare_models_CLS/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/probing/compare_models_CLS/probe_config.json'

from probe_ably.core.tasks.probing import TrainProbingTask
from probe_ably.core.tasks.metric_task import ProcessMetricTask
train_probing_task = TrainProbingTask()
process_metric_task = ProcessMetricTask()

#%%
#with Flow("Compare_Models") as flow:
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
probe_config = parse_probe_config.run(PROBE_CONFIG_FILE)
#%%
all_data_encodings = encode_from_config.run(encode_configs)

#%%
#%%
prepared_data = prep_data_for_probeably.run(all_data_encodings, 
											encode_configs)

#%%
# train_results = train_probing_task.run(prepared_data, probe_config)
#%%
SAVE_DIR = Path(encode_configs["shared_config"]["save_dir"])
RESULTS_DIR = SAVE_DIR.joinpath('results')
results_filepath = RESULTS_DIR.joinpath('results.pickle')
try:
	with open(results_filepath, 'rb') as results_file:
		processed_results = pickle.load(results_file)
except FileNotFoundError:
	processed_results = process_metric_task.run(
		train_results, probe_config
		)
# processed_results = process_metric_task.run(
# 	train_results, probe_config
# )
#%%
probe_model = probe_config['probing_models']['0']['probing_model_name']
fig_mono = plot_results(processed_results, encode_configs, 
		which_probe_model=probe_model,
		which_complexity_control='norm',	
		which_task='context_monotonicity')


fig_insertions = plot_results(processed_results, encode_configs, 
		which_probe_model=probe_model,
		which_complexity_control='norm',	
		which_task='insertion_rel')
#%%
with open(results_filepath, 'wb+') as results_file:
	pickle.dump(processed_results, results_file)

# %%
fig_mono.write_image(str(RESULTS_DIR)+'/plots/context_mono_probing.png')
fig_insertions.write_image(str(RESULTS_DIR)+'/plots/insertion_rel_probing.png')

# %%
