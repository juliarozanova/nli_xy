#%%
%load_ext autoreload
%autoreload true
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from prefect import Flow

#%%
nli_xy_root = Path(__file__).parent.parent.parent.parent
os.chdir(nli_xy_root)
sys.path.append('.')

from nli_xy.encoding.load_tokenizer_task import load_tokenizer
from nli_xy.encoding import parse_encode_config, encode_from_config, load_encoder_model
from nli_xy.probing import parse_probe_config, prep_data_for_probeably
from nli_xy.visualization import plot_all_probing_results, plot_results

PROBE_ABLY_DIR = '/media/julia/Data/Code/PhD/Probe-Ably/'
sys.path.append(PROBE_ABLY_DIR)

ENCODE_CONFIG_FILE = './experiments/probing/compare_models_XY/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/probing/compare_models_XY/probe_config.json'

from probe_ably.probing import TrainProbingTask
from probe_ably.metrics import ProcessMetricTask
train_probing_task = TrainProbingTask()
process_metric_task = ProcessMetricTask()
probe_config = parse_probe_config.run(PROBE_CONFIG_FILE)
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
#%%
#with Flow("Compare_Models") as flow:
all_data_encodings = encode_from_config.run(encode_configs)
prepared_data = prep_data_for_probeably.run(all_data_encodings, 
											encode_configs)
	# task['representations'] = [task["representations"][key] for key in task["representations"].keys()]

#%%
prepared_data = [prepared_data[key] for key in prepared_data.keys()]
#%%
for item in prepared_data:
	item["representations"] = [item["representations"][key] for key in item["representations"].keys()]

#%%
train_results = train_probing_task.run(prepared_data, probe_config)
processed_results = process_metric_task.run(
	train_results, probe_config
)
#%%
import pickle
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

#%%
# plot_results(processed_results, encode_configs)

probe_model = probe_config['probing_models'][1]['probing_model_name']
plot_results(processed_results,
		encode_configs, 
		which_probe_model=probe_model,
		which_complexity_control='hidden_size',	
		which_task='context_monotonicity')

#%%
with open(results_filepath, 'wb+') as results_file:
	pickle.dump(processed_results, results_file)

# %%