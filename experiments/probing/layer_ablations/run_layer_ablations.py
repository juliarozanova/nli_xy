#%%
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
from nli_xy.encoding import parse_encode_config, encode_from_config
from nli_xy.probing import parse_probe_config, prep_task_data_for_probeably

PROBE_ABLY_DIR = '/data/Code/PhD/Probe-Ably/'
sys.path.append(PROBE_ABLY_DIR)
DATA_DIR = './data/nlixy_small/'
ENCODE_CONFIG_FILE = './experiments/probing/layer_ablations/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/probing/layer_ablations/probe_config.json'

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
import pickle
SAVE_DIR = Path(encode_configs["shared_config"]["save_dir"])
RESULTS_DIR = SAVE_DIR.joinpath('results')
results_filepath = RESULTS_DIR.joinpath('results.pickle')

with open(results_filepath, 'wb+') as results_file:
	pickle.dump(processed_results, results_file)
