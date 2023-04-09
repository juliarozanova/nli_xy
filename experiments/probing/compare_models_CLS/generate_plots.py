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
prepared_data = prep_data_for_probeably.run(all_data_encodings, 
											encode_configs)

# %%
rep_name = 'roberta-large-mnli'

reps = all_data_encodings[rep_name]['test']['representations']
# %%
