import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from torch.utils.data import TensorDataset
from loguru import logger

# PROBE_ABLY_DIR = '/media/julia/Data/Code/PhD/Probe-Ably'
PROBE_ABLY_DIR = '/home/julia/Code/PhD/Probe-Ably'
sys.path.append(PROBE_ABLY_DIR)
ENCODE_CONFIG_FILE = './experiments/probing/compare_models_XY/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/probing/compare_models_XY/probe_config.json'

from nli_xy.encoding import parse_encode_config, encode_from_config, load_encoder_model
from nli_xy.probing import parse_probe_config, prep_data_for_probeably
from probe_ably import ProbingExperiment

##  nli_xy create representations
encode_configs = parse_encode_config(ENCODE_CONFIG_FILE)

all_data_encodings = encode_from_config(encode_configs)

prepared_data = prep_data_for_probeably(all_data_encodings, encode_configs)
tasks = [prepared_data[key] for key in prepared_data.keys()]
for item in tasks:
    item["representations"] = [item["representations"][key] for key in item["representations"].keys()]

experiment = ProbingExperiment.from_json(PROBE_CONFIG_FILE)
experiment.load_tasks(tasks)
results = experiment.run()

SAVE_DIR = Path(encode_configs["shared_config"]["save_dir"])
RESULTS_DIR = SAVE_DIR.joinpath('results')
results_filepath = RESULTS_DIR.joinpath(f'results_{datetime.now()}.pickle')

# save results to file
with open(results_filepath, 'wb+') as results_file:
	pickle.dump(results, results_file)

# save plots to folder
for model_index in [0,1]:
    probe_model = experiment.probing_config['probing_models'][model_index]['probing_model_name']
    fig_mono = plot_results(processed_results, encode_configs, 
            which_probe_model=probe_model,
            which_complexity_control='norm',	
            which_task='context_monotonicity')


    fig_insertions = plot_results(processed_results, encode_configs, 
            which_probe_model=probe_model,
            which_complexity_control='norm',	
            which_task='insertion_rel')

    fig_mono.write_image(str(RESULTS_DIR)+f'/plots/context_mono_probing_{model_index}.png')
    fig_insertions.write_image(str(RESULTS_DIR)+f'/plots/insertion_rel_probing_{model_index}.png')