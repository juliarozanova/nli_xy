import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from torch.utils.data import TensorDataset

PROBE_ABLY_DIR = '/media/julia/Data/Code/PhD/Probe-Ably'
sys.path.append(PROBE_ABLY_DIR)
ENCODE_CONFIG_FILE = './experiments/probing/compare_models_CLS/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/probing/compare_models_CLS/probe_config.json'

from nli_xy.encoding import parse_encode_config, encode_from_config, load_encoder_model
from probe_ably import ProbingExperiment

##  nli_xy create representations
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
# probe_config = parse_probe_config.run(PROBE_CONFIG_FILE)
all_data_encodings = encode_from_config.run(encode_configs)

# for a single representation
experiment = ProbingExperiment.from_json(PROBE_CONFIG_FILE)

# Create a list of ProbingTask objects
# TODO: create Probing Task Objects
probing_tasks = []
experiment.load_tasks = probing_tasks

print(all_data_encodings['roberta-large-mnli']['train'].keys())

# experiment.from_split_datasets(trainloader, devloader, testloader)
# for a set of representations: one task (one label set)