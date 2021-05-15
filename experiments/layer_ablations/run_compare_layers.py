#%%
import os
from re import split
os.chdir('../../')

#%%
%load_ext autoreload
%autoreload 2

#%%
from json import load
import sys
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from nli_xy.encoding import parse_encode_config, build_split_datasets, \
	encode_split_datasets, load_tokenizer, load_encoder_model
from nli_xy.probing import prep_data_for_probeably

from prefect import Flow
from prefect.engine.flow_runner import FlowRunner
from probe_ably.core.tasks.probing import TrainProbingTask
from probe_ably.core.tasks.metric_task import ProcessMetricTask
#%%

# generate encode_configs for each model

PROBE_ABLY_DIR = '/data/Code/PhD/Probe-Ably/'
sys.path.append(PROBE_ABLY_DIR)
DATA_DIR = './data/nlixy_small/'
ENCODE_CONFIG_FILE = './experiments/compare_models/encode_configs.json'
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)

#%% 
# take rep_name as an input

rep_names = ["roberta-large-mnli-help", "roberta-large-mnli"]
models_data = {}
for rep_name in rep_names:
	encode_config = encode_configs["representations"][rep_name]
	tokenizer = load_tokenizer.run(encode_config)
	split_datasets = build_split_datasets.run(DATA_DIR, encode_config, tokenizer)
	encoder_model = load_encoder_model.run(encode_config)
	
	model_data = encode_split_datasets.run(split_datasets, 
													encoder_model, 
													encode_config,
													device='cuda')
	models_data[rep_name] = model_data							

prepared_data = prep_data_for_probeably.run("monotonicity_compare_models", models_data)


# %%
probe_config = {
   "probing_setup":{
      "train_size":0.50,
      "dev_size":0.25,
      "test_size":0.25,
      "intra_metric":"probe_ably.core.metrics.accuracy.AccuracyMetric",
      "inter_metric":"probe_ably.core.metrics.selectivity.SelectivityMetric",
      "probing_models":{0:
         {
            "probing_model_name":"probe_ably.core.models.linear.LinearModel",
            "batch_size":64,
            "epochs":10,
            "number_of_models":50
         },
         1:{
            "probing_model_name":"probe_ably.core.models.mlp.MLPModel",
            "batch_size":64,
            "epochs":10,
            "number_of_models": 50
         }
	  }
   }
}

# %%
# %%


train_probing_task = TrainProbingTask()
process_metric_task = ProcessMetricTask()

def probe_from_dataloaders(config_dict, prepared_data):
	train_results = train_probing_task.run(prepared_data, config_dict["probing_setup"])
	processed_results = process_metric_task.run(
		train_results, config_dict["probing_setup"]
	)
	return processed_results
# %%

processed_results = probe_from_dataloaders(probe_config, prepared_data["tasks"])
# %%
processed_results
# %%
