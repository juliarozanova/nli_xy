#%%
from nli_xy.probing.parse_probe_config_task import parse_probe_config
import os
from re import split

from torch._C import Value
os.chdir('../../')

#%%
%load_ext autoreload
%autoreload 2

#%%
import sys
from json import load
import pandas as pd
import numpy as np
from prefect import Flow
from prefect.engine.flow_runner import FlowRunner
from torch.utils.data import Dataset
from nli_xy.encoding import parse_encode_config, encode_from_config
from nli_xy.probing import prep_task_data_for_probeably

PROBE_ABLY_DIR = '/data/Code/PhD/Probe-Ably/'
sys.path.append(PROBE_ABLY_DIR)
DATA_DIR = './data/nlixy_small/'
ENCODE_CONFIG_FILE = './experiments/compare_models/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/compare_models/probe_config.json'

from probe_ably.core.tasks.probing import TrainProbingTask
from probe_ably.core.tasks.metric_task import ProcessMetricTask
#%%


#%% 
# take rep_name as an input

rep_names = encode_configs["representations"].keys()
all_data_encodings = {}
for rep_name in rep_names:
	encode_config = encode_configs["representations"][rep_name]
	tokenizer = load_tokenizer.run(encode_config)
	split_datasets = build_split_datasets.run(DATA_DIR, encode_config, tokenizer)
	encoder_model = load_encoder_model.run(encode_config)

	encoded_data = encode_split_datasets.run(split_datasets, 
													encoder_model, 
													encode_config,
													device='cuda')
	all_data_encodings[rep_name] = encoded_data


#%%

train_probing_task = TrainProbingTask()
process_metric_task = ProcessMetricTask()

encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
all_data_encodings = encode_from_config.run(encode_configs)
#%%
prepared_data = prep_task_data_for_probeably.run(all_data_encodings, 
												task_name=encode_configs["shared_config"]["task_label"])

#%%
probe_config = parse_probe_config.run(PROBE_CONFIG_FILE)
train_results = train_probing_task.run(prepared_data, probe_config)
processed_results = process_metric_task.run(
	train_results, probe_config]
)
# %%
probe_config = {
   "probing_setup":{
      "intra_metric":"probe_ably.core.metrics.accuracy.AccuracyMetric",
      "inter_metric":"probe_ably.core.metrics.selectivity.SelectivityMetric",
      "probing_models":{0:
         {
            "probing_model_name":"probe_ably.core.models.linear.LinearModel",
            "batch_size":64,
            "epochs":20,
            "number_of_models": 20
         },
         1:{
            "probing_model_name":"probe_ably.core.models.mlp.MLPModel",
            "batch_size":64,
            "epochs":20,
            "number_of_models": 20
         }
	  }
   }
}
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
# %%
import plotly as px
import plotly.graph_objects as go
import pandas as pd
#%%

# %%
def plot_results(processed_results, 
				encode_configs,
				which_task='context_monotonicity',
				which_probe_model='probe_ably.core.models.linear.LinearModel', 
				which_complexity_control='norm', 
				which_metric='accuracy'):

	fig = go.Figure()
	rep_names = encode_configs["representations"].keys()

	for rep_name in rep_names:
		results_df = isolate_probe_ably_result(processed_results,
												rep_name,
												which_task,
												which_probe_model,
												which_complexity_control,
												which_metric)

		fig.add_trace(go.Scatter(x=results_df.x, y=results_df.y,
							mode='lines+markers',
							name='lines+markers'))

	fig.show()

#%%
def isolate_probe_ably_result(processed_results,
							rep_name='roberta-large-mnli',
							which_task='context_monotonicity', 
							which_probe_model='probe_ably.core.models.linear.LinearModel', 
							which_complexity_control='norm', 
							which_metric='accuracy'):

	results_df = None

	for task_results in processed_results:
		if task_results['name']==which_task:
			for probe_model_type in task_results['probings']:
				if probe_model_type['model_name']==which_probe_model:
					for result_set in probe_model_type['probing_results']:
						if result_set['x_axis']==which_complexity_control and result_set['y_axis'].lower()==which_metric:
							for rep_results in result_set['chart_data']:
								print(rep_results)
								if rep_results['id']==rep_name:
									results_df = pd.DataFrame(rep_results['data'], columns=["x", "y"])
	return results_df

# %%

plot_results(processed_results, encode_configs, which_task='monotonicity_compare_models')