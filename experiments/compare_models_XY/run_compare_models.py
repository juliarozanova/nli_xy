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
ENCODE_CONFIG_FILE = './experiments/compare_models/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/compare_models/probe_config.json'

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
from nli_xy.visualization import plot_all_results
plot_all_results.run(processed_results, encode_configs, probe_config)
#%%
flow.run()

# %%
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
								if rep_results['id']==rep_name:
									results_df = pd.DataFrame(rep_results['data'], columns=["x", "y"])
	return results_df

# %%

plot_results(processed_results, encode_configs, which_task=encode_configs["shared_config"]["task_label"])
# %%
