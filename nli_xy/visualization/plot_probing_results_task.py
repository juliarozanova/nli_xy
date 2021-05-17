from prefect import task
import plotly.graph_objects as go
import pandas as pd

@task
def plot_all_probing_results(processed_results, encode_configs, probe_config):
    print(probe_config['probing_models'])
    probe_models = [model_info['probing_model_name'] for model_id, model_info in probe_config['probing_models'].items()]
    print([name for name in probe_models])


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
							name=rep_name))

	fig.show()

def isolate_probe_ably_result(processed_results,
							rep_name='roberta-large-mnli',
							which_task='context_monotonicity', 
							which_probe_model='probe_ably.core.models.linear.LinearModel', 
							which_complexity_control='norm', 
							which_metric='accuracy'):

	results_df = None

	task_results = processed_results[0]
	if task_results['name']==which_task:
		for probe_model_type in task_results['probings']:
			if probe_model_type['model_name']==which_probe_model:
				for result_set in probe_model_type['probing_results']:
					if result_set['x_axis']==which_complexity_control and result_set['y_axis'].lower()==which_metric:
						for rep_results in result_set['chart_data']:
							if rep_results['id']==rep_name:
								results_df = pd.DataFrame(rep_results['data'], columns=["x", "y"])
	return results_df