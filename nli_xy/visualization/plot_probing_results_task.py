from prefect import task
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle
import pandas as pd

symbols = ['circle','square','circle-open',  'cross', 'triangle-up', 'bowtie', 'star']
colors = px.colors.qualitative.Plotly
colors = px.colors.qualitative.Bold[:1]+px.colors.qualitative.Bold[4:-1] +  ['black']

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

	fig = make_subplots(rows=1, cols=2)
	rep_names = encode_configs["representations"].keys()

	for metric in ['accuracy', 'selectivity']:
		cols = {
			'accuracy':1, 
			'selectivity':2,
		}

		cycle_colors = cycle(colors)
		cycle_symbols = cycle(symbols)
		for rep_name in rep_names:
			color=next(cycle_colors)
			symbol=next(cycle_symbols)

			print('rep_name:', rep_name)
			print('which_task:', which_task)
			print('which_complexity_control', which_complexity_control)
			results_df = isolate_probe_ably_result(processed_results,
													rep_name,
													which_task,
													which_probe_model,
													which_complexity_control,
													which_metric=metric)
			print(results_df)

			fig.add_trace(go.Scatter(x=results_df.x, y=results_df.y,
								mode='lines+markers',
								marker_size=8,
								marker_symbol=symbol,
								legendgroup=rep_name,
								showlegend=(True if metric=='accuracy' else False), 
								line=dict(color=color),
								name=rep_name),
								row=1, 
								col=cols[metric])

	if which_task == 'context_monotonicity':
		title = "Context Montonicity Probing Results:"
	elif which_task == 'insertion_rel':
		title = "XY Insertion Relation Probing Results:"
	else:
		title = None
 
	fig.update_layout(
		title=title,
		title_x=0.5,
		title_y=0.98,
		title_font_size=25,
		width=1300,
		margin=dict(
			t=125,
		),
		legend=dict(
		orientation="h",
		yanchor="bottom",
		tracegroupgap=10,
		y=1.05,
		xanchor="center",
		x=0.5,
			font=dict(size=16)))

	if which_complexity_control=='norm':
		# Update xaxis properties
		fig.update_xaxes(title_text="Probe Complexity (Nuclear Norm)", row=1, col=1)
		fig.update_xaxes(title_text="Probe Complexity (Nuclear Norm)", row=1, col=2)

	# Update yaxis properties
	fig.update_yaxes(title_text="Probe Test Accuracy", range=[0,1], row=1, col=1)
	fig.update_yaxes(title_text="Selectivity", range=[-0.2, 0.5],row=1, col=2)

	fig.show()
	return fig

def plot_old_results():
	fig = make_subplots(rows=1, cols=2)
	try:
		cycle_colors = cycle(colors)
		cycle_symbols = cycle(symbols)
		for exp_id in exp_ids:
			color = next(cycle_colors)
			symbol = next(cycle_symbols)
			add_linear_norm_results(fig, exp_id, color, symbol)
			add_linear_selectivity_results(fig, exp_id, color, symbol)
	except FileNotFoundError:
		pass
		
	fig.update_layout(
		title="Monotonicity Probing: X Embedding (Premise Context Only)",
		title_x=0.5,
		title_y=1,
		title_font_size=25,
		width=1500,
		margin=dict(
			t=125,
		),
		legend=dict(
		orientation="h",
		yanchor="bottom",
		tracegroupgap=10,
		y=1.05,
		xanchor="center",
		x=0.5,
			font=dict(size=16)))

	# Update xaxis properties
	fig.update_xaxes(title_text="Nuclear Norm of Linear Probe", row=1, col=1)
	fig.update_xaxes(title_text="Nuclear Norm of Linear Probe", row=1, col=2)

	# Update yaxis properties
	fig.update_yaxes(title_text="Probe Test Accuracy", range=[0.4,1], row=1, col=1)
	fig.update_yaxes(title_text="Entropic Selectivity", range=[-0.05, 0.55],row=1, col=2)

	fig.show()

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