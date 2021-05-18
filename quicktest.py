#%%
from nli_xy.encoding import parse_encode_config, encode_from_config, load_encoder_model, load_tokenizer
from nli_xy.probing import parse_probe_config, prep_data_for_probeably
from nli_xy.visualization import plot_all_probing_results, plot_results

PROBE_ABLY_DIR = '/data/Code/PhD/Probe-Ably/'
sys.path.append(PROBE_ABLY_DIR)
ENCODE_CONFIG_FILE = './experiments/probing/compare_models_CLS/encode_configs.json'
PROBE_CONFIG_FILE = './experiments/probing/compare_models_CLS/probe_config.json'

#%%
from probe_ably.core.tasks.probing import TrainProbingTask
from probe_ably.core.tasks.metric_task import ProcessMetricTask
train_probing_task = TrainProbingTask()
process_metric_task = ProcessMetricTask()
#%%
#%%
#with Flow("Compare_Models") as flow:
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
# %%
rep_name = 'facebook-bart-large-mnli'
encode_config = encode_configs['representations'][rep_name]
# %%
tokenizer = load_tokenizer.run(encode_config)
# %%
