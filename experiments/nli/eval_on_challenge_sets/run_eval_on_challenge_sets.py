#%%
%load_ext autoreload
%autoreload 2
import sys
from prefect import Flow
import torch
from pathlib import Path


#%%
nli_xy_root = Path(__file__).parent.parent.parent.parent
os.chdir(nli_xy_root)
sys.path.append('.')

from nli_xy.encoding import parse_encode_config
from nli_xy.analysis import eval_on_nli_datasets

ENCODE_CONFIG_FILE = './experiments/nli/eval_on_challenge_sets/encode_configs.json'
EVAL_SETS_DIR = './data/nli_challenge_sets/'
#%%

#with Flow("Compare_Models") as flow:
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
eval_outputs = eval_on_nli_datasets.run(encode_configs, EVAL_SETS_DIR=EVAL_SETS_DIR)

# %%
results = eval_outputs['results']
# %%
