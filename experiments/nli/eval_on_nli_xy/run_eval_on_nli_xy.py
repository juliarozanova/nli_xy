#%%
%load_ext autoreload
%autoreload 2
import sys
from prefect import Flow
from pathlib import Path

#%%
nli_xy_root = Path(__file__).parent.parent.parent.parent
os.chdir(nli_xy_root)
from nli_xy.encoding import parse_encode_config
from nli_xy.analysis import eval_on_nli_datasets

#%%
ENCODE_CONFIG_FILE = './experiments/nli/eval_on_nli_xy/encode_configs.json'
encode_configs = parse_encode_config.run(ENCODE_CONFIG_FILE)
results = eval_on_nli_datasets.run(encode_configs, from_nli_xy_datasets=True)

# %%
# Error Breakdowns
