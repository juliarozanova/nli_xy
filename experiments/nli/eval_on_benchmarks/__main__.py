#%%
# %load_ext autoreload
# %autoreload 2
import os
import sys
import torch
from pathlib import Path
from loguru import logger
import pandas as pd

#%%
nli_xy_root = Path(__file__).parent.parent.parent.parent
os.chdir(nli_xy_root)
sys.path.append('.')
from nli_xy.encoding import parse_encode_config
from nli_xy.analysis import eval_on_nli_datasets
from datasets import load_dataset

#%%
ENCODE_CONFIG_FILE = './experiments/nli/eval_on_benchmarks/encode_configs.json'
encode_configs = parse_encode_config(ENCODE_CONFIG_FILE)

dataset_names = ['snli', 'multi-nli']
test_split_names = {
    'snli': 'test',
    'multi_nli': 'validation_matched',
}

for dataset_name in dataset_names:
    logger.info(f'Running evaluations on the dataset {dataset_name}:')
    dataset = load_dataset(dataset_name, split=test_split_names[dataset_name])
    dataset = dataset.filter(lambda x: x["label"]not in [-1, '-1'])
    eval_outputs = eval_on_nli_datasets(encode_configs, from_huggingface_dataset=dataset)
    meta_dfs = eval_outputs['meta_dfs']
    results = eval_outputs['results']

# Error Breakdowns
#%%
# for rep_name, encode_config in encode_configs["representations"].items():
# 	print(rep_name)
# 	meta_df = meta_dfs[rep_name]

#%%
rep_name = 'roberta-large-mnli'
meta_df = meta_dfs[rep_name]
