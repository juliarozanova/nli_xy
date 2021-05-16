from prefect import task
from nli_xy.datasets import NLI_XY_Dataset
import os

@task
def build_split_datasets(DATA_DIR, config, tokenizer):

    splits = ['train', 'dev', 'test']
    # check exists_dir
    data_dirs = [os.path.join(DATA_DIR, f'{split}/') for split in splits]

    split_datasets = dict(zip(splits, 
            [build_dataset.run(data_dir, config, tokenizer) for data_dir in data_dirs]))

    return split_datasets

@task
def build_dataset(data_dir, config, tokenizer):
    dataset = NLI_XY_Dataset(config, tokenizer)
    dataset.from_contexts(data_dir)

    return dataset