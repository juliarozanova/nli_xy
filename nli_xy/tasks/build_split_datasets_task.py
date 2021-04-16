from prefect import task
from nli_xy.tasks import build_dataset
import os

@task
def build_split_datasets(DATA_DIR, config, tokenizer):

    splits = ['train', 'dev', 'test']
    # check exists_dir
    data_dirs = [os.path.join(DATA_DIR, f'{split}/') for split in splits]

    datasets = dict(zip(splits, 
            [build_dataset.run(data_dir, config, tokenizer) for data_dir in data_dirs]))

    return datasets
