from prefect import task
from nli_xy.datasets import NLI_XY_Dataset

@task
def build_dataset(data_dir, config, tokenizer):
    dataset = NLI_XY_Dataset(config, tokenizer)
    dataset.from_contexts(data_dir)

    return dataset

