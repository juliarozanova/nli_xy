from prefect import task
import pandas as pd
from .nlixy_dataset import NLI_XY_Dataset
from .nli_dataset import NLI_2label_Dataset

@task
def convert_nlixy_to_nli(nlixy_dataset: NLI_XY_Dataset):
    device = nlixy_dataset.device
    tokenizer = nlixy_dataset.tokenizer
    nli_df = nlixy_dataset.meta_df
    nli_df['sentence1'] = nli_df['premise']
    nli_df['sentence2'] = nli_df['hypothesis']

    return NLI_2label_Dataset(nlixy_dataset.meta_df, tokenizer, device)
