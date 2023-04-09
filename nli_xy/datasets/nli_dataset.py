import pandas as pd
from nli_xy.datasets.preprocessing import instantiate, set_of_insertions_into_context, prepare_model_inputs, expand_with_opposite_relations, gold_labeller
from transformers import AutoTokenizer
import torch
import pdb
import os
from torch.utils.data import Dataset
from prefect import task
import pandas as pd
import torch


class NLI_2label_Dataset(Dataset):
    def __init__(self, df, tokenizer, device):
        self.df = df
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = self.calculate_max_length()
        # self.df['y_true'] = self.df.gold_label.apply(relabel)

    def __len__(self):
        return len(self.df)
    
    def calculate_max_length(self):
        tokenized_sentence_pairs = self.df.apply(lambda row: self.encode_row(
                                                                row['sentence1'],
                                                                row['sentence2'],
                                                                padding=False,
                                                                ), axis=1).to_list()
        tokenized_sentence_pairs = [item['input_ids'] for item in tokenized_sentence_pairs]
        max_length = max([len(tok_list) for tok_list in tokenized_sentence_pairs])
        return max_length

    def encode_row(self, sentence1, sentence2, *args, **kwargs):
        row_outputs = self.tokenizer.encode_plus(sentence1,
                                             sentence2,
                                             *args, **kwargs
                                             )
        return row_outputs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        outputs = self.encode_row(sentence1, sentence2, padding='max_length', 
                                    max_length=self.max_length)

        input_ids = torch.tensor(outputs['input_ids']).to(self.device)
        attention_mask = torch.tensor(outputs['attention_mask']).to(self.device)
        # label = torch.tensor(row['y_true']).to(self.device)
        
        return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                }

# def relabel(label):
#     entailment_labels = ['ENTAILMENT', 'entailment', 'True', True]
#     non_entailment_labels = ['CONTRADICTION', 'NEUTRAL', 'neutral', 'contradiction',
#             'non-entailment', 'Non-Entailment', 'nonentailment', 'False', False]
#     if label in entailment_labels:
#         return 1
#     elif label in non_entailment_labels:
#         return 0
#     if label in entailment_labels:
#         return 1
#     elif label in non_entailment_labels:
#         return 0
#     else:
#         raise ValueError(f'Unexpected entailment label! expected one of: {entailment_labels} \
#              or {non_entailment_labels}')

def load_nli_data(filepath, tokenizer, device='cuda'):
    if is_tsv(filepath):
        try:
           df = pd.read_csv(filepath, sep='\t', header=0)
        except KeyError:
            # for fragments format
            df = pd.read_csv(filepath, sep='\t', names=['sentence1', 'sentence2', 'gold_label'])
    elif filepath.endswith('.jsonl'):
        df = pd.read_json(filepath, orient='records', lines=True)

    dataset = NLI_2label_Dataset(df, tokenizer, device=device)

    return dataset

def is_tsv(filepath):
    try:
        return filepath.endswith('.tsv')
    except AttributeError:
        return filepath.suffix=='.tsv'
    else:
        raise ValueError('Invalid NLI dataset file format: expected ".tsv".')

