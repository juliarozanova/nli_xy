import pandas as pd
from nli_xy.datasets.preprocessing import instantiate, set_of_insertions_into_context, \
 prepare_model_inputs, expand_with_opposite_relations, gold_labeller
from transformers import AutoTokenizer
import torch
import pdb
import os

def get_classifier_token_id(tokenizer):
    if tokenizer.name_or_path in ['facebook/bart-large-mnli']:
        return tokenizer.eos_token_id
    else:
        return tokenizer.cls_token_id
    


class NLI_XY_Dataset():
    def __init__(self, rep_config, tokenizer):
        self.input_df = None
        self.meta_df = None
        self.rep_config = rep_config
        self.max_length = self.rep_config['max_length']
        self.tokenizer=tokenizer
        self.classifier_token_id = get_classifier_token_id(self.tokenizer)
        # todo: calculate a new max_len for a given dataset?
        self.device = rep_config['device']

    def from_contexts(self, data_dir):
        contexts_tsv = os.path.join(data_dir, 'contexts.tsv')
        insertions_tsv = os.path.join(data_dir, 'insertions.tsv')
        contexts_df = pd.read_csv(contexts_tsv, sep='\t')
        insertions_df = pd.read_csv(insertions_tsv, sep='\t')

        # ensure space at start of every insertion string
        insertions_df[['x','y']] = insertions_df[['x','y',]].apply(lambda x: ' ' + x)

        # include reverse relations
        insertions_df = expand_with_opposite_relations(insertions_df)
        
        # initilize dfs for model inputs and for meta info
        input_dfs_per_context = []
        meta_dfs_per_context = [] 

        for context_row in contexts_df.itertuples(index=False):
            input_sub_df, meta_sub_df = set_of_insertions_into_context(context_row,
                                                    insertions_df,
                                                    self.tokenizer)
            monotonicity = context_row.monotonicity
            context = context_row.context
            source = context_row.source

            meta_sub_df['context_monotonicity'] = monotonicity
            meta_sub_df['context'] = context
            meta_sub_df['source'] = source

            input_dfs_per_context += [input_sub_df]
            meta_dfs_per_context += [meta_sub_df]
            
        # TODO: reindex! 
        self.input_tokens_df = pd.concat(input_dfs_per_context)
        self.meta_df = pd.concat(meta_dfs_per_context)
        
        self.meta_df['gold_label'] = self.meta_df.apply(gold_labeller, axis=1)

        self.input_df, X_ranges, Y_ranges = prepare_model_inputs(self.input_tokens_df,
                self.tokenizer, 
                self.max_length, 
                self.device,
                self.rep_config['context_option'])

        for index, input_ids in self.input_df['input_ids'].items():
            range_span = input_ids[X_ranges[index][0]:X_ranges[index][1]]
            target_span = self.tokenizer.convert_tokens_to_ids(self.meta_df['X_tokens'].iloc[index])
            try:
                assert range_span.tolist() == target_span
            except AssertionError:
                print('Poorly chosen MAX_LENGTH, target ids out of range! \n',
                      self.tokenizer.convert_ids_to_tokens(range_span),
                      self.tokenizer.convert_ids_to_tokens(target_span),
                      len(input_ids))


        self.meta_df.drop(['X_tokens','Y_tokens'], axis=1, inplace=True)

        self.meta_df['X_range'] = X_ranges
        self.meta_df['Y_range'] = Y_ranges

    def from_sentence_pairs(self, premise_hypothesis_tsv):
        pass

    def __len__(self): 
        return len(self.input_df)

    def __getitem__(self, index):
        row = self.input_df.iloc[index]
        meta_row = self.meta_df.iloc[index]

        return {
                'example_index': index,
                'input_ids': row['input_ids'].to(self.device), 
                'attention_mask': row['attention_mask'].to(self.device),
                'CLS_token_index': row['input_ids'].tolist().index(self.classifier_token_id), 
                'X_range': meta_row['X_range'],
                'Y_range': meta_row['Y_range'],
                'context': meta_row['context'],
                'insertion_pair': meta_row['insertion_pair'],
                'insertion_rel': meta_row['insertion_rel']
                }
