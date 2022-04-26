from nli_xy.encoding.build_dataset_task import build_dataset
from prefect import task
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from nli_xy.encoding import load_encoder_model, load_tokenizer, build_dataset
from nli_xy.datasets import load_nli_data, convert_nlixy_to_nli
from nli_xy.analysis.utils import accuracy_from_meta_df

@task 
def eval_on_nli_datasets(encode_configs, EVAL_SETS_DIR=None, from_nli_xy_datasets=False):
    if not EVAL_SETS_DIR and not from_nli_xy_datasets:
        raise ValueError('Must include either directory path of tsv eval dataset files \
                        or a NLI_XY_Dataset object!')

    SAVE_DIR = Path(encode_configs['shared_config']['save_dir']).joinpath('results')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILEPATH = SAVE_DIR.joinpath('summary_results.tsv')
    results = {}
    meta_dfs = {}

    for rep_name, encode_config in encode_configs["representations"].items():
        logger.info(f'Evaluating the {rep_name} model: \n')
        results[rep_name] = {}
        REP_SAVE_DIR = SAVE_DIR.joinpath(rep_name)
        REP_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        device = encode_config['device']
        batch_size=encode_config['batch_size']

        if from_nli_xy_datasets:
            NLI_XY_DIR = encode_configs['shared_config']['data_dir']
            REP_META_DF_FILEPATH = REP_SAVE_DIR.joinpath(f'nli_xy_meta.tsv')

            try:
                meta_df = pd.read_csv(REP_META_DF_FILEPATH, sep='\t')
                logger.info(f'Using cached meta outputs file for {rep_name} for \
                    the nli_xy dataset.')
            except FileNotFoundError:
                tokenizer = load_tokenizer.run(encode_config)
                encoder_model = load_encoder_model.run(encode_config)
                nli_xy_dataset = build_dataset.run(NLI_XY_DIR, encode_config, tokenizer)
                nli_dataset = convert_nlixy_to_nli.run(nli_xy_dataset)
                meta_df = eval_on_nli_dataset.run(nli_dataset, 
                                                    encoder_model, 
                                                    rep_name,
                                                    batch_size)
            
            accuracy = accuracy_from_meta_df(meta_df)

            with open(REP_META_DF_FILEPATH, 'w+') as meta_file:
                meta_file.write(meta_df.to_csv(sep='\t', index=False))

            results['index'] = ['nli_xy']
            results[rep_name] = accuracy
            meta_dfs[rep_name] = meta_df

        else:
            EVAL_SETS_DIR = Path(EVAL_SETS_DIR)
            eval_set_filenames = os.listdir(EVAL_SETS_DIR)
            results['index'] = eval_set_filenames
            meta_dfs[rep_name] = {}

            for eval_set_filename in eval_set_filenames:
                eval_set_name = eval_set_filename.strip('.tsv')
                eval_set_path = EVAL_SETS_DIR.joinpath(eval_set_filename)
                REP_META_DF_FILEPATH = REP_SAVE_DIR.joinpath(f'{eval_set_name}_meta.tsv')

                try:
                    meta_df = pd.read_csv(REP_META_DF_FILEPATH, sep='\t')
                    logger.info(f'Using cached meta outputs file for {rep_name} for \
                        the {eval_set_name} dataset.')
                except FileNotFoundError:
                    tokenizer = load_tokenizer.run(encode_config)
                    encoder_model = load_encoder_model.run(encode_config)
                    nli_dataset = load_nli_data.run(eval_set_path, tokenizer, device)
                    meta_df = eval_on_nli_dataset.run(nli_dataset, 
                                                    encoder_model, 
                                                    rep_name,
                                                    batch_size)

                accuracy = accuracy_from_meta_df(meta_df)

                with open(REP_META_DF_FILEPATH, 'w+') as meta_file:
                    meta_file.write(meta_df.to_csv(sep='\t', index=False))

                results[rep_name][eval_set_name] = accuracy
                meta_dfs[rep_name][eval_set_name] = meta_df



    with open(RESULTS_FILEPATH, 'w+') as results_file:
        try:
            results_df = pd.DataFrame(results)
            results_file.write(results_df.to_csv(sep='\t', index='index'))
        except:
            pass

    return {
        'results':results, 
        'meta_dfs':meta_dfs
        }
        
@task
def eval_on_nli_dataset(nli_dataset, encoder_model, encoder_model_name, batch_size=64):
    eval_loader = DataLoader(nli_dataset, batch_size)

    with torch.no_grad():
        y_pred = []
        for inputs in tqdm(eval_loader):
            batch_outputs = encoder_model(inputs['input_ids'], inputs['attention_mask'])
            batch_logits = batch_outputs['logits'].to('cpu')
            batch_predictions = np.argmax(batch_logits, axis=1)
            y_pred += batch_predictions

    meta_df = nli_dataset.df
    meta_df['y_pred'] = y_pred
    meta_df['y_pred'] = meta_df['y_pred'].apply(int)

    three_class_models = ['roberta-large-mnli', 
        'facebook/bart-large-mnli', 
        'facebook/bart-large-mnli-help', 
        'facebook-bart-large-mnli', 
        'facebook-bart-large-mnli-help', 
        'roberta-large-mnli-double-finetuning',
        'bert-base-uncased-snli-help',
        'bert-base-uncased-snli',
        'microsoft/deberta-large-mnli']

    if encoder_model_name in three_class_models:
        meta_df['y_pred'] = meta_df['y_pred'].apply(relabel_three_class_predictions)
    

    meta_df['correct'] = (meta_df['y_true']==meta_df['y_pred']).apply(int)
    return meta_df


def relabel_three_class_predictions(three_class_label):
    if three_class_label in ['2', 2]:
        # entailment in three classes maps to entailment in two classes
        return 1
    elif three_class_label in ['0', '1', 0, 1]:
        # contradiction and neutral in three classes maps to non-entailment in two classes
        return 0