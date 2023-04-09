from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from nli_xy.encoding import load_encoder_model, load_tokenizer, build_dataset
from nli_xy.datasets import load_nli_data, convert_nlixy_to_nli, NLI_2label_Dataset
from nli_xy.analysis.utils import accuracy_from_meta_df

three_class_models_2_is_entailment = ['roberta-large-mnli', 
    ]

three_class_models_1_is_entailment = [
    'bert-base-uncased-snli',
    'bert-base-uncased-snli-help',
    'roberta-large-mnli-help'
]

two_class_models_0_is_entailment = [
    'roberta-large-mnli-double-finetuning',
    'roberta-large-mnli-help-contexts-help'
]


# TODO: split dataloading and eval script

def eval_on_nli_datasets(
    encode_configs,
    EVAL_SETS_DIR=None, 
    from_nli_xy_datasets=False, 
    from_folder=False,
    from_huggingface_dataset=None):

    if not EVAL_SETS_DIR and not from_nli_xy_datasets and not from_huggingface_dataset:
        raise ValueError('Must include either directory path of tsv eval dataset files \
                        or a NLI_XY_Dataset object!')
    
    if EVAL_SETS_DIR:
        from_folder = True

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

        tokenizer = load_tokenizer(encode_config)
        encoder_model = load_encoder_model(encode_config)

        if from_nli_xy_datasets:
            NLI_XY_DIR = encode_configs['shared_config']['data_dir']
            REP_META_DF_FILEPATH = REP_SAVE_DIR.joinpath(f'nli_xy_meta.tsv')

            # try:
            #     meta_df = pd.read_csv(REP_META_DF_FILEPATH, sep='\t')
            #     logger.info(f'Using cached meta outputs file for {rep_name} for \
            #         the nli_xy dataset.')
            # except FileNotFoundError:
            nli_xy_dataset = build_dataset(NLI_XY_DIR, encode_config, tokenizer)
            nli_dataset = convert_nlixy_to_nli(nli_xy_dataset)
            meta_df = eval_on_nli_dataset(nli_dataset, 
                                                encoder_model, 
                                                rep_name,
                                                batch_size)
            
            accuracy = accuracy_from_meta_df(meta_df)

            with open(REP_META_DF_FILEPATH, 'w+') as meta_file:
                meta_file.write(meta_df.to_csv(sep='\t', index=False))

            results['index'] = ['nli_xy']
            results[rep_name] = accuracy
            meta_dfs[rep_name] = meta_df

        elif from_folder:
            EVAL_SETS_DIR = Path(EVAL_SETS_DIR)
            eval_set_filenames = os.listdir(EVAL_SETS_DIR)
            results['index'] = eval_set_filenames
            meta_dfs[rep_name] = {}

            for eval_set_filename in eval_set_filenames:
                logger.info(f'Evaluating on the dataset {eval_set_filename}:')
                eval_set_name = eval_set_filename.strip('.tsv')
                eval_set_path = EVAL_SETS_DIR.joinpath(eval_set_filename)
                REP_META_DF_FILEPATH = REP_SAVE_DIR.joinpath(f'{eval_set_name}_meta.tsv')

                try:
                    meta_df = pd.read_csv(REP_META_DF_FILEPATH, sep='\t')
                    logger.info(f'Using cached meta outputs file for {rep_name} for \
                        the {eval_set_name} dataset.')
                except FileNotFoundError:
                    nli_dataset = load_nli_data(eval_set_path, tokenizer, device)
                    meta_df = eval_on_nli_dataset(nli_dataset, 
                                                    encoder_model, 
                                                    rep_name,
                                                    batch_size)
                accuracy = accuracy_from_meta_df(meta_df)

                with open(REP_META_DF_FILEPATH, 'w+') as meta_file:
                    meta_file.write(meta_df.to_csv(sep='\t', index=False))

                results[rep_name][eval_set_name] = accuracy
                meta_dfs[rep_name][eval_set_name] = meta_df

        elif from_huggingface_dataset:
            dataset = from_huggingface_dataset
            dataset.set_format(type='pd')
            df = dataset[0:]

            df['sentence1'] = df['premise']
            df['sentence2'] = df['hypothesis']
            # this is dataset dependent (specifically mnli/snli), can we dedce the interpretation function from the dataset object?
            df['gold_label'] = df['label'].apply(interpret_three_class_label_as_string)

            logger.info(df[['sentence1', 'premise', 'hypothesis']].loc[df.label==0].head(20))
            logger.info(df[['label', 'gold_label']].loc[df.label==0].head(20))

            nli_dataset = NLI_2label_Dataset(df, tokenizer, device)
            meta_df = eval_on_nli_dataset(nli_dataset, 
                                                encoder_model, 
                                                rep_name,
                                                batch_size)

            accuracy = accuracy_from_meta_df(meta_df)



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
        
def eval_on_nli_dataset(nli_dataset, encoder_model, encoder_model_name, batch_size=64):
    eval_loader = DataLoader(nli_dataset, batch_size)
    encoder_model.eval()

    with torch.no_grad():
        y_pred = []
        for inputs in tqdm(eval_loader):
            batch_outputs = encoder_model(inputs['input_ids'], inputs['attention_mask'])
            batch_logits = batch_outputs['logits'].to('cpu')
            batch_predictions = np.argmax(batch_logits, axis=1)
            y_pred += batch_predictions

    meta_df = nli_dataset.df
    # interpret as 2 class label
    meta_df['y_true'] = meta_df['gold_label'].apply(lambda x: interpret_gold_label(x, encoder_model_name))

    meta_df['y_pred'] = y_pred
    meta_df['y_pred'] = meta_df['y_pred'].apply(int)
    meta_df['y_pred_before_interp'] = meta_df['y_pred']
    meta_df['y_pred'] = meta_df['y_pred'].apply(lambda x: interpret_predicted_label(x, encoder_model_name))
    
    meta_df['correct'] = (meta_df['y_true']==meta_df['y_pred']).apply(int)

    # ??? 
    # test_head = meta_df[['sentence1', 'sentence2', 'gold_label', 'y_true', 'y_pred_before_interp', 'y_pred']].loc[meta_df.label==0].head(20)
    # logger.info(test_head)

    return meta_df


def interpret_three_class_label_as_string(label):
    if label in [1, '1',2,'2']:
        return 'non-entailment'
    elif label in [0, '0']:
        return 'entailment'
    else: 
        raise ValueError(f'Label {label} out of recognised three class range!')

def interpret_gold_label(gold_label, rep_name):
    if gold_label in ['entailment', 'ENTAILMENT']:
        return 0
    elif gold_label in ['non-entailment', 'NON-ENTAILMENT', 'contradiction', 'CONTRADICTION', 'neutral', 'NEUTRAL']:
        return 1
    else:
        raise ValueError(f'Unrecognised gold label {gold_label}!')

# Roberta-Large-MNLI label:
def interpret_predicted_label(predicted_label, rep_name):
    if rep_name in three_class_models_2_is_entailment:
        if predicted_label in ['2', 2]:
            # entailment in three classes maps to entailment in two classes
            return 0  
        elif predicted_label in ['0', '1', 0, 1]:
            # contradiction and neutral in three classes maps to non-entailment in two classes
            return 1 
        else:
            raise ValueError(f'Unexpected predicted label "{predicted_label}" found while running predictions for {rep_name}.')

    elif rep_name in three_class_models_1_is_entailment:
        if predicted_label in ['1', 1]:
            # entailment in three classes maps to entailment in two classes
            return 0  
        elif predicted_label in ['2', '0', 2, 0]:
            # contradiction and neutral in three classes maps to non-entailment in two classes
            return 1 
        else:
            raise ValueError(f'Unexpected predicted label "{predicted_label}" found while running predictions for {rep_name}.')

    elif rep_name in two_class_models_0_is_entailment:
        if predicted_label in ['0', 0]:
            # entailment in three classes maps to entailment in two classes
            return 0  
        elif predicted_label in ['2', '1', 2, 1]:
            # contradiction and neutral in three classes maps to non-entailment in two classes
            return 1 
        else:
            raise ValueError(f'Unexpected predicted label "{predicted_label}" found while running predictions for {rep_name}.')

    # an unfortonate choice in the HELP training labels
    elif rep_name in ['roberta-large-mnli-help']:
        if predicted_label in [1,'1']:
            return 0
        elif predicted_label in [0, '0']:
            return 1
        else:
            raise ValueError(f'Unexpected predicted label "{predicted_label}" found while running predictions for {rep_name}.')
    else: 
        raise ValueError(f'Unrecognised model {rep_name}: Please configure the interpretation of its prediction labels!')


# def relabel_three_class_predictions(three_class_label):
#     if three_class_label in ['2', 2]:
#         # entailment in three classes maps to entailment in two classes
#         return 1
#     elif three_class_label in ['0', '1', 0, 1]:
#         # contradiction and neutral in three classes maps to non-entailment in two classes
#         return 0