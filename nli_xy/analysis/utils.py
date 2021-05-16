import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from nli_xy.encoding.utils import get_rep_paths

def accuracy_from_meta_df(meta_df):
    accuracy = accuracy_score(meta_df.y_pred, meta_df.y_true)
    logger.info(f'\n Accuracy: {accuracy}')
    return accuracy

def fetch_full_meta(rep_name, encode_config):

    REP_SAVE_DIR, REP_SAVE_FILEPATHS, LABELS_SAVE_FILEPATHS, \
    META_DF_SAVE_FILEPATHS = get_rep_paths(rep_name, encode_config['save_dir'])

    splits = ['train', 'dev', 'test']
    meta_paths = [META_DF_SAVE_FILEPATHS[split] for split in splits]
    meta_dfs = [pd.read_csv(meta_path, sep='\t', index_col=0) for meta_path in meta_paths]

    return pd.concat(meta_dfs)

def reformat_meta_df(meta_df, encode_config):
    '''
    Perform some relabeling for grammatical partitioning 
    and error analysis
    '''

    meta_df['grammar_set'] = meta_df.apply(lambda row: 
            merge_grammar_classes(row['X_grammar'], row['Y_grammar']), axis=1)
    meta_df['grammar_class'] = pd.Categorical(meta_df['grammar_set'])
    meta_df = add_iscorrect_col(meta_df, encode_config["encoder_model"])
    return meta_df

def add_iscorrect_col(meta_df, encoder_model_name):
    two_class_models = [
        './models/roberta-large-mnli-help',
        'roberta-large-mnli-help',
    ]
    three_class_models = [
        'roberta-large-mnli'
    ]

    if encoder_model_name in two_class_models:
        meta_df['y_true'] = meta_df['gold_label'].apply(lambda x: 1 if x else 0)
        meta_df['y_pred'] = meta_df['model_predictions'].apply(lambda x: 1 if x==1 else 0)
    
    if encoder_model_name in three_class_models:
        meta_df['y_true'] = meta_df['gold_label'].apply(lambda x: 1 if x else 0)
        meta_df['y_pred'] = meta_df['model_predictions'].apply(lambda x: 1 if x==2 else 0)

    meta_df['correct'] = meta_df.y_true == meta_df.y_pred
    meta_df['correct'] = meta_df['correct'].apply(int)

    return meta_df


def merge_grammar_classes(x_grammar, y_grammar):
    return set_name(set([x_grammar, y_grammar]))

def set_name(grammar_set):
    if grammar_set==set(['m', 'p', 's']):
        return 'mps'
    if grammar_set==set(['m', 'p']):
        return 'mp'
    if grammar_set==set(['m', 's']):
        return 'ms'
    if grammar_set==set(['p', 's']):
        return 'ps'
    if grammar_set==set(['s']):
        return 's'
    if grammar_set==set(['p']):
        return 'p'
    if grammar_set==set(['m']):
        return 'm'