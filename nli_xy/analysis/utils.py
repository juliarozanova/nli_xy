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
