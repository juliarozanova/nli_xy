from prefect import task
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import torch
import pdb

def get_rep_paths(rep_name, SAVE_DIR):
    REP_SAVE_DIR = Path(SAVE_DIR).joinpath('processed_data',rep_name)

    REP_SAVE_FILEPATHS = {}
    LABELS_SAVE_FILEPATHS = {}
    META_DF_SAVE_FILEPATHS = {}
    for split in ['train', 'dev', 'test']:
        SPLIT_SAVE_DIR = REP_SAVE_DIR.joinpath(f"{split}")

        REP_SAVE_FILEPATHS[split] = SPLIT_SAVE_DIR.joinpath('representations.pt')
        LABELS_SAVE_FILEPATHS[split] = SPLIT_SAVE_DIR.joinpath('labels.pt')
        META_DF_SAVE_FILEPATHS[split] = SPLIT_SAVE_DIR.joinpath('meta.tsv')

    return REP_SAVE_DIR, REP_SAVE_FILEPATHS, LABELS_SAVE_FILEPATHS, \
            META_DF_SAVE_FILEPATHS

def get_target_reps(hidden, X_ranges, Y_ranges, CLS_token_indices, config, device):
    '''
    Args
    ____ 

    hidden: 
                tuple of length (#hidden layers), with each entry being a:
                torch.tensor of shape (batch_size, max_length, hidden_size)

    X_ranges: 
                list of length 2, each entry being a torch.tensor of shape
                (batch_size), containing the start and end value of each 
            
                

    config: dict
    '''

    layers = get_focus_layers(hidden, config['layer_range'], config['layer_summary'])
    batch_size = layers.shape[0]

    if config['X_or_Y'] != 'neither':
        if config['X_or_Y'] in ['both','XY']:
            X_phrase_tokens = get_phrase_tokens(layers, X_ranges)
            Y_phrase_tokens = get_phrase_tokens(layers, Y_ranges)

        if config['X_or_Y'] == 'X_only':
            X_phrase_tokens = get_phrase_tokens(layers, X_ranges)
            Y_phrase_tokens = None

        if config['X_or_Y'] == 'Y_only':
            Y_phrase_tokens = get_phrase_tokens(layers, Y_ranges)
            X_phrase_tokens = None

        X = summarise_phrase(X_phrase_tokens, strategy=config['phrase_summary'])
        Y = summarise_phrase(Y_phrase_tokens, strategy=config['phrase_summary'])
        reps_list = [X,Y]
    else:
        reps_list = []


    if config['include_cls']:

        
        CLS = get_phrase_tokens(layers, [CLS_token_indices,
                                         CLS_token_indices + torch.ones(batch_size, dtype=torch.int64)])
        CLS = summarise_phrase(CLS, strategy='mean')
        reps_list = [CLS] + reps_list


    reps_list = [item for item in reps_list if item is not None]
    target_reps = summarise_final(reps_list, strategy=config['pair_summary']).to(device)
    assert target_reps.shape[0] == batch_size

    return target_reps


def get_focus_layers(hidden, layer_range='-1', layer_summary='single'):
    '''
    Args
    ____

    hidden:
                tuple of length # model depth (e.g. 25 for RoBERTa)

    layer_range:
                int or pair of of ints

    layer_summary:
                str, 
                either 'single', 'mean' or 'concat'.  

    '''
    #TODO: mulitlayer options

    return hidden[layer_range]

def get_phrase_tokens(layers, token_ranges):
    '''
    Args
    ____

    layers:
                size (batch_size, max_length, hidden_size)
                hidden size may be larger than originally, depending on
                the layer summary options (e.g, concatenating multiple layers)

    token_ranges:
                list of length 2, each entry being a torch.tensor of shape
                (batch_size), containing the start and end value of each 

    Returns
    _______

    phrase_token_reps:
                tuple of length (batch_size), each entry is a 
                torch.tensor of shape (phrase_tokens_length, hidden_size)
                 
                
    '''
    # non trivial!!

    stacked_ranges = torch.stack(token_ranges, dim=1)

    layers = torch.unbind(layers)
    phrase_token_reps = []

    for i, (start,end) in enumerate(stacked_ranges):
        phrase_token_reps.append(layers[i][start:end,:])


    return  phrase_token_reps

def summarise_phrase(token_reps_list, strategy='mean'):
    '''
    Args:
    ____ 
    
    token_reps_list: 
                    tuple of length (batch_size), each entry is a 
                    torch.tensor of shape (phrase_tokens_length, hidden_size)

    Returns:
    _______ 
        summary representation of shape (batch_size, hidden_size)

    '''
    if token_reps_list:
        if strategy == 'mean':
            # average across number of tokens 
            mean_tensor = lambda x: torch.mean(x, dim=0)
            summary_tensors = list(map(mean_tensor, token_reps_list))
            return torch.stack(summary_tensors)

        if strategy == 'first':
            return phrase_tokens[:,0,:]

        if strategy == 'concat':
            return torch.cat(token_reps_list, dim=1)

def summarise_final(rep_list, strategy='concat'):
    '''
    Args
        rep_list: list of torch.tensors, each of size (batch_size, embedding_size).
        Typically, some combination of [CLS], X and Y summaries

    '''

    if not rep_list:
        raise ValueError('Empty representation tensor!') 

    if strategy == 'concat':
        # concatenate along hidden_size dimension,
        # so that output size is (batch_size, 2*hidden_size)
        return torch.cat(rep_list, dim=1)

    if strategy == 'mean':
        pair_stacked = torch.stack(rep_list)
        return torch.mean(pair_stacked, dim=0)