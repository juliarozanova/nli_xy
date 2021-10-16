from prefect import task
from pathlib import Path
import pandas as pd
import torch
from loguru import logger
from nli_xy.encoding import parse_encode_config, build_split_datasets, \
	encode_split_datasets, load_tokenizer, load_encoder_model

@task
def encode_from_config(encode_configs, save_encoded=True):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    encode_configs : dict 
        Config dict with an encoding configurations for each named 
        representation format.
        Preferably parsed from json by nli_xy.encoding.parse_encode_config.
        It should follow this schema:
            {
                "shared_config":{
                    "data_dir": (dir of nli_xy dataset), 
                    "save_dir": (dir for saved representations),
                    "task_label": (column of the NliXYDataset meta_df to use as task labels),
                    "tokenizer": (transformer model name or path to local tokenizer),
                    "encoder_model": (transformer model name or path to local model), 
                    "X_or_Y": ("both", "X", "Y" or "neither"),
                    "layer_range": (an :int or int range),
                    "layer_summary": (single, mean or concat),
                    "phrase_summary": ("mean" or "concat"),
                    "pair_summary": ("mean" or "concat"),
                    "embedding_size": (:int),
                    "include_cls": ("True" or "False"),
                    "context_option": ("all", "Premise" or "Hypothesis"),
                    "max_length": (max length for tokenized padding),
                    "batch_size": (:int),
                    "device": ("cuda" or "gpu")    
                }
                "representations": {
                    (representation name) : {
                        (any specific overrides for the shared_config settings)
                        }
                }
            }
    write_encoded : bool, optional
        Save encoded representations to save_dir, by default True. Not recommended if space is 
        limited.

    Return
    -------
    all_data_encodings : dict
        Dictionary with a encoded_data dictionary for each representation configuration 
        in the config. The dictionary has the schema:
            {
                (representation name): {
                    "train": {
                        "representations": (torch.Tensor, on cpu)
                        "meta_df": (DataFrame)
                    }
                    "dev": {
                        "representations": (torch.Tensor, on cpu)
                        "meta_df": (DataFrame)
                        }
                    "test": {
                        "representations": (torch.Tensor, on cpu)
                        "meta_df": (DataFrame)
                    }
                }
                ...
            }


    """
    SAVE_DIR = Path(encode_configs['shared_config']['save_dir']).joinpath('processed_data')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info('Running encoding task for each model in config:\n')

    all_data_encodings = {}
    rep_names = encode_configs['representations'].keys()

    for rep_name in rep_names:
        logger.info(f'Encoding data for {rep_name}: ')
        REP_SAVE_DIR = SAVE_DIR.joinpath(rep_name)

        try:
            encoded_data = load_encoded_data(REP_SAVE_DIR)
            logger.info('Loaded from saved embeddings file. \n')
        except FileNotFoundError:
            encode_config = encode_configs['representations'][rep_name]
            tokenizer = load_tokenizer.run(encode_config)
            split_datasets = build_split_datasets.run(encode_config['data_dir'], encode_config, tokenizer)
            encoder_model = load_encoder_model.run(encode_config)
            encoded_data = encode_split_datasets.run(split_datasets, 
                                                            encoder_model, 
                                                            encode_config,
                                                            device='cuda')
            if save_encoded:
                save_encoded_data(encoded_data, REP_SAVE_DIR, split_datasets)

        all_data_encodings[rep_name] = encoded_data

    return all_data_encodings

def load_encoded_data(REP_SAVE_DIR):

    encoded_data = {}
    for split in ['train', 'dev', 'test']:
        encoded_data[split] = {}

        SPLIT_SAVE_DIR = REP_SAVE_DIR.joinpath(f"{split}")
        REP_SAVE_FILEPATH = SPLIT_SAVE_DIR.joinpath('representations.pt')
        META_DF_SAVE_FILEPATH = SPLIT_SAVE_DIR.joinpath('meta.tsv')

        encoded_data[split]['representations'] = torch.load(REP_SAVE_FILEPATH)
        split_meta = pd.read_csv(META_DF_SAVE_FILEPATH, sep='\t') 

            #split_labels = torch.tensor(split_meta[task_label].numpy()).flatten()
            #encoded_data[split]['labels'] = split_labels

        encoded_data[split]["meta_df"] = split_meta

    return encoded_data

def save_encoded_data(encoded_data, REP_SAVE_DIR, split_datasets):

    for split in ['train', 'dev', 'test']:
        SPLIT_SAVE_DIR = REP_SAVE_DIR.joinpath(f'{split}')
        SPLIT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        REP_SAVE_FILEPATH = SPLIT_SAVE_DIR.joinpath('representations.pt')
        META_DF_SAVE_FILEPATH = SPLIT_SAVE_DIR.joinpath('meta.tsv')

        torch.save(encoded_data[split]['representations'], REP_SAVE_FILEPATH)
        split_meta = split_datasets[split].meta_df

        split_meta['composite'] = split_meta.apply(create_composite_label, axis=1)

        with open(META_DF_SAVE_FILEPATH, 'w+') as meta_file:
            meta_file.write(split_meta.to_csv(sep='\t'))

def create_composite_label(row):
    pair = row['context_monotonicity'], row['insertion_rel']
    mapping = {
        ('up', 'leq'): 0,
        ('down', 'geq'): 1,
        ('up', 'geq'): 2,
        ('down','leq'): 3,
        ('up', 'none'):4,
        ('down', 'none'):5
    }

    return mapping[pair]
