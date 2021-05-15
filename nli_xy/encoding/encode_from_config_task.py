from prefect import task
from nli_xy.encoding import parse_encode_config, build_split_datasets, \
	encode_split_datasets, load_tokenizer, load_encoder_model

@task
def encode_from_config(encode_configs, write_encoded=True):
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
                        "labels": (torch.Tensor, on cpu)
                    }
                    "dev": {
                        "representations": (torch.Tensor, on cpu)
                        "labels": (torch.Tensor, on cpu)
                        }
                    "test": {
                        "representations": (torch.Tensor, on cpu)
                        "labels": (torch.Tensor, on cpu)
                    }
                }
                ...
            }


    """
    rep_names = encode_configs["representations"].keys()
    all_data_encodings = {}

    for rep_name in rep_names:
        encode_config = encode_configs["representations"][rep_name]
        tokenizer = load_tokenizer.run(encode_config)
        split_datasets = build_split_datasets.run(encode_config['data_dir'], encode_config, tokenizer)
        encoder_model = load_encoder_model.run(encode_config)
        encoded_data = encode_split_datasets.run(split_datasets, 
                                                        encoder_model, 
                                                        encode_config,
                                                        device='cuda')
        all_data_encodings[rep_name] = encoded_data

    return all_data_encodings