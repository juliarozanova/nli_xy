from re import T
from nli_xy.encoding.encode_from_config_task import encode_from_config
from prefect import task
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import torch

@task
def prep_data_for_probeably(all_data_encodings, encode_configs):
    """
    :return: Dictonary of processed data in the format:
    { task_id:
        {'task_name': str,
        'models':
            {model_id:
                {"model_name": str,
                    "model": {"train":,  
                              "dev": , 
                              "test": },
                    "control": {"train": , 
                                "dev": , 
                                "test": },
                    "representation_size": int,
                    "number_of_classes": int,
                    "default_control": boolean (False if user inputs control task)
                }
            }
        }
        ...
    }
    :rtype: Dict
    """

    task_labels = encode_configs['shared_config']['task_labels']
    all_prepared_task_data = [prep_task_data_for_probeably.run(all_data_encodings, 
                                        task_label, encode_configs) for task_label in task_labels]

    return dict(enumerate(all_prepared_task_data))

@task
def prep_task_data_for_probeably(all_data_encodings, task_label, encode_configs):

    prepared_task_data = {
            "task_name": task_label,
            "representations": dict(enumerate([prepare_model_contents(rep_name, 
                                        task_label,
                                        encode_configs, 
                                        all_data_encodings) 
                                        for rep_name in all_data_encodings.keys()]))
        }

    return prepared_task_data


def prepare_model_contents(rep_name, task_label, encode_configs, all_data_encodings):
    encoded_data = all_data_encodings[rep_name]

    splits = ['train', 'dev', 'test']

    model_data_dict = dict(zip(splits, 
                          [prepare_entries(encoded_data, 
                                           split, 
                                           task_label, 
                                           encode_configs) for split in splits]))
    control_data_dict = dict(zip(splits, 
                          [prepare_entries(encoded_data, 
                                           split, 
                                           task_label, 
                                           encode_configs,
                                           control=True) for split in splits]))
    return {"representation_name": rep_name,
                "representation": model_data_dict,
                "control": control_data_dict,
                "representation_size": encoded_data["train"]["representations"].shape[1],
                "number_of_classes": get_num_classes(task_label, encoded_data),
                "default_control": False
	}

    #TODO: return RepresentationData Object

def get_num_classes(task_label, encoded_data):
    return len(encoded_data['train']['meta_df'][task_label].unique())

def load_control_labels_from_file(task_label, encode_configs, split):
    '''
    Expects control labels in each train/dev/split partition directory of the data as
    a tsv file with name schema "{task_label}_control_labels.tsv", where task_label
    matches the task_label in the encode_configs file.
    '''

    DATA_DIR = Path(encode_configs['shared_config']['data_dir'])
    split_control_labels_filepath = DATA_DIR.joinpath(split, f'{task_label}_control_labels.tsv')
    split_control_labels = pd.read_csv(split_control_labels_filepath, 
                                        sep='\t',
                                        squeeze=True,
                                        header=None)
    split_control_labels = torch.tensor(split_control_labels.to_list(), 
                            dtype=torch.long).to('cpu')
    return split_control_labels


def prepare_entries(encoded_data, split, task_label, encode_configs, control=False):
    encoded_data_split = encoded_data[split]
    vectors = encoded_data_split["representations"].to('cpu')

    if not control:
        try:
            categorical_labels = encoded_data_split["meta_df"][task_label]
            try:
                labels = torch.tensor(categorical_labels.values,
                                    dtype=torch.long).to('cpu')
            except:
                coded_labels = categorical_labels.apply(lambda x: relabel(x, task_label))
                labels = torch.tensor(coded_labels.values,
                                        dtype=torch.long).to('cpu')
        except KeyError:
            raise ValueError(f'''Invalid task label {task_label}
            specified, not found in meta dataframe columns!''')
    elif control:
        labels = load_control_labels_from_file(task_label, encode_configs, split)
    else:
        raise ValueError(f'No valid labels for {task_label}!')

    dataset = dict()
    for i in range(0, len(vectors)):
        dataset[i] = {"representation": vectors[i], "label": labels[i]}

    return TorchDataset(dataset)

class TorchDataset(Dataset):
    def __init__(self, dataset):

        self.dataset = list(dataset.values())
        self.labels = np.array([data["label"] for data in self.dataset])
        self.keys = list(dataset.keys())

    def __getitem__(self, index):
        instance = self.dataset[index]
        return (
            torch.FloatTensor(instance["representation"]),
            instance["label"],
            index,
        )

    def get_id(self, index):
        return self.keys[index]

    def __len__(self):
        return len(self.dataset)

def relabel(label, task):
        if task=='context_monotonicity':
            return 0 if label=='down' else 1

        if task=='insertion_rel':
            if label=='leq':
                return 0
            if label=='geq':
                return 1
            if label=='none':
                return 2