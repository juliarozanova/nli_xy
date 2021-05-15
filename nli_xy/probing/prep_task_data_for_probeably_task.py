from prefect import task
from torch.utils.data import Dataset
import numpy as np
import torch

@task
def prep_task_data_for_probeably(all_data_encodings, task_name):
    """
    :return: Dictonary of processed data in the format:
    { task_id:
        {'task_name': str,
        'models':
            {model_id:
                {"model_name": str,
                    "model": {"train": numpy.ndarray,  
                              "dev": numpy.ndarray, 
                              "test": numpy.ndarray},
                    "control": {"train": numpy.ndarray, 
                                "dev": numpy.ndarray, 
                                "test": numpy.ndarray},
                    "representation_size": int,
                    "number_of_classes": int,
                    "default_control": boolean (False if user inputs control task)
                }
            }
        }
    }
    :rtype: Dict
    """

    # we consistently use one task per encode config file
    prepared_data =  {
        0:
        {
            "task_name": task_name,
            "models": dict(enumerate([prepare_model_contents(rep_name, all_data_encodings) \
                                        for rep_name in all_data_encodings.keys()]))
        }
    }

    return prepared_data


def prepare_model_contents(rep_name, all_data_encodings):
    encoded_data = all_data_encodings[rep_name]

    return {"model_name": rep_name,
                "model": {"train": prepare_entries(encoded_data["train"]), 
                    "dev": prepare_entries(encoded_data["dev"]),
                    "test": prepare_entries(encoded_data["test"])},
                "control": {"train": prepare_entries(encoded_data["train"]), 
                    "dev": prepare_entries(encoded_data["dev"]),
                    "test": prepare_entries(encoded_data["test"])},
                "representation_size": encoded_data["train"]["representations"].shape[1],
                "number_of_classes": len(np.unique(encoded_data["train"]["labels"])),
                "default_control": False
	}

def prepare_entries(encoded_data_split):
    vectors = encoded_data_split["representations"].to('cpu')
    labels = encoded_data_split["labels"].to('cpu')

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