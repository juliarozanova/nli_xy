from prefect import task
from torch.utils.data import Dataset
import numpy as np
import torch

@task
def prep_task_data_for_probeably(task_name, models_data):
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

    prepared_data =  {
        0:
        {
            "task_name": task_name,
            "models": dict(enumerate([prepare_model_contents(rep_name, models_data) \
                                        for rep_name in models_data.keys()]))
        }
    }

    return prepared_data


def prepare_model_contents(rep_name, models_data):
    model_data = models_data[rep_name]

    return {"model_name": rep_name,
                "model": {"train": prepare_entries(model_data["train_data"]), 
                    "dev": prepare_entries(model_data["dev_data"]),
                    "test": prepare_entries(model_data["test_data"])},
                "control": {"train": prepare_entries(model_data["train_data"]), 
                    "dev": prepare_entries(model_data["dev_data"]),
                    "test": prepare_entries(model_data["test_data"])},
                "representation_size": model_data["train_data"]["representations"].shape[1],
                "number_of_classes": len(np.unique(model_data["train_data"]["labels"])),
                "default_control": False
	}

def prepare_entries(model_data_split):
    vectors = model_data_split["representations"].to('cpu')
    labels = model_data_split["labels"].to('cpu')

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