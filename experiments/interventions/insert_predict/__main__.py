#%% 
#%%
from pathlib import Path
from tqdm import tqdm
import sys
from nli_xy.constants import AMNESIC_PATH, MODEL_NAMES, model_pretrained_paths, model_label_mapper
sys.path.append(AMNESIC_PATH)
import torch
import numpy as np
import scipy as sp
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from transformers import AutoModelForSequenceClassification as Model
from numpy.linalg import norm


# make label column a click input


# gold labels are always two-class
# predicted labels depend on model

#%%


def extract_model_classifier_head(model_name, model):
    if 'roberta' in model_name:
        nli_clf = model.classifier
    else:
        def nli_clf(x):
            return model.classifier(model.bert.pooler(x)) 
    # TODO. Fix to do pooling for Bart

    return nli_clf

def predict_on_reps(reps, nli_clf):
    if 'roberta' in model_name:
        features = torch.tensor(reps).view(reps.shape[0],-1,1024).type(torch.float32)
    elif 'bert-base' in model_name:
        features = torch.tensor(reps).view(reps.shape[0],-1,768).type(torch.float32)
    elif 'bart-large' in model_name:
        features = torch.tensor(reps).view(reps.shape[0],-1,1024).type(torch.float32)

    new_preds=nli_clf(features).detach().numpy()
    int_new_preds = [int(x) for x in list(np.argmax(new_preds, axis=-1))]
    predictions = [model_label_mapper[model_name](x) for x in int_new_preds]

    return predictions



# %%

    # debiased_acc = get_prediction_accuracy(debiased, model, gold_labels)
def generate_control(x_test, control_dims, intervened_dir=None):
    # get a controls based on rank/removed dims
    # rand_dims is the number removed
    rand_dims = x_test.shape[1] - control_dims
    rand_control, rand_direction_p = rand_direction_control(x_test, rand_dims)

    if intervened_dir:
        with open(intervened_dir.joinpath(f'control_{control_dims}.npy'), 'wb') as file:
            np.save(file, rand_control)
        with open(intervened_dir.joinpath(f'control_rand_direction_p_{control_dims}.npy'), 'wb') as file:
            np.save(file, rand_direction_p)

    return  rand_control, rand_direction_p


#  %%
# NOTE:
# 1. Run probing again on REMAINING semantic subspace

def predict_on_reps_from_path(reps_path, nli_clf, gold_labels):
    reps_name = reps_path.name
    reps = np.load(reps_path)
    predictions = predict_on_reps(reps, nli_clf)

    acc = accuracy_score(y_pred=predictions, y_true=gold_labels)

    return acc

if __name__=='__main__':

    layer = 'last'

    # for label_column in ['control']:
    for label_column in ['context_monotonicity', 'insertion_rel', 'composite', 'gold_label', 'control']:
        # for which_intervention in ["amnesic"]: #cross-amnesic:
        for which_intervention in ["mnesic", "amnesic"]: #cross-amnesic:

            for model_name in MODEL_NAMES:

                config = dict(
                    encoder=model_name,
                    property=label_column,
                    layer=layer
                )

                wandb.init(
                    name=f'{model_name}',
                    project=f"nli_xy_{which_intervention}_{label_column}",
                    tags=["nli_xy"],
                    config=config,
                    reinit=True
                )

                results_dir = Path(f'experiments/interventions/results/{label_column}/{model_name}/')
                amnesic_Ps_dir = results_dir.joinpath('Ps')
                amnesic_Ws_dir = results_dir.joinpath('Ws')
                if label_column=="control":
                    reps_dir = results_dir.joinpath(f'control_{which_intervention}_reps')
                else:
                    reps_dir = results_dir.joinpath(f'{which_intervention}_reps')
                meta_path = results_dir.joinpath('meta.tsv')

                meta_df = pd.read_csv(meta_path, sep='\t', index_col=0)
                gold_labels = meta_df['gold_label']

                print(f'Model Name: {model_name}')
                model = Model.from_pretrained(model_pretrained_paths[model_name])
                model.eval()
                nli_clf = extract_model_classifier_head(model_name, model)

                reps_paths = reps_dir.glob('*.npy')

                if which_intervention in ["rowspace"] or label_column=="control":
                    num_reps = len(list(reps_paths)) 
                elif which_intervention in ["amnesic", "mnesic", "cross_amnesic"]:
                    num_reps = len(list(reps_paths)) - 1 # Don't account for un-numbered P matrix

                #TODO: up to i, rather than up to P_i?
        # Prediction Stuff
                pbar = tqdm(range(num_reps))
                for i in pbar:
                    if label_column=="control":
                        reps_path = reps_dir.joinpath(f"control_control_{i+1}.npy")
                    elif which_intervention in ["amnesic", "mnesic", "cross_amnesic"]:
                        reps_path = reps_dir.joinpath(f"{which_intervention}_up_to_P_{i}.npy")
                    elif which_intervention=="rowspace":
                        reps_path = reps_dir.joinpath(f"{which_intervention}_up_to_Prow_{i}.npy")

                    acc = predict_on_reps_from_path(reps_path, nli_clf, gold_labels)
                    wandb.log({'task_acc': acc}, step=i)

                # RANKS

                # print(f'''
                #     original accuracy: {original_acc}\n
                #     debiased_accuracy: {debiased_acc}\n
                #     biased_accuracy: {biased_acc}\n
                # ''')

                # orig_rank = np.linalg.matrix_rank(x_test)
                # deb_rank = np.linalg.matrix_rank(debiased)
                # b_rank = np.linalg.matrix_rank(biased)
                # print(f'''Rank of original matrix: {orig_rank} \n 
                #             Rank of debiased matrix: {deb_rank} \n
                #             Rank of biased matrix: {b_rank} \n''')

                    # load model

                    # one at a time:
                        # load reps
                        # perform predictions
                        # write to W&B