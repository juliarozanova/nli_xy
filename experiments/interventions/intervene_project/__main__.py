import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy as sp
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import wandb
from nli_xy.constants import AMNESIC_PATH, MODEL_NAMES, model_label_mapper

sys.path.append(AMNESIC_PATH)
from amnesic_probing.tasks.utils import rand_direction_control

def load_full_data(TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH):
    x_test_train = torch.load(TRAIN_DATA_PATH.joinpath('representations.pt'))
    x_test_dev = torch.load(DEV_DATA_PATH.joinpath('representations.pt'))
    x_test_test = torch.load(TEST_DATA_PATH.joinpath('representations.pt'))

    test_meta_df_train = pd.read_csv(TRAIN_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    test_meta_df_dev = pd.read_csv(DEV_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    test_meta_df_test = pd.read_csv(TEST_DATA_PATH.joinpath('meta.tsv'), sep='\t')

    x_test = torch.vstack([x_test_train, x_test_dev, x_test_test]).numpy()
    test_meta_df = pd.concat([test_meta_df_train, test_meta_df_dev, test_meta_df_test])

    return x_test, test_meta_df


def prepare_data(model_name):
    # TRAIN_DATA_PATH = Path(f'./experiments/probing/compare_models_CLS/processed_data/{model_name}/train/')
    # DEV_DATA_PATH = Path(f'./experiments/probing/compare_models_CLS/processed_data/{model_name}/dev/')
    TEST_DATA_PATH = Path(f'./experiments/probing/compare_models_CLS/processed_data/{model_name}/test/')

    # x_test, test_meta_df = load_full_data(TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH)

    x_test = torch.load(TEST_DATA_PATH.joinpath('representations.pt'))

    test_meta_df = pd.read_csv(TEST_DATA_PATH.joinpath('meta.tsv'), sep='\t')

    # Get original predicted results 
    meta_df = test_meta_df.copy()
    # gold_labels = meta_df['gold_label']
    model_predictions_entailment = meta_df['model_predictions'].apply(model_label_mapper[model_name])
    meta_df['model_predictions'] = model_predictions_entailment
    # original_acc = accuracy_score(gold_labels, model_predictions_entailment)
    return x_test, meta_df 

def generate_control(x_test, control_dims, intervened_dir=None, which_control="amnesic"):
    # get a controls based on rank/removed dims
    # rand_dims is the number removed
    # rand_dims = x_test.shape[1] - control_dims
    if which_control=="mnestic":
        rand_dims = x_test.shape[1] - control_dims
    elif which_control=="amnesic":
        rand_dims =  control_dims
    elif which_control=="control":
        rand_dims = control_dims

    rand_control, rand_direction_p = rand_direction_control(x_test, rand_dims)

    if intervened_dir:
        with open(intervened_dir.joinpath(f'control_{which_control}_{control_dims}.npy'), 'wb') as file:
            np.save(file, rand_control)
        # with open(intervened_dir.joinpath(f'control_rand_direction_p_{control_dims}.npy'), 'wb') as file:
        #     np.save(file, rand_direction_p)

    return  rand_control, rand_direction_p

def project_onto_probe_nullspaces(x_test, results_dir):
    '''
    Given the following directory structure:

    results_dir/
    - Ps/
    - amnesic_reps/
    - mnesic_reps/

    Load the projection matrices from the "Ps" folder and calculate all nullspace projected representations.

    '''
    amnesic_Ps_dir = results_dir.joinpath('Ps/')
    amnesic_reps_dir = results_dir.joinpath('amnesic_reps/')

    amnesic_reps_path = amnesic_reps_dir.joinpath('amnesic_P.npy')
    P = np.load(Path(amnesic_Ps_dir).joinpath(f'P.npy'))
    amnesic_reps = np.dot(x_test, P)
    np.save(amnesic_reps_path, amnesic_reps)

    enumerate_matrices = range(len(list(amnesic_Ps_dir.glob('*.npy'))) -1)
    pbar = tqdm(enumerate_matrices)
    
    for n in pbar:
        # TODO: do it cumulatively to make linear  

        # cumulative multiplication by P_0 ... P_n
        P = np.load(Path(amnesic_Ps_dir).joinpath(f'P_{n}.npy'))
        amnesic_reps = x_test
        amnesic_reps = np.dot(amnesic_reps, P)

        amnesic_reps_path = amnesic_reps_dir.joinpath(f'amnesic_up_to_P_{n}.npy')
        np.save(amnesic_reps_path, amnesic_reps)
    
    return amnesic_reps

def project_onto_CROSS_probe_nullspaces(x_test, results_dir, cross_results_dir):
    '''
    Given the following directory structure:

    results_dir/
    - Ps/
    - amnesic_reps/
    - mnesic_reps/

    Load the projection matrices from the "Ps" folder and calculate all nullspace projected representations.

    '''
    cross_amnesic_Ps_dir = cross_results_dir.joinpath('Ps/')
    cross_amnesic_reps_dir = results_dir.joinpath('cross_amnesic_reps/')

    cross_amnesic_reps_path = cross_amnesic_reps_dir.joinpath('cross_amnesic_P.npy')
    P = np.load(Path(cross_amnesic_Ps_dir).joinpath(f'P.npy'))
    cross_amnesic_reps = np.dot(x_test, P)
    np.save(cross_amnesic_reps_path, cross_amnesic_reps)

    enumerate_matrices = range(len(list(cross_amnesic_Ps_dir.glob('*.npy'))) -1)
    pbar = tqdm(enumerate_matrices)
    
    # cumulative multiplication by P_0 ... P_n
    for n in pbar:
        P = np.load(Path(cross_amnesic_Ps_dir).joinpath(f'P_{n}.npy'))
        cross_amnesic_reps = x_test
        cross_amnesic_reps = np.dot(cross_amnesic_reps, P)

        cross_amnesic_reps_path = cross_amnesic_reps_dir.joinpath(f'cross_amnesic_up_to_P_{n}.npy')
        np.save(cross_amnesic_reps_path, cross_amnesic_reps)

def project_onto_probe_rowspaces(x_test, results_dir):
    '''
    Given the following directory structure:

    results_dir/
    - Ps/
    - Ws/
    - Prows/
    - amnesic_reps/
    - mnesic_reps/

    Load the projection matrices from the "Ps" folder and calculate all rowspace projected representations.

    '''
    amnesic_Prows_dir = results_dir.joinpath('Prows/')
    rowspace_reps_dir = results_dir.joinpath('rowspace_reps')

    enumerate_matrices = range(len(list(amnesic_Prows_dir.glob('*.npy'))))
    pbar = tqdm(enumerate_matrices)
    
    # TODO: Maybe cumulative, not singular?
    for n in pbar:
        Prow = np.load(Path(amnesic_Prows_dir).joinpath(f'Prow_{n}.npy'))
        rowspace_reps = x_test
        rowspace_reps = np.dot(rowspace_reps, Prow)

        rowspace_reps_path = rowspace_reps_dir.joinpath(f'rowspace_reps_Prow_{n}.npy')
        np.save(rowspace_reps_path, rowspace_reps)


def project_onto_probe_subspaces(x_test, results_dir):
    '''
    Given the following directory structure:

    results_dir/
    - Ps/
    - amnesic_reps/
    - mnesic_reps/

    Load the projection matrices from the "Ps" folder and calculate all rowspace projected representations.

    '''
    amnesic_Ps_dir = results_dir.joinpath('Ps/')
    mnesic_reps_dir = results_dir.joinpath('mnesic_reps/')

    P = np.load(Path(amnesic_Ps_dir).joinpath(f'P.npy'))
    mnesic_reps = np.dot(x_test, np.identity(P.shape[0])-P)

    mnesic_reps_path = mnesic_reps_dir.joinpath('mnesic_P.npy')
    np.save(mnesic_reps_path, mnesic_reps)


    enumerate_matrices = range(len(list(amnesic_Ps_dir.glob('*.npy'))) -1)
    pbar = tqdm(enumerate_matrices)
    
    # cumulative multiplication 
    for n in pbar:
        P = np.load(Path(amnesic_Ps_dir).joinpath(f'P_{n}.npy'))
        mnesic_reps = x_test
        mnesic_reps = np.dot(mnesic_reps, np.identity(P.shape[0])-P)

        mnesic_reps_path = mnesic_reps_dir.joinpath(f'mnesic_up_to_P_{n}.npy')
        np.save(mnesic_reps_path, mnesic_reps)

def create_and_save_amnesic_reps(x_test, results_dir):
    project_onto_probe_nullspaces(x_test, results_dir)

def create_and_save_cross_amnesic_reps(x_test, results_dir, cross_results_dir):
    half_way = project_onto_probe_nullspaces(x_test, results_dir)
    project_onto_CROSS_probe_nullspaces(half_way, results_dir, cross_results_dir)

def create_and_save_mnesic_reps(x_test, results_dir):
    project_onto_probe_subspaces(x_test, results_dir)

def create_and_save_rowspace_reps(x_test, results_dir):
    project_onto_probe_rowspaces(x_test, results_dir)


if __name__ == '__main__':

    for label_column in ['context_monotonicity', 'insertion_rel', 'composite', 'gold_label', 'control']:
    # for label_column in ['control']:
        for model_name in MODEL_NAMES:
            if label_column=="context_monotonicity":
                cross_label = "insertion_rel"
            elif label_column=="insertion_rel":
                cross_label = "context_monotonicity"

            results_dir = Path(f'experiments/interventions/results/{label_column}/{model_name}/')
            amnesic_reps_dir = results_dir.joinpath('amnesic_reps')
            amnesic_Ps_dir = results_dir.joinpath('Ps')
            mnesic_reps_dir = results_dir.joinpath('mnesic_reps')
            rowspace_reps_dir = results_dir.joinpath('rowspace_reps')
            control_amnesic_reps_dir = results_dir.joinpath('control_amnesic_reps')
            control_mnesic_reps_dir = results_dir.joinpath('control_mnesic_reps')
            meta_path = results_dir.joinpath('meta.tsv')


            if not os.path.exists(control_amnesic_reps_dir):
                os.makedirs(control_amnesic_reps_dir)
            if not os.path.exists(control_mnesic_reps_dir):
                os.makedirs(control_mnesic_reps_dir)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            if not os.path.exists(amnesic_reps_dir):
                os.makedirs(amnesic_reps_dir)
            if not os.path.exists(amnesic_Ps_dir):
                os.makedirs(amnesic_Ps_dir)
            if not os.path.exists(mnesic_reps_dir):
                os.makedirs(mnesic_reps_dir)
            if not os.path.exists(rowspace_reps_dir):
                os.makedirs(rowspace_reps_dir)
            

            x_test, meta_df = prepare_data(model_name)
            with open(meta_path, 'w+') as meta_file:
                meta_file.write(meta_df.to_csv(sep='\t', index=False))

            # create_and_save_cross_amnesic_reps(x_test, results_dir, cross_results_dir)

            if label_column=='control':

                # config = dict(
                #     encoder=model_name,
                #     property=label_column,
                #     layer='last'
                # )

                # wandb.init(
                #     name=f'{model_name}',
                #     project=f"nli_xy_controls",
                #     tags=["nli_xy"],
                #     config=config,
                #     reinit=True
                # )

                intervened_dir = control_amnesic_reps_dir
                for control_dims in tqdm(range(1, 50)):
                    # try:
                    #     with open(intervened_dir.joinpath(f'control_{control_dims}.npy'), 'rb') as file:
                    #         rand_control = np.load(file)

                    rand_control, rand_direction_p = generate_control(x_test, control_dims, intervened_dir=intervened_dir, which_control="control")
            else:
                pass
                # create_and_save_amnesic_reps(x_test, results_dir)
                # create_and_save_mnesic_reps(x_test, results_dir)
                # create_and_save_rowspace_reps(x_test, results_dir)



                        # remain_p = np.identity(x_test.shape[1]) - rand_direction_p
                        # print(remain_p.shape)
                        # rand_directions = sp.linalg.orth(remain_p)
                        # print(len(rand_directions))
                        # print(rand_directions[0].shape)

                        # rand_rank = np.linalg.matrix_rank(rand_control)
                        # print(f'Rank of Control Matrix: {rand_rank}\n')

                        # rand_accuracy = get_prediction_accuracy(rand_control, model, gold_labels)
                        # print(f'control accuracy with {control_dims} remaining dimensions: {rand_accuracy}')

                        # Ws = []
                        # for i in range(b_rank):
                        #     W = np.load(Path(out_dir).joinpath(f'W_{i}.npy'))
                        #     Ws.append(W)
                        # dot_matrix = np.asarray([[np.dot(r,w.T)/(norm(r)*norm(w)) for r in rand_directions] for w in Ws])
                        # dot_matrix = dot_matrix.reshape(len(rand_directions), len(Ws))
                        # print(f'DOT Product Matrix Shape: {dot_matrix.shape}')
                        # plt.imshow(dot_matrix, cmap='hot', interpolation='nearest')
                        # plt.show()

                        # project_and_show(rand_control, meta_df)

                    # if show:
                    #     project_and_show(x_test, meta_df)
                    #     project_and_show(debiased, meta_df)