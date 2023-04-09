#%%
import os
import sys
from pathlib import Path

from nli_xy.constants import AMNESIC_PATH, MODEL_NAMES
sys.path.append(AMNESIC_PATH)

import wandb
import torch
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import json
from amnesic_probing.tasks.utils import get_projection_matrix

# config
def run_amnesic_probing(model_name, label_column):
    token_choice = 'CLS'
    classification_type = label_column

    results_dir = Path(f'experiments/interventions/results/{label_column}/{model_name}/')
    writer = SummaryWriter(results_dir)

    amnesic_reps_dir = results_dir.joinpath('amnesic_reps')
    amnesic_Ps_dir = results_dir.joinpath('Ps')
    amnesic_Ws_dir = results_dir.joinpath('Ws')
    amnesic_Prows_dir = results_dir.joinpath('Prows')
    mnesic_reps_dir = results_dir.joinpath('mnesic_reps')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(amnesic_reps_dir):
        os.makedirs(amnesic_reps_dir)
    if not os.path.exists(amnesic_Ps_dir):
        os.makedirs(amnesic_Ps_dir)
    if not os.path.exists(amnesic_Prows_dir):
        os.makedirs(amnesic_Prows_dir)
    if not os.path.exists(amnesic_Ws_dir):
        os.makedirs(amnesic_Ws_dir)
    if not os.path.exists(mnesic_reps_dir):
        os.makedirs(mnesic_reps_dir)

    TRAIN_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/train/')
    DEV_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/dev/')
    TEST_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/test/')

    # combine train and dev sets to train amnesic probe

    x_train_a = torch.load(TRAIN_DATA_PATH.joinpath('representations.pt'))
    x_train_b = torch.load(DEV_DATA_PATH.joinpath('representations.pt'))
    x_train = torch.vstack([x_train_a, x_train_b]).numpy()
    train_meta_df_a = pd.read_csv(TRAIN_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    train_meta_df_b = pd.read_csv(DEV_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    train_meta_df = pd.concat([train_meta_df_a, train_meta_df_b])
    y_train = train_meta_df[label_column]

    # use nli_xy test set 
    x_dev = torch.load(TEST_DATA_PATH.joinpath('representations.pt')).numpy()
    dev_meta_df = pd.read_csv(TEST_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    y_dev = dev_meta_df[label_column]

    num_clfs = 200
    n_classes = len(set(y_train))
    majority = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))
    max_iter = 1000
    print('number of classes:', n_classes)
    print('most common class (dev):', majority)

    config = dict(
        encoder=model_name,
        property=label_column,
        layer=layer
    )

    wandb.init(
        name=f'{model_name}_{layer}_inlp',
        project=f"amnesic_{label_column}_{token_choice}",
        tags=["inlp", classification_type],
        config=config,
        reinit=True
    )

    P, rowspace_projections, Ws, all_projections, best_projection = get_projection_matrix(num_clfs,
                                                                x_train, y_train, x_dev, y_dev,
                                                                majority_acc=majority, max_iter=max_iter,
                                                                summary_writer=writer
                                                                )

    for i, projection in enumerate(all_projections):
        np.save(results_dir.joinpath(f'Ps/P_{i}.npy'), projection)
        np.save(results_dir.joinpath(f'Ws/W_{i}.npy'), Ws[i])
        np.save(results_dir.joinpath(f'Prows/Prow_{i}'), rowspace_projections[i])

    np.save(results_dir.joinpath('Ps/P.npy'), P)

    removed_directions = int((best_projection[1]) * n_classes)
        # in case of 2 classes, each inlp iteration we remove a single direction
    if n_classes == 2:
        removed_directions /= 2

    else:  # in regression tasks, each iteration we remove a single dimension
        removed_directions = int((best_projection[1]))

    meta_dic = {'best_i': best_projection[1],
                'n_classes': n_classes,
                'majority': majority,
                'removed_directions': removed_directions}

    wandb.run.summary['best_i'] = best_projection[1]
    wandb.run.summary['removed_directions'] = removed_directions

    json.dump(meta_dic, open(results_dir.joinpath('meta.json'), 'w+'))

#%%
if __name__==('__main__'):
    np.random.seed(42)
    layer = 'last'
    # for label_column in ['insertion_rel', 'context_monotonicity', 'composite']:
    for label_column in ['context_monotonicity', 'insertion_rel', 'composite', 'gold_label']:
        for model_name in MODEL_NAMES:
            run_amnesic_probing(model_name, label_column)