#%%
from pathlib import Path
import sys
from nli_xy.constants import AMNESIC_PATH 
sys.path.append(AMNESIC_PATH)

import torch
import pandas as pd
import numpy as np
from collections import Counter
import wandb
import json
from torch.utils.tensorboard import SummaryWriter
from amnesic_probing.tasks.utils import get_projection_matrix

#%%
if __name__=='__main__':
    np.random.seed(1)
    models = ['roberta-large-mnli', 'roberta-large-mnli-help', 'roberta-large-mnli-double-finetuning', 'bert-base-uncased-snli','bert-base-uncased-snli-help']

    label_column = 'context_monotonicity'
    token_choice = 'CLS'
    classification_type = label_column
    layer = 'last'

    flipped=False

    for model in models:
        out_dir = f'experiments/interventions/results/{label_column}/{token_choice}/{model}/'
        writer = SummaryWriter(out_dir)

        TRAIN_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model}/train/')
        DEV_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model}/dev/')
        TEST_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model}/test/')

        # combine train and dev sets to train amnesic probe

        x_train_a = torch.load(TRAIN_DATA_PATH.joinpath('representations.pt'))
        x_train_b = torch.load(DEV_DATA_PATH.joinpath('representations.pt'))
        x_train = torch.vstack([x_train_a, x_train_b]).numpy()
        train_meta_df_a = pd.read_csv(TRAIN_DATA_PATH.joinpath('meta.tsv'), sep='\t')
        train_meta_df_b = pd.read_csv(DEV_DATA_PATH.joinpath('meta.tsv'), sep='\t')
        train_meta_df = pd.concat([train_meta_df_a, train_meta_df_b])
        y_train = train_meta_df[label_column]

        # use nli_xy test set to 
        x_dev = torch.load(TEST_DATA_PATH.joinpath('representations.pt')).numpy()
        dev_meta_df = pd.read_csv(TEST_DATA_PATH.joinpath('meta.tsv'), sep='\t')
        y_dev = dev_meta_df[label_column]

        num_clfs = 100
        n_classes = len(set(y_train))
        majority = Counter(y_dev).most_common(1)[0][1] / float(len(y_dev))
        max_iter = 100000
        print('number of classes:', n_classes)
        print('most common class (dev):', majority)


        config = dict(
            encoder=model,
            property=label_column,
            layer=layer
        )

        wandb.init(
            name=f'{model}_{layer}_inlp',
            project=f"larger_set_amnesic_{label_column}_{token_choice}",
            tags=["inlp", classification_type],
            config=config,
            reinit=True
        )

        P, all_projections, best_projection = get_projection_matrix(num_clfs,
                                                                    x_train, y_train, x_dev, y_dev,
                                                                    majority_acc=majority, max_iter=max_iter,
                                                                    summary_writer=writer)

        for i, projection in enumerate(all_projections):
            np.save(out_dir + '/P_{}.npy'.format(i), projection)

        np.save(out_dir + '/P.npy', best_projection[0])

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

        json.dump(meta_dic, open(out_dir + '/meta.json', 'w'))

# %%
