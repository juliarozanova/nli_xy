#%% 
import os
os.chdir('../')
os.chdir('../')

#%%
from pathlib import Path
import sys
from nli_xy.constants import AMNESIC_PATH 
sys.path.append(AMNESIC_PATH)
from amnesic_probing.tasks.utils import rand_direction_control
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from transformers import AutoModelForSequenceClassification as Model
from loguru import logger

# model_names = ['roberta-large-mnli', 'roberta-large-mnli-help', 'roberta-large-mnli-double-finetuning', 'bert-base-uncased-snli','bert-base-uncased-snli-help', 'facebook-bart-large-mnli', 'facebook-bart-large-mnli-help']
model_names = ['roberta-large-mnli', 'roberta-large-mnli-help', 'roberta-large-mnli-double-finetuning', 'bert-base-uncased-snli','bert-base-uncased-snli-help']
# model_names = ['facebook/bart-large-mnli']

label_column = 'context_monotonicity'
token_choice = 'CLS'
classification_type = label_column
layer = 'last'
show = True
flipped = False

# gold labels are always two-class
# predicted labels depend on model

#%%
def two_class_relabel(two_class_label):
    if two_class_label in [1, '1']:
        out_label = 'entailment'
    elif two_class_label in [0, '0']:
        out_label = 'non-entailment'
    elif two_class_label in [2, '2']:
        out_label = 'non-entailment'
    else:
        raise ValueError(f'Unexpected two class label: {two_class_label} of type {type(two_class_label)}')
    return out_label

def three_class_relabel(three_class_label):
    if three_class_label in [2, '2']:
        out_label = 'entailment'
    elif three_class_label in [0, '0', 1, '1']:
        out_label = 'non-entailment'
    return out_label

def compare_debiased(debiased, baseline_debiased, model_name):
    model_pretrained_path = {
        'roberta-large-mnli': 'roberta-large-mnli',
        'roberta-large-mnli-help':  './models/roberta-large-mnli-help', 
        'roberta-large-mnli-double-finetuning':  './models/roberta-large-mnli-double_finetuning', 
        'facebook-bart-large-mnli': 'facebook/bart-large-mnli', 
        'facebook-bart-large-mnli-help': './models/facebook-bart-large-mnli-help',
        'bert-base-uncased-snli': 'textattack/bert-base-uncased-snli',
        'bert-base-uncased-snli-help': './models/bert-base-uncased-snli-help'
    }
    model = Model.from_pretrained(model_pretrained_path[model_name])
    model.eval()
    if 'roberta' in model_name:
        nli_clf = model.classifier
    else:
        def nli_clf(x):
            return model.classifier(model.bert.pooler(x)) 
    # TODO. Fix to do pooling for Bart
        


    meta_df = test_meta_df
    entailment_full = meta_df['gold_label']
    model_predictions_entailment = meta_df['model_predictions'].apply(output_label_mapper[model_name])
    meta_df['model_predictions'] = model_predictions_entailment

    meta_df['correct'] = model_predictions_entailment == meta_df['gold_label']

    # forward pass through debiased features and new predictions

    def predict_new_vecs(new_vecs):
        if 'roberta' in model_name:
            features = torch.tensor(new_vecs).view(new_vecs.shape[0],-1,1024).type(torch.float32)
        elif 'bert-base' in model_name:
            features = torch.tensor(new_vecs).view(new_vecs.shape[0],-1,768).type(torch.float32)
        elif 'bart-large' in model_name:
            features = torch.tensor(new_vecs).view(new_vecs.shape[0],-1,1024).type(torch.float32)

        new_preds=nli_clf(features).detach().numpy()
        int_new_preds = [int(x) for x in list(np.argmax(new_preds, axis=-1))]
        final_preds = [output_label_mapper[model_name](x) for x in int_new_preds]

        return final_preds


    from sklearn.metrics import accuracy_score

    debiased_preds = predict_new_vecs(debiased)
    debiased_acc = accuracy_score(entailment_full, debiased_preds)

    baseline_debiased_preds = predict_new_vecs(baseline_debiased)
    baseline_debiased_acc = accuracy_score(entailment_full, baseline_debiased_preds)
    #%%
    original_acc = accuracy_score(entailment_full, model_predictions_entailment)
    print(f'''\n Debiased Accuracy: {debiased_acc}, \n
            Baseline Debiased Accuracy: {baseline_debiased_acc} \n
            Original Accuracy: {original_acc} \n
            ''')

    return meta_df

output_label_mapper = {
    'roberta-large-mnli': three_class_relabel,
    'roberta-large-mnli-help': two_class_relabel,
    'roberta-large-mnli-double-finetuning': three_class_relabel,
    'bert-base-uncased-snli': three_class_relabel,
    'bert-base-uncased-snli-help': three_class_relabel, 
    'facebook-bart-large-mnli': three_class_relabel, 
    'facebook-bart-large-mnli-help':  three_class_relabel,
}

# %%
for model_name in model_names:
    print(f'Model Name: {model_name}')
    out_dir = f'experiments/interventions/results/{label_column}/{token_choice}/{model_name}/'

    # TRAIN_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/train/')
    # DEV_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/dev/')
    TEST_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/test/')

    x_test = torch.load(TEST_DATA_PATH.joinpath('representations.pt')).numpy()
    test_meta_df = pd.read_csv(TEST_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    y_test = test_meta_df[label_column]

    debiased = x_test

    for i in range(80):
        try:
            P = np.load(Path(out_dir).joinpath(f'P_{i}.npy'))
            debiased = np.dot(debiased, P)
            removed_dims = i
        except FileNotFoundError:
            print(f'Number of dimensions removed: {removed_dims}')
            break

    P = np.load(Path(out_dir).joinpath(f'P.npy'))
    # debiased = np.dot(debiased, np.identity(P.shape[0])- P)
    # debiased = np.dot(debiased, P)
    debiased = P.dot(x_test.T).T



    # choose_directions = np.random.choice([0,1], size=debiased.shape[1], p=[0.07, 0.93])
    # baseline_debiased = np.linalg.multi_dot([x_test, choose_directions.reshape(-1,1)])
    # baseline_debiased = np.dot(np.tile(choose_directions, (x_test.shape[1],1)), x_test.T).T
    baseline_debiased = rand_direction_control(x_test, 80)
    print(baseline_debiased.shape)

    orig_rank = np.linalg.matrix_rank(x_test)
    deb_rank = np.linalg.matrix_rank(debiased)
    rand_rank = np.linalg.matrix_rank(baseline_debiased)
    print(f'''Rank of original matrix: {orig_rank} \n 
                Rank of debiased matrix: {deb_rank} \n
                Rank of control matrix: {rand_rank} \n''')

    # TODO: weird inverse
    # debiased = np.dot(x_test, (np.identity(P.shape[0]) - P))

    if show:
        pca = PCA(n_components=3)
        projected = pca.fit_transform(debiased)
        projected = pd.DataFrame(projected, columns=['x', 'y', 'z'])
        pca = PCA(n_components=3)
        orig_projected = pca.fit_transform(x_test)
        orig_projected = pd.DataFrame(orig_projected, columns=['x', 'y', 'z'])

        pca = PCA(n_components=3)
        baseline_projected = pca.fit_transform(baseline_debiased)
        baseline_projected = pd.DataFrame(baseline_projected, columns=['x', 'y', 'z'])

        fig1 = px.scatter_3d(projected, 
        x=projected.x, 
        y=projected.y, 
        z=projected.z, 
        color=y_test,
        title=f'{model_name}',
        hover_data=projected.columns)

        fig2 = px.scatter_3d(baseline_projected, 
        x=baseline_projected.x, 
        y=baseline_projected.y, 
        z=baseline_projected.z, 
        color=y_test,
        title=f'{model_name}',
        hover_data=baseline_projected.columns)

        fig3 = px.scatter_3d(orig_projected, 
        x=orig_projected.x, 
        y=orig_projected.y, 
        z=orig_projected.z, 
        color=y_test,
        title=f'{model_name}',
        hover_data=orig_projected.columns)

        fig1.show()
        fig2.show()
        fig3.show()

    df = compare_debiased(debiased, baseline_debiased, model_name)

    # SANITY CHECK:
    #Ensure probing dataset model predictions and eval_on_nli_xy  predictions match up
#     other_df = pd.read_csv(f'experiments/nli/eval_on_nli_xy/results/{model_name}/nli_xy_meta.tsv', sep='\t')
#     other_df['model_predictions'] = other_df['y_pred'].apply(two_class_relabel)

#     for test_context in set(df.context.to_list()):
#         a = df.loc[(df.context==test_context)]
#         b = other_df.loc[other_df.context==test_context]

#         test_pairs  = a.insertion_pair.to_list()
#         for test_pair in test_pairs:
#             part_a = a.loc[a.insertion_pair==test_pair]
#             part_b = b.loc[b.insertion_pair==test_pair]

#             gold_a = set(part_a['model_predictions'].to_list())
#             gold_b = set(part_b['model_predictions'].to_list())

#             try:
#                 assert gold_a == gold_b
#             except AssertionError:
#                 print(test_pair)
#                 print(gold_a)
#                 print(gold_b)
#                 break

# #other_df always has two_label predictions
# # %%

