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
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification as Model
from loguru import logger

# model_names = ['roberta-large-mnli', 'roberta-large-mnli-help', 'roberta-large-mnli-double-finetuning', 'bert-base-uncased-snli','bert-base-uncased-snli-help']
# model_names = ['roberta-large-mnli-help', 'roberta-large-mnli-double-finetuning', 'bert-base-uncased-snli','bert-base-uncased-snli-help']
model_names = ['roberta-large-mnli-help']#, 'roberta-large-mnli-double-finetuning', 'bert-base-uncased-snli','bert-base-uncased-snli-help']
label_column = 'context_monotonicity'
token_choice = 'CLS'
classification_type = label_column
layer = 'last'
show = False

# gold labels are always two-class
# predicted labels depend on model

#%%
def collect_full_data(TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH):
    x_test_train = torch.load(TRAIN_DATA_PATH.joinpath('representations.pt'))
    x_test_dev = torch.load(DEV_DATA_PATH.joinpath('representations.pt'))
    x_test_test = torch.load(TEST_DATA_PATH.joinpath('representations.pt'))

    test_meta_df_train = pd.read_csv(TRAIN_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    test_meta_df_dev = pd.read_csv(DEV_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    test_meta_df_test = pd.read_csv(TEST_DATA_PATH.joinpath('meta.tsv'), sep='\t')

    x_test = torch.vstack([x_test_train, x_test_dev, x_test_test]).numpy()
    test_meta_df = pd.concat([test_meta_df_train, test_meta_df_dev, test_meta_df_test])

    return x_test, test_meta_df


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


def project_onto_probe_nullspaces(x_test, out_dir, probe_ids=None):
    debiased = x_test
    if probe_ids == None:
        P = np.load(Path(out_dir).joinpath(f'P.npy'))
        debiased = np.dot(debiased, P)
    else:
        for i in probe_ids:
            try:
                P = np.load(Path(out_dir).joinpath(f'P_{i}.npy'))
                debiased = np.dot(debiased, P)
                removed_dims = i
            except FileNotFoundError:
                # print(f'Number of dimensions removed: {removed_dims}')
                break
    return debiased


def project_onto_probe_rowspaces(x_test, out_dir, probe_ids=None):
    biased = x_test
    if probe_ids == None:
        P = np.load(Path(out_dir).joinpath(f'P.npy'))
        biased = np.dot(biased, np.identity(P.shape[0])-P)
    else:
        for i in probe_ids:
            try:
                P = np.load(Path(out_dir).joinpath(f'P_{i}.npy'))
                biased = np.dot(biased, np.identity(P.shape[0])-P)
                removed_dims = i
            except FileNotFoundError:
                # print(f'Number of dimensions removed: {removed_dims}')
                break
    return biased


def project_and_show(input_matrix, labels):
    pca = PCA(n_components=3)
    projected = pca.fit_transform(input_matrix)
    projected = pd.DataFrame(projected, columns=['x', 'y', 'z'])
    projected['labels'] = labels

    fig = px.scatter_3d(projected, 
    x=projected.x, 
    y=projected.y, 
    z=projected.z, 
    color=projected.labels,
    title=f'{model_name}',
    hover_data=projected.columns)
    
    fig.show()

def extract_model_classifier_head(model):
    if 'roberta' in model_name:
        nli_clf = model.classifier
    else:
        def nli_clf(x):
            return model.classifier(model.bert.pooler(x)) 
    # TODO. Fix to do pooling for Bart

    return nli_clf

def predict_on_new_reps(new_representations, nli_clf):
    if 'roberta' in model_name:
        features = torch.tensor(new_representations).view(new_representations.shape[0],-1,1024).type(torch.float32)
    elif 'bert-base' in model_name:
        features = torch.tensor(new_representations).view(new_representations.shape[0],-1,768).type(torch.float32)
    elif 'bart-large' in model_name:
        features = torch.tensor(new_representations).view(new_representations.shape[0],-1,1024).type(torch.float32)

    new_preds=nli_clf(features).detach().numpy()
    int_new_preds = [int(x) for x in list(np.argmax(new_preds, axis=-1))]
    predictions = [output_label_mapper[model_name](x) for x in int_new_preds]

    return predictions

def get_prediction_accuracy(representations, model, gold_labels):

    # forward pass through debiased features and new predictions
    nli_clf = extract_model_classifier_head(model)
    predictions = predict_on_new_reps(representations, nli_clf)
    accuracy = accuracy_score(gold_labels, predictions)
    #%%
    return accuracy

output_label_mapper = {
    'roberta-large-mnli': three_class_relabel,
    'roberta-large-mnli-help': two_class_relabel,
    'roberta-large-mnli-double-finetuning': three_class_relabel,
    'bert-base-uncased-snli': three_class_relabel,
    'bert-base-uncased-snli-help': three_class_relabel, 
    'facebook-bart-large-mnli': three_class_relabel, 
    'facebook-bart-large-mnli-help':  three_class_relabel,
}

model_pretrained_path = {
    'roberta-large-mnli': 'roberta-large-mnli',
    'roberta-large-mnli-help':  './models/roberta-large-mnli-help', 
    'roberta-large-mnli-double-finetuning':  './models/roberta-large-mnli-double_finetuning', 
    'facebook-bart-large-mnli': 'facebook/bart-large-mnli', 
    'facebook-bart-large-mnli-help': './models/facebook-bart-large-mnli-help',
    'bert-base-uncased-snli': 'textattack/bert-base-uncased-snli',
    'bert-base-uncased-snli-help': './models/bert-base-uncased-snli-help'
}
# %%
for model_name in model_names:
    print(f'Model Name: {model_name}')
    model = Model.from_pretrained(model_pretrained_path[model_name])
    model.eval()

    out_dir = f'experiments/interventions/results/{label_column}/{token_choice}/{model_name}/'

    TRAIN_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/train/')
    DEV_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/dev/')
    TEST_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/test/')

    x_test, test_meta_df = collect_full_data(TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH)
    y_test = test_meta_df[label_column]

    # Get original predicted results 
    meta_df = test_meta_df.copy()
    gold_labels = meta_df['gold_label']
    model_predictions_entailment = meta_df['model_predictions'].apply(output_label_mapper[model_name])
    meta_df['model_predictions'] = model_predictions_entailment
    original_acc = accuracy_score(gold_labels, model_predictions_entailment)



    probe_ids = None
    # create all representations and predict
    debiased = project_onto_probe_nullspaces(x_test, out_dir, probe_ids)
    debiased_acc = get_prediction_accuracy(debiased, model, gold_labels)

    biased = project_onto_probe_rowspaces(x_test, out_dir, probe_ids)
    biased_acc = get_prediction_accuracy(biased, model, gold_labels)

    print(f'''
        original accuracy: {original_acc}\n
        debiased_accuracy: {debiased_acc}\n
        biased_accuracy: {biased_acc}\n
    ''')


    orig_rank = np.linalg.matrix_rank(x_test)
    deb_rank = np.linalg.matrix_rank(debiased)
    b_rank = np.linalg.matrix_rank(biased)
    print(f'''Rank of original matrix: {orig_rank} \n 
                Rank of debiased matrix: {deb_rank} \n
                Rank of biased matrix: {b_rank} \n''')

    # get all controls based on rank/removed dims
    for control_dims in range(1, deb_rank):
        rand_dims = x_test.shape[1] - control_dims
        rand_control = rand_direction_control(x_test, rand_dims)
        rand_accuracy = get_prediction_accuracy(rand_control, model, gold_labels)
        print(f'control accuracy with {control_dims} remaining dimensions: {rand_accuracy}')
        # rand_rank = np.linalg.matrix_rank(rand_control)
        # print(f'Rank of Control Matrix: {rand_rank}\n')

    if show:
        project_and_show(x_test, meta_df)
        project_and_show(debiased, meta_df)
        project_and_show(rand_control, meta_df)
    # MOST COMPELLING SO FAR (or... it just flips stuff so performs poorly? compare reflection)
    # debiased = (P-np.identity(P.shape[0])).dot(x_test.T).T

    # TODO: weird inverse
    # debiased = np.dot(x_test, (np.identity(P.shape[0]) - P))

    # SANITY CHECK4
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

# NOTE:

# 1. Run probing again on REMAINING semantic subspace
# %%