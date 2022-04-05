#%%
import os
os.chdir('../')

#%%
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from sklearn.decomposition import PCA
from transformers import AutoModelForSequenceClassification as Model

model_names = ['roberta-large-mnli', 'roberta-large-mnli-help', 'bert-base-uncased-snli','bert-base-uncased-snli-help']

label_column = 'insertion_rel'
token_choice = 'CLS'
classification_type = label_column
layer = 'last'
show = True
flipped = False


#%%
def compare_debiased(debiased, model_name):
    if model_name == 'roberta-large-mnli-help':
        model_name = './models/roberta-large-mnli-help'
    if model_name == 'facebook-bart-large-mnli':
        model_name = 'facebook/bart-large-mnli'
    if model_name == 'facebook-bart-large-mnli':
        model_name = './models/facebook-bart-large-mnli-help'
    if model_name == 'bert-base-uncased-snli':
        model_name = 'textattack/bert-base-uncased-snli'
    if model_name == 'bert-base-uncased-snli-help':
        model_name = './models/bert-base-uncased-snli-help'

    nli_clf = Model.from_pretrained(model_name).classifier
    if 'roberta' in model_name:
        features = torch.tensor(debiased).view(debiased.shape[0],-1,1024).type(torch.float32)
    elif 'snli' in model_name:
        features = torch.tensor(debiased).view(debiased.shape[0],-1,768).type(torch.float32)

    new_preds=nli_clf(features).detach()

    def relabel(three_class_label):
        if three_class_label == 2:
            two_class_label = 'entailment'
        if three_class_label in [0,1]:
            two_class_label = 'non-entailment'
        return two_class_label
        

    entailment_dev = dev_meta_df['gold_label']
    model_predictions_entailment = dev_meta_df['model_predictions'].apply(relabel)
    dev_meta_df['correct'] = model_predictions_entailment == dev_meta_df['gold_label']

    int_new_preds = list(np.argmax(new_preds, axis=1))
    final_preds = [relabel(x) for x in int_new_preds]

    from sklearn.metrics import accuracy_score
    debiased_acc = accuracy_score(entailment_dev, final_preds)
    #%%
    original_acc = accuracy_score(entailment_dev, model_predictions_entailment)
    print(f'''Model: {model_name}, \n Debiased Accuracy: {debiased_acc}, \n
            Original Accuracy: {original_acc} \n
            ''')

    return dev_meta_df

# %%
for model_name in model_names:
    out_dir = f'experiments/interventions/results/{label_column}/{token_choice}/{model_name}/'

    # TRAIN_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/train/')
    DEV_DATA_PATH = Path(f'./experiments/probing/compare_models_{token_choice}/processed_data/{model_name}/train/')
    # x_train = torch.load(TRAIN_DATA_PATH.joinpath('representations.pt')).numpy()
    # train_meta_df = pd.read_csv(TRAIN_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    # y_train = train_meta_df[label_column]

    x_dev = torch.load(DEV_DATA_PATH.joinpath('representations.pt')).numpy()
    dev_meta_df = pd.read_csv(DEV_DATA_PATH.joinpath('meta.tsv'), sep='\t')
    y_dev = dev_meta_df[label_column]

    # if flipped:
    #     x_train, y_train, x_dev, y_dev = x_dev, y_dev, x_train, y_train

    P = np.load(Path(out_dir).joinpath('P.npy'))
    debiased = np.dot(x_dev, P)

    # debiased = x_train
    if show:
        pca = PCA(n_components=3)
        projected = pca.fit_transform(debiased)
        projected = pd.DataFrame(projected, columns=['x', 'y', 'z'])
        pca = PCA(n_components=3)
        orig_projected = pca.fit_transform(x_dev)
        orig_projected = pd.DataFrame(orig_projected, columns=['x', 'y', 'z'])
        # fig = px.scatter_3d(projected)
        fig1 = px.scatter_3d(projected, 
        x=projected.x, 
        y=projected.y, 
        z=projected.z, 
        color=y_dev,
        title=f'{model_name}',
        hover_data=projected.columns)

        fig2 = px.scatter_3d(orig_projected, 
        x=orig_projected.x, 
        y=orig_projected.y, 
        z=orig_projected.z, 
        color=y_dev,
        title=f'{model_name}',
        hover_data=projected.columns)

        fig1.show()
        fig2.show()

    df = compare_debiased(debiased, model_name)
#%% 