# %% 
from prefect import task
import pandas as pd
from tqdm import tqdm
import os

#%% 
@task
def prep_for_probeably(folder_path, auxiliary_task_column, output_name):
    embed_filepath = os.path.join(folder_path, 'embedding.tsv')
    meta_filepath = os.path.join(folder_path, 'meta.tsv')

    reps = pd.read_csv(embed_filepath, sep='\t', header=None)
    meta = pd.read_csv(meta_filepath, sep='\t')


    reps['label'] = pd.factorize(meta[auxiliary_task_column])[0]

    out_filepath = os.path.join(folder_path, f'{output_name}.tsv')
    with open(out_filepath, 'w+') as out_file:
        out_file.write(reps.to_csv(sep='\t', header=False))

@task
def prep_set_for_probeably(root_folder_path, auxiliary_task_column, output_name):
    child_dirs = os.listdir(root_folder_path)
    for child_dir in tqdm(child_dirs):
        folder_path = os.path.join(root_folder_path, child_dir)
        prep_for_probeably.run(folder_path, auxiliary_task_column, output_name)

#%%
root_folder_path = 'experiments/layer_ablations/roberta-large-mnli-help/'

reps = prep_set_for_probeably.run(root_folder_path, 'context_monotonicity', 'probe_reps_and_labels')

# %%
