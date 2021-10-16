from prefect import task
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from nli_xy.encoding.utils import get_target_reps

@task
def encode_split_datasets(split_datasets, encoder_model, encode_config, device):
    splits = ['train', 'dev', 'test']
    # check exists_dir
    encoded_splits = dict(zip(splits,
                        [encode_dataset.run(split_datasets[split], encoder_model, encode_config, device)
                             for split in splits]  
                        ))
                                                
    return {
        'train': encoded_splits['train'],
        'dev': encoded_splits['dev'],
        'test': encoded_splits['test']
    }  

@task
def encode_dataset(dataset, encoder_model, config, device, write_to_file=False):
    '''
        representations: torch.tensor() of size (dataset_size, embedding_size)
        task: string, 'monotonicity' or 'rel'

    '''

    dataloader = DataLoader(dataset, config['batch_size'])

    # iterate encoding over batches
    # TODO: either remove or update the write_to_file functionality here

    # check whether the model is an NLI model (we will ignore its predictions if not)
    # is_it_nli = encoder_is_nli_model(rep_config['encoder_model']) 

    model_predictions = []

    if write_to_file:
        embed_file = open(RESULTS_DIR+'embeddings.tsv', 'a+')
        embed_file.truncate(0) # ensure clear file before appending

    batch_rep_list = []
    batch_meta_list = []

    encoder_model.eval()
    with torch.no_grad():

        for inputs in tqdm(dataloader):

            outputs = encoder_model(inputs['input_ids'], inputs['attention_mask'])
            try:
                hidden = outputs['hidden_states']
            except KeyError:
                hidden = outputs['encoder_hidden_states']

            logits = outputs['logits']
            batch_predictions = torch.argmax(logits, dim=1)    
            model_predictions.append(batch_predictions)

            target_reps = get_target_reps(hidden,
                                            inputs['X_range'],
                                            inputs['Y_range'], 
                                            inputs['CLS_token_index'],
                                            config, 
                                            device)

            if write_to_file:
                np.savetxt(embed_file,
                          target_reps.cpu().numpy(),
                          delimiter='\t')
            # else:
        
            batch_rep_list.append(target_reps.cpu())
            #todo: write logits prediction to meta file

    model_predictions = torch.cat(model_predictions).cpu()
    dataset.meta_df['model_predictions'] = model_predictions

    dataset.meta_df.drop(labels=['X_range', 'Y_range'],
                  axis=1,
                  inplace=True)

    if write_to_file:
        with open(results_dir+'meta.tsv', 'w+') as meta_file:
            meta_file.write(dataset.meta_df.to_csv(sep='\t'))

    #representations = torch.tensor(torch.cat(batch_rep_list), dtype=torch.float)
    representations = torch.cat(batch_rep_list).type(torch.float)

    task='monotonicity'
    def relabel(label, task):
        if task=='monotonicity':
            return 0 if label=='down' else 1

        if task=='rel':
            if label=='leq':
                return 0
            if label=='geq':
                return 1
            if label=='none':
                return 2

    if task=='monotonicity':
        labels = dataset.meta_df['context_monotonicity'].apply(lambda x: relabel(x, task))
    if task=='rel':
        labels = dataset.meta_df['insertion_rel'].apply(lambda x: relabel(x, task))

    labels = torch.tensor(labels.values.tolist()).to(device)

    return {
            'representations': representations.to('cpu'),
            'labels': labels.to('cpu')
           }