from prefect import task
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

@task
def get_reps(dataset, encoder_model, config):
    encoder_model.eval()
    dataloader = DataLoader(dataset, batch_size=config['batch_size'])
    reps_list = []
    meta_list = []
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            outputs = encoder_model(**{
                'input_ids': inputs.pop('input_ids'),
                'attention_mask': inputs.pop('attention_mask')
                })
            #outputs = encoder_model(inputs['input_ids'], inputs['attention_mask'])

            reps_list.append([outputs.hidden_states[i].to('cpu') for i in range(len(outputs.hidden_states))])
            meta_list.append(inputs)


    return reps_list, meta_list

def encode_dataset(dataset, encoder_model, task, rep_config, device, write_to_file=False):
    '''
    Returns
    _______
        representations: torch.tensor() of size (dataset_size, embedding_size)
        task: string, 'monotonicity' or 'rel'

    '''

    dataloader = DataLoader(dataset, rep_config['batch_size'])

    # iterate encoding over batches
    # TODO: either remove or update the write_to_file functionality here

    # check whether the model is an NLI model (we will ignore its predictions if not)
    is_it_nli = encoder_is_nli_model(rep_config['encoder_model']) 

    if is_it_nli:
        model_predictions = []

    if write_to_file:
        embed_file = open(RESULTS_DIR+'embeddings.tsv', 'a+')
        embed_file.truncate(0) # ensure clear file before appending

    batch_rep_list = []
    encoder_model.eval()
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            # forward pass
            if rep_config['encoder_model'] in ['random_lookup_768', 'random_lookup_1024']:
                try:
                    random_size = rep_config['random_size']
                except ValueError:
                    print("Failed to specify random embedding size!")

                batch_list = torch.unbind(inputs['input_ids'].to('cpu'))
                reps_list = []
                for sequence in batch_list:
                    reps_list.append([lookup(encoder_model, token_id, random_size)\
                                      for token_id in sequence])

                hidden = (torch.tensor(reps_list).to(device),) # tuple of length one, as in one hidden layer
                target_reps = get_target_reps(hidden,
                                              inputs['X_range'],
                                              inputs['Y_range'], 
                                              rep_config, 
                                              device)

            else:
                outputs = encoder_model(inputs['input_ids'], inputs['attention_mask'])
                try:
                    hidden = outputs['hidden_states']
                except KeyError:
                    hidden = outputs['encoder_hidden_states']

                if is_it_nli:
                    logits = outputs['logits']
                    batch_predictions = torch.argmax(logits, dim=1)    
                    model_predictions.append(batch_predictions)

                target_reps = get_target_reps(hidden,
                                              inputs['X_range'],
                                              inputs['Y_range'], 
                                              rep_config, 
                                              device)

            if write_to_file:
                np.savetxt(embed_file,
                          target_reps.cpu().numpy(),
                          delimiter='\t')
            else:
                batch_rep_list.append(target_reps.cpu())


            #todo: write logits prediction to meta file

    if is_it_nli:
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

    return representations, labels

def get_target_reps(hidden, X_ranges, Y_ranges, rep_config, device):
    '''
    Args
    ____ 

    hidden: 
                tuple of length (#hidden layers), with each entry being a:
                torch.tensor of shape (batch_size, max_length, hidden_size)

    X_ranges: 
                list of length 2, each entry being a torch.tensor of shape
                (batch_size), containing the start and end value of each 
            
                

    config: nlixy.parse_config.Config object
    '''

    layers = get_focus_layers(hidden, rep_config['layer_range'], rep_config['layer_summary'])
    batch_size = layers.shape[0]

    if rep_config['X_or_Y'] != 'neither':
        if rep_config['X_or_Y'] in ['both','XY']:
            X_phrase_tokens = get_phrase_tokens(layers, X_ranges)
            Y_phrase_tokens = get_phrase_tokens(layers, Y_ranges)

        if rep_config['X_or_Y'] == 'X_only':
            X_phrase_tokens = get_phrase_tokens(layers, X_ranges)
            Y_phrase_tokens = None

        if rep_config['X_or_Y'] == 'Y_only':
            Y_phrase_tokens = get_phrase_tokens(layers, Y_ranges)
            X_phrase_tokens = None

        X = summarise_phrase(X_phrase_tokens, strategy=rep_config['phrase_summary'])
        Y = summarise_phrase(Y_phrase_tokens, strategy=rep_config['phrase_summary'])
        reps_list = [X,Y]
    else:
        reps_list = []


    if rep_config['include_cls']:
        CLS = get_phrase_tokens(layers, [torch.zeros(batch_size, dtype=torch.int64),
                                         torch.ones(batch_size, dtype=torch.int64)])
        CLS = summarise_phrase(CLS, strategy='mean')
        reps_list = [CLS] + reps_list


    reps_list = [item for item in reps_list if item is not None]
    target_reps = summarise_final(reps_list, strategy=rep_config['pair_summary']).to(device)
    assert target_reps.shape[0] == batch_size

    return target_reps


def get_focus_layers(hidden, layer_range='-1', layer_summary='single'):
    '''
    Args
    ____

    hidden:
                tuple of length # model depth (e.g. 25 for RoBERTa)

    layer_range:
                int or pair of of ints

    layer_summary:
                str, 
                either 'single', 'mean' or 'concat'.  

    '''
    #TODO: mulitlayer options

    return hidden[layer_range]



def get_phrase_tokens(layers, token_ranges):
    '''
    Args
    ____

    layers:
                size (batch_size, max_length, hidden_size)
                hidden size may be larger than originally, depending on
                the layer summary options (e.g, concatenating multiple layers)

    token_ranges:
                list of length 2, each entry being a torch.tensor of shape
                (batch_size), containing the start and end value of each 

    Returns
    _______

    phrase_token_reps:
                tuple of length (batch_size), each entry is a 
                torch.tensor of shape (phrase_tokens_length, hidden_size)
                 
                
    '''
    # non trivial!!

    stacked_ranges = torch.stack(token_ranges, dim=1)

    layers = torch.unbind(layers)
    phrase_token_reps = []

    for i, (start,end) in enumerate(stacked_ranges):
        phrase_token_reps.append(layers[i][start:end,:])


    return  phrase_token_reps

def summarise_phrase(token_reps_list, strategy='mean'):
    '''
    Args:
    ____ 
    
    token_reps_list: 
                    tuple of length (batch_size), each entry is a 
                    torch.tensor of shape (phrase_tokens_length, hidden_size)

    Returns:
    _______ 
        summary representation of shape (batch_size, hidden_size)

    '''
    if token_reps_list:
        if strategy == 'mean':
            # average across number of tokens 
            mean_tensor = lambda x: torch.mean(x, dim=0)
            summary_tensors = list(map(mean_tensor, token_reps_list))
            return torch.stack(summary_tensors)

        if strategy == 'first':
            return phrase_tokens[:,0,:]

        if strategy == 'concat':
            return torch.cat(token_reps_list, dim=1)

def summarise_final(rep_list, strategy='concat'):
    '''
    Args
        rep_list: list of torch.tensors, each of size (batch_size, embedding_size).
        Typically, some combination of [CLS], X and Y summaries

    '''

    if not rep_list:
        raise ValueError('Empty representation tensor!') 

    if strategy == 'concat':
        # concatenate along hidden_size dimension,
        # so that output size is (batch_size, 2*hidden_size)
        return torch.cat(rep_list, dim=1)

    if strategy == 'mean':
        pair_stacked = torch.stack(rep_list)
        return torch.mean(pair_stacked, dim=0)
        
