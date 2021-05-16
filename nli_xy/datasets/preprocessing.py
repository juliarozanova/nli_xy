import pandas as pd 
import torch
import pdb

# sentence pair -> context + x,y (tokens)

# context + insertions -> context + x, y (tokens)

# context + 

variable = ' x '

def instantiate(sent_tokens, np_tokens, variable='x'):
    '''
    Replace a variable/placeholder in a sentence (e.g. 'x') with the given noun phrase. 
    Inputs should already be tokenized. 
    '''
    # replace variable token with np instantiation
    try:
        # find indices of 'x'
        var_index = sent_tokens.index(variable)
    except ValueError:
        var_index = sent_tokens.index('Ġ'+variable)
        #else:
            #raise ValueError(f'Variable token {variable} not among sentence tokens!')


    full_sent_tokens = sent_tokens[:var_index] + np_tokens + sent_tokens[var_index+1:]
    inserted_end = var_index + len(np_tokens)
    inserted_start = var_index
    return full_sent_tokens, inserted_start, inserted_end 

def filter_insertions_by_grammar(context_row, insertions_df):
    if context_row.singular!='x':
        insertions_df = insertions_df.loc[insertions_df.x_grammar != 's']
        insertions_df = insertions_df.loc[insertions_df.y_grammar != 's']
    if context_row.mass!='x':
        insertions_df = insertions_df.loc[insertions_df.x_grammar != 'm']
        insertions_df = insertions_df.loc[insertions_df.y_grammar != 'm']
    if context_row.plural!='x':
        insertions_df = insertions_df.loc[insertions_df.x_grammar != 'p']
        insertions_df = insertions_df.loc[insertions_df.y_grammar != 'p']
    return insertions_df

def set_of_insertions_into_context(context_row, insertions_df, tokenizer):
    '''
    For one context, populate (x,y) with all insertion pairs in an insertion_df.

    Args
    ____ 
        context: str
        insertions_df: pd.DataFrame with columns 'x', 'y' and 'insertion_rel'. 
    '''

    raw_context = context_row.context
    context = tokenizer.tokenize(raw_context)

    insertions_df = filter_insertions_by_grammar(context_row, insertions_df)
    try:
        assert not insertions_df.empty
    except AssertionError:
        raise ValueError(f'''
                No grammatically valid insertions for this context! Check data labels for
                the context: \"{raw_context}\".''')

    token_inputs_df = pd.DataFrame({
            'sentence1_toks': [],
            'sentence2_toks': [],
            'raw_X_range':[],
            'raw_y_range':[],
            })

    meta_df = pd.DataFrame({
            'premise': [],
            'hypothesis':[],
            'insertion_pair':[],
            'X_grammar':[],
            'Y_grammar':[],
            'X_tokens':[],
            'Y_tokens':[],
            })

    # tokenize insertions
    meta_df['X_tokens'] = insertions_df['x'].apply(tokenizer.tokenize)
    meta_df['Y_tokens'] = insertions_df['y'].apply(tokenizer.tokenize)

    # populate data for model inputs
    token_inputs_df['sentence1_toks'] =  meta_df.apply(lambda row: 
            instantiate(context, row['X_tokens'])[0], axis=1)
    token_inputs_df['sentence2_toks'] =  meta_df.apply(lambda row: 
            instantiate(context, row['Y_tokens'])[0], axis=1)

    token_inputs_df['raw_X_range'] =  meta_df.apply(lambda row: 
            instantiate(context, row['X_tokens'])[1:], axis=1)
    token_inputs_df['raw_Y_range'] =  meta_df.apply(lambda row: 
            instantiate(context, row['Y_tokens'])[1:], axis=1)

    # populate meta data
    meta_df['premise'] = token_inputs_df['sentence1_toks'].apply(
                                tokenizer.convert_tokens_to_string)
    meta_df['hypothesis'] = token_inputs_df['sentence2_toks'].apply(
                                tokenizer.convert_tokens_to_string)
    meta_df['insertion_pair'] = insertions_df['x'] +', ' + insertions_df['y']
    meta_df['insertion_rel'] = insertions_df['insertion_rel']
    meta_df['X_grammar'] = insertions_df['x_grammar']
    meta_df['Y_grammar'] = insertions_df['y_grammar']
    meta_df['source'] = context_row.source

    return token_inputs_df, meta_df

def prepare_model_inputs(token_inputs_df,
        tokenizer, 
        max_length, 
        device, 
        context_option='all'):
    '''
    Convert tokens to ids, adding special tokens, padding, etc. 
    Returns:
    ________
        inputs_df: A pandas DataFrame with columns 'input_ids', 'attention_mask'

    '''

    inputs_and_ranges = token_inputs_df.apply(lambda row:
                                encode_from_tokens(row['sentence1_toks'],
                                                   row['sentence2_toks'],
                                                   row['raw_X_range'],
                                                   row['raw_Y_range'],
                                                   tokenizer,
                                                   max_length,
                                                   context_option,
                                                   device), axis=1)
    inputs_and_ranges_df = pd.DataFrame(inputs_and_ranges.to_list())

    inputs_df = inputs_and_ranges_df[['input_ids', 'attention_mask']]
    X_ranges = inputs_and_ranges_df['X_range']
    Y_ranges = inputs_and_ranges_df['Y_range']


    return inputs_df, X_ranges, Y_ranges

def encode_from_tokens(sentence1_tokens, 
                       sentence2_tokens, 
                       raw_X_range,
                       raw_Y_range,
                       tokenizer, 
                       max_length, 
                       context_option,
                       device):
    '''
    Returns
    _______
        (dict) with keys 'input_ids', 'attention_mask'. Appropriate input for transformer lm model.
    '''

    if context_option=='all':
        inputs = tokenizer.encode_plus(sentence1_tokens, 
                                       sentence2_tokens, 
                                       padding='max_length',
                                       truncation=True,
                                       max_length=max_length,
                                       return_special_tokens_mask=True)

        X_range, Y_range = recalculate_insertion_range(raw_X_range, 
                                                       raw_Y_range,
                                                       inputs['special_tokens_mask'],
                                                       context_option)
        inputs_and_ranges_dict = {
                'input_ids': torch.tensor(inputs['input_ids']).to(device),
                'attention_mask': torch.tensor(inputs['attention_mask']).to(device),
                'X_range': X_range,
                'Y_range': Y_range
                } 

    if context_option=='premise_only':
        inputs = tokenizer.encode_plus(sentence1_tokens,  
                                       padding='max_length',
                                       truncation=True,
                                       max_length=max_length,
                                       return_special_tokens_mask=True)

        X_range, Y_range = recalculate_insertion_range(raw_X_range, 
                                                       raw_Y_range,
                                                       inputs['special_tokens_mask'],
                                                        context_option)
        inputs_and_ranges_dict = {
                'input_ids': torch.tensor(inputs['input_ids']).to(device),
                'attention_mask': torch.tensor(inputs['attention_mask']).to(device),
                'X_range': X_range,
                'Y_range': Y_range
                } 

    return inputs_and_ranges_dict

def recalculate_insertion_range(raw_X_range, raw_Y_range, special_tokens_mask, context_option):
    count_first_specials, remaining_mask = count_until_encounter(0, special_tokens_mask)
    add_to_X_ranges = count_first_specials
    count_first_sentence_ids, remaining_mask = count_until_encounter(1,
                                                                    remaining_mask)
    if context_option =='all':
        count_second_specials, remaining_mask = count_until_encounter(0, remaining_mask)
        add_to_Y_ranges = sum([count_first_specials,
                                count_first_sentence_ids,
                                count_second_specials])
        Y_range = (raw_Y_range[0]+add_to_Y_ranges, raw_Y_range[1]+add_to_Y_ranges)

    elif context_option == 'premise_only':
        Y_range = (0,0)

    X_range = (raw_X_range[0]+add_to_X_ranges, raw_X_range[1]+add_to_X_ranges)


    return X_range, Y_range

def count_until_encounter(value, value_list):
    i = value_list.pop(0)
    cumulative=0

    if i is value:
        return cumulative, value_list
    elif i is not value:
        cumulative+=1

    while value_list[0] is not value:
        cumulative += 1
        i = value_list.pop(0)

    return cumulative, value_list
        

def expand_with_opposite_relations(insertions_df):
    reverse_insertions_df = pd.DataFrame()
    reverse_insertions_df['x'] = insertions_df['y']
    reverse_insertions_df['x_grammar'] = insertions_df['y_grammar']
    reverse_insertions_df['y'] = insertions_df['x']
    reverse_insertions_df['y_grammar'] = insertions_df['x_grammar']
    reverse_insertions_df['source'] = insertions_df['source']
    reverse_insertions_df['insertion_rel'] = insertions_df['insertion_rel'].apply(reverse_labels)

    return pd.concat([insertions_df, reverse_insertions_df]).drop_duplicates()

def reverse_labels(label_string):
    if label_string == 'leq':
        return 'geq'
    if label_string == 'eq':
        return 'eq'
    if label_string == 'none':
        return 'none'
    if label_string == 'geq':
        return 'leq'
    else:
        print(label_string)
        raise ValueError()

def gold_labeller(meta_df_row):
    insertion_rel = meta_df_row['insertion_rel']
    monotonicity = meta_df_row['context_monotonicity']

    if insertion_rel=='leq' and monotonicity in ['up', 'upward_monotone']:
        return 'entailment'
    elif insertion_rel=='geq' and monotonicity in ['down', 'downward_monotone']:
        return 'entailment'
    else:
        return 'non-entailment'