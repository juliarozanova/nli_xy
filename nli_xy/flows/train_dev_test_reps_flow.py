from nli_xy.tasks import load_tokenizer, parse_config, build_split_datasets, load_encoder_model, get_reps
import pandas as pd
import torch
DATA_DIR = './data/nlixy_small/'
CONF_FILE = './configs/compare_layers_conf.json'

config = parse_config.run(CONF_FILE)
config['device'] = 'cuda'
tokenizer = load_tokenizer.run(config)
encoder_model = load_encoder_model.run(config)

datasets = build_split_datasets.run(DATA_DIR, config, tokenizer)
rep_batches, meta_batches = get_reps.run(datasets['train'], encoder_model, config)

batches = zip(rep_batches, meta_batches)
flat_batches = []

# test on the first few batches!
for batch_num, (rep_batch, meta_batch) in enumerate(batches):
    for layer_num, layer in enumerate(rep_batch):
        for token_num, token_reps in enumerate(layer.unbind(dim=1)):
            meta_out = pd.DataFrame({
                    'batch_num': [batch_num]*token_reps.shape[0],
                    'layer_num': [layer_num]*token_reps.shape[0],
                    'token_num': [token_num]*token_reps.shape[0],
                    'X_range': torch.stack(meta_batch['X_range'], dim=1).tolist(),
                    'Y_range': torch.stack(meta_batch['Y_range'], dim=1).tolist()
                    })
            reps_out = pd.DataFrame(token_reps.numpy())
            flat_batches.append(pd.concat([meta_out, reps_out]))

