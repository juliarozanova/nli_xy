from prefect import task
from transformers import AutoModelForSequenceClassification 

@task
def load_encoder_model(encode_config):
    encoder_model = AutoModelForSequenceClassification.from_pretrained(encode_config['encoder_model'],
                                                                output_hidden_states=True,
                                                                return_dict=True)
    encoder_model.to(encode_config['device'])
    encoder_model.eval()
    return encoder_model
