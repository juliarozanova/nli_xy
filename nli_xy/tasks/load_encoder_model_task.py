from prefect import task
from transformers import AutoModelForSequenceClassification 

@task
def load_encoder_model(config):
    encoder_model = AutoModelForSequenceClassification.from_pretrained(config['encoder_model'],
                                                                output_hidden_states=True,
                                                                return_dict=True)
    encoder_model.to(config['device'])
    return encoder_model
