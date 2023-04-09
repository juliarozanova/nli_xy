from prefect import task
from transformers import AutoTokenizer

def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], use_fast=False)
    return tokenizer