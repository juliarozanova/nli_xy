from prefect import task
import json

@task
def parse_config(CONF_FILE):
    with open(CONF_FILE, 'r') as file:
        config = json.loads(file.read())
    return config
