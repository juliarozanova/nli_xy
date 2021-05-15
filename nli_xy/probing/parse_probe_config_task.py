from prefect import task
import json

@task
def parse_probe_config(PROBE_CONFIG_FILE):

    with open(PROBE_CONFIG_FILE, 'r') as file:
        loaded_config = json.loads(file.read())

    return loaded_config