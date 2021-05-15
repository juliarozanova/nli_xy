from prefect import task

@task
def parse_probe_config(PROBE_CONFIG_FILE):

    with open(PROBE_CONFIG_FILE, 'r') as file:
        loaded_config = json.loads(file.read())

    return loaded_config