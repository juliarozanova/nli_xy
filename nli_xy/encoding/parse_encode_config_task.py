from prefect import task
import json

@task
def parse_encode_config(ENCODE_CONFIG_FILE: str):


    with open(ENCODE_CONFIG_FILE, 'r') as file:
        loaded_config = json.loads(file.read())

    shared_config = loaded_config["shared_config"] 

    parsed_config = {"shared_config": shared_config,
                    "representations": {}}

    for rep_name in list(loaded_config["representations"].keys()):
        rep_config = loaded_config["representations"][rep_name]
        full_rep_config =  shared_config
        full_rep_config.update(rep_config)
        parsed_config["representations"][rep_name] = verify_types(full_rep_config.copy())

    return parsed_config


def verify_types(config: dict):
    try:
        if config['include_cls'] == 'True':
            config['include_cls'] = True
        else:
            config['include_cls'] = False
    except KeyError:
        pass

    return config

