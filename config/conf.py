import yaml

# TODO: pydantic model for configs
with open("./config/config.yaml", "r") as f:
    conf = yaml.safe_load(f)