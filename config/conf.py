import yaml

# TODO: pydantic model for configs
with open("/global/homes/t/tomerh/gw_torch/config/config.yaml", "r") as f:
    conf = yaml.safe_load(f)
print("blablabla")