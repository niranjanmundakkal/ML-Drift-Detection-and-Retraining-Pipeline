import json
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_all_configs(config_dir="config"):
    config_dir = Path(config_dir)

    schema = load_json(config_dir / "schema.json")
    pipeline_config = load_json(config_dir / "pipeline_config.json")
    model_config = load_json(config_dir / "model_config.json")

    return schema, pipeline_config, model_config