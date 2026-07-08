from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs
from core.metadata_builder import build_metadata

schema, pipeline_config, model_config = load_all_configs()
metadata = build_metadata(schema=schema, pipeline_config=pipeline_config, model_config=model_config)

print("SCHEMA:")
print(json.dumps(schema, indent=2))

print("\nPIPELINE CONFIG:")
print(json.dumps(pipeline_config, indent=2))

print("\nMODEL CONFIG:")
print(json.dumps(model_config, indent=2))

print("\nMETADATA:")
print(json.dumps(metadata, indent=2))