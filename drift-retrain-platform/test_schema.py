import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs
from core.metadata_builder import build_metadata
from core.schema_validator import validate_reference_schema, validate_batch_schema

schema, pipeline_config, model_config = load_all_configs()
metadata = build_metadata(schema, pipeline_config, model_config)

reference_df = pd.read_csv(metadata["reference_data_path"])
validate_reference_schema(reference_df, schema)
print("Reference schema validation passed.")

batch_df = pd.read_csv(metadata["incoming_data_dir"] + "batch_1.csv")
validate_batch_schema(batch_df, schema)
print("Batch schema validation passed.")
