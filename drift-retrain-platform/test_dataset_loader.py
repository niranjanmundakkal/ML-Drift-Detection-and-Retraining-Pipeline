from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs
from core.metadata_builder import build_metadata
from ingestion.dataset_loader import load_reference_data, load_batch_data

schema, pipeline_config, model_config = load_all_configs()
metadata = build_metadata(schema, pipeline_config, model_config)

reference_df = load_reference_data(metadata, schema)
print("Reference data loaded successfully.")
print(reference_df.head())

batch_df = load_batch_data("data/incoming/batch_1.csv", metadata, schema)
print("\nBatch data loaded successfully.")
print(batch_df.head())
