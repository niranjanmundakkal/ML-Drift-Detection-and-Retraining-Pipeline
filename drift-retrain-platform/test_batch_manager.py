from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs
from core.metadata_builder import build_metadata
from ingestion.batch_manager import get_sorted_batches

schema, pipeline_config, model_config = load_all_configs()
metadata = build_metadata(schema, pipeline_config, model_config)

batch_files = get_sorted_batches(metadata, pipeline_config)

print("Batch files found:")
for file in batch_files:
    print(file)
