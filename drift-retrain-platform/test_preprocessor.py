from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs
from core.metadata_builder import build_metadata
from ingestion.dataset_loader import load_reference_data
from preprocessing.preprocessor import DynamicPreprocessor

schema, pipeline_config, model_config = load_all_configs()
metadata = build_metadata(schema, pipeline_config, model_config)

reference_df = load_reference_data(metadata, schema)

preprocessor = DynamicPreprocessor(metadata, pipeline_config)
X_ref, y_ref = preprocessor.fit_transform(reference_df, include_target=True)

print("Preprocessing successful.")
print("X_ref shape:", X_ref.shape)
print("y_ref shape:", y_ref.shape if y_ref is not None else None)
print("\nFeature names:")
print(preprocessor.get_feature_names())
print("\nTransformed data sample:")
print(X_ref.head())
