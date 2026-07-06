from src.core.config_loader import load_all_configs

schema, pipeline_config, model_config = load_all_configs()

print(schema)
print(pipeline_config)
print(model_config)