"""Dynamic preprocessing utilities for schema-driven ML pipelines."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DynamicPreprocessor:
    def __init__(self, metadata, pipeline_config):
        self.metadata = metadata
        self.pipeline_config = pipeline_config

        self.target_column = metadata.get("target_column")
        self.id_column = metadata.get("id_column")

        self.feature_columns = metadata.get("feature_columns", [])
        self.numerical_columns = metadata.get("numerical_columns", [])
        self.categorical_columns = metadata.get("categorical_columns", [])

        self.preprocessor = None
        self.feature_names_ = None

    def _build_preprocessor(self):
        preprocessing_cfg = self.pipeline_config.get("preprocessing", {})

        num_missing_strategy = (
            preprocessing_cfg.get("missing_value_strategy", {}).get("numerical", "median")
        )
        cat_missing_strategy = (
            preprocessing_cfg.get("missing_value_strategy", {}).get("categorical", "most_frequent")
        )
        if cat_missing_strategy == "mode":
            cat_missing_strategy = "most_frequent"
        categorical_encoding = (
            preprocessing_cfg.get("encoding", {}).get("categorical", "onehot")
        )
        numerical_scaling = (
            preprocessing_cfg.get("scaling", {}).get("numerical", "standard")
        )

        numerical_steps = [
            ("imputer", SimpleImputer(strategy=num_missing_strategy))
        ]

        if numerical_scaling == "standard":
            numerical_steps.append(("scaler", StandardScaler()))

        numerical_pipeline = Pipeline(steps=numerical_steps)

        categorical_steps = [
            ("imputer", SimpleImputer(strategy=cat_missing_strategy))
        ]

        if categorical_encoding == "onehot":
            encoder_kwargs = {"handle_unknown": "ignore"}
            try:
                encoder = OneHotEncoder(**encoder_kwargs, sparse_output=False)
            except TypeError:
                encoder = OneHotEncoder(**encoder_kwargs, sparse=False)
            categorical_steps.append(("encoder", encoder))

        categorical_pipeline = Pipeline(steps=categorical_steps)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, self.numerical_columns),
                ("cat", categorical_pipeline, self.categorical_columns),
            ],
            remainder="drop"
        )

    def _prepare_feature_frame(self, df):
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns during preprocessing: {missing_features}")

        X = df[self.feature_columns].copy()
        return X

    def fit(self, df):
        X = self._prepare_feature_frame(df)

        self._build_preprocessor()
        self.preprocessor.fit(X)

        self.feature_names_ = self._get_transformed_feature_names()
        return self

    def transform(self, df, include_target=True):
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")

        X_raw = self._prepare_feature_frame(df)
        X_transformed = self.preprocessor.transform(X_raw)

        X_transformed = pd.DataFrame(
            X_transformed,
            columns=self.feature_names_,
            index=df.index
        )

        y = None
        if include_target and self.target_column in df.columns:
            y = df[self.target_column].copy()

        return X_transformed, y

    def fit_transform(self, df, include_target=True):
        self.fit(df)
        return self.transform(df, include_target=include_target)

    def _get_transformed_feature_names(self):
        feature_names = []

        if self.numerical_columns:
            feature_names.extend(self.numerical_columns)

        if self.categorical_columns:
            cat_pipeline = self.preprocessor.named_transformers_["cat"]
            encoder = cat_pipeline.named_steps.get("encoder")

            if encoder is not None:
                cat_feature_names = encoder.get_feature_names_out(self.categorical_columns)
                feature_names.extend(cat_feature_names.tolist())
            else:
                feature_names.extend(self.categorical_columns)

        return feature_names

    def get_feature_names(self):
        if self.feature_names_ is None:
            raise ValueError("Feature names are not available before fitting the preprocessor.")
        return self.feature_names_
