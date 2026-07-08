import pandas as pd

from src.preprocessing.feature_alignment import align_features

training_feature_names = [
    "tenure",
    "monthly_charges",
    "gender_Female",
    "gender_Male"
]

X_batch = pd.DataFrame({
    "gender_Male": [1, 0],
    "tenure": [12, 24],
    "monthly_charges": [50.0, 80.0]
})

aligned = align_features(X_batch, training_feature_names)

print(aligned)
print(aligned.columns.tolist())
