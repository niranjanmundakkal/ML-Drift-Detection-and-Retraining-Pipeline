import pandas as pd
from scipy.stats import chi2_contingency


def categorical_drift(reference_column, batch_column):

    reference_counts = (
        pd.Series(reference_column)
        .value_counts()
    )

    batch_counts = (
        pd.Series(batch_column)
        .value_counts()
    )

    categories = reference_counts.index.union(
        batch_counts.index
    )

    reference_counts = (
        reference_counts
        .reindex(categories, fill_value=0)
    )

    batch_counts = (
        batch_counts
        .reindex(categories, fill_value=0)
    )

    contingency = pd.DataFrame({
        "reference": reference_counts,
        "batch": batch_counts
    })

    statistic, p_value, _, _ = chi2_contingency(contingency)

    return statistic, p_value