from scipy.stats import ks_2samp


def numerical_drift(reference_column, batch_column):

    statistic, p_value = ks_2samp(
        reference_column,
        batch_column
    )

    return statistic, p_value