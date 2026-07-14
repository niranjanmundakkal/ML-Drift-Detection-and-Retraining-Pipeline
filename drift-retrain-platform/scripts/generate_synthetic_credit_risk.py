import os
import numpy as np
import pandas as pd

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_credit_risk_data(
    n_rows,
    drift_level=0.0,
    start_id=1
):
    """
    Generates a realistic synthetic Credit Risk dataset.

    drift_level:
        0.0 -> Reference
        0.2 -> Mild Drift
        0.5 -> Medium Drift
        0.9 -> Heavy Drift
    """

    applicant_id = [
        f"A{start_id+i:06d}"
        for i in range(n_rows)
    ]

    age = np.random.randint(21, 65, n_rows)

    annual_income = np.random.normal(
        loc=750000 - drift_level * 180000,
        scale=120000,
        size=n_rows
    )

    annual_income = np.clip(
        annual_income,
        150000,
        2500000
    )

    employment_status = np.random.choice(
        [
            "Salaried",
            "Self-employed",
            "Business",
            "Unemployed"
        ],
        size=n_rows,
        p=[
            0.55 - drift_level * 0.10,
            0.20,
            0.15,
            0.10 + drift_level * 0.10
        ]
    )

    years_employed = np.random.randint(
        0,
        35,
        n_rows
    )

    credit_score = np.random.normal(
        loc=730 - drift_level * 80,
        scale=45,
        size=n_rows
    )

    credit_score = np.clip(
        credit_score,
        300,
        900
    )

    loan_amount = np.random.normal(
        loc=250000 + drift_level * 180000,
        scale=90000,
        size=n_rows
    )

    loan_amount = np.clip(
        loan_amount,
        50000,
        1200000
    )

    loan_term = np.random.choice(
        [12,24,36,48,60],
        size=n_rows,
        p=[0.1,0.2,0.3,0.2,0.2]
    )

    interest_rate = np.random.normal(
        loc=9 + drift_level * 5,
        scale=1.5,
        size=n_rows
    )

    interest_rate = np.clip(
        interest_rate,
        5,
        22
    )

    debt_to_income_ratio = np.random.normal(
        loc=28 + drift_level * 12,
        scale=8,
        size=n_rows
    )

    debt_to_income_ratio = np.clip(
        debt_to_income_ratio,
        5,
        80
    )

    existing_loans = np.random.poisson(
        lam=1.5 + drift_level,
        size=n_rows
    )

    payment_history = np.random.choice(
        [
            "Excellent",
            "Good",
            "Average",
            "Poor"
        ],
        size=n_rows,
        p=[
            0.30-drift_level*0.10,
            0.40-drift_level*0.10,
            0.20+drift_level*0.10,
            0.10+drift_level*0.10
        ]
    )

    home_ownership = np.random.choice(
        [
            "Own",
            "Mortgage",
            "Rent"
        ],
        size=n_rows,
        p=[
            0.30,
            0.40,
            0.30
        ]
    )

    marital_status = np.random.choice(
        [
            "Single",
            "Married",
            "Divorced"
        ],
        size=n_rows,
        p=[
            0.35,
            0.55,
            0.10
        ]
    )

    loan_purpose = np.random.choice(
        [
            "Home",
            "Car",
            "Education",
            "Personal",
            "Business"
        ],
        size=n_rows
    )

    region = np.random.choice(
        [
            "North",
            "South",
            "East",
            "West"
        ],
        size=n_rows
    )

    default_score = (
        1.5 * (credit_score < 600).astype(int)
        + 1.2 * (debt_to_income_ratio > 45).astype(int)
        + 1.0 * (interest_rate > 13).astype(int)
        + 1.0 * (loan_amount > 500000).astype(int)
        + 1.2 * (payment_history == "Poor").astype(int)
        + 0.8 * (employment_status == "Unemployed").astype(int)
        + 0.6 * (existing_loans >= 3).astype(int)
        - 0.8 * (annual_income > 1000000).astype(int)
        - 0.5 * (credit_score > 750).astype(int)
        - 0.3 * (payment_history == "Excellent").astype(int)
        + drift_level * 0.8
    )

    default_probability = sigmoid(default_score - 2.3)

    default = np.where(
        np.random.rand(n_rows) < default_probability,
        1,
        0
    )


    df = pd.DataFrame({

        "applicant_id": applicant_id,

        "age": age,

        "annual_income": np.round(annual_income, 2),

        "employment_status": employment_status,

        "years_employed": years_employed,

        "credit_score": np.round(credit_score),

        "loan_amount": np.round(loan_amount, 2),

        "loan_term": loan_term,

        "interest_rate": np.round(interest_rate, 2),

        "debt_to_income_ratio": np.round(
            debt_to_income_ratio,
            2
        ),

        "existing_loans": existing_loans,

        "payment_history": payment_history,

        "home_ownership": home_ownership,

        "marital_status": marital_status,

        "loan_purpose": loan_purpose,

        "region": region,

        "default": default

    })

    return df
def introduce_missingness(df, frac=0.02):
    """
    Introduce small amounts of missing values
    to simulate real-world datasets.
    """

    df = df.copy()

    for col in [
        "annual_income",
        "credit_score",
        "loan_amount",
        "employment_status"
    ]:
        mask = np.random.rand(len(df)) < frac
        df.loc[mask, col] = np.nan

    return df


def main():

    os.makedirs("data/reference", exist_ok=True)
    os.makedirs("data/incoming", exist_ok=True)

    # -----------------------------
    # Reference Dataset
    # -----------------------------
    reference_df = generate_credit_risk_data(
        n_rows=3000,
        drift_level=0.0,
        start_id=1
    )

    reference_df = introduce_missingness(
        reference_df,
        frac=0.02
    )

    reference_df.to_csv(
        "data/reference/reference.csv",
        index=False
    )

    # -----------------------------
    # Batch 1 (Very Low Drift)
    # -----------------------------
    batch_001 = generate_credit_risk_data(
        n_rows=700,
        drift_level=0.10,
        start_id=10001
    )

    batch_001 = introduce_missingness(batch_001,0.02)

    batch_001.to_csv(
        "data/incoming/batch_001.csv",
        index=False
    )

    # -----------------------------
    # Batch 2 (Moderate Drift)
    # -----------------------------
    batch_002 = generate_credit_risk_data(
        n_rows=700,
        drift_level=0.30,
        start_id=11001
    )

    batch_002 = introduce_missingness(batch_002,0.03)

    batch_002.to_csv(
        "data/incoming/batch_002.csv",
        index=False
    )

    # -----------------------------
    # Batch 3 (High Drift)
    # -----------------------------
    batch_003 = generate_credit_risk_data(
        n_rows=700,
        drift_level=0.60,
        start_id=12001
    )

    batch_003 = introduce_missingness(batch_003,0.04)

    batch_003.to_csv(
        "data/incoming/batch_003.csv",
        index=False
    )

    # -----------------------------
    # Batch 4 (Extreme Drift)
    # -----------------------------
    batch_004 = generate_credit_risk_data(
        n_rows=700,
        drift_level=0.90,
        start_id=13001
    )

    batch_004 = introduce_missingness(batch_004,0.05)

    batch_004.to_csv(
        "data/incoming/batch_004.csv",
        index=False
    )

    print("=" * 60)
    print("Synthetic Credit Risk Dataset Generated Successfully")
    print("=" * 60)
    print("Reference Dataset : data/reference/reference.csv")
    print("Incoming Batches  : data/incoming/")
    print("=" * 60)


if __name__ == "__main__":
    main()
    