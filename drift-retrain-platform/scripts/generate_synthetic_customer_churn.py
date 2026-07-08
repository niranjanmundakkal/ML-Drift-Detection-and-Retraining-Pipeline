import os
import numpy as np
import pandas as pd

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_customer_churn_data(n_rows, drift_level=0.0, start_id=1):
    customer_id = [f"C{start_id + i:03d}" for i in range(n_rows)]

    gender = np.random.choice(["Male", "Female"], size=n_rows, p=[0.5, 0.5])
    senior_citizen = np.random.choice(["Yes", "No"], size=n_rows, p=[0.15 + drift_level * 0.1, 0.85 - drift_level * 0.1])
    partner = np.random.choice(["Yes", "No"], size=n_rows, p=[0.45, 0.55])
    dependents = np.random.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7])

    tenure = np.random.randint(max(1, int(1 + drift_level * 3)), max(2, int(72 - drift_level * 10)), size=n_rows)

    contract_probs = {
        0.0: [0.55, 0.25, 0.20],
        0.3: [0.65, 0.22, 0.13],
        0.6: [0.75, 0.18, 0.07],
        1.0: [0.82, 0.13, 0.05],
    }

    if drift_level <= 0.15:
        probs = contract_probs[0.0]
    elif drift_level <= 0.45:
        probs = contract_probs[0.3]
    elif drift_level <= 0.8:
        probs = contract_probs[0.6]
    else:
        probs = contract_probs[1.0]

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=probs)

    region = np.random.choice(["North", "South", "East", "West"], size=n_rows, p=[0.25, 0.25, 0.25, 0.25])
    internet_service = np.random.choice(["DSL", "Fiber", "No"], size=n_rows, p=[0.4, 0.35, 0.25])
    tech_support = np.random.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7])
    payment_method = np.random.choice(["Electronic check", "Bank transfer", "Credit card", "Mailed check"], size=n_rows, p=[0.35, 0.25, 0.2, 0.2])

    monthly_charges = np.random.normal(loc=65 + drift_level * 18, scale=12 + drift_level * 3, size=n_rows)
    monthly_charges = np.clip(monthly_charges, 20, 150)

    total_charges = monthly_charges * tenure + np.random.normal(0, 100, size=n_rows)
    total_charges = np.clip(total_charges, 0, None)

    age = np.random.randint(18, 75, size=n_rows)
    avg_monthly_usage_gb = np.random.randint(0, 200, size=n_rows)
    support_calls_last_6m = np.random.randint(0, 10, size=n_rows)
    payment_delay_days = np.random.randint(0, 30, size=n_rows)

    churn_score = (
        0.9 * (contract == "Month-to-month").astype(int)
        + 0.5 * (senior_citizen == "Yes").astype(int)
        + 0.4 * (partner == "No").astype(int)
        + 0.003 * monthly_charges
        - 0.02 * tenure
        + drift_level * 0.8
    )

    churn_prob = sigmoid(churn_score - 1.2)
    churn = np.where(np.random.rand(n_rows) < churn_prob, 1, 0)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "region": region,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "contract": contract,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": np.round(total_charges, 2),
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "support_calls_last_6m": support_calls_last_6m,
        "payment_delay_days": payment_delay_days,
        "churn": churn,
    })
    return df


def introduce_missingness(df, frac=0.02):
    df = df.copy()
    for col in ["monthly_charges", "total_charges", "partner"]:
        mask = np.random.rand(len(df)) < frac
        df.loc[mask, col] = np.nan
    return df


def main():
    os.makedirs("data/reference", exist_ok=True)
    os.makedirs("data/incoming", exist_ok=True)

    reference_df = generate_customer_churn_data(n_rows=3000, drift_level=0.0, start_id=1)
    reference_df = introduce_missingness(reference_df, frac=0.02)
    reference_df.to_csv("data/reference/reference.csv", index=False)

    batch_001 = generate_customer_churn_data(n_rows=700, drift_level=0.05, start_id=10001)
    batch_001 = introduce_missingness(batch_001, frac=0.02)
    batch_001.to_csv("data/incoming/batch_001.csv", index=False)

    batch_002 = generate_customer_churn_data(n_rows=700, drift_level=0.25, start_id=11001)
    batch_002 = introduce_missingness(batch_002, frac=0.03)
    batch_002.to_csv("data/incoming/batch_002.csv", index=False)

    batch_003 = generate_customer_churn_data(n_rows=700, drift_level=0.55, start_id=12001)
    batch_003 = introduce_missingness(batch_003, frac=0.04)
    batch_003.to_csv("data/incoming/batch_003.csv", index=False)

    batch_004 = generate_customer_churn_data(n_rows=700, drift_level=0.90, start_id=13001)
    batch_004 = introduce_missingness(batch_004, frac=0.05)
    batch_004.to_csv("data/incoming/batch_004.csv", index=False)

    print("Synthetic customer churn datasets generated successfully.")


if __name__ == "__main__":
    main()
