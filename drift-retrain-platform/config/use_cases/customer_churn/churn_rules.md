# Customer churn generation rules

## Feature relationships
- `total_charges` is derived from `monthly_charges * tenure + small noise`.
- `monthly_charges` depends on contract type.
- `support_calls_last_6m` is higher for customers with higher churn risk.
- Customers with tech support generally have lower churn risk.
- Customers with payment delays are more likely to churn.

## Churn probability heuristics
- Add risk for Month-to-month contracts.
- Add risk for high support calls.
- Add risk for payment delays above 20 days.
- Add risk for short tenure under 12 months.
- Add risk for customers without tech support.
- Reduce risk for two-year contracts.
- Reduce risk for long tenure above 36 months.
- Reduce risk for autopay customers.

## Drift scenarios
- Batch 4 increases monthly charges mean.
- Batch 5 shifts contract distribution toward month-to-month.
- Batch 6 increases support calls intensity.
- Batch 8 introduces stronger drift to trigger retraining.
