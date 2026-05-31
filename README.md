# Credit Risk Probability Model – Bati Bank

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence the need for an interpretable and well-documented model?

Basel II (Pillar 2 – Supervisory Review, and Pillar 3 – Market Discipline) requires banks to validate and document internal risk models. The model must be interpretable so that regulators and auditors can understand why a particular applicant receives a certain risk score. A black‑box model would fail compliance.  
Thus, we must:
- Use explainable features (e.g., Weight of Evidence transformations).
- Document every modeling choice: proxy variable definition, clustering logic, feature engineering steps.
- Provide ongoing performance monitoring and back‑testing.

### 2. Why is a proxy variable necessary without a direct default label? What business risks does proxy-based prediction introduce?

The raw dataset contains only transaction behavior – no historical loan defaults. A proxy target (e.g., “disengaged customer” = high risk) is needed to approximate default risk.

Risks introduced:
- Proxy bias – A customer may be disengaged from the e‑commerce platform but still pay a loan reliably.
- Concept drift – Behavioral patterns change over time; today’s “disengaged” may not mean future default.
- Adverse selection – The model might reject thrifty but creditworthy customers.

Mitigations: Monitor proxy performance against real loan outcomes once the service launches, and periodically re‑calibrate the proxy definition.

### 3. Key trade-offs between a simple interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context.

| Aspect                 | Logistic Regression (with WoE)          | Gradient Boosting (XGBoost)               |
|------------------------|------------------------------------------|--------------------------------------------|
| Interpretability   | High – coefficients show direct impact   | Low – complex interactions, black‑box     |
| Regulatory acceptance | Preferred (scorecard format)         | Requires additional explainability (SHAP, LIME) |
| Predictive power   | Moderate                                 | High (captures non‑linear relationships)  |
| Maintenance        | Easy to retrain and adjust               | More complex, longer retraining cycles    |

Practical recommendation:  
Use an ensemble approach – train an XGBoost for higher accuracy, then map its probability output to a transparent scorecard via logistic regression on WoE‑binned features. This balances regulatory needs with performance.
