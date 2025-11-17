# Agentic AI System for Breast Cancer Detection (EDA · ML · SHAP · Email)

This repository implements an **agentic AI workflow** for breast cancer diagnosis using the **Breast Cancer Wisconsin Diagnostic** dataset (via `sklearn.datasets.load_breast_cancer`). The system combines:

- **EDA & Feature Analysis**
- **Model benchmarking & selection**
- **Explainable AI (SHAP)**
- **Synthetic “Test Lab” for virtual patients**
- **Simulated Email Agent** that drafts follow-up summaries when predicted risk is malignant

The project is designed to support both **hands-on experimentation** and **reproducible, agent-like orchestration** for a graduate-level AI/ML course.

---

## 1. Project Goals

- Build a reproducible classification pipeline (benign vs malignant).
- Compare multiple ML models (e.g., Logistic Regression, Random Forest, SVM, XGBoost).
- Provide **global** and **local** explainability via SHAP.
- Simulate an **agentic workflow** with specialized modules:
  - EDA / Data Agent
  - Modeling / Tuning Agent
  - Explainability Agent
  - Notification / Email Agent
- Offer a **synthetic patient test lab** with triage-style interaction.

---

## 2. High-Level Architecture

### Agents / Modules

- **Data & EDA Agent**
  - Loads `sklearn` breast cancer dataset
  - Splits train/test
  - Performs summary statistics, distributions, correlations, and boxplots

- **Modeling Agent**
  - Trains several candidate models (e.g., LR, RF, SVM, XGBoost)
  - Uses cross-validation and metrics such as:
    - Accuracy
    - F1-score
    - ROC AUC
  - Selects a “champion” model

- **Explainability Agent (SHAP)**
  - Wraps the chosen model with a SHAP explainer
  - Produces global feature importance
  - Generates local explanations for individual patients

- **Synthetic Test Lab & Triage Agent**
  - Generates synthetic patients by sampling from the original feature distribution
  - Lets the user pick a `patient_id` to:
    - Get a benign/malignant prediction
    - View SHAP explanations

- **Email / Notification Agent (Simulated)**
  - After each prediction, composes a **simulated email** summary:
    - Patient ID
    - Model prediction (benign vs malignant)
    - Top contributing features (from SHAP)
    - If malignant, appends an extra call to action:
      > “Follow-up needed – please schedule further diagnostic review.”

> Note: In this repository, the email agent is configured as a **simulation** (safe by default). To actually send emails, you will need to add credentials (SMTP, API keys) in a local config file and **never commit secrets to GitHub**.

---

## 3. Repository Structure (Suggested)

You can adapt as needed, but a clean layout might look like:

```text
Breast-Cancer-Detection-EDA-ML-Agentic-SHAP-email/
├─ src/
│  ├─ data_agent.py
│  ├─ modeling_agent.py
│  ├─ explainability_agent.py
│  ├─ email_agent.py
│  └─ synthetic_test_lab.py
├─ notebooks/
│  └─ 01_agentic_breast_cancer_pipeline.ipynb
├─ config/
│  └─ email_config_example.json
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ CONTRIBUTING.md
