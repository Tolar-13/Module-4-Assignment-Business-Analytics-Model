# T-Mobile Retention Analytics: Churn Prediction Model

This project supports T-Mobile’s customer retention strategy by developing a machine learning model that predicts customer churn. The model enables early identification of at-risk customers, helping the business take proactive steps to improve retention and reduce revenue loss.

---

## Project Purpose

The goal is to build and evaluate a predictive model using historical customer data. This solution provides the analytical foundation for deploying churn alerts and retention workflows in production environments.

---

## Contents

| Filename                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `data_cleaning.ipynb`     | Cleans the original Telco dataset and prepares a model-ready CSV.           |
| `telco_model_ready.csv`   | Final cleaned dataset used for training.                                    |
| `Churn_model.ipynb`       | Main notebook for model training, testing, threshold tuning, SHAP analysis. |
| `model.pkl`               | Final trained Logistic Regression model, saved with `joblib`.              |
| `model_threshold.txt`     | Contains optimal decision threshold (0.39) for binary classification.       |
| `README.md`               | This documentation file.                                                    |

---

## Implementation Plan

1. **Data Cleaning**  
   - Handled in `data_cleaning.ipynb`: removed nulls, encoded variables, exported `telco_model_ready.csv`.

2. **Model Development**  
   - Logistic Regression with `class_weight='balanced'`  
   - Model trained on 80% of data, tested on 20%  
   - Threshold tuned to `0.39` for better recall  
   - Model saved with `joblib`

3. **Evaluation Summary**

   **Test Set (Threshold = 0.39):**
   - Accuracy: **0.69**
   - Precision: **0.46**
   - Recall: **0.89**
   - F1-Score: **0.60**
   - ROC AUC: **0.8377**

4. **Explainability**
   - SHAP analysis identifies key features influencing churn probability.
   - Visuals included in `Churn_model.ipynb`.

5. **Model Persistence**
   - Model saved to `model.pkl`
   - Classification threshold saved to `model_threshold.txt` for consistent deployment

---

## How to Run (Manual)

1. Open Jupyter Notebook or VS Code
2. Upload the following files:
   - `data_cleaning.ipynb`
   - `Churn_model.ipynb`
   - `telco_model_ready.csv`
3. Run `Churn_model.ipynb` step-by-step to:
   - Load the data
   - Train & test the model
   - Evaluate and explain performance
   - Save the model and threshold

---

## Requirements

Below are the required Python packages (tested on Python 3.11):

```text
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
shap==0.45.0
joblib==1.4.0
```

Install all dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib
```

---

## Deployment Note

The saved `model.pkl` and `model_threshold.txt` are ready for deployment in a Python-based API (Flask, FastAPI) or Streamlit app. SHAP visualizations can also be integrated into dashboards for interpretability.
