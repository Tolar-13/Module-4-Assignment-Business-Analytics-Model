# T-Mobile Retention Model

This project presents a churn prediction model built for T-Mobile using a cleaned and pre-processed customer dataset. The goal is to accurately identify customers likely to churn, allowing T-Mobile to proactively implement retention strategies. The project follows a structured machine learning workflow, including data preparation, model development, explainability, and saving of artifacts for deployment.

---

## Project Structure

```text
.
├── data_cleaning.ipynb       # Notebook used in Module 3 to clean and prepare the dataset
├── Churn_model.ipynb         # Main model development, training, evaluation and SHAP explainability
├── final_model.pkl           # Trained logistic regression model saved for deployment
├── scaler_final.pkl          # Fitted StandardScaler for preprocessing numerical inputs during inference
├── shap_explainer.pkl        # Saved SHAP Explainer object for interpreting model predictions
├── threshold.txt             # Stored threshold value (0.39) used for decision boundary adjustment
├── README.md                 # This file
```

---

## How to Run the Project (Manual Steps)

1. Clone or download this repository manually.
2. Launch Jupyter Notebook and open `Churn_model.ipynb`.
3. Run all cells sequentially to:
   - Load the model-ready dataset
   - Train and evaluate the logistic regression model
   - Visualize performance metrics
   - Generate and visualise SHAP values
   - Save final model and artifacts

---

## Model Evaluation

The final model was trained using logistic regression with a threshold of 0.39.

**Test Set Evaluation (Threshold = 0.39):**

```text
Accuracy: 0.69
Precision: 0.46
Recall: 0.89
ROC AUC: 0.8377
```

This balance prioritises identifying churners while controlling false positives, aligning with T-Mobile’s business objective of proactive customer retention.

---

## Explainability

SHAP (SHapley Additive exPlanations) was used to interpret both global and local feature importance, helping stakeholders understand key drivers behind churn predictions. The saved `shap_explainer.pkl` can be loaded during deployment for real-time explanation.

---

## Requirements

```bash
scikit-learn==1.3.0
pandas==1.5.3
numpy==1.23.5
matplotlib==3.7.1
seaborn==0.12.2
shap==0.44.0
joblib==1.3.2
```
Install them using:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn shap joblib
```
