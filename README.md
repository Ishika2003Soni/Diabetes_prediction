# 🩺 Diabetes Risk Prediction Suite

This Streamlit-based application is designed to predict the risk of diabetes based on patient data using a machine learning model trained on the Pima Indians Diabetes Dataset. It also provides advanced model interpretation using SHAP values to explain predictions.

---

## 🚀 Features

- 🔍 **Single Patient Prediction**  
  Enter patient information manually to predict the diabetes risk and visualize feature contributions using a SHAP waterfall chart.

- 📄 **Batch Prediction from CSV**  
  Upload a CSV file containing multiple patient records for bulk prediction and download the results.

- 📊 **Dataset Explorer**  
  Explore the Pima Diabetes dataset with interactive summary statistics, correlation heatmaps, and distribution plots.

- 📈 **Model Metrics & SHAP Explainability**  
  View performance metrics like confusion matrix, ROC curve, and global SHAP summary plot for model interpretation.

---

## 🛠️ Technologies Used

| Tool                | Purpose                                 |
|---------------------|------------------------------------------|
| `Python`            | Core programming language                |
| `Streamlit`         | Web application UI                      |
| `scikit-learn`      | ML training and evaluation               |
| `shap`              | Explainability (SHAP values)            |
| `pandas`, `numpy`   | Data manipulation                        |
| `matplotlib`, `seaborn` | Visualization libraries             |
| `joblib`, `json`    | Model and metrics persistence            |

---

## 📂 Project Structure


---

## ▶️ Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt

**Run the App**
streamlit run app.py

📤 Input Format for Batch Prediction
A CSV file with the following columns:
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
2,120,70,25,80,32.0,0.4,29
...

🧠 Model Details
Algorithm: Random Forest (default)
Dataset: Pima Indians Diabetes Dataset
Metrics: Accuracy, AUC, Confusion Matrix, ROC Curve
Interpretability: SHAP waterfall and summary plots

📌 Notes
Ensure the following files exist before running:
best_model.pkl
shap_explainer.pkl
metrics.json
diabetes.csv
You can retrain the model using your own train_model.py.

📷 Screenshots
Single Prediction with SHAP Waterfall
SHAP Summary Plot
