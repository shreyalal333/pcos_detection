# 🩺 Predictive Modeling for Early Diagnosis of Polycystic Ovary Syndrome (PCOS)

## 📌 Overview

Polycystic Ovary Syndrome (PCOS) is one of the most common endocrine disorders affecting women of reproductive age. Delayed diagnosis can lead to complications such as infertility, type 2 diabetes, cardiovascular disease, obesity, and psychological disorders.

This project presents a machine learning-based predictive system for the early detection of PCOS using clinical, hormonal, and physiological parameters. The trained model is deployed using **Streamlit**, allowing users to upload datasets for model evaluation and perform individual risk assessment through an interactive web application.

---

## 🎯 Objectives

- Develop a machine learning model for early PCOS prediction.
- Perform preprocessing and evaluation of clinical datasets.
- Provide individual PCOS risk assessment based on user inputs.
- Visualize model performance using evaluation metrics.
- Promote awareness by providing preventive measures and nearby gynecologist search.

---

## ✨ Features

### 📖 1. Introduction
- Overview of PCOS
- Common symptoms
- Importance of early diagnosis
- Project objective

### 📊 2. Dataset & Model Evaluation
- Upload CSV or Excel dataset
- Automatic preprocessing
- Confusion Matrix
- Classification Report
- ROC Curve
- Area Under Curve (AUC)

### 👩 3. Individual Risk Assessment
Users can enter:

- Age
- Height
- Weight
- BMI (automatically calculated)
- Blood Pressure
- LH
- FSH
- AMH
- TSH
- Prolactin

The application predicts the probability of PCOS using the trained machine learning model.

### 🌿 4. Prevention & Doctor Finder

- Lifestyle recommendations
- Dietary guidance
- Exercise recommendations
- Google Maps link to locate nearby gynecologists

---

## 🧠 Machine Learning Workflow

Dataset Collection

↓

Data Cleaning

↓

Missing Value Handling

↓

Feature Selection

↓

Feature Scaling (StandardScaler)

↓

Model Training (XGBoost Classifier)

↓

Model Evaluation

↓

Model Deployment using Streamlit

---

## ⚙️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Pickle

---

## 📂 Project Structure

```
PCOS_Detection/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── PCOS Dataset.csv
└── images/
```

---

## 📈 Model Evaluation

The trained model achieved:

| Metric | Score |
|---------|--------|
| Accuracy | 98% |
| Precision | 0.97 |
| Recall | 0.94 |
| F1-Score | 0.95 |

The model also demonstrates excellent discrimination capability using the ROC Curve and Area Under the Curve (AUC).

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/pcos-detection.git
```

Navigate to the project

```bash
cd pcos-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Input Parameters

The application accepts the following clinical features:

- Age
- Height
- Weight
- Body Mass Index (BMI)
- Blood Pressure (Systolic & Diastolic)
- Luteinizing Hormone (LH)
- Follicle Stimulating Hormone (FSH)
- Anti-Müllerian Hormone (AMH)
- Thyroid Stimulating Hormone (TSH)
- Prolactin (PRL)

---

## 📷 Application Pages

1. Introduction
2. Dataset Upload & Model Evaluation
3. Individual PCOS Risk Prediction
4. Prevention & Nearby Gynecologist Finder

---

## 📌 Future Enhancements

- Integration of ultrasound findings
- Lifestyle recommendation engine
- Explainable AI (SHAP/LIME)
- Cloud database support
- Mobile application deployment
- Electronic Health Record (EHR) integration

---

## ⚠️ Disclaimer

This application is intended for educational and research purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Individuals should consult qualified healthcare professionals for medical evaluation and diagnosis.

---

## 👩‍💻 Author

**Shreya Lal**

Department of Biotechnology

R.V. College of Engineering

Bengaluru, India
