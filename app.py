import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
feature_cols = scaler.feature_names_in_


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="PCOS Early Diagnosis", layout="wide")

menu = st.sidebar.radio(
    "Navigation",
    ["Introduction", "Dataset & Model Evaluation", "Individual Risk Assessment", "Prevention & Doctor Finder"]
)

# ---------------- PAGE 1 ----------------
if menu == "Introduction":
    st.title("🩺 Polycystic Ovary Syndrome (PCOS)")
    st.write("""
    PCOS is a common hormonal disorder affecting women of reproductive age.
    It is associated with metabolic, reproductive, and psychological complications. """)

    st.subheader("Who does it affect?")
    st.markdown("""
    - Women aged **12–50 years**
    - ~6–13% prevalence globally
    - Higher prevalence in South Asian populations""")

    st.subheader("Common Symptoms")
    st.markdown("""
    - Irregular or absent periods  
    - Acne and oily skin  
    - Excess hair growth (hirsutism)  
    - Weight gain  
    - Infertility  
    - Insulin resistance / diabetes  """)

    st.subheader("Why Early Detection Matters")
    st.write("""
    Early screening helps prevent long-term complications such as:
    diabetes, cardiovascular disease, infertility, and mental health issues.
    """)

    st.info("⚠️ This tool is for educational screening only, not diagnosis.")

# ---------------- PAGE 2 ----------------
elif menu == "Dataset & Model Evaluation":
    st.title("📊 Dataset Upload & Model Performance")

    file = st.file_uploader("Upload PCOS Dataset", type=["csv", "xlsx"])

    if file:
        if file.name.endswith("xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        df = df.drop(columns=['Sl. No', 'Patient File No.'])
        df.fillna(df.mean(numeric_only=True), inplace=True)
        FEATURE_COLUMNS = scaler.feature_names_in_

        X = df.drop('PCOS (Y/N)', axis=1)
        X = X[FEATURE_COLUMNS]   # enforce correct order

        y = df['PCOS (Y/N)']

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X.fillna(X.mean(numeric_only=True), inplace=True)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:,1]
        # Predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
        # Ensure correct types
        y = y.astype(int)
        y_pred = y_pred.astype(int)
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)

        

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0,1],[0,1],'--')
        ax.legend()
        st.pyplot(fig)

# ---------------- PAGE 3 ----------------
if st.button("Predict PCOS Risk"):

    row = dict.fromkeys(feature_cols, 0)

    # map known features by exact training names
    def set_if_exists(name, value):
        if name in row:
            row[name] = value

    set_if_exists("Age (yrs)", age)
    set_if_exists("Height(Cm)", height)
    set_if_exists("Weight (Kg)", weight)
    set_if_exists("BMI", bmi)
    set_if_exists("BP _Systolic (mmHg)", systolic)
    set_if_exists("BP _Diastolic (mmHg)", diastolic)
    set_if_exists("LH", lh)
    set_if_exists("FSH", fsh)
    set_if_exists("FSH/LH", fsh/lh if lh != 0 else 0)
    set_if_exists("AMH", amh)
    set_if_exists("TSH", tsh)
    set_if_exists("PRL", prl)

    input_df = pd.DataFrame([row])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if probability > 0.65:
        st.error(f"⚠️ High Risk of PCOS ({probability*100:.2f}%)")
    elif probability > 0.35:
        st.warning(f"🟠 Moderate Risk ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({probability*100:.2f}%)")


# ---------------- PAGE 4 ----------------
else:
    st.title("🌿 Prevention & Gynecologist Finder")

    st.subheader("Preventive Measures")
    st.write("""
    - Maintain healthy body weight
    - Regular physical activity
    - Balanced diet (low sugar, high fiber)
    - Stress management
    - Regular hormonal screening
    """)

    st.subheader("Find Nearby Gynecologist")
    st.write("Click below to locate nearby gynecologists:")
    st.markdown("[🔍 Find Gynecologist Near Me](https://www.google.com/maps/search/gynecologist+near+me)")
