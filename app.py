import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


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
elif menu == "Individual Risk Assessment":
    st.title("🧍 Individual PCOS Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 15, 50)
        height = st.number_input("Height (cm)")
        weight = st.number_input("Weight (kg)")
        systolic = st.number_input("BP Systolic")
        diastolic = st.number_input("BP Diastolic")

    with col2:
        lh = st.number_input("LH")
        fsh = st.number_input("FSH")
        amh = st.number_input("AMH")
        tsh = st.number_input("TSH")
        prl = st.number_input("Prolactin")

    bmi = weight / ((height/100)**2) if height > 0 else 0
    st.write(f"**Calculated BMI:** {bmi:.2f}")

    if st.button("Predict PCOS Risk"):

        # ✅ get exact training feature order
        feature_cols = scaler.feature_names_in_

        # ✅ create zero row with correct columns
        row = dict.fromkeys(feature_cols, 0)

        # ✅ safely map inputs if column exists
        def put(col, val):
            if col in row:
                row[col] = val

        put("Age (yrs)", age)
        put("Height(Cm)", height)
        put("Weight (Kg)", weight)
        put("BMI", bmi)
        put("BP _Systolic (mmHg)", systolic)
        put("BP _Diastolic (mmHg)", diastolic)
        put("LH", lh)
        put("FSH", fsh)
        put("FSH/LH", fsh/lh if lh != 0 else 0)
        put("AMH", amh)
        put("TSH", tsh)
        put("PRL", prl)

        # build dataframe in correct order
        input_df = pd.DataFrame([row])
        input_df = input_df[feature_cols]

        # scale + predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Result")

        if probability > 0.45:
            st.error(f"⚠️ High Risk of PCOS ({probability*100:.2f}%)")
        elif probability > 0.20:
            st.warning(f"🟠 Moderate Risk ({probability*100:.2f}%)")
        else:
            st.success(f"🟠 High Risk of PCOS ({probability*120:.2f}%)")



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
