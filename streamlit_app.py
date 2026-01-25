# =========================
# INSTALL MISSING LIBRARIES
# =========================
!pip install streamlit reportlab

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# SETTINGS
# =========================
st.set_page_config(page_title="Diabetes Screening & Awareness", layout="centered")
THRESHOLD = 0.4  # your tuned threshold
MODEL_PATH = "best_pipe.pkl"

st.title("🩺 Diabetes Risk Screening + Awareness (Sri Lanka)")
st.caption("Educational screening prototype (not a diagnosis).")


# =========================
# ✅ 1) PUT YOUR REAL TRAINING FEATURES HERE
# =========================
# How to get this list from notebook:
# print(X.columns.tolist())
#
# Replace the example list below with your exact list.
FEATURE_COLUMNS = [
    # ---- EXAMPLE (replace with your real columns) ----
    "Age",
    "BMI (kg/m²)",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
    "Waist circumference (cm)",
    "Family history of diabetes",
    "Physical activity level",
    "Smoking status",
    "Alcohol consumption",
    "Sleep duration"
]


# =========================
# Load pipeline safely
# =========================
@st.cache_resource
def load_pipe():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    pipe = load_pipe()
except Exception as e:
    st.error("❌ Could not load best_pipe.pkl")
    st.info("Make sure best_pipe.pkl is uploaded to GitHub in the same folder as streamlit_app.py")
    st.code(str(e))
    st.stop()


# =========================
# Feature importance
# =========================
def feature_importance_df(pipeline, top_k=12):
    if not hasattr(pipeline, "named_steps"):
        return None

    model = pipeline.named_steps.get("model")
    prep = pipeline.named_steps.get("preprocess") or pipeline.named_steps.get("prep") or pipeline.named_steps.get("preprocessor")

    if model is None or prep is None:
        return None

    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = None

    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_.ravel())
        if names is None:
            names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        return df.head(top_k)

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if names is None:
            names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        return df.head(top_k)

    return None


# =========================
# PDF leaflet
# =========================
def make_leaflet_pdf(name, risk_prob, risk_label, tips_list):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Type 2 Diabetes – Awareness Leaflet")
    y -= 25

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Name: {name if name else 'N/A'}")
    y -= 16
    c.drawString(50, y, f"Screening Risk: {risk_label}")
    y -= 16
    if risk_prob is not None:
        c.drawString(50, y, f"Estimated probability: {risk_prob:.2f}")
    else:
        c.drawString(50, y, "Estimated probability: N/A")
    y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Quick Awareness")
    y -= 18

    c.setFont("Helvetica", 10)
    for line in [
        "Type 2 diabetes happens when the body cannot use insulin properly.",
        "High blood sugar over time can damage heart, kidneys, eyes, and nerves.",
        "Early screening + lifestyle improvements reduce complications."
    ]:
        c.drawString(50, y, f"- {line}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Personal Tips (Sri Lanka-friendly)")
    y -= 18

    c.setFont("Helvetica", 10)
    tips_list = tips_list or []
    for tip in tips_list[:10]:
        if y < 80:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"• {tip}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: Educational screening tool only. Not a medical diagnosis.")
    c.save()

    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================
# Session state
# =========================
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "last_label" not in st.session_state:
    st.session_state.last_label = "N/A"
if "last_tips" not in st.session_state:
    st.session_state.last_tips = []
if "last_name" not in st.session_state:
    st.session_state.last_name = ""


# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk Prediction", "Why risk is high?", "Awareness Quiz", "Download Leaflet (PDF)"]
)


# =========================
# TAB 1: Prediction
# =========================
with tab1:
    st.subheader("Risk Prediction")
    st.write("Enter details. This is a **screening estimate (not a diagnosis)**.")

    name = st.text_input("Name (optional)", value=st.session_state.last_name)

    # Build form for every feature
    st.markdown("### Patient inputs")
    user_row = {}

    # Default options
    yes_no = ["No", "Yes"]
    gender_opts = ["Male", "Female", "Other"]
    activity_opts = ["Never", "1–2 days/week", "3–5 days/week", "Almost daily"]
    smoke_opts = ["Never", "Former", "Current"]
    alcohol_opts = ["No", "Occasionally", "Weekly", "Daily"]
    sleep_opts = ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"]

    for col in FEATURE_COLUMNS:
        lc = col.lower()

        # Numeric guessing based on keyword
        if "age" in lc:
            user_row[col] = st.number_input(col, 0, 120, 40)
        elif "bmi" in lc:
            user_row[col] = st.number_input(col, 5.0, 80.0, 25.0)
        elif "waist" in lc:
            user_row[col] = st.number_input(col, 30.0, 200.0, 85.0)
        elif "systolic" in lc:
            user_row[col] = st.number_input(col, 60, 250, 120)
        elif "diastolic" in lc:
            user_row[col] = st.number_input(col, 30, 150, 80)

        # Categorical guessing
        elif "gender" in lc or "sex" in lc:
            user_row[col] = st.selectbox(col, gender_opts)
        elif "physical" in lc or "exercise" in lc or "activity" in lc:
            user_row[col] = st.selectbox(col, activity_opts)
        elif "smok" in lc:
            user_row[col] = st.selectbox(col, smoke_opts)
        elif "alcohol" in lc:
            user_row[col] = st.selectbox(col, alcohol_opts)
        elif "sleep" in lc:
            user_row[col] = st.selectbox(col, sleep_opts)

        # Default yes/no
        else:
            user_row[col] = st.selectbox(col, yes_no)

    if st.button("Predict"):
        st.session_state.last_name = name
        try:
            X_input = pd.DataFrame([user_row])
            prob = float(pipe.predict_proba(X_input)[:, 1][0])

            st.session_state.last_prob = prob
            if prob >= THRESHOLD:
                st.session_state.last_label = "Higher Risk"
                st.error(f"Estimated probability: {prob:.2f}  →  Higher Risk (threshold={THRESHOLD})")
            else:
                st.session_state.last_label = "Lower Risk"
                st.success(f"Estimated probability: {prob:.2f}  →  Lower Risk (threshold={THRESHOLD})")

            st.info("If you have thirst, frequent urination, fatigue, blurred vision, or slow healing wounds, please check FBS/HbA1c.")
        except Exception as e:
            st.error("Prediction failed (feature mismatch).")
            st.code(str(e))
            st.warning("Fix: FEATURE_COLUMNS must exactly match your training X.columns (same spelling and spacing).")


# =========================
# TAB 2: Feature importance
# =========================
with tab2:
    st.subheader("Why risk is high?")
    st.write("Shows which features influence the trained model most.")

    imp = feature_importance_df(pipe, top_k=12)
    if imp is None or imp.empty:
        st.warning("Feature importance not available for this model.")
    else:
        fig = plt.figure()
        plt.barh(imp["feature"][::-1], imp["importance"][::-1])
        plt.title("Top features (importance)")
        plt.xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)


# =========================
# TAB 3: Awareness quiz
# =========================
with tab3:
    st.subheader("Awareness Quiz (5 questions)")
    st.write("Answer quickly — you’ll get personalised tips (Sri Lanka-friendly).")

    q1 = st.radio("1) How often do you exercise (at least 30 mins)?", ["Rarely", "1–2 days/week", "3–5 days/week", "Almost daily"])
    q2 = st.radio("2) Sweet tea / sugary drinks per day?", ["0", "1", "2", "3 or more"])
    q3 = st.radio("3) Your usual rice portion?", ["Small", "Medium", "Large"])
    q4 = st.radio("4) Family history of diabetes?", ["No", "Yes"])
    q5 = st.radio("5) Sleep most nights?", ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"])

    if st.button("Get my tips"):
        tips = []

        if q1 in ["Rarely", "1–2 days/week"]:
            tips.append("Try a 30-minute walk after dinner (5 days/week helps insulin sensitivity).")
            tips.append("If busy: 10 minutes walk × 3 times/day.")

        if q2 in ["1", "2", "3 or more"]:
            tips.append("Reduce sweet tea/soft drinks gradually (half sugar → quarter → none).")
            tips.append("Replace with water or unsweetened drinks (plain tea).")

        if q3 == "Large":
            tips.append("Reduce rice portion and increase vegetables (gotukola, mukunuwenna, beans, cabbage).")
            tips.append("Try red/brown rice sometimes; portion control still matters.")

        if q4 == "Yes":
            tips.append("Family history increases risk: do regular screening (FBS/HbA1c) and maintain healthy waist/weight.")

        if q5 == "<6 hours":
            tips.append("Aim for 7–8 hours sleep. Poor sleep increases cravings and insulin resistance.")

        tips.append("Add fibre/protein: dhal, chickpeas, eggs, fish, chicken, leafy salads.")
        tips.append("Snack ideas: roasted gram (kadala), plain yogurt, fruit (portion control), nuts (small portion).")
        tips.append("If BP is high, reduce salt and follow medical advice.")

        st.session_state.last_tips = tips

        st.markdown("### Your Tips")
        for t in tips:
            st.write("•", t)


# =========================
# TAB 4: PDF leaflet
# =========================
with tab4:
    st.subheader("Download Leaflet (PDF)")
    st.write("Generates a one-page awareness leaflet with your last quiz tips + last risk result.")

    leaflet_name = st.text_input("Name on leaflet", value=st.session_state.last_name)

    if st.button("Generate PDF"):
        pdf_bytes = make_leaflet_pdf(
            name=leaflet_name,
            risk_prob=st.session_state.last_prob,
            risk_label=st.session_state.last_label,
            tips_list=st.session_state.last_tips
        )

        st.download_button(
            label="⬇️ Download PDF leaflet",
            data=pdf_bytes,
            file_name="diabetes_awareness_leaflet.pdf",
            mime="application/pdf"
        )

st.divider()
st.caption("⚠️ Disclaimer: Educational screening tool only. Not a medical diagnosis.")
