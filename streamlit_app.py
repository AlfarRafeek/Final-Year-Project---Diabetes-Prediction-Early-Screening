import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Diabetes Screening & Awareness", layout="centered")

st.title("🩺 Diabetes Risk Screening + Awareness (Sri Lanka)")
st.caption("Educational screening prototype (not a diagnosis).")


# =========================
# Load model (pipeline)
# =========================
@st.cache_resource
def load_pipe(path="best_pipe.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

PIPE_PATH = "best_pipe.pkl"

try:
    pipe = load_pipe(PIPE_PATH)
except Exception as e:
    st.error("❌ Model file could not be loaded.")
    st.info("Make sure `best_pipe.pkl` is in the same GitHub folder as `streamlit_app.py`.")
    st.code(str(e))
    st.stop()


# =========================
# Detect expected input columns
# =========================
def detect_expected_columns(pipeline):
    """
    Best-effort detection of input columns expected by the pipeline.
    Works for many sklearn / imblearn pipelines.
    """
    # 1) If pipeline stores feature_names_in_
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    # 2) If preprocessing step stores feature_names_in_
    if hasattr(pipeline, "named_steps"):
        prep = pipeline.named_steps.get("preprocess") or pipeline.named_steps.get("prep") or pipeline.named_steps.get("preprocessor")
        if prep is not None and hasattr(prep, "feature_names_in_"):
            return list(prep.feature_names_in_)

    return None


expected_cols = detect_expected_columns(pipe)

# If we can't detect, allow manual list (you can paste X.columns from notebook here)
MANUAL_FALLBACK_COLS = None  # Example: ["Age", "BMI (kg/m²)", "Systolic BP (mmHg)", ...]

if expected_cols is None and MANUAL_FALLBACK_COLS:
    expected_cols = MANUAL_FALLBACK_COLS


# =========================
# Feature importance helper
# =========================
def get_feature_importance(pipeline, top_k=12):
    """
    Supports:
    - LogisticRegression: abs(coef)
    - Tree models: feature_importances_
    Will attempt to use preprocess.get_feature_names_out()
    """
    if not hasattr(pipeline, "named_steps"):
        return None

    model = pipeline.named_steps.get("model")
    prep = pipeline.named_steps.get("preprocess") or pipeline.named_steps.get("prep") or pipeline.named_steps.get("preprocessor")

    if model is None or prep is None:
        return None

    # Feature names after preprocessing
    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = None

    # Logistic regression coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        imp = np.abs(coefs)
        if feat_names is None:
            feat_names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
        return df.head(top_k)

    # Tree-based feature importances
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if feat_names is None:
            feat_names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
        return df.head(top_k)

    return None


# =========================
# PDF Leaflet helper
# =========================
def make_leaflet_pdf(name, risk_prob, risk_label, tips_list):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 60
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
    lines = [
        "Type 2 Diabetes happens when the body cannot use insulin properly.",
        "High blood sugar over time can damage the heart, kidneys, eyes, and nerves.",
        "Early screening helps reduce complications."
    ]
    for line in lines:
        c.drawString(50, y, f"- {line}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Personal Tips (Sri Lanka-friendly)")
    y -= 18
    c.setFont("Helvetica", 10)
    for tip in (tips_list or [])[:10]:
        if y < 80:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"• {tip}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "When to check blood sugar (FBS / HbA1c)")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, "If you have thirst, frequent urination, fatigue, blurred vision, or slow healing wounds, visit a clinic.")
    y -= 30

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: Educational screening tool only. Not a medical diagnosis.")
    c.save()

    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================
# Session state (store last results)
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
# UI Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk Prediction", "Why risk is high?", "Awareness Quiz", "Download Leaflet (PDF)"]
)

THRESHOLD = 0.4  # your tuned threshold


# =========================
# TAB 1: Risk Prediction (dynamic feature form)
# =========================
with tab1:
    st.subheader("Risk Prediction")
    st.write("Enter details. This is a **screening estimate (not a diagnosis)**.")

    # Show expected columns so you can confirm it's correct
    with st.expander("Show model input columns (for debugging)"):
        if expected_cols is None:
            st.error("Could not detect expected input columns from your pipeline.")
            st.info("Fix: set MANUAL_FALLBACK_COLS in code OR re-save the pipeline with feature names.")
        else:
            st.write(expected_cols)

    name = st.text_input("Name (optional)", value=st.session_state.last_name)

    if expected_cols is None:
        st.warning("Prediction disabled because the app cannot detect required input columns.")
    else:
        st.markdown("### Fill inputs")
        user_row = {}

        # Simple dropdown options (safe defaults)
        yes_no = ["No", "Yes"]
        freq4 = ["Never", "1–2 days/week", "3–5 days/week", "Almost daily"]
        freq3 = ["Low", "Medium", "High"]
        gender_opts = ["Male", "Female", "Other"]

        # Heuristic form builder based on column names
        for col in expected_cols:
            lc = col.lower()

            # numeric-like
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
            elif "chol" in lc:
                user_row[col] = st.number_input(col, 80.0, 400.0, 180.0)
            elif "glucose" in lc or "sugar" in lc:
                user_row[col] = st.number_input(col, 50.0, 400.0, 100.0)

            # categorical-like
            elif "gender" in lc or "sex" in lc:
                user_row[col] = st.selectbox(col, gender_opts)
            elif "exercise" in lc or "physical activity" in lc:
                user_row[col] = st.selectbox(col, freq4)
            elif "smok" in lc:
                user_row[col] = st.selectbox(col, ["Never", "Former", "Current"])
            elif "alcohol" in lc:
                user_row[col] = st.selectbox(col, ["No", "Occasionally", "Weekly", "Daily"])
            elif "sleep" in lc:
                user_row[col] = st.selectbox(col, ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"])
            elif "diet" in lc or "food" in lc:
                user_row[col] = st.selectbox(col, freq3)

            # default: yes/no
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

                st.info(
                    "If you have symptoms (frequent urination, thirst, fatigue, blurred vision, slow healing), "
                    "please visit a clinic and check FBS/HbA1c."
                )

            except Exception as e:
                st.warning("Prediction failed due to input mismatch.")
                st.code(str(e))
                st.info(
                    "Fix: Your pipeline expects specific categories/values. "
                    "Tell me your expected_cols list and I can make exact dropdown options."
                )


# =========================
# TAB 2: Why risk is high? (Feature importance)
# =========================
with tab2:
    st.subheader("Why risk is high? (Feature importance)")
    st.write("This explains which features influence the trained model most.")

    imp_df = get_feature_importance(pipe, top_k=12)

    if imp_df is None or imp_df.empty:
        st.warning("Feature importance is not available for this saved model.")
    else:
        fig = plt.figure()
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        plt.title("Top Features (importance)")
        plt.xlabel("Importance (abs coef or tree importance)")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "For Logistic Regression: importance = absolute coefficient after preprocessing. "
            "For tree models: importance = built-in feature importance."
        )


# =========================
# TAB 3: Awareness quiz (Sri Lanka-focused)
# =========================
with tab3:
    st.subheader("Awareness Quiz (5 Questions)")
    st.write("Answer quickly — you’ll get personalised tips (Sri Lanka-friendly).")

    q1 = st.radio("1) How often do you exercise (at least 30 mins)?", ["Rarely", "1–2 days/week", "3–5 days/week", "Almost daily"])
    q2 = st.radio("2) How many sugary drinks / sweet tea per day?", ["0", "1", "2", "3 or more"])
    q3 = st.radio("3) Your usual rice portion at lunch/dinner?", ["Small", "Medium", "Large"])
    q4 = st.radio("4) Do you have close family history of diabetes?", ["No", "Yes"])
    q5 = st.radio("5) How is your sleep most nights?", ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"])

    if st.button("Get my tips"):
        tips = []

        if q1 in ["Rarely", "1–2 days/week"]:
            tips.append("Try a 30-minute walk after dinner (5 days/week helps insulin sensitivity).")
            tips.append("If busy: do 10 mins walk × 3 times/day.")

        if q2 in ["1", "2", "3 or more"]:
            tips.append("Reduce sweet tea/soft drinks gradually (half sugar → quarter → none).")
            tips.append("Replace with water or unsweetened drinks (plain tea, herbal drinks).")

        if q3 == "Large":
            tips.append("Reduce rice portion slightly and increase vegetables (gotukola, mukunuwenna, beans, cabbage).")
            tips.append("Try red/brown rice sometimes; portion control still matters.")

        if q4 == "Yes":
            tips.append("Family history increases risk: do regular screening (FBS/HbA1c) and maintain healthy weight/waist.")

        if q5 == "<6 hours":
            tips.append("Aim for 7–8 hours sleep. Poor sleep can increase cravings and insulin resistance.")

        tips.append("Add fibre/protein: dhal, chickpeas, eggs, fish, chicken, leafy salads.")
        tips.append("Snack ideas: roasted gram (kadala), plain yogurt, fruit (portion control), nuts (small portion).")
        tips.append("If BP is high, reduce salt and follow medical advice.")

        st.session_state.last_tips = tips

        st.markdown("### Your Tips")
        for t in tips:
            st.write("•", t)

        st.warning("Educational only. For diagnosis, please consult a healthcare professional.")


# =========================
# TAB 4: PDF Leaflet Download
# =========================
with tab4:
    st.subheader("Download Leaflet (PDF)")
    st.write("Generate a one-page diabetes awareness leaflet (includes your last quiz tips and last risk result).")

    leaflet_name = st.text_input("Name on leaflet (optional)", value=st.session_state.last_name)

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

    st.caption("Tip: Run a prediction and the quiz first so the leaflet contains personalised info.")

st.divider()
st.caption("⚠️ Disclaimer: This tool is for education/screening only and does not provide a medical diagnosis.")
