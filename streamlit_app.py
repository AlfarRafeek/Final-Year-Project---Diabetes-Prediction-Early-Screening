# ============================================================
# Diabetes Risk Analytics Dashboard (Sri Lanka) - PPS aligned
# Author: Alfar Rafeek
# Purpose: Visualize diabetes risk patterns + key predictive factors
# Notes:
# - Upload your Excel survey (.xlsx) in the app
# - This app trains a simple model inside the app for explainability
#   (avoids best_pipe.pkl version mismatch + missing-columns issues)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, brier_score_loss


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Diabetes Risk Analytics (Sri Lanka)", layout="wide")
st.title("📊 Diabetes Risk Analytics Dashboard (Sri Lanka)")
st.caption("Data analytics dashboard to visualize diabetes risk levels and significant predictive factors (PPS-aligned).")
st.info("⚠️ Educational analytics only — not a medical diagnosis.")


# -------------------------
# Constants
# -------------------------
TARGET_COL = "Have you ever been diagnosed with diabetes?"
DROP_COLS_LEAKAGE = [
    "Are you currently taking any medications for diabetes or related conditions?"
]
# Optional: late symptoms can inflate performance if your goal is 'early screening'
LATE_SYMPTOMS = [
    "Do you experience frequent urination?",
    "Do you often feel unusually thirsty?",
    "Do you feel unusually fatigued or tired?",
    "Do you have blurred vision or slow-healing wounds?",
]

NUMERIC_COLS_CANDIDATES = [
    "Waist circumference (cm)",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
    "BMI (kg/m²)",
]

RANDOM_STATE = 42


# -------------------------
# Helpers
# -------------------------
def safe_read_excel(file) -> pd.DataFrame:
    # pandas needs openpyxl to read xlsx (added in requirements.txt)
    df = pd.read_excel(file)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    return df


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().map({"Yes": 1, "No": 0})
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_rule_based_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Simple screening segmentation for analytics (not diagnosis)."""
    df = df.copy()

    needed = ["BMI (kg/m²)", "Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)"]
    if not all(c in df.columns for c in needed):
        df["Risk Score (rule)"] = np.nan
        df["Risk Band"] = "Unknown"
        return df

    bmi = df["BMI (kg/m²)"]
    waist = df["Waist circumference (cm)"]
    sbp = df["Systolic BP (mmHg)"]
    dbp = df["Diastolic BP (mmHg)"]

    score = np.zeros(len(df), dtype=float)

    # BMI
    score += (bmi >= 25).fillna(False).astype(int)
    score += (bmi >= 30).fillna(False).astype(int)  # extra point for obesity

    # Waist (simple unified threshold for demonstration)
    score += (waist >= 90).fillna(False).astype(int)

    # BP
    score += (sbp >= 130).fillna(False).astype(int)
    score += (dbp >= 85).fillna(False).astype(int)

    band = np.where(score <= 1, "Low", np.where(score <= 3, "Medium", "High"))
    df["Risk Score (rule)"] = score
    df["Risk Band"] = band
    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre


def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def make_leaflet_pdf(summary_lines, tips_lines):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Diabetes Awareness Leaflet (Sri Lanka)")
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary")
    y -= 18
    c.setFont("Helvetica", 10)

    for line in summary_lines:
        c.drawString(50, y, f"• {line}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Practical Tips")
    y -= 18
    c.setFont("Helvetica", 10)

    for tip in tips_lines[:14]:
        if y < 80:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"• {tip}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: Educational analytics tool only. Not a medical diagnosis.")
    c.save()

    pdf = buf.getvalue()
    buf.close()
    return pdf


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Upload data")
uploaded = st.sidebar.file_uploader("Upload your survey Excel (.xlsx)", type=["xlsx"])

st.sidebar.header("Settings")
remove_leakage = st.sidebar.checkbox("Remove leakage question (medication)", value=True)
remove_late_symptoms = st.sidebar.checkbox("Remove late symptoms (harder early-screening)", value=True)
test_size = st.sidebar.slider("Test size", 0.15, 0.40, 0.25, 0.05)
threshold = st.sidebar.slider("Risk threshold (for model probability)", 0.20, 0.80, 0.40, 0.05)

st.sidebar.caption("Tip: Removing late symptoms avoids unrealistically high accuracy.")


# -------------------------
# Load data
# -------------------------
if not uploaded:
    st.warning("Please upload your Excel file to start (your Diabetes Risk Survey Responses).")
    st.stop()

df = safe_read_excel(uploaded)
df = clean_columns(df)
df = map_target(df)
df = coerce_numeric(df)

# Drop leakage columns if selected
drop_cols = []
if remove_leakage:
    drop_cols += [c for c in DROP_COLS_LEAKAGE if c in df.columns]
if remove_late_symptoms:
    drop_cols += [c for c in LATE_SYMPTOMS if c in df.columns]

if drop_cols:
    df_model = df.drop(columns=drop_cols, errors="ignore")
else:
    df_model = df.copy()

df_model = add_rule_based_risk(df_model)

if TARGET_COL not in df_model.columns:
    st.error(f"Target column not found: {TARGET_COL}")
    st.stop()

# Separate X/y
X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL].astype(int)


# -------------------------
# Tabs (analytics dashboard style)
# -------------------------
tab_overview, tab_quality, tab_risk, tab_model, tab_awareness = st.tabs(
    ["Overview", "Data Quality", "Risk Analytics", "Model + Explainability", "Awareness + PDF"]
)

# ============================================================
# TAB 1: Overview
# ============================================================
with tab_overview:
    st.subheader("Dataset overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_model.shape[0])
    c2.metric("Features", X.shape[1])
    c3.metric("Diabetes (Yes) count", int((y == 1).sum()))

    st.markdown("### Sample of the dataset")
    st.dataframe(df_model.head(10), use_container_width=True)

    st.markdown("### Class distribution (Plot 1)")
    counts = y.value_counts().sort_index()
    fig = plt.figure()
    plt.bar(["No (0)", "Yes (1)"], [counts.get(0, 0), counts.get(1, 0)])
    plt.ylabel("Count")
    plt.title("Plot 1: Diabetes target distribution")
    plt.tight_layout()
    st.pyplot(fig)

    st.download_button(
        "⬇️ Download Plot 1 as PNG",
        data=fig_to_png_bytes(fig),
        file_name="Plot01_Class_Distribution.png",
        mime="image/png"
    )

# ============================================================
# TAB 2: Data Quality (missingness + distributions + outliers)
# ============================================================
with tab_quality:
    st.subheader("Data quality checks")

    st.markdown("### Missing values (Plot 2)")
    missing = df_model.isna().sum().sort_values(ascending=False)
    top_missing = missing[missing > 0].head(15)

    fig = plt.figure()
    if len(top_missing) == 0:
        plt.text(0.1, 0.5, "No missing values found (top 15).", fontsize=12)
        plt.axis("off")
    else:
        plt.bar(top_missing.index.astype(str), top_missing.values)
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Missing count")
        plt.title("Plot 2: Missing values (top columns)")
        plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Numeric distributions (Plots 3–6)")
    numeric_to_plot = [c for c in NUMERIC_COLS_CANDIDATES if c in df_model.columns]
    cols = st.columns(2)

    plot_num = 3
    for idx, col in enumerate(numeric_to_plot[:4]):
        data = pd.to_numeric(df_model[col], errors="coerce").dropna()
        fig = plt.figure()
        if data.empty:
            plt.text(0.1, 0.5, f"No numeric data available for {col}", fontsize=12)
            plt.axis("off")
        else:
            plt.hist(data, bins=20)
            plt.title(f"Plot {plot_num}: Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
        cols[idx % 2].pyplot(fig)
        plot_num += 1

    st.markdown("### Correlation heatmap (numeric only) (Plot 7)")
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df_model[num_cols].corr(numeric_only=True)
        fig = plt.figure()
        plt.imshow(corr.values, aspect="auto")
        plt.title("Plot 7: Correlation heatmap (numeric)")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# ============================================================
# TAB 3: Risk Analytics (PPS aligned visuals)
# ============================================================
with tab_risk:
    st.subheader("Risk analytics (PPS-aligned)")

    st.markdown("### Risk band distribution (Plot 8)")
    band_counts = df_model["Risk Band"].value_counts()
    fig = plt.figure()
    plt.bar(band_counts.index, band_counts.values)
    plt.ylabel("Count")
    plt.title("Plot 8: Risk bands (rule-based using BMI/Waist/BP)")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Risk band vs diabetes outcome (stacked) (Plot 9)")
    ct = pd.crosstab(df_model["Risk Band"], df_model[TARGET_COL], normalize="index")
    fig = plt.figure()
    x_idx = np.arange(len(ct.index))
    p0 = ct.get(0, pd.Series([0]*len(ct.index), index=ct.index)).values
    p1 = ct.get(1, pd.Series([0]*len(ct.index), index=ct.index)).values
    plt.bar(x_idx, p0, label="No (0)")
    plt.bar(x_idx, p1, bottom=p0, label="Yes (1)")
    plt.xticks(x_idx, ct.index)
    plt.ylabel("Proportion")
    plt.title("Plot 9: Diabetes proportion within each risk band")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Diabetes rate heatmap: Age × Diet (Plot 10)")
    if "Age" in df_model.columns and "How would you describe your diet?" in df_model.columns:
        piv = df_model.pivot_table(
            values=TARGET_COL,
            index="Age",
            columns="How would you describe your diet?",
            aggfunc="mean"
        )
        fig = plt.figure()
        plt.imshow(piv.values, aspect="auto")
        plt.xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
        plt.yticks(range(len(piv.index)), piv.index)
        plt.colorbar(label="Diabetes rate")
        plt.title("Plot 10: Diabetes rate by Age group and Diet type")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Age and Diet columns not available for heatmap.")

    st.markdown("### BMI vs Waist scatter (colored by outcome) (Plot 11)")
    if "BMI (kg/m²)" in df_model.columns and "Waist circumference (cm)" in df_model.columns:
        d0 = df_model[df_model[TARGET_COL] == 0]
        d1 = df_model[df_model[TARGET_COL] == 1]
        fig = plt.figure()
        plt.scatter(d0["BMI (kg/m²)"], d0["Waist circumference (cm)"], alpha=0.6, label="No (0)")
        plt.scatter(d1["BMI (kg/m²)"], d1["Waist circumference (cm)"], alpha=0.6, label="Yes (1)")
        plt.xlabel("BMI (kg/m²)")
        plt.ylabel("Waist circumference (cm)")
        plt.title("Plot 11: BMI vs Waist — by diabetes outcome")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("BMI and Waist columns not available.")

    st.markdown("### Symptoms vs outcome (top yes-rate) (Plot 12)")
    symptom_cols = [c for c in df_model.columns if c.startswith("Do you") or c.startswith("Have you noticed") or c.startswith("Do you feel")]
    symptom_cols = [c for c in symptom_cols if c != TARGET_COL]

    # compute yes rate among diabetic vs non-diabetic
    rows = []
    for c in symptom_cols:
        if c in df_model.columns:
            vals = df_model[c].astype(str).str.strip()
            if vals.isin(["Yes", "No"]).any():
                tmp = df_model.copy()
                tmp[c] = vals.map({"Yes": 1, "No": 0})
                rate_yes_diab = tmp.loc[tmp[TARGET_COL] == 1, c].mean()
                rate_yes_nond = tmp.loc[tmp[TARGET_COL] == 0, c].mean()
                if pd.notna(rate_yes_diab) and pd.notna(rate_yes_nond):
                    rows.append((c, rate_yes_diab, rate_yes_nond))

    if rows:
        sym_df = pd.DataFrame(rows, columns=["Symptom", "Yes-rate (Diabetes=1)", "Yes-rate (Diabetes=0)"])
        sym_df = sym_df.sort_values("Yes-rate (Diabetes=1)", ascending=False).head(10)

        fig = plt.figure()
        y_pos = np.arange(len(sym_df))
        plt.barh(y_pos - 0.15, sym_df["Yes-rate (Diabetes=0)"], height=0.3, label="No diabetes")
        plt.barh(y_pos + 0.15, sym_df["Yes-rate (Diabetes=1)"], height=0.3, label="Diabetes")
        plt.yticks(y_pos, sym_df["Symptom"])
        plt.xlabel("Proportion 'Yes'")
        plt.title("Plot 12: Symptom yes-rate comparison")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No symptom columns detected in Yes/No format.")

# ============================================================
# TAB 4: Model + Explainability (train in-app)
# ============================================================
with tab_model:
    st.subheader("Model + explainability (trained inside the dashboard)")

    st.write(
        "This section trains a few models using your dataset (no saved pickle needed). "
        "It reports **ROC-AUC, F1, calibration**, learning curve, and **permutation importance**."
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=RANDOM_STATE, stratify=y
    )

    pre = make_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    best_name = None
    best_auc = -1
    best_pipe = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    st.markdown("### Cross-validated ROC-AUC (Model selection) (Plot 13)")
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        auc_mean = float(scores.mean())
        auc_std = float(scores.std())
        results.append((name, auc_mean, auc_std))

        if auc_mean > best_auc:
            best_auc = auc_mean
            best_name = name
            best_pipe = pipe

    res_df = pd.DataFrame(results, columns=["Model", "CV ROC-AUC Mean", "CV ROC-AUC Std"]).sort_values("CV ROC-AUC Mean", ascending=False)
    st.dataframe(res_df, use_container_width=True)

    fig = plt.figure()
    plt.bar(res_df["Model"], res_df["CV ROC-AUC Mean"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("ROC-AUC")
    plt.title("Plot 13: Cross-validated ROC-AUC (mean)")
    plt.tight_layout()
    st.pyplot(fig)

    st.success(f"Selected model: **{best_name}** (best mean CV ROC-AUC ≈ {best_auc:.3f})")

    # Fit best model
    best_pipe.fit(X_train, y_train)
    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= float(threshold)).astype(int)

    # Metrics
    st.markdown("### Test metrics (threshold applied)")
    m = {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "Brier (calibration)": brier_score_loss(y_test, proba),
    }
    st.write(pd.DataFrame([m]).round(4))

    # Confusion matrix (Plot 14)
    st.markdown("### Confusion matrix (Plot 14)")
    cm = confusion_matrix(y_test, pred)
    fig = plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.title("Plot 14: Confusion matrix")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    st.pyplot(fig)

    # ROC (Plot 15)
    st.markdown("### ROC curve (Plot 15)")
    fig = plt.figure()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("Plot 15: ROC curve (test)")
    plt.tight_layout()
    st.pyplot(fig)

    # PR curve (Plot 16)
    st.markdown("### Precision–Recall curve (Plot 16)")
    fig = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title("Plot 16: Precision–Recall curve (test)")
    plt.tight_layout()
    st.pyplot(fig)

    # Calibration curve (Plot 17)
    st.markdown("### Calibration curve (Plot 17)")
    fig = plt.figure()
    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, strategy="quantile")
    plt.title("Plot 17: Calibration curve (reliability)")
    plt.tight_layout()
    st.pyplot(fig)

    # Learning curve (Plot 18)
    st.markdown("### Learning curve (ROC-AUC) (Plot 18)")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=best_pipe,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig = plt.figure()
    plt.plot(train_sizes, train_mean, marker="o", label="Train ROC-AUC")
    plt.plot(train_sizes, val_mean, marker="o", label="CV ROC-AUC")
    plt.xlabel("Training set size")
    plt.ylabel("ROC-AUC")
    plt.title("Plot 18: Learning curve")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Permutation importance (Plot 19)
    st.markdown("### Top drivers (Permutation importance) (Plot 19)")
    st.caption("Permutation importance shows which variables reduce ROC-AUC most when shuffled (model-agnostic).")

    perm = permutation_importance(
        best_pipe, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc"
    )
    # Feature names from original columns (safe and readable)
    perm_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": perm.importances_mean
    }).sort_values("Importance", ascending=False).head(12)

    fig = plt.figure()
    plt.barh(perm_df["Feature"][::-1], perm_df["Importance"][::-1])
    plt.xlabel("ROC-AUC drop (higher = more important)")
    plt.title("Plot 19: Permutation importance (top 12)")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Why risk is high? (user-level explanation)")
    st.write("Pick one record and compare it against the dataset + key drivers.")

    idx = st.number_input("Choose a row index (0-based)", min_value=0, max_value=max(0, len(df_model)-1), value=0)
    row = df_model.iloc[int(idx)].copy()

    # Build explanation from rule risk + top features
    summary_lines = []
    if "Risk Band" in row:
        summary_lines.append(f"Rule-based risk band: {row.get('Risk Band')}")
    if "Risk Score (rule)" in row:
        summary_lines.append(f"Rule-based risk score: {row.get('Risk Score (rule)')}")

    # Show selected record
    st.markdown("**Selected record (preview):**")
    st.dataframe(pd.DataFrame([row]), use_container_width=True)

    # Compare numeric to dataset percentiles (Plot 20)
    st.markdown("### Compare to population (percentiles) (Plot 20)")
    numeric_cols_present = [c for c in NUMERIC_COLS_CANDIDATES if c in df_model.columns]
    if numeric_cols_present:
        fig = plt.figure()
        labels = []
        vals = []
        percs = []
        for c in numeric_cols_present:
            v = pd.to_numeric(row.get(c), errors="coerce")
            series = pd.to_numeric(df_model[c], errors="coerce").dropna()
            if pd.notna(v) and not series.empty:
                pct = float((series < v).mean() * 100)
                labels.append(c)
                vals.append(v)
                percs.append(pct)

        if labels:
            y_pos = np.arange(len(labels))
            plt.barh(y_pos, percs)
            plt.yticks(y_pos, labels)
            plt.xlabel("Percentile in dataset (%)")
            plt.title("Plot 20: Where this person stands vs dataset")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Not enough valid numeric values for percentile plot.")
    else:
        st.info("No numeric measures available for percentile comparison.")


# ============================================================
# TAB 5: Awareness + PDF
# ============================================================
with tab_awareness:
    st.subheader("Awareness quiz + downloadable leaflet")

    st.markdown("### Awareness quiz (12 questions)")
    q1 = st.radio("1) Exercise per week?", ["Never", "1–3 times", "3–6 times", "Daily"])
    q2 = st.radio("2) Sweet tea / sugary drinks per day?", ["0", "1", "2", "3+"])
    q3 = st.radio("3) Rice portion size?", ["Small", "Medium", "Large"])
    q4 = st.radio("4) Family history of diabetes?", ["No", "Yes"])
    q5 = st.radio("5) Sleep hours?", ["0-5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "8+"])
    q6 = st.radio("6) Fried foods?", ["Rarely", "1–2/week", "3–4/week", "Almost daily"])
    q7 = st.radio("7) Fruits/vegetables per day?", ["<2 portions", "2–3", "4–5", "5+"])
    q8 = st.radio("8) BP check frequency?", ["Never", "Sometimes", "Yearly", "Often"])
    q9 = st.radio("9) Stress level?", ["Low", "Moderate", "High"])
    q10 = st.radio("10) Daily sitting time?", ["<4 hours", "4–6 hours", "6–8 hours", "8+ hours"])
    q11 = st.radio("11) Do you smoke?", ["No", "Yes"])
    q12 = st.radio("12) Do you eat late night often?", ["No", "Yes"])

    tips = []
    if st.button("Get my tips"):
        if q1 in ["Never", "1–3 times"]:
            tips.append("Walk 30 minutes/day (even 10 minutes × 3 times helps).")
        if q2 in ["2", "3+"]:
            tips.append("Reduce sugar in tea gradually; replace with water/unsweetened drinks.")
        if q3 == "Large":
            tips.append("Reduce rice portion and add vegetables (gotukola, mukunuwenna, beans, cabbage).")
        if q4 == "Yes":
            tips.append("Family history increases risk — do regular FBS/HbA1c screening.")
        if q5 in ["0-5 hours", "5-6 hours"]:
            tips.append("Aim for 7–8 hours sleep; poor sleep increases insulin resistance.")
        if q6 in ["3–4/week", "Almost daily"]:
            tips.append("Limit fried foods; choose boiled/steamed/grilled options.")
        if q7 in ["<2 portions", "2–3"]:
            tips.append("Increase fruits/vegetables to at least 4–5 portions/day.")
        if q9 == "High":
            tips.append("Manage stress with relaxation, breathing, walking, prayer, or hobbies.")
        if q10 in ["6–8 hours", "8+ hours"]:
            tips.append("Stand up every hour; short walks reduce sedentary risk.")
        if q11 == "Yes":
            tips.append("Stop smoking; it increases insulin resistance and cardiovascular risk.")
        if q12 == "Yes":
            tips.append("Avoid late-night heavy meals; prefer light dinner + earlier timing.")

        tips.append("If BP is high, reduce salt and follow medical advice.")
        tips.append("If symptoms persist, consult a clinician for proper lab tests.")

        st.markdown("### Your tips")
        for t in tips:
            st.write("•", t)

    st.markdown("### Download PDF leaflet")
    name_for_pdf = st.text_input("Name on leaflet (optional)", value="")
    if st.button("Generate PDF leaflet"):
        summary_lines = [
            f"Name: {name_for_pdf if name_for_pdf else 'N/A'}",
            "Tool: Diabetes Risk Analytics Dashboard (Sri Lanka)",
            "Reminder: This is educational analytics, not a diagnosis."
        ]
        pdf = make_leaflet_pdf(summary_lines, tips_lines=tips if tips else ["Maintain healthy diet, exercise, sleep and regular screening."])
        st.download_button(
            label="⬇️ Download PDF leaflet",
            data=pdf,
            file_name="diabetes_awareness_leaflet.pdf",
            mime="application/pdf"
        )

st.caption("© Academic prototype — built for final year project submission.")
