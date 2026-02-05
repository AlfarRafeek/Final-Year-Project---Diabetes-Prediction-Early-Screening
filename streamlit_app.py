import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder


# ---------------------------
# Page settings
# ---------------------------
st.set_page_config(
    page_title="Diabetes Risk Analytics Dashboard (Sri Lanka)",
    layout="wide"
)

st.title("📊 Diabetes Risk Analytics Dashboard (Sri Lanka)")
st.caption("Data analytics dashboard to visualize diabetes risk levels and significant predictive factors (PPS-aligned).")
st.info("Educational analytics only — this tool does not provide a medical diagnosis.")


# ---------------------------
# Constants (match your dataset)
# ---------------------------
TARGET_COL = "Have you ever been diagnosed with diabetes?"
DROP_COLS_IF_EXIST = ["Timestamp"]


# ---------------------------
# Helpers
# ---------------------------
def safe_read_excel(uploaded_file):
    """Read Excel safely. Streamlit Cloud needs openpyxl in requirements.txt."""
    return pd.read_excel(uploaded_file, engine="openpyxl")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning consistent with your Google Form survey."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for c in DROP_COLS_IF_EXIST:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Standardize target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().replace({"Yes": 1, "No": 0})
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Convert numeric candidates (sometimes come as text)
    numeric_candidates = [
        "Waist circumference (cm)",
        "Systolic BP (mmHg)",
        "Diastolic BP (mmHg)",
        "BMI (kg/m²)"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def plot_distribution(df, col, title_prefix="Distribution"):
    """Histogram for numeric columns; bar chart for categorical columns."""
    s = df[col].dropna()

    fig = plt.figure()
    if is_numeric_series(s):
        plt.hist(s, bins=20)
        plt.xlabel(col)
        plt.ylabel("Frequency")
    else:
        vc = s.value_counts().head(15)
        plt.bar(vc.index.astype(str), vc.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")

    plt.title(f"{title_prefix}: {col}")
    plt.tight_layout()
    return fig


def diabetes_rate_by_group(df, group_col):
    """Observed diabetes rate by a grouping column."""
    temp = df[[group_col, TARGET_COL]].dropna()
    g = temp.groupby(group_col)[TARGET_COL].mean().sort_values(ascending=False)
    return g


def correlation_heatmap_numeric(df):
    """Correlation heatmap for numeric columns only."""
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr(numeric_only=True)

    fig = plt.figure()
    plt.imshow(corr.values)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    return fig


def heatmap_from_pivot(piv: pd.DataFrame, title: str, xlabel: str, ylabel: str):
    """Simple heatmap (matplotlib) from a pivot table of rates."""
    fig = plt.figure()
    plt.imshow(piv.values, aspect="auto")
    plt.xticks(range(len(piv.columns)), piv.columns.astype(str), rotation=45, ha="right")
    plt.yticks(range(len(piv.index)), piv.index.astype(str))
    plt.colorbar(label="Diabetes rate")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig


def stacked_rate_bar(df: pd.DataFrame, group_col: str, top_n=10):
    """Stacked bar showing proportions of (0/1) outcome inside each group."""
    ct = pd.crosstab(df[group_col], df[TARGET_COL], normalize="index")

    # keep only top groups by count to avoid extremely long charts
    counts = df[group_col].astype(str).value_counts()
    keep = counts.head(top_n).index
    ct = ct.loc[ct.index.astype(str).isin(keep)]

    p0 = ct.get(0, pd.Series([0] * len(ct), index=ct.index)).values
    p1 = ct.get(1, pd.Series([0] * len(ct), index=ct.index)).values

    fig = plt.figure()
    x = np.arange(len(ct.index))
    plt.bar(x, p0, label="No (0)")
    plt.bar(x, p1, bottom=p0, label="Yes (1)")
    plt.xticks(x, ct.index.astype(str), rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Proportion")
    plt.title(f"Outcome breakdown by {group_col} (top {top_n} groups)")
    plt.legend()
    plt.tight_layout()
    return fig


def mutual_info_importance(df, top_k=15):
    """
    Mutual Information for mixed survey data (categorical + numeric).
    Useful for analytics dashboards: highlights which inputs are most associated with the outcome.
    """
    if TARGET_COL not in df.columns:
        return None

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    # Remove columns that are completely empty
    X = X.loc[:, X.notna().sum() > 0]

    # Separate numeric and categorical
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Fill missing values
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Missing")

    # One-hot encode categoricals
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = ohe.fit_transform(X[cat_cols])
        cat_names = ohe.get_feature_names_out(cat_cols)
    else:
        X_cat = np.zeros((len(X), 0))
        cat_names = []

    X_num = X[num_cols].values if len(num_cols) > 0 else np.zeros((len(X), 0))
    X_all = np.hstack([X_num, X_cat])
    feature_names = list(num_cols) + list(cat_names)

    if X_all.shape[1] == 0:
        return None

    mi = mutual_info_classif(X_all, y, random_state=42)
    mi_df = pd.DataFrame({"feature": feature_names, "mi": mi}).sort_values("mi", ascending=False)
    return mi_df.head(top_k)


def build_awareness_leaflet(summary_lines, tip_lines):
    """Create a short PDF leaflet (educational use only)."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, page_height = A4

    y = page_height - 60
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Diabetes Awareness Leaflet (Sri Lanka)")
    y -= 22

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Summary")
    y -= 18
    pdf.setFont("Helvetica", 10)

    for line in summary_lines:
        pdf.drawString(50, y, f"• {line}")
        y -= 14

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Personal tips")
    y -= 18
    pdf.setFont("Helvetica", 10)

    for tip in tip_lines[:16]:
        if y < 80:
            pdf.showPage()
            y = page_height - 60
            pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y, f"• {tip}")
        y -= 14

    y -= 10
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(50, y, "Disclaimer: Educational analytics only — not a medical diagnosis.")
    pdf.save()

    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------
# Sidebar: data source
# ---------------------------
st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload your Excel survey file (.xlsx)", type=["xlsx"])
st.sidebar.caption("If Streamlit Cloud shows an openpyxl error, add openpyxl to requirements.txt.")

if uploaded is None:
    st.info("Upload your **Diabetes Risk Survey (Responses).xlsx** to begin.")
    st.stop()


# ---------------------------
# Load + clean
# ---------------------------
try:
    df_raw = safe_read_excel(uploaded)
except Exception as e:
    st.error("I couldn't read the Excel file. On Streamlit Cloud, you must have **openpyxl** installed.")
    st.code(str(e))
    st.stop()

df = clean_dataset(df_raw)

if TARGET_COL not in df.columns:
    st.error(f"Target column not found: {TARGET_COL}")
    st.write("Columns detected:", list(df.columns))
    st.stop()


# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

filter_cols = []
for col in ["Age", "Gender", "Occupation", "How would you describe your diet?"]:
    if col in df.columns:
        filter_cols.append(col)

df_f = df.copy()

for col in filter_cols:
    options = ["All"] + sorted(df_f[col].dropna().astype(str).unique().tolist())
    selected = st.sidebar.selectbox(f"{col}", options)
    if selected != "All":
        df_f = df_f[df_f[col].astype(str) == selected]


# ---------------------------
# KPIs
# ---------------------------
total = len(df_f)
pos = int(df_f[TARGET_COL].sum())
neg = int((df_f[TARGET_COL] == 0).sum())
rate = (pos / total) if total > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total responses", f"{total}")
k2.metric("Diabetes (Yes)", f"{pos}")
k3.metric("No Diabetes", f"{neg}")
k4.metric("Diabetes rate", f"{rate * 100:.1f}%")


# ---------------------------
# Tabs (dashboard layout)
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Risk patterns",
    "Significant factors",
    "Data quality",
    "Awareness + Quiz"
])


# ===========================
# TAB 1: Overview
# ===========================
with tab1:
    st.subheader("Overview")

    c1, c2 = st.columns(2)

    with c1:
        fig = plt.figure()
        counts = df_f[TARGET_COL].value_counts().sort_index()
        plt.bar(["No (0)", "Yes (1)"], [counts.get(0, 0), counts.get(1, 0)])
        plt.title("Class distribution (diabetes outcome)")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("**Preview of the filtered dataset**")
        st.dataframe(df_f.head(20), use_container_width=True)

    st.markdown("### Variable distributions (auto chart type)")
    cols_to_show = [c for c in df_f.columns if c != TARGET_COL]
    pick = st.multiselect("Choose variables to plot", cols_to_show, default=cols_to_show[:4])

    for col in pick:
        st.pyplot(plot_distribution(df_f, col))


# ===========================
# TAB 2: Risk patterns (PPS aligned + advanced visuals)
# ===========================
with tab2:
    st.subheader("Risk patterns (risk levels + key variables)")

    left, right = st.columns(2)

    # Diabetes rate by Age / Gender / Diet
    with left:
        for group_col in ["Age", "Gender", "How would you describe your diet?"]:
            if group_col in df_f.columns:
                g = diabetes_rate_by_group(df_f, group_col)
                fig = plt.figure()
                plt.bar(g.index.astype(str), g.values)
                plt.xticks(rotation=45, ha="right")
                plt.ylim(0, 1)
                plt.ylabel("Diabetes rate")
                plt.title(f"Observed diabetes rate by {group_col}")
                plt.tight_layout()
                st.pyplot(fig)

    # Numeric comparisons by class
    with right:
        numeric_cols = [
            c for c in ["BMI (kg/m²)", "Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)"]
            if c in df_f.columns
        ]
        if len(numeric_cols) == 0:
            st.info("No numeric columns found for BMI/Waist/BP comparisons.")
        else:
            chosen = st.selectbox("Numeric variable vs diabetes outcome", numeric_cols)
            fig = plt.figure()
            data0 = df_f[df_f[TARGET_COL] == 0][chosen].dropna()
            data1 = df_f[df_f[TARGET_COL] == 1][chosen].dropna()
            plt.boxplot([data0, data1], labels=["No (0)", "Yes (1)"])
            plt.title(f"{chosen} by diabetes status")
            plt.ylabel(chosen)
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("### Correlation (numeric only)")
    fig_corr = correlation_heatmap_numeric(df_f)
    if fig_corr is None:
        st.info("Not enough numeric columns for a correlation heatmap.")
    else:
        st.pyplot(fig_corr)

    st.markdown("### Symptom prevalence (Yes/No) vs diabetes")
    symptom_cols = [
        "Do you experience frequent urination?",
        "Do you often feel unusually thirsty?",
        "Have you noticed unexplained weight loss or gain?",
        "Do you feel unusually fatigued or tired?",
        "Do you have blurred vision or slow-healing wounds?",
    ]
    symptom_cols = [c for c in symptom_cols if c in df_f.columns]

    if len(symptom_cols) == 0:
        st.info("Symptom columns not found in the uploaded file.")
    else:
        rows = []
        for c in symptom_cols:
            sub = df_f[df_f[c].astype(str).str.strip() == "Yes"]
            if len(sub) > 0:
                rows.append((c, sub[TARGET_COL].mean(), len(sub)))

        if len(rows) > 0:
            sym_df = pd.DataFrame(rows, columns=["Symptom", "Diabetes rate (Yes responders)", "N (Yes responders)"])
            sym_df = sym_df.sort_values("Diabetes rate (Yes responders)", ascending=False)

            fig = plt.figure()
            plt.barh(sym_df["Symptom"][::-1], sym_df["Diabetes rate (Yes responders)"][::-1])
            plt.xlabel("Diabetes rate among 'Yes' answers")
            plt.title("Symptoms linked with higher observed diabetes rate")
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(sym_df, use_container_width=True)
        else:
            st.info("No 'Yes' symptom responses found after filtering.")

    # -----------------------------
    # Advanced visualisations
    # -----------------------------
    st.markdown("## Advanced visualisations")

    # A) Heatmap: Age × Gender diabetes rate
    if "Age" in df_f.columns and "Gender" in df_f.columns:
        piv = df_f.pivot_table(values=TARGET_COL, index="Age", columns="Gender", aggfunc="mean")
        st.pyplot(heatmap_from_pivot(piv, "Diabetes rate heatmap: Age × Gender", "Gender", "Age group"))
    else:
        st.info("Age × Gender heatmap needs both Age and Gender columns.")

    # B) Stacked proportion chart: choose categorical variable
    st.markdown("### Outcome breakdown by category (stacked proportions)")
    cat_candidates = [
        c for c in df_f.columns
        if c != TARGET_COL and not pd.api.types.is_numeric_dtype(df_f[c])
    ]
    cat_candidates = [c for c in cat_candidates if df_f[c].notna().sum() > 0]

    if len(cat_candidates) > 0:
        pick_cat = st.selectbox("Choose a categorical variable", cat_candidates, key="stack_pick")
        st.pyplot(stacked_rate_bar(df_f, pick_cat, top_n=12))
    else:
        st.info("No categorical columns available for stacked breakdown chart.")

    # C) Scatter matrix (numeric only)
    st.markdown("### Numeric relationship matrix (scatter matrix)")
    numeric_cols = [c for c in df_f.select_dtypes(include=["number"]).columns if c != TARGET_COL]
    numeric_cols = [c for c in numeric_cols if df_f[c].dropna().shape[0] > 5]

    if len(numeric_cols) >= 2:
        chosen_cols = st.multiselect(
            "Pick numeric columns (2–4 recommended)",
            numeric_cols,
            default=numeric_cols[:3],
            key="scatter_matrix_cols"
        )[:4]

        if len(chosen_cols) >= 2:
            fig = plt.figure()
            pd.plotting.scatter_matrix(
                df_f[chosen_cols].dropna(),
                figsize=(7, 7),
                diagonal="hist",
                alpha=0.6
            )
            plt.suptitle("Scatter matrix (numeric features)", y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for scatter matrix.")

    # D) Lift-style ranking (example using BMI)
    st.markdown("## Lift-style analysis (ranking by BMI as an example)")
    if "BMI (kg/m²)" in df_f.columns:
        tmp = df_f[["BMI (kg/m²)", TARGET_COL]].dropna().copy()
        if len(tmp) > 20:
            tmp = tmp.sort_values("BMI (kg/m²)", ascending=False)
            tmp["decile"] = pd.qcut(np.arange(len(tmp)), 10, labels=False)

            lift = tmp.groupby("decile")[TARGET_COL].mean().sort_index(ascending=True)
            overall = tmp[TARGET_COL].mean()

            fig = plt.figure()
            plt.plot(range(1, 11), lift.values, marker="o", label="Decile diabetes rate")
            plt.axhline(overall, linestyle="--", label="Overall rate")
            plt.xlabel("Decile (1 = highest BMI group)")
            plt.ylabel("Diabetes rate")
            plt.title("Lift-style view: Diabetes rate by BMI-ranked deciles")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)

            st.caption("This is not a prediction model — it simply shows how risk concentrates when ranking by one factor.")
        else:
            st.info("Not enough BMI data for a decile lift chart.")
    else:
        st.info("BMI column not available for lift chart.")


# ===========================
# TAB 3: Significant factors
# ===========================
with tab3:
    st.subheader("Significant predictive factors (data-driven)")

    st.write(
        "This section uses **Mutual Information** to estimate which variables are most strongly associated with the "
        "diabetes outcome in your survey data (works with both categorical and numeric inputs)."
    )

    mi_df = mutual_info_importance(df_f, top_k=15)

    if mi_df is None or mi_df.empty:
        st.warning("I couldn't compute significant factors (check your filtered data).")
    else:
        fig = plt.figure()
        plt.barh(mi_df["feature"][::-1], mi_df["mi"][::-1])
        plt.xlabel("Mutual Information (higher = stronger association)")
        plt.title("Top factors linked to diabetes outcome")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(mi_df, use_container_width=True)

    st.markdown("### Download cleaned data (useful for appendix/evidence)")
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download cleaned filtered dataset (CSV)",
        data=csv,
        file_name="cleaned_diabetes_survey_filtered.csv",
        mime="text/csv"
    )

    st.markdown("## Segment insights (which groups show the highest observed rate?)")
    st.caption("This is useful for stakeholder reporting: it highlights which combinations show higher observed diabetes rate.")

    segment_cols = [c for c in ["Age", "Gender", "How would you describe your diet?", "How often do you exercise per week?", "Average sleep hours per night"] if c in df_f.columns]
    if len(segment_cols) >= 2:
        col1 = st.selectbox("Segment column 1", segment_cols, index=0, key="seg1")
        col2 = st.selectbox("Segment column 2", segment_cols, index=1, key="seg2")

        seg = (
            df_f.groupby([col1, col2])[TARGET_COL]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Diabetes rate", "count": "N"})
            .sort_values(["Diabetes rate", "N"], ascending=[False, False])
        )
        st.dataframe(seg.head(15), use_container_width=True)
    else:
        st.info("Need at least two segment columns (Age/Gender/Diet/Exercise/Sleep etc.).")

    st.markdown("## Mutual Information grouped by original question")
    if mi_df is not None and not mi_df.empty:
        base = mi_df.copy()

        def base_name(f):
            # Convert one-hot names like "Gender_Male" -> "Gender"
            if "_" in str(f):
                return str(f).split("_")[0]
            return str(f)

        base["question"] = base["feature"].astype(str).apply(base_name)
        grouped = base.groupby("question")["mi"].sum().sort_values(ascending=False).reset_index()

        fig = plt.figure()
        plt.barh(grouped["question"][::-1], grouped["mi"][::-1])
        plt.xlabel("Total MI (summed across categories)")
        plt.title("Question-level importance (grouped MI)")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(grouped.head(15), use_container_width=True)


# ===========================
# TAB 4: Data quality
# ===========================
with tab4:
    st.subheader("Data quality checks")

    missing = df_f.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values detected in the filtered dataset.")
    else:
        st.write("Missing values by column:")
        st.dataframe(
            missing.reset_index().rename(columns={"index": "Column", 0: "Missing"}),
            use_container_width=True
        )

        fig = plt.figure()
        top = missing.head(15)
        plt.bar(top.index.astype(str), top.values)
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Missing count")
        plt.title("Missing values (top 15 columns)")
        plt.tight_layout()
        st.pyplot(fig)



# ===========================
# TAB 5: Awareness + Quiz
# ===========================
with tab5:
    st.subheader("🧠 Awareness Quiz & Prevention Tips")
    st.caption("Quick lifestyle check + practical advice. For awareness and academic use only.")

    # Keep tips in session state so they remain available for PDF download after button clicks
    if "awareness_tips" not in st.session_state:
        st.session_state.awareness_tips = []

    st.markdown("### Short quiz")

    exercise = st.radio("1) How often do you exercise per week?", ["Never", "1–3 times", "3–6 times", "Daily"])
    sugary_drinks = st.radio("2) How many sugary drinks do you have per day?", ["0", "1", "2", "3+"])
    rice_size = st.radio("3) Your usual rice portion size:", ["Small", "Medium", "Large"])
    family_history = st.radio("4) Do you have a family history of diabetes?", ["No", "Yes"])
    sleep = st.radio("5) Average sleep per night:", ["0–5 hours", "5–6 hours", "6–7 hours", "7–8 hours", "8+ hours"])
    fried_food = st.radio("6) How often do you eat fried foods?", ["Rarely", "1–2/week", "3–4/week", "Almost daily"])
    fruits_veg = st.radio("7) Fruits/vegetables per day:", ["<2 portions", "2–3", "4–5", "5+"])
    bp_check = st.radio("8) How often do you check your blood pressure?", ["Never", "Sometimes", "Yearly", "Often"])
    stress = st.radio("9) Your stress level:", ["Low", "Moderate", "High"])
    sitting = st.radio("10) Daily sitting time:", ["<4 hours", "4–6 hours", "6–8 hours", "8+ hours"])
    smoking = st.radio("11) Do you smoke?", ["No", "Yes"])
    late_meals = st.radio("12) Do you often eat heavy meals late at night?", ["No", "Yes"])

    if st.button("Get my tips"):
        tips = []

        if exercise in ["Never", "1–3 times"]:
            tips.append("Try to add more movement (e.g., a 30-minute walk daily, or 10 minutes × 3).")
        if sugary_drinks in ["2", "3+"]:
            tips.append("Cut down sugary drinks gradually; replace with water or unsweetened options.")
        if rice_size == "Large":
            tips.append("Reduce rice portion size and increase vegetables for better balance.")
        if family_history == "Yes":
            tips.append("Family history increases risk—consider regular screening (FBS/HbA1c) if advised.")
        if sleep in ["0–5 hours", "5–6 hours"]:
            tips.append("Aim for 7–8 hours of sleep; poor sleep is linked with higher metabolic risk.")
        if fried_food in ["3–4/week", "Almost daily"]:
            tips.append("Limit fried foods where possible; choose boiled/steamed/grilled foods more often.")
        if fruits_veg in ["<2 portions", "2–3"]:
            tips.append("Increase fruits/vegetables—try to reach around 4–5 portions per day.")
        if stress == "High":
            tips.append("Work on stress management (walking, breathing exercises, hobbies, mindfulness).")
        if sitting in ["6–8 hours", "8+ hours"]:
            tips.append("Break up sitting time—stand and move at least once every hour.")
        if smoking == "Yes":
            tips.append("Consider quitting smoking; it increases long-term cardiovascular and metabolic risk.")
        if late_meals == "Yes":
            tips.append("Try to avoid heavy late dinners; a lighter earlier meal is usually better.")

        # General reminders (always shown)
        tips.append("If blood pressure is elevated, reduce salt and follow medical guidance.")
        tips.append("If symptoms are concerning or persistent, consult a clinician for proper lab testing.")

        st.session_state.awareness_tips = tips

        st.markdown("### ✅ Your personalised tips")
        for item in tips:
            st.write("•", item)

    st.divider()
    st.subheader("⬇️ Download leaflet (PDF)")

    person_name = st.text_input("Name (optional)", value="")
    if st.button("Generate PDF leaflet"):
        summary_lines = [
            f"Name: {person_name if person_name else 'N/A'}",
            f"Filtered dataset size in dashboard: {len(df_f)}",
            "This leaflet is based on quiz answers (not a diagnosis)."
        ]

        tips_for_pdf = st.session_state.awareness_tips
        if not tips_for_pdf:
            tips_for_pdf = ["Maintain a healthy diet, stay active, sleep well, and do regular screening if at risk."]

        pdf_bytes = build_awareness_leaflet(summary_lines, tips_for_pdf)

        st.download_button(
            "⬇️ Download PDF leaflet",
            data=pdf_bytes,
            file_name="diabetes_awareness_leaflet.pdf",
            mime="application/pdf"
        )

st.caption("© Academic prototype — built for final year project submission.")
