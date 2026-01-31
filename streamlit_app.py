import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# ---------------------------
# Constants (match your dataset)
# ---------------------------
TARGET_COL = "Have you ever been diagnosed with diabetes?"
DROP_COLS_IF_EXIST = ["Timestamp"]


# ---------------------------
# Helpers
# ---------------------------
def safe_read_excel(uploaded_file):
    """Read Excel safely (Streamlit Cloud needs openpyxl in requirements.txt)."""
    return pd.read_excel(uploaded_file, engine="openpyxl")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning consistent with your survey."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for c in DROP_COLS_IF_EXIST:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Standardize target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = (
            df[TARGET_COL].astype(str).str.strip().replace({"Yes": 1, "No": 0})
        )
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Convert true numeric columns (if any come as text)
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
    """Histogram for numeric; bar chart for categorical."""
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
    """Return diabetes rate table by any categorical column."""
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


def mutual_info_importance(df, top_k=15):
    """
    Significant factors using Mutual Information (works for mixed types).
    This is strong for a Data Analytics dashboard: highlights variables linked to diabetes.
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

    # Fill missing
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Missing")

    # OneHot for categoricals
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


# ---------------------------
# Sidebar: data source
# ---------------------------
st.sidebar.header("Data Source")

uploaded = st.sidebar.file_uploader("Upload your Excel survey file (.xlsx)", type=["xlsx"])

st.sidebar.markdown("**Tip:** If Streamlit Cloud shows `openpyxl` error, add it to requirements.txt.")

if uploaded is None:
    st.info("Upload your **Diabetes Risk Survey (Responses).xlsx** to start.")
    st.stop()


# ---------------------------
# Load + clean
# ---------------------------
try:
    df_raw = safe_read_excel(uploaded)
except Exception as e:
    st.error("Could not read Excel. On Streamlit Cloud, you must have **openpyxl** installed.")
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
k4.metric("Diabetes rate", f"{rate*100:.1f}%")


# ---------------------------
# Tabs (dashboard layout)
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Risk patterns",
    "Significant factors",
    "Data quality"
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
        plt.title("Class Distribution (Diabetes target)")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("**Preview of filtered dataset**")
        st.dataframe(df_f.head(20), use_container_width=True)

    # Distributions (auto numeric/categorical)
    st.markdown("### Variable distributions (auto chart type)")
    cols_to_show = [c for c in df_f.columns if c != TARGET_COL]
    pick = st.multiselect("Choose variables to plot", cols_to_show, default=cols_to_show[:4])

    for col in pick:
        st.pyplot(plot_distribution(df_f, col))


# ===========================
# TAB 2: Risk patterns (PPS aligned)
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
                plt.title(f"Diabetes rate by {group_col}")
                plt.tight_layout()
                st.pyplot(fig)

    # Numeric comparisons by class
    with right:
        numeric_cols = [c for c in ["BMI (kg/m²)", "Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)"] if c in df_f.columns]
        if len(numeric_cols) == 0:
            st.info("No numeric columns found for BMI/Waist/BP comparisons.")
        else:
            chosen = st.selectbox("Numeric variable vs diabetes", numeric_cols)
            fig = plt.figure()
            # simple boxplot by class
            data0 = df_f[df_f[TARGET_COL] == 0][chosen].dropna()
            data1 = df_f[df_f[TARGET_COL] == 1][chosen].dropna()
            plt.boxplot([data0, data1], labels=["No (0)", "Yes (1)"])
            plt.title(f"{chosen} by Diabetes status")
            plt.ylabel(chosen)
            plt.tight_layout()
            st.pyplot(fig)

    # Correlation heatmap numeric only
    st.markdown("### Correlation (numeric only)")
    fig_corr = correlation_heatmap_numeric(df_f)
    if fig_corr is None:
        st.info("Not enough numeric columns for a correlation heatmap.")
    else:
        st.pyplot(fig_corr)

    # Symptom prevalence plot
    st.markdown("### Symptom prevalence (Yes/No) vs Diabetes")
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
        # rate of diabetes among those who answered Yes
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
            plt.title("Symptoms associated with higher diabetes rate")
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(sym_df, use_container_width=True)
        else:
            st.info("No 'Yes' symptom responses found after filtering.")


# ===========================
# TAB 3: Significant factors
# ===========================
with tab3:
    st.subheader("Significant predictive factors (data-driven)")
    st.write(
        "This uses **Mutual Information** to estimate which variables are most associated with the diabetes outcome "
        "(works with mixed categorical + numeric survey data)."
    )

    mi_df = mutual_info_importance(df_f, top_k=15)

    if mi_df is None or mi_df.empty:
        st.warning("Could not compute significant factors (check data after filtering).")
    else:
        fig = plt.figure()
        plt.barh(mi_df["feature"][::-1], mi_df["mi"][::-1])
        plt.xlabel("Mutual Information (higher = stronger association)")
        plt.title("Top factors linked to diabetes outcome")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(mi_df, use_container_width=True)

    st.markdown("### Download cleaned data (for appendix / evidence)")
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download cleaned filtered dataset (CSV)",
        data=csv,
        file_name="cleaned_diabetes_survey_filtered.csv",
        mime="text/csv"
    )


# ===========================
# TAB 4: Data quality
# ===========================
with tab4:
    st.subheader("Data quality checks")

    # Missing values
    missing = df_f.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values detected in filtered dataset.")
    else:
        st.write("Missing values by column:")
        st.dataframe(missing.reset_index().rename(columns={"index": "Column", 0: "Missing"}), use_container_width=True)

        fig = plt.figure()
        top = missing.head(15)
        plt.bar(top.index.astype(str), top.values)
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Missing count")
        plt.title("Missing values (top 15 columns)")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("### Basic validation notes (for your report)")
    st.info(
        "• Categorical ranges (Age/Height/Weight/Sleep) should be visualized as bar charts, not histograms.\n"
        "• Numeric measurements (BMI/BP/Waist) can be summarized using boxplots and correlation.\n"
        "• Significant factors are computed from the survey itself (Mutual Information), supporting PPS dashboard goals."
    )
