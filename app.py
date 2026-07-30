import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import os
from fpdf import FPDF

# ======================================================================
# BACKEND
# Uses the model, scaler, and column order exactly as saved by the
# notebook — no reconstruction needed, everything was saved correctly.
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "heart_disease_scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "heart_disease_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "cardiac_arrest.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")

TARGET_COLUMN = "target"

# Human-readable option labels -> underlying numeric code the model expects
CP_OPTIONS = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
RESTECG_OPTIONS = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE_OPTIONS = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL_OPTIONS = {"Unknown": 0, "Fixed defect": 1, "Normal": 2, "Reversible defect": 3}
YES_NO = {"No": 0, "Yes": 1}
SEX_OPTIONS = {"Female": 0, "Male": 1}

# Test-set metrics for this exact model, evaluated on the same 80/20 split
# used in the notebook (random_state=42).
MODEL_INFO = {"accuracy": 0.836, "precision": 0.788, "recall": 0.897, "f1": 0.839, "roc_auc": 0.879, "n_estimators": 100}

# Friendly display names + normal clinical reference ranges (for context only)
FEATURE_LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting blood pressure",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise angina", "oldpeak": "ST depression",
    "slope": "ST slope", "ca": "Major vessels", "thal": "Thalassemia",
}


def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop_duplicates()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_columns():
    return joblib.load(COLUMNS_PATH)


def encode_input(raw_input: dict, feature_names: list) -> pd.DataFrame:
    return pd.DataFrame([[raw_input[c] for c in feature_names]], columns=feature_names)


def predict_risk(model, scaler, X: pd.DataFrame):
    """Returns (predicted_class, probability_of_disease, confidence)."""
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])
    tree_preds = np.array([t.predict(X_scaled)[0] for t in model.estimators_])
    agreement = (tree_preds == pred).mean() * 100
    return pred, proba, agreement


def get_risk_level(proba: float):
    if proba >= 0.66:
        return {"label": "High risk", "severity": "high",
                "advice": "Strong indicators of heart disease risk. Please consult a cardiologist promptly for a full clinical evaluation.",
                "steps": ["Schedule a cardiology consultation within 1-2 weeks", "Bring this report and recent lab results", "Avoid strenuous activity until cleared by a doctor"]}
    elif proba >= 0.33:
        return {"label": "Moderate risk", "severity": "medium",
                "advice": "Some risk indicators are present. A check-up with a physician is recommended to assess further.",
                "steps": ["Book a routine check-up in the next month", "Monitor blood pressure and cholesterol", "Discuss lifestyle changes with your doctor"]}
    else:
        return {"label": "Low risk", "severity": "low",
                "advice": "Low likelihood of heart disease based on these inputs. Continue regular health check-ups.",
                "steps": ["Maintain a heart-healthy diet and regular exercise", "Continue annual check-ups", "Re-screen if new symptoms appear"]}


def get_top_risk_factors(model, feature_names, raw_input, raw_df, top_n=3):
    """Combines the model's global feature importances with how far this
    patient's values deviate from the dataset average, to surface the
    factors most likely driving this specific prediction."""
    importances = dict(zip(feature_names, model.feature_importances_))
    factors = []
    for col in feature_names:
        avg = raw_df[col].mean()
        std = raw_df[col].std() or 1
        deviation = abs(raw_input[col] - avg) / std
        score = importances[col] * (0.5 + deviation)
        factors.append((col, score, raw_input[col], avg))
    factors.sort(key=lambda x: -x[1])
    return factors[:top_n]


def append_to_history(record: dict):
    df_row = pd.DataFrame([record])
    if os.path.exists(HISTORY_PATH):
        df_row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORY_PATH, mode="w", header=True, index=False)


def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def build_pdf_report(inputs_display: dict, proba: float, risk: dict, confidence: float, top_factors_display: list) -> bytes:
    pdf = FPDF(format="A4")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 65, 90)
    pdf.cell(0, 12, "Clinical Risk Assessment Report", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    fill = {"high": (250, 227, 224), "medium": (253, 243, 223), "low": (227, 242, 232)}[risk["severity"]]
    pdf.set_fill_color(*fill)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Risk Level: {risk['label']} ({proba*100:.1f}% probability)", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 9, f"Model Confidence: {confidence:.1f}%", ln=True, fill=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, risk["advice"])
    pdf.ln(4)

    if top_factors_display:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 65, 90)
        pdf.cell(0, 10, "Key Contributing Factors", ln=True)
        pdf.set_draw_color(180, 180, 180)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(20, 20, 20)
        for line in top_factors_display:
            pdf.cell(0, 8, f"- {line}", ln=True)
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 65, 90)
    pdf.cell(0, 10, "Patient Input Details", ln=True)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(20, 20, 20)
    for key, value in inputs_display.items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(70, 8, f"{key}:")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"{value}", ln=True)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 40, 30)
    pdf.multi_cell(0, 5, "Disclaimer: This report is generated by a machine learning model for educational/screening purposes only. It is NOT a medical diagnosis. Please consult a qualified healthcare professional for any medical concerns.")

    return bytes(pdf.output())


# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Clinical Diagnostics AI | Heart Disease Risk Engine",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FAFBFC", "figure.facecolor": "#FAFBFC"})

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

:root {
    --brand-dark: #0B3556;
    --brand: #135C8C;
    --brand-light: #2F8BC7;
    --brand-pale: #E4F1FA;
    --danger: #C0392B;
    --warn: #B9770E;
    --ok: #1E8449;
    --ink: #1B1E22;
    --muted: #62686F;
    --surface: #FFFFFF;
    --border: #E1E6EB;
}

.stApp { background: linear-gradient(180deg,#F8FAFC 0%, #EEF3F7 100%); }

.hero {
    background: linear-gradient(120deg, var(--brand-dark) 0%, var(--brand) 55%, var(--brand-light) 100%);
    border-radius: 18px; padding: 28px 32px; color: white; margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(11,53,86,0.18);
}
.hero h1 { margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p { margin: 6px 0 0; opacity: 0.9; font-size: 0.95rem; }

.disclaimer {
    background: #FFF6E5; border: 1px solid #F0D9A8; border-radius: 12px;
    padding: 12px 16px; font-size: 0.82rem; color: #6B4E12; margin-bottom: 1.4rem;
    display: flex; gap: 10px; align-items: flex-start;
}

.metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 1px 3px rgba(20,20,20,0.04);
}
.metric-card .label { font-size: 0.76rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.metric-card .value { font-size: 1.65rem; font-weight: 800; color: var(--ink); }
.metric-card .sub { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

.result-hero {
    border-radius: 18px; padding: 28px; color: white; text-align: center;
    box-shadow: 0 10px 28px rgba(20,20,20,0.16);
}
.result-high   { background: linear-gradient(135deg, #7A1F13, #C0392B); }
.result-medium { background: linear-gradient(135deg, #7A5A0C, #B9770E); }
.result-low    { background: linear-gradient(135deg, #145A32, #1E8449); }
.result-hero .tag { font-size: 0.8rem; opacity: 0.88; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600; }
.result-hero .num { font-size: 2.6rem; font-weight: 800; margin: 6px 0; letter-spacing: -0.02em; }
.result-hero .stats { display:flex; justify-content:center; gap: 28px; margin-top: 10px; font-size: 0.85rem; opacity: 0.92; }

.gauge-track { background: rgba(255,255,255,0.25); border-radius: 999px; height: 10px; margin-top: 14px; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 999px; background: rgba(255,255,255,0.95); }

.rec-card { border-radius: 14px; padding: 18px 20px; margin-top: 16px; display: flex; gap: 14px; align-items: flex-start; border: 1px solid transparent; }
.rec-high   { background: #FCECE5; border-color: #F0B79C; }
.rec-medium { background: #FDF3DF; border-color: #F0CD8F; }
.rec-low    { background: #E9F7EF; border-color: #A9DFBF; }
.rec-card .rec-icon { font-size: 1.6rem; line-height: 1; }
.rec-card h4 { margin: 0 0 4px; font-size: 1.02rem; font-weight: 700; color: var(--ink); }
.rec-card p { margin: 0; font-size: 0.87rem; color: #4A4A44; }

.factor-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 14px; border-radius: 10px; background: var(--surface); border: 1px solid var(--border); margin-bottom: 6px; font-size: 0.85rem; }

.steps-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px; margin-top: 14px; }
.steps-card li { font-size: 0.87rem; color: var(--ink); margin-bottom: 6px; }

.section-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: 6px 0 12px; display:flex; align-items:center; gap:8px; }
.section-sub { font-size: 0.82rem; color: var(--muted); margin-top: -8px; margin-bottom: 14px; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B3556 0%, #135C8C 100%); }
section[data-testid="stSidebar"] * { color: #E9F3FA !important; }

.stButton>button, .stDownloadButton>button {
    background: var(--brand); color: white; border-radius: 10px; border: none;
    font-weight: 600; padding: 0.55rem 1rem; transition: all 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover { background: var(--brand-dark); transform: translateY(-1px); }

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; }

footer, #MainMenu { visibility: hidden; }
.app-footer { text-align:center; color: var(--muted); font-size: 0.78rem; padding: 18px 0 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================================
# LOAD MODEL & DATA
# ======================================================================
model = load_model()
scaler = load_scaler()
feature_names = load_columns()
raw_df = load_raw_data()

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="font-size:1.8rem;">🫀</div>
            <div>
                <div style="font-weight:800; font-size:1.05rem; line-height:1.1;">Clinical Diagnostics AI</div>
                <div style="font-size:0.72rem; opacity:0.75;">Heart Disease Risk Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin:10px 0 16px;'>", unsafe_allow_html=True)
    st.caption(
        "Estimates heart disease risk from clinical measurements and "
        "flags cases that may need further medical evaluation."
    )
    st.markdown(
        "<div style='font-size:0.72rem; opacity:0.6; margin-top:20px;'>v1.1 · For educational/screening use only</div>",
        unsafe_allow_html=True,
    )

# ======================================================================
# HERO HEADER
# ======================================================================
st.markdown(
    """
    <div class="hero">
        <h1>🫀 Clinical Disease Diagnostics Engine</h1>
        <p>AI-assisted heart disease risk screening from clinical measurements.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
        <span style="font-size:1.1rem;">⚠️</span>
        <span><b>Medical disclaimer:</b> This tool provides an educational risk estimate only and is
        <b>not a medical diagnosis</b>. Always consult a qualified healthcare professional for any
        health concerns or before making medical decisions.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_dashboard, tab_batch, tab_history = st.tabs(
    ["🔮  Predict", "📊  Dashboard", "📁  Batch Prediction", "🕘  History"]
)

# ======================================================================
# PREDICT TAB
# ======================================================================
with tab_predict:
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="section-title">🩺 Patient Details</div>', unsafe_allow_html=True)
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=1, max_value=120, value=54, step=1)
                sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()))
                cp_label = st.selectbox("Chest pain type", list(CP_OPTIONS.keys()))
                trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=60, max_value=260, value=130, step=1, help="Normal resting range: approximately 90-120 mm Hg")
                chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=650, value=246, step=1, help="Desirable: under 200 mg/dl")
                fbs_label = st.selectbox("Fasting blood sugar > 120 mg/dl", list(YES_NO.keys()))
                restecg_label = st.selectbox("Resting ECG results", list(RESTECG_OPTIONS.keys()))
            with c2:
                thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=250, value=150, step=1, help="Typical max HR estimate: roughly 220 minus age")
                exang_label = st.selectbox("Exercise-induced angina", list(YES_NO.keys()))
                oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
                slope_label = st.selectbox("Slope of peak exercise ST segment", list(SLOPE_OPTIONS.keys()))
                ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3, 4])
                thal_label = st.selectbox("Thalassemia", list(THAL_OPTIONS.keys()), index=2)

            submitted = st.form_submit_button("🔮  Predict Risk", width="stretch")

    with right:
        st.markdown('<div class="section-title">📈 Risk Assessment</div>', unsafe_allow_html=True)

        if not submitted:
            st.markdown(
                """
                <div style="border:1px dashed var(--border); border-radius:14px; padding:40px 20px; text-align:center; color:var(--muted); background:var(--surface);">
                    <div style="font-size:2rem;">🫀</div>
                    <p style="margin-top:8px; font-size:0.88rem;">Fill in the patient details and click <b>Predict Risk</b> to see results here.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            raw_input = {
                "age": age,
                "sex": SEX_OPTIONS[sex_label],
                "cp": CP_OPTIONS[cp_label],
                "trestbps": trestbps,
                "chol": chol,
                "fbs": YES_NO[fbs_label],
                "restecg": RESTECG_OPTIONS[restecg_label],
                "thalach": thalach,
                "exang": YES_NO[exang_label],
                "oldpeak": oldpeak,
                "slope": SLOPE_OPTIONS[slope_label],
                "ca": ca,
                "thal": THAL_OPTIONS[thal_label],
            }

            X = encode_input(raw_input, feature_names)
            pred, proba, confidence = predict_risk(model, scaler, X)
            risk = get_risk_level(proba)
            gauge_pct = round(proba * 100, 1)

            st.markdown(
                f"""
                <div class="result-hero result-{risk['severity']}">
                    <div class="tag">Predicted disease probability</div>
                    <div class="num">{gauge_pct:.1f}%</div>
                    <div class="stats">
                        <span>🧬 {risk['label']}</span>
                        <span>🎯 {confidence:.0f}% model confidence</span>
                    </div>
                    <div class="gauge-track"><div class="gauge-fill" style="width:{gauge_pct}%;"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            icons = {"high": "🆘", "medium": "🟡", "low": "✅"}
            sev_class = f"rec-{risk['severity']}"
            st.markdown(
                f"""
                <div class="rec-card {sev_class}">
                    <div class="rec-icon">{icons[risk['severity']]}</div>
                    <div>
                        <h4>{risk['label']}</h4>
                        <p>{risk['advice']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            top_factors = get_top_risk_factors(model, feature_names, raw_input, raw_df, top_n=3)
            st.markdown('<div style="margin-top:18px; font-weight:700; font-size:0.92rem;">Key contributing factors</div>', unsafe_allow_html=True)
            top_factors_display = []
            for col, score, patient_val, avg_val in top_factors:
                direction = "above" if patient_val > avg_val else "below"
                line = f"{FEATURE_LABELS.get(col, col)}: {patient_val} ({direction} average of {avg_val:.1f})"
                top_factors_display.append(line)
                st.markdown(
                    f"""<div class="factor-row"><span>{FEATURE_LABELS.get(col, col)}</span>
                    <span style="color:var(--muted);">{patient_val} · {direction} avg ({avg_val:.1f})</span></div>""",
                    unsafe_allow_html=True,
                )

            steps_html = "".join(f"<li>{s}</li>" for s in risk["steps"])
            st.markdown(
                f"""<div class="steps-card"><div style="font-weight:700; font-size:0.92rem; margin-bottom:8px;">Recommended next steps</div>
                <ul style="margin:0; padding-left:18px;">{steps_html}</ul></div>""",
                unsafe_allow_html=True,
            )

            display_inputs = {
                "Age": age, "Sex": sex_label, "Chest pain type": cp_label,
                "Resting BP": f"{trestbps} mm Hg", "Cholesterol": f"{chol} mg/dl",
                "Fasting blood sugar > 120": fbs_label, "Resting ECG": restecg_label,
                "Max heart rate": thalach, "Exercise angina": exang_label,
                "ST depression": oldpeak, "ST slope": slope_label,
                "Major vessels": ca, "Thalassemia": thal_label,
            }

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **display_inputs,
                "Probability (%)": gauge_pct,
                "Risk Level": risk["label"],
                "Model Confidence (%)": round(confidence, 1),
            }
            append_to_history(record)

            pdf_bytes = build_pdf_report(display_inputs, proba, risk, confidence, top_factors_display)
            st.download_button(
                "⬇️  Download PDF Report",
                data=pdf_bytes,
                file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                width="stretch",
            )

# ======================================================================
# DASHBOARD TAB
# ======================================================================
with tab_dashboard:
    st.markdown('<div class="section-title">📊 Clinical Data Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Insights derived from the training dataset.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total Patients", f"{len(raw_df):,}", "training samples"),
        ("Disease Prevalence", f"{raw_df[TARGET_COLUMN].mean()*100:.1f}%", "of dataset"),
        ("Avg Age", f"{raw_df['age'].mean():.0f} yrs", "all patients"),
        ("Avg Cholesterol", f"{raw_df['chol'].mean():.0f} mg/dl", "all patients"),
    ]
    for col, (label, value, sub) in zip([k1, k2, k3, k4], kpis):
        col.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div>
            <div class="value">{value}</div><div class="sub">{sub}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Disease Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        counts = raw_df[TARGET_COLUMN].value_counts().sort_index()
        sns.barplot(x=["No Disease", "Disease"], y=counts.values, palette=["#1E8449", "#C0392B"], ax=ax)
        ax.set_ylabel("Patients")
        st.pyplot(fig)

    with c2:
        st.markdown("**Age Distribution by Disease Status**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.histplot(data=raw_df, x="age", hue=TARGET_COLUMN, kde=True, bins=20, palette=["#1E8449", "#C0392B"], ax=ax)
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Max Heart Rate vs Age**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.scatterplot(data=raw_df, x="age", y="thalach", hue=TARGET_COLUMN, palette=["#1E8449", "#C0392B"], alpha=0.7, ax=ax)
        st.pyplot(fig)

    with c4:
        st.markdown("**Chest Pain Type by Disease Status**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.countplot(data=raw_df, x="cp", hue=TARGET_COLUMN, palette=["#1E8449", "#C0392B"], ax=ax)
        ax.set_xlabel("Chest pain type (0-3)")
        st.pyplot(fig)

    st.markdown("**Feature Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(raw_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    st.markdown("<hr style='border-color:var(--border); margin:24px 0 18px;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌲 Model Card</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (label, value) in zip(
        [m1, m2, m3, m4, m5],
        [
            ("Algorithm", "Random Forest"),
            ("Accuracy", f"{MODEL_INFO['accuracy']*100:.1f}%"),
            ("Precision", f"{MODEL_INFO['precision']*100:.1f}%"),
            ("Recall", f"{MODEL_INFO['recall']*100:.1f}%"),
            ("ROC-AUC", f"{MODEL_INFO['roc_auc']:.3f}"),
        ],
    ):
        col.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div><div class="value" style="font-size:1.2rem;">{value}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("**Top Predictive Features**")
    imp_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(c, c) for c in feature_names],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    st.bar_chart(imp_df.set_index("Feature")["Importance"], color="#135C8C")

# ======================================================================
# BATCH PREDICTION TAB
# ======================================================================
with tab_batch:
    st.markdown('<div class="section-title">📁 Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">Upload a CSV with columns: <code>{", ".join(feature_names)}</code> (raw numeric-coded values, same format as the training data).</div>',
        unsafe_allow_html=True,
    )

    bc1, bc2 = st.columns([1, 2])
    with bc1:
        template = raw_df[feature_names].head(3)
        st.download_button(
            "⬇️  Sample Template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="batch_template.csv",
            mime="text/csv",
            width="stretch",
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in feature_names if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                probs, risks, confs = [], [], []
                for _, row in batch_df.iterrows():
                    X = pd.DataFrame([[row[c] for c in feature_names]], columns=feature_names)
                    pred, proba, conf = predict_risk(model, scaler, X)
                    risk = get_risk_level(proba)
                    probs.append(round(proba * 100, 1))
                    risks.append(risk["label"])
                    confs.append(round(conf, 1))

                batch_df["Probability (%)"] = probs
                batch_df["Risk Level"] = risks
                batch_df["Model Confidence (%)"] = confs

                m1, m2, m3 = st.columns(3)
                m1.markdown(f"""<div class="metric-card"><div class="label">Rows Processed</div><div class="value">{len(batch_df)}</div></div>""", unsafe_allow_html=True)
                m2.markdown(f"""<div class="metric-card"><div class="label">Avg Probability</div><div class="value">{np.mean(probs):.1f}%</div></div>""", unsafe_allow_html=True)
                m3.markdown(f"""<div class="metric-card"><div class="label">High Risk Count</div><div class="value">{risks.count('High risk')}</div></div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
                st.dataframe(batch_df, width="stretch")

                st.download_button(
                    "⬇️  Download Results CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"batch_risk_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ======================================================================
# HISTORY TAB
# ======================================================================
with tab_history:
    st.markdown('<div class="section-title">🕘 Prediction History</div>', unsafe_allow_html=True)

    hist_df = load_history()
    if hist_df.empty:
        st.markdown(
            """
            <div style="border:1px dashed var(--border); border-radius:14px; padding:40px 20px; text-align:center; color:var(--muted); background:var(--surface);">
                <div style="font-size:2rem;">📭</div>
                <p style="margin-top:8px; font-size:0.88rem;">No predictions logged yet. Make one on the Predict tab.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        h1, h2, h3 = st.columns(3)
        h1.markdown(f"""<div class="metric-card"><div class="label">Total Predictions</div><div class="value">{len(hist_df)}</div></div>""", unsafe_allow_html=True)
        h2.markdown(f"""<div class="metric-card"><div class="label">Avg Probability</div><div class="value">{hist_df['Probability (%)'].mean():.1f}%</div></div>""", unsafe_allow_html=True)
        h3.markdown(f"""<div class="metric-card"><div class="label">Most Common Risk</div><div class="value" style="font-size:1.1rem;">{hist_df['Risk Level'].mode()[0]}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.dataframe(hist_df.sort_values("timestamp", ascending=False), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Predicted Probability Over Time**")
            st.line_chart(hist_df.set_index("timestamp")["Probability (%)"], color="#135C8C")
        with c2:
            st.markdown("**Risk Level Breakdown**")
            st.bar_chart(hist_df["Risk Level"].value_counts(), color="#C0392B")

        st.download_button(
            "⬇️  Download Full History CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

st.markdown(
    '<div class="app-footer">🫀 Clinical Diagnostics AI · Built with Streamlit &amp; scikit-learn · For educational/screening use only — not a substitute for professional medical advice</div>',
    unsafe_allow_html=True,
)
