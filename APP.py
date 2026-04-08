import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# =========================
# 1. 页面配置 (Page Config)
# =========================
st.set_page_config(
    page_title="Hepatic Lesion Risk Predictor | Precision Hepatology",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 2. 高级医学期刊风格 CSS
# =========================
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .title-box {
        background: linear-gradient(135deg, #0A2540 0%, #1750A1 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .title-box h1 { margin: 0; font-size: 2.2rem; font-weight: 700; font-family: 'Helvetica Neue', sans-serif;}
    .title-box p { margin-top: 10px; font-size: 1.1rem; opacity: 0.9; }
    .clinical-note {
        background-color: #EBF4FA;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 6px solid #1750A1;
        margin-bottom: 2rem;
        color: #0A2540;
        font-size: 1rem;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    .card-title {
        color: #0A2540;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #64748B;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. 页面头与图片
# =========================
st.markdown(
    """
    <div class="title-box">
        <h1>Machine Learning Framework for Predicting Malignancy Risk in Hepatic Lesions</h1>
        <p>Explainable Artificial Intelligence (XAI) for Pre-biopsy Triage & Decision Support</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="clinical-note">
    <b>Objective:</b> This tool leverages an advanced Gradient Boosting algorithm to predict the malignancy risk of primary hepatic space-occupying lesions (PHSOLs) based on a 9-biomarker signature. <br>
    <b>Note for Clinicians:</b> This is a supplementary decision-support tool intended for research purposes. Final medical decisions regarding biopsy or surgery should always be made by a qualified hepatobiliary specialist in conjunction with imaging (CT/MRI) and comprehensive clinical evaluations.
    </div>
    """,
    unsafe_allow_html=True
)

# 插入横幅图片 (已经修正为正确的 use_column_width)
try:
    if os.path.exists("Fig.png"):
        st.image("Fig.png", use_column_width=True)
    elif os.path.exists("fig.png"):
        st.image("fig.png", use_column_width=True)
except Exception:
    pass

# =========================
# 4. 数据与模型加载
# =========================
MODEL_FILE = "GB.pkl"
DATA_FILE = "Final_Cleaned_Data.xlsx"
FEATURES = ['Mb', 'PIVKA', 'DBIL', 'CL', 'EO', 'GGT', 'Urea', 'AFP', 'RDW-CV']

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_excel(DATA_FILE)
    else:
        dummy_data = {
            'Mb': np.random.uniform(10, 500, 100),
            'PIVKA': np.random.uniform(10, 2000, 100),
            'DBIL': np.random.uniform(1, 50, 100),
            'CL': np.random.uniform(90, 110, 100),
            'EO': np.random.uniform(0.01, 0.5, 100),
            'GGT': np.random.uniform(10, 300, 100),
            'Urea': np.random.uniform(2, 15, 100),
            'AFP': np.random.uniform(1, 1000, 100),
            'RDW-CV': np.random.uniform(11, 18, 100)
        }
        return pd.DataFrame(dummy_data)

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Error loading model `{MODEL_FILE}`. Details: {e}")
    st.stop()

X_f = df[FEATURES] if all(f in df.columns for f in FEATURES) else df

# =========================
# 5. 用户输入面板
# =========================
st.markdown('<div class="card"><div class="card-title">📝 Step 1: Patient Biomarker Input</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3, col1, col2, col3, col1, col2, col3]

input_vals = []
for idx, f in enumerate(FEATURES):
    with columns[idx]:
        if pd.api.types.is_numeric_dtype(X_f[f]):
            min_val = float(X_f[f].min())
            max_val = float(X_f[f].max())
            median_val = float(X_f[f].median())
            
            v = st.number_input(
                f"{f}",
                min_value=0.0,
                max_value=max_val * 10,
                value=median_val,
                help=f"Dataset Reference Range: {min_val:.2f} - {max_val:.2f}"
            )
        else:
            opts = X_f[f].unique().tolist()
            v = st.selectbox(f"{f}", opts)
        input_vals.append(v)
        
st.markdown('</div>', unsafe_allow_html=True)
X_in = pd.DataFrame([input_vals], columns=FEATURES)

# =========================
# 6. 预测与可视化
# =========================
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    predict_btn = st.button("🚀 Run Personalized Risk Assessment", type="primary", use_container_width=True)

if predict_btn:
    st.markdown('<div class="card"><div class="card-title">📊 Step 2: Prediction Results & Interpretation</div>', unsafe_allow_html=True)
    # 侦察代码开始
    expected_features = model.get_booster().feature_names
    st.error(f"⚠️ 破案了！模型内部死记硬背的特征顺序和名称是：{expected_features}")
    st.stop()
    # 侦察代码结束

    prob_pos = model.predict_proba(X_in)[0][1] * 100

    res_c1, res_c2 = st.columns([1.2, 1])

    with res_c1:
        st.markdown('#### 🩺 Clinical Conclusion')
        if prob_pos >= 50:
            st.error("### ⚠️ High Malignancy Risk Detected")
            st.write("The model indicates a **higher likelihood** of the hepatic lesion being malignant (e.g., HCC). Expedited pathological biopsy or aggressive intervention is highly recommended.")
        else:
            st.success("### ✅ Low Risk / Benign Detected")
            st.write("The model indicates a **lower likelihood** of malignancy. The lesion is highly likely to be a benign entity (e.g., hemangioma, FNH). Routine radiographic surveillance may be considered.")
            
        st.info(f"**Calculated Probability of Malignancy:** **{prob_pos:.2f}%**")
        st.write("*Interpretation: A probability closer to 100% indicates higher risk. Please review the SHAP explainability plots below to understand which specific biomarkers are driving this patient's risk profile.*")

    with res_c2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pos,
            number={"suffix": "%", "font": {"size": 40, "color": "#0A2540"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "#1750A1"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "rgba(40, 167, 69, 0.2)"},
                    {"range": [30, 70], "color": "rgba(255, 193, 7, 0.2)"},
                    {"range": [70, 100], "color": "rgba(220, 53, 69, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": prob_pos
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # 7. SHAP 分析
    # =========================
    st.markdown('<div class="card"><div class="card-title">🔍 Step 3: AI Explainability (SHAP Analysis)</div>', unsafe_allow_html=True)
    st.write("The plots below unpack the 'black box' of the AI, showing exactly how each biomarker pushes the patient's risk higher (Red) or lower (Blue) compared to the baseline.")

    try:
        with st.spinner('Calculating SHAP values for personalized explainability...'):
            bg_data = shap.sample(X_f[FEATURES], min(100, len(X_f)))
            
            try:
                explainer = shap.TreeExplainer(model)
                shap_values_raw = explainer.shap_values(X_in)
                if isinstance(shap_values_raw, list):
                    sv_values = shap_values_raw[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    sv_values = shap_values_raw[0]
                    base_val = explainer.expected_value
                    if isinstance(base_val, (list, np.ndarray)):
                         base_val = base_val[0]
                sv_in_plot = shap.Explanation(values=sv_values, base_values=base_val, data=X_in.iloc[0].values, feature_names=FEATURES)
            except Exception:
                explainer = shap.KernelExplainer(model.predict_proba, bg_data)
                shap_values_raw = explainer.shap_values(X_in)
                if isinstance(shap_values_raw, list):
                    sv_values = shap_values_raw[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    sv_values = shap_values_raw[0, :, 1] if shap_values_raw.ndim == 3 else shap_values_raw[0]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                sv_in_plot = shap.Explanation(values=sv_values, base_values=base_val, data=X_in.iloc[0].values, feature_names=FEATURES)

            p1, p2 = st.columns(2)

            with p1:
                st.markdown("**SHAP Waterfall Plot**")
                fig_wf, ax_wf = plt.subplots(figsize=(6, 5), dpi=150)
                shap.plots.waterfall(sv_in_plot, max_display=10, show=False)
                st.pyplot(fig_wf)
                plt.close(fig_wf)

            with p2:
                st.markdown("**Feature Contribution Ranking**")
                abs_sv = np.abs(sv_values)
                total = abs_sv.sum() if abs_sv.sum() != 0 else 1.0
                pct = abs_sv / total * 100

                contrib_df = pd.DataFrame({
                    "Biomarker": FEATURES,
                    "Patient Value": X_in.iloc[0].values,
                    "Effect": ["⬆️ Increased Malignancy Risk" if v > 0 else "⬇️ Decreased Malignancy Risk" for v in sv_values],
                    "Contribution Impact": pct
                }).sort_values("Contribution Impact", ascending=False)

                st.dataframe(
                    contrib_df.style.format({
                        "Patient Value": "{:.2f}",
                        "Contribution Impact": "{:.1f}%"
                    }).background_gradient(subset=['Contribution Impact'], cmap='Reds'),
                    use_container_width=True,
                    hide_index=True
                )
    except Exception as e:
        st.warning(f"⚠️ Could not generate SHAP explanation. Details: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 8. 专业页脚
# =========================
st.markdown(
    """
    <div class="footer">
        <b>Clinical Research Team</b><br>
        <i>Powered by Streamlit | Developed for clinical research and precision hepatology.</i>
    </div>
    """,
    unsafe_allow_html=True
)
