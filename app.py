#app.py — CardioAI main entry point.
#Run with: streamlit run app.py

import streamlit as st
import os, sys, pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

from models.model_utils import load_all_models, load_scaler, load_metrics, MODEL_DIR
from tabs import tab_predict, tab_analysis, tab_advisor

st.set_page_config(
    page_title="CardioAI — Cardiovascular Disease Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #0a0f1e;
        color: #e2e8f0;
    }
    .stApp { background-color: #0a0f1e; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2e 0%, #0a0f1e 100%) !important;
        border-right: 1px solid #1e293b;
    }

    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        text-decoration: none !important;
        color: inherit !important;
        pointer-events: none !important;
    }
    h1 .header-anchor, h2 .header-anchor,
    h3 .header-anchor, h4 .header-anchor {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 16px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: #4F8EF7 !important;
        color: white !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        transition: background 0.2s ease !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        box-shadow: none !important;
        transform: none !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F8EF7, #7c3aed) !important;
        border: none !important;
        color: white !important;
        padding: 0.65rem 1.5rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb, #5b21b6) !important;
        box-shadow: none !important;
        transform: none !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #334155 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    [data-baseweb="select"] > div { background: #1e293b !important; border-color: #334155 !important; }
    [data-testid="stNumberInput"] input { background: #1e293b !important; border-color: #334155 !important; }
    .stInfo    { background: #1e3a5f !important; border: 1px solid #3b82f6 !important; border-radius: 10px !important; }
    .stWarning { background: #2d1f00 !important; border: 1px solid #f59e0b !important; border-radius: 10px !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

#Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <h1 style='color:#4F8EF7; font-size:1.4rem; margin:0.5rem 0;'>CardioAI</h1>
        <p style='color:#64748b; font-size:0.85rem; margin:0;'>Cardiovascular Disease Prediction</p>
    </div>
    <hr style='border-color:#1e293b; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#94a3b8; font-size:0.88rem; line-height:1.7;'>
        <strong style='color:#e2e8f0; font-size:0.95rem;'>About This Project</strong><br><br>
        CardioAI is a machine learning-powered application designed to predict the
        likelihood of cardiovascular disease based on a patient's clinical and lifestyle data.<br><br>
        <strong style='color:#4F8EF7;'>7 machine learning models</strong> are trained on a
        real-world dataset of 70,000 patient records. Their predictions are combined through
        an ensemble vote for a reliable overall verdict.<br><br>
        <strong style='color:#e2e8f0;'>Models:</strong><br>
        <span style='color:#4F8EF7;'>Logistic Regression</span> &nbsp;·&nbsp;
        <span style='color:#34D399;'>Random Forest</span> &nbsp;·&nbsp;
        <span style='color:#F59E0B;'>SVM</span> &nbsp;·&nbsp;
        <span style='color:#A78BFA;'>KNN</span><br>
        <span style='color:#F87171;'>Gradient Boosting</span> &nbsp;·&nbsp;
        <span style='color:#38BDF8;'>Naive Bayes</span> &nbsp;·&nbsp;
        <span style='color:#8A0991;'>Neural Network</span><br><br>
        An integrated <strong style='color:#a78bfa;'>AI Health Advisor</strong> powered by Gemma 2 (2B)
        interprets results and gives personalised health advice.<br><br>
        <strong style='color:#e2e8f0;'>Input features:</strong> age, gender, height, weight,
        blood pressure, cholesterol, glucose, smoking, alcohol intake.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='color:#475569; font-size:0.75rem;'>
        <em>For educational purposes only.
        Not a substitute for professional medical advice.</em>
    </div>
    """, unsafe_allow_html=True)

#Load resources (pkls always present/No fallbacks)
@st.cache_resource(show_spinner="Loading models...")
def load_resources():
    return load_all_models(), load_scaler()

def get_metrics():
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = load_metrics()
    return st.session_state["metrics"]

models, scaler = load_resources()
metrics        = get_metrics()

#Header
st.markdown("""
<div style='text-align:center; padding:1.5rem 0 1rem 0;'>
    <h1 style='font-size:2.2rem; background:linear-gradient(135deg,#4F8EF7,#a78bfa);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
        CardioAI Prediction System
    </h1>
    <p style='color:#64748b; margin:0.5rem 0 0 0;'>
        Multi-model cardiovascular disease detection with AI-powered health advice
    </p>
</div>
""", unsafe_allow_html=True)

#Tabs
tab1, tab2, tab3 = st.tabs([
    "  Prediction  ",
    "  Analysis  ",
    "  AI Advisor  "
])

with tab1:
    tab_predict.render(models, scaler)

with tab2:
    tab_analysis.render(metrics)

with tab3:
    tab_advisor.render()