#Ollama-powered local AI health advice (gemma2:2b).

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.llm_advisor import stream_llm_advice, _check_ollama

RESET_KEYS = ["patient_data", "predictions", "ensemble", "llm_advice"]


def render():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
         padding:2rem; border-radius:16px; margin-bottom:2rem;
         border:1px solid rgba(167,139,250,0.3)'>
        <h2 style='color:#a78bfa; margin:0;'>AI Health Advisor</h2>
        <p style='color:#94a3b8; margin:0.5rem 0 0 0;'>
            Powered by Gemma 2 (2B) running locally via Ollama.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #Checking Ollama status
    ok, msg = _check_ollama()
    if ok:
        st.markdown("""
        <div style='background:#14532d22; border:1px solid #16a34a; border-radius:10px;
             padding:0.7rem 1rem; margin-bottom:1rem;'>
            <span style='color:#16a34a; font-weight:600;'>Ollama is running</span>
            <span style='color:#94a3b8; font-size:0.88rem;'> — gemma2:2b is ready.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#2d1f0022; border:1px solid #f59e0b; border-radius:10px;
             padding:0.9rem 1.2rem; margin-bottom:1rem;'>
            <span style='color:#f59e0b; font-weight:600;'>Ollama not detected:</span>
            <span style='color:#94a3b8; font-size:0.88rem;'> {msg}</span>
            <div style='color:#64748b; font-size:0.82rem; margin-top:0.5rem;'>
                Setup: &nbsp;
                <code style='background:#0f172a; padding:2px 6px; border-radius:4px;'>ollama pull gemma2:2b</code>
                &nbsp; then &nbsp;
                <code style='background:#0f172a; padding:2px 6px; border-radius:4px;'>ollama serve</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    #Empty state Handling
    if "patient_data" not in st.session_state or "predictions" not in st.session_state:
        st.markdown("""
        <div style='background:#1e293b; border-radius:16px; padding:3rem; text-align:center;
             border:2px dashed #334155;'>
            <h3 style='color:#94a3b8; margin:1rem 0;'>No Data Yet</h3>
            <p style='color:#64748b;'>
                Please go to the <strong style='color:#e2e8f0;'>Prediction</strong> tab first,
                fill in your medical data, and click <em>Analyse &amp; Predict</em>.
                Then come back here for your personalised AI health advice.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    patient_data = st.session_state["patient_data"]
    predictions  = st.session_state["predictions"]
    ensemble     = st.session_state["ensemble"]

    is_disease     = ensemble["verdict"] == "Cardiovascular Disease Detected"
    risk_label     = ensemble.get("risk_label", "High" if is_disease else "Low")
    color          = {"Low": "#16a34a", "Mild": "#d97706", "High": "#dc2626"}[risk_label]
    age_display    = patient_data.get("_age_years", round(patient_data["age"] / 365.25, 1))
    gender_display = patient_data.get("_gender_label", "N/A")

    #Patient summary card
    st.markdown(f"""
    <div style='background:#1e293b; border-radius:16px; padding:1.5rem; margin-bottom:1.5rem;
         display:flex; gap:2rem; flex-wrap:wrap; border:1px solid #334155;'>
        <div>
            <div style='color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;'>Patient</div>
            <div style='color:#e2e8f0; font-size:1.1rem; font-weight:bold;'>{gender_display}, Age {age_display}</div>
        </div>
        <div>
            <div style='color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;'>BP (Sys / Dia)</div>
            <div style='color:#e2e8f0; font-size:1.1rem; font-weight:bold;'>{patient_data['ap_hi']} / {patient_data['ap_lo']} mmHg</div>
        </div>
        <div>
            <div style='color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;'>ML Verdict</div>
            <div style='color:{color}; font-size:1.1rem; font-weight:bold;'>Risk: {risk_label}</div>
        </div>
        <div>
            <div style='color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;'>Disease Probability</div>
            <div style='color:{color}; font-size:1.1rem; font-weight:bold;'>{ensemble['avg_probability']}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "llm_advice" not in st.session_state:
        st.session_state["llm_advice"] = None

    generate = st.button("Generate AI Health Advice", use_container_width=True, type="primary")

    if st.session_state["llm_advice"] and not generate:
        if st.button("Regenerate Advice", use_container_width=True):
            st.session_state["llm_advice"] = None
            st.rerun()

    if generate:
        #Clear previous advice before streaming new one
        st.session_state["llm_advice"] = None

        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e1b4b 0%,#1e293b 100%);
             border-radius:16px; padding:0.5rem 1.5rem; margin-top:1rem;
             border:1px solid rgba(167,139,250,0.3);'>
        """, unsafe_allow_html=True)

        full_response = st.write_stream(
            stream_llm_advice(patient_data, ensemble, predictions)
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state["llm_advice"] = full_response

    elif st.session_state["llm_advice"]:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e1b4b 0%,#1e293b 100%);
             border-radius:16px; padding:0.5rem 1.5rem; margin-top:1rem;
             border:1px solid rgba(167,139,250,0.3);'>
        """, unsafe_allow_html=True)
        st.markdown(st.session_state["llm_advice"])
        st.markdown("</div>", unsafe_allow_html=True)

    #Disclaimer + Restore (only after advice is shown)
    if st.session_state.get("llm_advice"):
        st.markdown("""
        <div style='background:#1e293b; border-radius:8px; padding:0.8rem 1rem;
             margin-top:1rem; border-left:3px solid #fbbf24;'>
            <span style='color:#fbbf24; font-weight:bold;'>Disclaimer:</span>
            <span style='color:#94a3b8; font-size:0.85rem;'>
                AI-generated advice for informational purposes only.
                Always consult a qualified healthcare professional for medical decisions.
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        if st.button("Restore / Start Over", key="restore_advisor"):
            for k in RESET_KEYS:
                st.session_state.pop(k, None)
            st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
            st.rerun()