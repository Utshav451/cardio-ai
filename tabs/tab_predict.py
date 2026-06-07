#Patient data input form and ML prediction.

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model_utils import predict_single, ensemble_prediction, MODEL_COLORS

#Keys to clear on restore
RESET_KEYS = ["patient_data", "predictions", "ensemble", "llm_advice"]


def render(models: dict, scaler):
    #Initialise form version — incrementing this resets the entire form
    if "form_version" not in st.session_state:
        st.session_state["form_version"] = 0

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
         padding:2rem; border-radius:16px; margin-bottom:2rem;
         border:1px solid rgba(79,142,247,0.3)'>
        <h2 style='color:#4F8EF7; margin:0; font-size:1.5rem;'>Patient Medical Assessment</h2>
        <p style='color:#94a3b8; margin:0.5rem 0 0 0;'>
            Enter your medical data below for an instant cardiovascular risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #form key changes on every restore → Streamlit fully re-renders the form blank
    with st.form(f"prediction_form_{st.session_state['form_version']}"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Demographics & Body")

            age_years = st.number_input(
                "Age (years)", min_value=1, max_value=120,
                value=None, placeholder="Enter age in years"
            )
            gender = st.selectbox(
                "Gender",
                options=["-- Select Gender --", "Female", "Male"],
                index=0
            )
            height = st.number_input(
                "Height (cm)", min_value=50, max_value=250,
                value=None, placeholder="e.g. 168"
            )
            weight = st.number_input(
                "Weight (kg)", min_value=10.0, max_value=300.0,
                value=None, placeholder="e.g. 72.0", step=0.5
            )
            smoke = st.selectbox(
                "Smoker",
                options=["-- Select --", "No", "Yes"],
                index=0
            )
            alco = st.selectbox(
                "Alcohol Intake",
                options=["-- Select --", "No", "Yes"],
                index=0
            )

        with col2:
            st.markdown("#### Blood Pressure & Labs")

            ap_hi = st.number_input(
                "Systolic Blood Pressure (mm Hg)", min_value=60, max_value=280,
                value=None, placeholder="e.g. 120"
            )
            ap_lo = st.number_input(
                "Diastolic Blood Pressure (mm Hg)", min_value=40, max_value=180,
                value=None, placeholder="e.g. 80"
            )
            cholesterol = st.selectbox(
                "Cholesterol Level",
                options=[
                    "-- Select Cholesterol --",
                    "Normal (< 200 mg/dl)",
                    "Above Normal (200–239 mg/dl)",
                    "Well Above Normal (≥ 240 mg/dl)"
                ],
                index=0
            )
            gluc = st.selectbox(
                "Fasting Glucose Level",
                options=[
                    "-- Select Glucose --",
                    "Normal (< 100 mg/dl)",
                    "Above Normal (100–125 mg/dl)",
                    "Well Above Normal (≥ 126 mg/dl)"
                ],
                index=0
            )

        submitted = st.form_submit_button(
            "Analyse & Predict", use_container_width=True, type="primary"
        )

    if submitted:
        #Validation of inputs
        errors = []
        if age_years is None:             errors.append("Age is required.")
        if gender.startswith("--"):       errors.append("Gender must be selected.")
        if height is None:                errors.append("Height is required.")
        if weight is None:                errors.append("Weight is required.")
        if ap_hi is None:                 errors.append("Systolic Blood Pressure is required.")
        if ap_lo is None:                 errors.append("Diastolic Blood Pressure is required.")
        if cholesterol.startswith("--"):  errors.append("Cholesterol level must be selected.")
        if gluc.startswith("--"):         errors.append("Glucose level must be selected.")
        if smoke.startswith("--"):        errors.append("Smoking status must be selected.")
        if alco.startswith("--"):         errors.append("Alcohol intake must be selected.")

        if errors:
            for e in errors:
                st.error(e)
            return

        #Parse values
        age_days  = int(age_years * 365.25)
        gender_val = 1 if gender == "Female" else 2
        #Cholesterol: map position to 1/2/3
        chol_map  = {
            "Normal (< 200 mg/dl)":          1,
            "Above Normal (200–239 mg/dl)":   2,
            "Well Above Normal (≥ 240 mg/dl)": 3
        }
        #glucose: map position to 1/2/3
        gluc_map  = {
            "Normal (< 100 mg/dl)":           1,
            "Above Normal (100–125 mg/dl)":    2,
            "Well Above Normal (≥ 126 mg/dl)": 3
        }
        chol_val  = chol_map[cholesterol]
        gluc_val  = gluc_map[gluc]
        smoke_val = 1 if smoke == "Yes" else 0
        alco_val  = 1 if alco  == "Yes" else 0

        patient_data = {
            "age": age_days, "gender": gender_val,
            "height": int(height), "weight": float(weight),
            "ap_hi": int(ap_hi), "ap_lo": int(ap_lo),
            "cholesterol": chol_val, "gluc": gluc_val,
            "smoke": smoke_val, "alco": alco_val,
            "_age_years":    age_years,
            "_gender_label": gender,
        }

        with st.spinner("Running models..."):
            predictions = predict_single(patient_data, models, scaler)
            ensemble    = ensemble_prediction(predictions)

        st.session_state["patient_data"] = patient_data
        st.session_state["predictions"]  = predictions
        st.session_state["ensemble"]     = ensemble

        #Verdict banner
        risk_label   = ensemble["risk_label"]
        prob         = ensemble["avg_probability"]
        banner_color = {"Low": "#16a34a", "Mild": "#d97706", "High": "#dc2626"}[risk_label]

        st.markdown(f"""
        <div style='background:{banner_color}22; border:2px solid {banner_color};
             border-radius:16px; padding:2rem; text-align:center; margin:1.5rem 0;'>
            <h2 style='color:{banner_color}; margin:0.5rem 0; font-size:1.8rem;'>
                Risk of Cardiovascular Disease: {risk_label}
            </h2>
            <p style='color:#e2e8f0; font-size:1.1rem; margin:0;'>
                Average disease probability: <strong style='color:{banner_color}'>{prob}%</strong>
                &nbsp;|&nbsp; {ensemble['disease_votes']} of {ensemble['total_models']} models flagged disease
            </p>
        </div>
        """, unsafe_allow_html=True)

        #Individual Model cards
        st.markdown("#### Individual Model Predictions")
        cols = st.columns(4)
        for i, (name, result) in enumerate(predictions.items()):
            with cols[i % 4]:
                prob = result["probability"]
                if prob < 35:
                    risk  = "Low"
                    color = "#16a34a"
                elif prob < 60:
                    risk  = "Mild"
                    color = "#d97706"
                else:
                    risk  = "High"
                    color = "#dc2626"
                st.markdown(f"""
                <div style='background:#1e293b; border-radius:12px; padding:1rem;
                     margin-bottom:1rem; border-left:4px solid {color};'>
                    <div style='font-size:0.78rem; color:#94a3b8;'>{name}</div>
                    <div style='font-size:1rem; font-weight:bold; color:{color}; margin:0.3rem 0;'>{risk}</div>
                    <div style='background:#0f172a; border-radius:8px; height:7px; margin-top:0.4rem;'>
                        <div style='background:{color}; height:7px; border-radius:8px;
                             width:{prob}%;'></div>
                    </div>
                    <div style='font-size:0.72rem; color:#94a3b8; margin-top:0.3rem;'>
                        {prob}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <p style='color:#94a3b8; font-size:0.88rem; margin-top:0.5rem;'>
            Switch to the <strong>Analysis</strong> or <strong>AI Advisor</strong> tab for more details.
        </p>
        """, unsafe_allow_html=True)

        st.divider()
        if st.button("Restore / Reset", key="restore_predict"):
            for k in RESET_KEYS:
                st.session_state.pop(k, None)
            st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
            st.rerun()