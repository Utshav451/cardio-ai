import requests
import os

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "gemma2:2b"

SYSTEM_PROMPT = """You are a compassionate and knowledgeable cardiovascular health advisor.
Your role is to provide personalised, detailed, and actionable health advice based on a
patient's medical data and ML model predictions.

Structure your response with these exact sections using markdown:
1. **Risk Assessment** — Explain the overall cardiovascular risk based on the ML verdict and key numbers.
2. **Key Health Concerns** — Highlight specific risk factors (BP, cholesterol, glucose, smoking, alcohol). Be specific with actual values.
3. **Lifestyle Recommendations** — Concrete, practical advice on diet, exercise, sleep, and stress management tailored to this patient.
4. **Medical Action Plan** — What to do next, including when/why to see a doctor.
5. **A Final Word** — Close with an encouraging, warm message.

Guidelines:
- Be warm, empathetic, and non-alarmist while being honest.
- Reference the patient's actual values (e.g. "your BP of 160/100 is Stage 2 hypertension").
- If disease is detected, clearly but calmly urge consulting a cardiologist.
- If healthy, reinforce positive habits and give preventive tips.
- Write at least 350 words. Use bullet points inside each section.
- Never make a definitive diagnosis — these are ML predictions only.
"""


def build_patient_summary(patient_data: dict, ensemble_result: dict, model_predictions: dict) -> str:
    p           = patient_data
    age_display = p.get("_age_years", round(p["age"] / 365.25, 1))
    gender      = "Female" if p.get("gender", 2) == 1 else "Male"
    chol_map    = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    gluc_map    = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}

    model_lines = "\n".join(
        f"  - {name}: {'Disease' if r['prediction']==1 else 'Healthy'} ({r['probability']}%)"
        for name, r in model_predictions.items()
    )

    return f"""Patient Medical Profile:
- Age: {age_display} years ({gender})
- Height: {p.get('height','N/A')} cm  |  Weight: {p.get('weight','N/A')} kg
- Systolic BP: {p.get('ap_hi','N/A')} mm Hg  |  Diastolic BP: {p.get('ap_lo','N/A')} mm Hg
- Cholesterol: {chol_map.get(p.get('cholesterol',1),'N/A')}
- Glucose: {gluc_map.get(p.get('gluc',1),'N/A')}
- Smoker: {'Yes' if p.get('smoke',0)==1 else 'No'}
- Alcohol Intake: {'Yes' if p.get('alco',0)==1 else 'No'}

ML Prediction Results:
- Overall Verdict: {ensemble_result['verdict']}
- Average Disease Probability: {ensemble_result['avg_probability']}%
- Model Consensus: {ensemble_result['disease_votes']} of {ensemble_result['total_models']} models flagged disease
- Individual Results:
{model_lines}

Please provide detailed, personalised health advice for this patient."""


def _check_ollama() -> tuple:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, "Ollama is not running. Please start it with: ollama serve"
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any("gemma2" in m for m in models):
            return False, "Model not found. Please run: ollama pull gemma2:2b"
        return True, "ok"
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running. Please start it with: ollama serve"
    except Exception as e:
        return False, str(e)


#Calling gemma2:2b model
def stream_llm_advice(patient_data: dict, ensemble_result: dict, model_predictions: dict):
    ok, msg = _check_ollama()
    if not ok:
        yield _fallback_advice(patient_data, ensemble_result)
        return

    summary = build_patient_summary(patient_data, ensemble_result, model_predictions)

    payload = {
        "model":    MODEL,
        "stream":   True,                             
        "options":  {"temperature": 0.7, "num_predict": 800},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": summary}
        ]
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
    except requests.exceptions.Timeout:
        yield "\n\n_Ollama timed out. Please try again._"
    except Exception as e:
        yield _fallback_advice(patient_data, ensemble_result)


# def get_llm_advice(patient_data: dict, ensemble_result: dict, model_predictions: dict) -> tuple:
#     """
#     Non-streaming fallback. Returns (advice_text, error_message).
#     Use stream_llm_advice() for live streaming instead.
#     """
#     ok, msg = _check_ollama()
#     if not ok:
#         return _fallback_advice(patient_data, ensemble_result), msg

#     summary = build_patient_summary(patient_data, ensemble_result, model_predictions)

#     payload = {
#         "model":    MODEL,
#         "stream":   False,
#         "options":  {"temperature": 0.7, "num_predict": 800},
#         "messages": [
#             {"role": "system",  "content": SYSTEM_PROMPT},
#             {"role": "user",    "content": summary}
#         ]
#     }

#     try:
#         response = requests.post(OLLAMA_URL, json=payload, timeout=300)
#         response.raise_for_status()
#         data = response.json()
#         return data["message"]["content"], None
#     except requests.exceptions.Timeout:
#         return _fallback_advice(patient_data, ensemble_result), \
#                "Ollama timed out. The model may still be loading — try again in a moment."
#     except Exception as e:
#         return _fallback_advice(patient_data, ensemble_result), str(e)

#Fallback Method
def _fallback_advice(patient_data: dict, ensemble_result: dict) -> str:
    verdict    = ensemble_result.get("verdict", "")
    prob       = ensemble_result.get("avg_probability", 50)
    ap_hi      = patient_data.get("ap_hi", 120)
    ap_lo      = patient_data.get("ap_lo", 80)
    chol       = patient_data.get("cholesterol", 1)
    gluc       = patient_data.get("gluc", 1)
    smoker     = patient_data.get("smoke", 0)
    alcohol    = patient_data.get("alco", 0)
    age        = patient_data.get("_age_years", round(patient_data.get("age", 18000) / 365.25, 1))
    is_disease = verdict == "Cardiovascular Disease Detected"

    sections = []

    if is_disease:
        ra = f"**Risk Assessment**\nBased on your medical data, our ensemble of ML models has flagged a cardiovascular disease probability of **{prob}%**. This is a significant result that warrants prompt medical attention. This is a machine learning prediction — not a clinical diagnosis — but it should not be ignored."
    elif prob > 35:
        ra = f"**Risk Assessment**\nYour data shows a moderate cardiovascular risk ({prob}% average probability). While no disease was definitively flagged, several health markers require monitoring and lifestyle attention."
    else:
        ra = f"**Risk Assessment**\nYour cardiovascular risk profile looks relatively low at {prob}% average probability. The majority of models did not flag disease. Maintaining heart-healthy habits now is the best way to stay this way as you age."
    sections.append(ra)

    concerns = []
    if ap_hi >= 140 or ap_lo >= 90:
        concerns.append(f"- **Blood Pressure {ap_hi}/{ap_lo} mm Hg**: Hypertension range. High BP is a leading cause of heart attacks and strokes.")
    elif ap_hi >= 130:
        concerns.append(f"- **Blood Pressure {ap_hi}/{ap_lo} mm Hg**: Elevated/Stage 1 hypertension. Lifestyle changes can help.")
    if chol == 3:
        concerns.append("- **Cholesterol (Well Above Normal)**: Severely elevated — accelerates arterial plaque. Discuss statins with your doctor.")
    elif chol == 2:
        concerns.append("- **Cholesterol (Above Normal)**: Elevated. Reduce saturated fats and increase fibre.")
    if gluc == 3:
        concerns.append("- **Glucose (Well Above Normal)**: May indicate diabetes — a strong cardiovascular risk multiplier. Request a fasting glucose test.")
    elif gluc == 2:
        concerns.append("- **Glucose (Above Normal)**: Warrants monitoring. Reduce sugar and refined carbohydrates.")
    if smoker:
        concerns.append("- **Smoking**: One of the most powerful cardiovascular risk factors. Quitting is the single best thing you can do for your heart.")
    if alcohol:
        concerns.append("- **Alcohol Intake**: Raises blood pressure and weakens heart muscle over time. Consider reducing or eliminating.")
    if age > 55:
        concerns.append(f"- **Age ({age} years)**: Cardiovascular risk increases significantly with age. Regular screenings are essential.")

    sections.append("**Key Health Concerns**\n" + ("\n".join(concerns) if concerns else "Your individual markers appear within acceptable ranges. Continue to monitor regularly."))

    sections.append("""**Lifestyle Recommendations**
- **Diet**: Heart-healthy diet — vegetables, fruits, whole grains, lean proteins. Reduce sodium (<2,000 mg/day), saturated fats, and added sugars. The Mediterranean diet is strongly evidence-backed.
- **Exercise**: At least 150 minutes of moderate aerobic activity per week plus 2 days of strength training.
- **Sleep**: Target 7–9 hours. Poor sleep is strongly linked to high BP and heart disease.
- **Stress Management**: Chronic stress raises cortisol and BP. Try mindfulness, meditation, or leisure activities.
- **Weight**: A 5–10% reduction in body weight can meaningfully improve BP, cholesterol, and glucose.""")

    if is_disease:
        sections.append("""**Medical Action Plan**
- **Consult a cardiologist as soon as possible.** Do not delay.
- Request a full cardiac workup: ECG, echocardiogram, lipid panel, blood glucose, stress test.
- Do not stop current medications without medical guidance.
- Schedule follow-up appointments every 3–6 months.""")
    else:
        sections.append("""**Medical Action Plan**
- Schedule a routine check-up and share your BP and cholesterol values with your GP.
- Request a full lipid panel and fasting blood glucose test if not done recently.
- Recheck your cardiovascular risk annually.
- If any symptoms appear (chest pain, shortness of breath, palpitations), seek immediate medical attention.""")

    close = (
        "**A Final Word**\nReceiving a high-risk result can be unsettling — but knowledge is power. Many cardiovascular conditions are highly manageable when caught early. Taking action today can dramatically change your trajectory. Your health is worth fighting for."
        if is_disease else
        "**A Final Word**\nYou are in a great position. Your numbers suggest your heart is holding up well. Keep making the choices that got you here, and stay proactive with regular check-ups. A healthy heart is one of your most valuable long-term assets."
    )
    sections.append(close)
    return "\n\n".join(sections)