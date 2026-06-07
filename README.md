# CardioAI — Cardiovascular Disease Prediction System

A machine learning web application that predicts cardiovascular disease risk using an ensemble of 7 ML models, with a local AI health advisor powered by **Ollama (Gemma 2 2B)** — no internet or API keys required.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange)
![Ollama](https://img.shields.io/badge/Ollama-Gemma2%3A2b-green)

---

## ✨ Features

- **7 ML Models** trained on 70,000 real patient records — Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, Naive Bayes, Neural Network
- **Ensemble voting** — combines all model predictions for a reliable final verdict
- **3-level risk classification** — Low / Mild / High with colour-coded UI (green / amber / red)
- **Live AI advice** — Gemma 2 (2B) streams personalised health advice token by token via Ollama
- **Interactive analysis** — bar charts, radar chart, per-patient prediction breakdown
- **No cloud dependency** — models run locally, AI advisor runs locally via Ollama
- **Modular codebase** — each tab is an independent Python module

---

## 🗂️ Project Structure

```
cardio_project/
│
├── app.py                          # Main Streamlit entry point
│
├── models/
│   ├── model_utils.py              # Load pkls, predict, ensemble logic
│   ├── scaler.pkl
│   ├── metrics.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── knn.pkl
│   ├── gradient_boosting.pkl
│   ├── naive_bayes.pkl
│   └── neural_network.pkl
│
├── tabs/
│   ├── tab_predict.py              # Tab 1: Patient input form + prediction
│   ├── tab_analysis.py             # Tab 2: Model performance charts
│   └── tab_advisor.py              # Tab 3: Ollama AI health advice
│
├── utils/
│   └── llm_advisor.py              # Ollama streaming integration
│
├── resources/
│   ├── cardio_train.csv            # Dataset
│   └── cardio_AI.ipynb             # Google Colab training notebook
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed (for AI advisor)

### 1. Clone the repository

```bash
git clone https://github.com/Utshav451/cardio-ai
cd cardio-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Ollama (AI Advisor)

```bash
# Pull the model (1.6 GB, one-time download)
ollama pull gemma2:2b

# Ollama runs automatically in the background after installation
```

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📤 Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit — CardioAI cardiovascular disease prediction"
git remote add origin https://github.com/Utshav451/cardio-ai.git
git branch -M main
git push -u origin main
```

---

## 📊 Input Features

| Feature | Description |
|---|---|
| Age | Patient age in years (converted to days internally) |
| Gender | Female / Male |
| Height | Height in cm |
| Weight | Weight in kg |
| Systolic BP | Systolic blood pressure (mm Hg) |
| Diastolic BP | Diastolic blood pressure (mm Hg) |
| Cholesterol | Normal / Above Normal / Well Above Normal |
| Fasting Glucose | Normal / Above Normal / Well Above Normal |
| Smoker | Yes / No |
| Alcohol Intake | Yes / No |

---

## 🤖 Machine Learning Models

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline classifier |
| Random Forest | 150-tree ensemble |
| SVM | Support Vector Machine (RBF kernel) |
| KNN | K-Nearest Neighbours (k=7) |
| Gradient Boosting | 150-estimator boosting |
| Naive Bayes | Gaussian probabilistic model |
| Neural Network | MLP (128 → 64 → 32) |

> **Note:** Typical accuracy on this dataset is ~72–74%. This is the known ceiling for this dataset with traditional ML — published research papers report the same range.

---

## 🎨 App Tabs

### Tab 1 — Prediction
- Fill in 10 medical parameters
- Click **Analyse & Predict** to run all 7 models
- See a colour-coded risk banner (Low / Mild / High)
- View per-model risk cards with probability bars

### Tab 2 — Analysis
- Model performance leaderboard
- Bar charts for Accuracy, ROC-AUC, F1, Recall (0–100 range)
- Radar chart comparing all 7 models
- Your personal prediction breakdown (if submitted)

### Tab 3 — AI Advisor
- Live streaming health advice from Gemma 2 (2B) via Ollama
- Personalised to your exact medical values and ML verdict
- Structured: Risk Assessment → Key Concerns → Lifestyle Recommendations → Action Plan
- Falls back to rule-based advice if Ollama is not running

---

## 📁 Dataset

This project uses the [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle.

- **70,000 patient records**
- **12 features** (age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, physical activity)
- **Binary target**: presence or absence of cardiovascular disease

---

## 🛡️ Disclaimer

This application is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 👨‍💻 Author

**Utshav Nath**

---

## 📜 License

This project is developed for educational and portfolio purposes.
