#Model performance analysis — rankings, bar charts, radar.

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.model_utils import MODEL_COLORS

DARK_BG  = "#0f172a"
CARD_BG  = "#1e293b"
TEXT_CLR = "#e2e8f0"
ACCENT   = "#4F8EF7"

RESET_KEYS = ["patient_data", "predictions", "ensemble", "llm_advice"]


def _style(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT)
        for s in ax.spines.values():
            s.set_edgecolor("#334155")


def render(metrics: dict):
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
         padding:2rem; border-radius:16px; margin-bottom:2rem;
         border:1px solid rgba(79,142,247,0.3)'>
        <h2 style='color:#4F8EF7; margin:0;'>Model Performance Analysis</h2>
        <p style='color:#94a3b8; margin:0.5rem 0 0 0;'>
            Comprehensive evaluation of all 7 ML models trained on the cardiovascular dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics:
        st.warning("No metrics found. Ensure metrics.pkl is in the models/ folder.")
        return

    names  = list(metrics.keys())
    colors = [MODEL_COLORS.get(n, ACCENT) for n in names]

    if "ensemble" in st.session_state and "predictions" in st.session_state:
        _render_patient_summary()

    #Rankings Models based on accuracy
    st.markdown("### Overall Model Rankings")
    sorted_models = sorted(metrics.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    rank_labels   = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
    for rank, (name, m) in enumerate(sorted_models):
        color = MODEL_COLORS.get(name, ACCENT)
        st.markdown(f"""
        <div style='background:#1e293b; border-radius:12px; padding:0.9rem 1.5rem;
             margin-bottom:0.6rem; border-left:4px solid {color};'>
            <span style='font-size:0.85rem; color:#475569; min-width:36px; display:inline-block;'>{rank_labels[rank]}</span>
            <span style='font-size:1rem; color:{color}; font-weight:bold; min-width:190px; display:inline-block;'>{name}</span>
            <span style='color:#94a3b8; font-size:0.82rem;'>
                Acc: <strong style='color:#e2e8f0'>{m['accuracy']}%</strong> &nbsp;|&nbsp;
                AUC: <strong style='color:#e2e8f0'>{m['roc_auc']}%</strong> &nbsp;|&nbsp;
                F1: <strong style='color:#e2e8f0'>{m['f1']}%</strong> &nbsp;|&nbsp;
                Recall: <strong style='color:#e2e8f0'>{m['recall']}%</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)

    #Bar charts
    st.markdown("### Performance Comparison")
    t_acc, t_auc, t_f1, t_rec = st.tabs(["Accuracy", "ROC-AUC", "F1 Score", "Recall"])
    for tab, key, label in [
        (t_acc, "accuracy", "Accuracy (%)"),
        (t_auc, "roc_auc",  "ROC-AUC (%)"),
        (t_f1,  "f1",       "F1 Score (%)"),
        (t_rec, "recall",   "Recall (%)"),
    ]:
        with tab:
            vals = [metrics[n][key] for n in names]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.5)
            ax.set_xlim(0, 108)
            ax.set_xlabel(label)
            ax.set_title(f"{label} by Model")
            for bar, v in zip(bars, vals):
                ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{v}%", va="center", color=TEXT_CLR, fontsize=9)
            _style(fig, ax)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, color=TEXT_CLR)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(axis="x", color="#334155", linewidth=0.5, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    #Radar chart
    st.markdown("### Multi-Metric Radar Chart — All Models")
    _render_radar(metrics, names, colors)

    if "ensemble" in st.session_state:
        st.divider()
        if st.button("Restore / Reset", key="restore_analysis"):
            for k in RESET_KEYS:
                st.session_state.pop(k, None)
            st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
            st.rerun()


def _render_patient_summary():
    predictions = st.session_state["predictions"]
    ensemble    = st.session_state["ensemble"]

    st.markdown("### Your Prediction Results")
    names   = list(predictions.keys())
    probs   = [predictions[n]["probability"] for n in names]
    preds   = [predictions[n]["prediction"]  for n in names]
    bcolors = []
    for p in probs:
        if p < 35:
            bcolors.append("#16a34a")
        elif p < 60:
            bcolors.append("#d97706")
        else:
            bcolors.append("#dc2626")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names, probs, color=bcolors, height=0.5, edgecolor="none")
    ax.axvline(50, color="#fbbf24", linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Disease Probability (%)")
    ax.set_title("Model Predictions for Your Data")
    for bar, v in zip(bars, probs):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                f"{v}%", va="center", color=TEXT_CLR, fontsize=9)
    _style(fig, ax)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=TEXT_CLR)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color="#334155", linewidth=0.5, linestyle="--")
    handles = [
        mpatches.Patch(color="#16a34a", label="Low Risk (< 35%)"),
        mpatches.Patch(color="#d97706", label="Mild Risk (35–59%)"),
        mpatches.Patch(color="#dc2626", label="High Risk (≥ 60%)"),
        plt.Line2D([0],[0], color="#fbbf24", linestyle="--", label="50% Threshold"),
    ]
    ax.legend(handles=handles, facecolor=CARD_BG, labelcolor=TEXT_CLR, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    risk_label = ensemble.get("risk_label", "Low")
    color = {"Low": "#16a34a", "Mild": "#d97706", "High": "#dc2626"}[risk_label]
    st.markdown(f"""
    <div style='background:{color}22; border:2px solid {color}; border-radius:12px;
         padding:1.2rem; text-align:center; margin:1rem 0;'>
        <h3 style='color:{color}; margin:0;'>
            Risk of Cardiovascular Disease: {risk_label}
        </h3>
        <p style='color:#94a3b8; margin:0.3rem 0 0 0;'>
            {ensemble['disease_votes']} / {ensemble['total_models']} models detected disease &nbsp;|&nbsp;
            Avg probability: <strong style='color:{color}'>{ensemble['avg_probability']}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


def _render_radar(metrics: dict, names: list, colors: list):
    keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    N      = len(keys)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_CLR, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20","40","60","80","100"], color="#475569", fontsize=7)
    ax.grid(color="#334155", linewidth=0.5)
    ax.spines["polar"].set_edgecolor("#334155")

    for name, color in zip(names, colors):
        vals = [metrics[name][k] for k in keys] + [metrics[name][keys[0]]]
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, alpha=0.08, color=color)
        ax.scatter(angles[:-1], vals[:-1], color=color, s=35, zorder=5)

    patches = [mpatches.Patch(color=c, label=n) for n, c in zip(names, colors)]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.45, 1.15),
              facecolor=CARD_BG, labelcolor=TEXT_CLR, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()