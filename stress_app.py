
import io
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stress & Deception AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CNN model files
MODEL_PATH  = "stress_cnn_model.keras"
NORM_PATH   = "normalization_stats.json"
CONFIG_PATH = "model_config.json"

# Behavioral model files
CLF_PATH    = "lie_detection_model_Classification.pkl"
REG_PATH    = "best_regression_model.pkl"

CHANNELS = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp", "ecg"]


# ─────────────────────────────────────────────────────────────
#  LOAD MODELS (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model_and_config():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(NORM_PATH) as f:
        norm = json.load(f)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return model, norm, cfg


@st.cache_resource
def load_behavioral_models():
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING — same as CNN training
# ─────────────────────────────────────────────────────────────
def resample_signal(signal, orig_fs, target_fs):
    signal = np.asarray(signal).squeeze()
    n = int(len(signal) * target_fs / orig_fs)
    return resample(signal, n)


def preprocess_pkl(pkl_data, target_fs=4):
    """Replicates the CNN training preprocessing for WESAD .pkl files."""
    signals = pkl_data["signal"]
    labels  = np.asarray(pkl_data["label"]).squeeze()

    # Resample labels (700 Hz -> 4 Hz)
    n_target  = int(len(labels) * target_fs / 700)
    indices   = np.linspace(0, len(labels) - 1, n_target).astype(int)
    labels_rs = labels[indices]
    mask      = np.isin(labels_rs, [1, 2])

    # Resample all signals to 4 Hz
    acc_raw = np.asarray(signals["wrist"]["ACC"])
    bvp_raw = np.asarray(signals["wrist"]["BVP"]).squeeze()
    eda_raw = np.asarray(signals["wrist"]["EDA"]).squeeze()
    tmp_raw = np.asarray(signals["wrist"]["TEMP"]).squeeze()
    ecg_raw = np.asarray(signals["chest"]["ECG"]).squeeze()

    acc = np.column_stack([
        resample_signal(acc_raw[:, i], 32, target_fs) for i in range(3)
    ])
    bvp  = resample_signal(bvp_raw, 64, target_fs)
    eda  = eda_raw
    temp = tmp_raw
    ecg  = resample_signal(ecg_raw, 700, target_fs)

    min_len = min(len(mask), len(acc), len(bvp), len(eda), len(temp), len(ecg))
    mask = mask[:min_len]
    acc  = acc[:min_len]
    bvp  = bvp[:min_len].reshape(-1, 1)
    eda  = eda[:min_len].reshape(-1, 1)
    temp = temp[:min_len].reshape(-1, 1)
    ecg  = ecg[:min_len].reshape(-1, 1)

    X = np.hstack([acc, bvp, eda, temp, ecg])
    X = X[mask]
    y = (labels_rs[:min_len][mask] == 2).astype(int)

    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True)
    X = (X - mean) / (std + 1e-8)

    return X, y


def make_windows(X, y, win_len=240, step=60):
    """75% overlap (step = 60 of 240)."""
    Xw, yw = [], []
    for s in range(0, len(X) - win_len + 1, step):
        e = s + win_len
        if len(np.unique(y[s:e])) == 1:
            Xw.append(X[s:e])
            yw.append(y[s])
    return np.array(Xw), np.array(yw)


@st.cache_data(show_spinner=False)
def run_inference(file_bytes: bytes):
    """Process a .pkl file and run the CNN. Cached on file content."""
    data = pickle.loads(file_bytes, encoding="latin1")
    X_proc, y_true = preprocess_pkl(data)
    X_win, y_win = make_windows(X_proc, y_true)

    if len(X_win) == 0:
        return None, None, None

    _model, _, _ = load_cnn_model_and_config()
    probs = _model.predict(X_win, verbose=0).flatten()
    return X_win, y_win, probs


def compute_metrics_at_threshold(y_true, probs, thr):
    preds = (probs >= thr).astype(int)
    out = {
        "accuracy":  accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall":    recall_score(y_true, preds, zero_division=0),
        "f1":        f1_score(y_true, preds, zero_division=0),
        "preds":     preds,
    }
    try:
        out["auc"] = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        out["auc"] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────
#  BEHAVIORAL FEATURES (Classification & Regression)
# ─────────────────────────────────────────────────────────────
FEATURES = {
    "Face Expression": {
        "desc": "Visible tension, micro-expressions, or stress in facial muscles",
        "options": ["Normal", "Tense / Stressed"],
    },
    "Voice Vibration": {
        "desc": "Audible tremor or shakiness detected during speech",
        "options": ["Stable", "Shaking / Trembling"],
    },
    "Eye Contact": {
        "desc": "Ability to maintain steady gaze during questioning",
        "options": ["Maintained", "Avoided / Shifty"],
    },
    "Blink Rate": {
        "desc": "Frequency of blinking compared to baseline",
        "options": ["Normal", "Rapid / Excessive"],
    },
    "Hand Movement": {
        "desc": "Restlessness or fidgeting observed in hands and arms",
        "options": ["Still", "Fidgeting / Restless"],
    },
    "Posture": {
        "desc": "Body posture and openness during the interaction",
        "options": ["Upright / Open", "Slouched / Closed"],
    },
    "Speech Speed": {
        "desc": "Speaking pace relative to the subject's normal baseline",
        "options": ["Normal", "Noticeably Changed"],
    },
    "Sweating Level": {
        "desc": "Visible perspiration beyond what context would explain",
        "options": ["Dry", "Sweating"],
    },
}


# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #185FA5 0%, #7C3AED 50%, #E63946 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .stress-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        padding: 1.8rem;
        border-radius: 14px;
        border-left: 6px solid #dc2626;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.08);
    }
    .calm-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%);
        padding: 1.8rem;
        border-radius: 14px;
        border-left: 6px solid #16a34a;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.08);
    }
    .mid-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fde68a 100%);
        padding: 1.8rem;
        border-radius: 14px;
        border-left: 6px solid #f59e0b;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.08);
    }
    .verdict-text  { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }
    .verdict-prob  { font-size: 1.05rem; color: #4b5563; margin-top: 0.5rem; line-height: 1.6; }

    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6b7280; }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e7eb;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-info     { background: #dbeafe; color: #1e40af; }
    .badge-elevated { background: #fee2e2; color: #b91c1c; }
    .badge-normal   { background: #dcfce7; color: #15803d; }

    .model-info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #185FA5;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        font-size: 0.92rem;
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 Stress & Deception AI</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Three machine learning models in one interface — '
            'CNN for physiological stress, Logistic Regression for lie classification, '
            'and Linear Regression for deception scoring</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Models in this App")
    st.markdown("""
    **1️⃣ 1D CNN** — *Stress Detection*
    Deep learning on wearable sensor signals.

    **2️⃣ Logistic Regression** — *Lie Classification*
    Binary classifier on 8 behavioral features.

    **3️⃣ Linear Regression** — *Deception Score*
    Continuous score from behavioral signals.
    """)
    st.markdown("---")

    try:
        _, _, _cnn_cfg = load_cnn_model_and_config()
        DEFAULT_THRESHOLD = float(_cnn_cfg.get("threshold", 0.30))
        st.markdown("### 📊 CNN Training Metrics")
        st.caption("Subject-independent test set")
        metrics = _cnn_cfg.get("test_metrics", {})
        col1, col2 = st.columns(2)
        col1.metric("AUC",      f"{metrics.get('auc',      0):.3f}")
        col2.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        col1.metric("F1",       f"{metrics.get('f1',       0):.3f}")
        col2.metric("Recall",   f"{metrics.get('recall',   0)*100:.1f}%")
    except Exception as e:
        DEFAULT_THRESHOLD = 0.30
        st.warning("⚠️ CNN model files not found")

    st.markdown("---")
    st.markdown("### ⚙️ CNN Threshold")
    THRESHOLD = st.slider(
        "Probability cutoff",
        min_value=0.10, max_value=0.90,
        value=DEFAULT_THRESHOLD, step=0.01,
        help="CNN predictions above this value → STRESSED."
    )
    st.caption(f"Default from training: **{DEFAULT_THRESHOLD:.2f}**")


# ─────────────────────────────────────────────────────────────
#  TOP-LEVEL TABS
# ─────────────────────────────────────────────────────────────
tab_cnn, tab_clf, tab_reg = st.tabs([
    "🧠 1D CNN — Stress Detection",
    "🎯 Logistic Regression — Lie Classification",
    "📊 Linear Regression — Deception Score",
])


# ═════════════════════════════════════════════════════════════
#  TAB 1 — CNN STRESS DETECTION
# ═════════════════════════════════════════════════════════════
with tab_cnn:
    st.markdown("""
    <div class="model-info-box">
        <b>Model 1 of 3 — 1D Convolutional Neural Network</b><br>
        Trained on the WESAD dataset to detect stress from wearable sensor signals.
        7 input channels (ACC×3, BVP, EDA, TEMP, ECG), 60-second windows at 4 Hz, ~25K parameters.
    </div>
    """, unsafe_allow_html=True)

    try:
        cnn_model, norm_stats, config = load_cnn_model_and_config()
    except Exception as e:
        st.error(f"❌ Could not load CNN model files: {e}")
        st.info("Required files: `stress_cnn_model.keras`, `normalization_stats.json`, `model_config.json`")
        st.stop()

    cnn_mode = st.radio(
        "Choose input mode:",
        ["📁 Upload WESAD subject (.pkl)", "🎲 Try with demo data"],
        horizontal=True,
        key="cnn_mode_radio",
    )

    # ─── Mode 1: Upload PKL ───
    if cnn_mode == "📁 Upload WESAD subject (.pkl)":
        uploaded = st.file_uploader(
            "Upload a WESAD subject .pkl file (e.g. S2.pkl, S3.pkl ...)",
            type=["pkl"],
            help="The file should follow the standard WESAD format with 'signal' and 'label' keys."
        )

        if uploaded is None:
            st.info("👆 Drag & drop a `.pkl` file from a WESAD subject folder to begin.")
        else:
            file_bytes = uploaded.getvalue()
            st.markdown(f"<span class='badge badge-info'>📦 {uploaded.name}</span> "
                        f"<span class='badge badge-info'>{len(file_bytes)/1e6:.1f} MB</span>",
                        unsafe_allow_html=True)

            with st.spinner("📥 Loading, preprocessing, and running inference..."):
                try:
                    X_win, y_win, probs = run_inference(file_bytes)
                except KeyError as e:
                    st.error(f"❌ Missing key in PKL file: {e}. This doesn't look like a WESAD-format file.")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Could not process file: {type(e).__name__}: {e}")
                    st.exception(e)
                    st.stop()

            if X_win is None or len(X_win) == 0:
                st.warning("⚠️ No usable windows found in this file.")
            else:
                m = compute_metrics_at_threshold(y_win, probs, THRESHOLD)
                preds = m["preds"]

                st.success(f"✅ Processed **{len(X_win)} windows** ({len(X_win):.0f} minutes of data)")

                st.markdown(f'<div class="section-header">📊 Performance at threshold = '
                            f'{THRESHOLD:.2f}</div>', unsafe_allow_html=True)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
                c2.metric("Precision", f"{m['precision']*100:.1f}%")
                c3.metric("Recall",    f"{m['recall']*100:.1f}%")
                c4.metric("F1",        f"{m['f1']:.3f}")
                c5.metric("AUC",       f"{m['auc']:.3f}" if not np.isnan(m['auc']) else "—")

                st.caption("💡 Move the threshold slider in the sidebar — these metrics update live.")

                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Total Windows", f"{len(X_win):,}")
                cc2.metric("Predicted Stress", f"{preds.sum():,}", f"{100*preds.mean():.1f}%")
                cc3.metric("Actual Stress",    f"{y_win.sum():,}", f"{100*y_win.mean():.1f}%")
                cc4.metric("Match Rate",       f"{(preds == y_win).mean()*100:.1f}%")

                # Confusion matrix
                st.markdown('<div class="section-header">🎯 Confusion Matrix</div>',
                            unsafe_allow_html=True)
                cm = confusion_matrix(y_win, preds, labels=[0, 1])
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Predicted: Calm", "Predicted: Stress"],
                    y=["Actual: Calm", "Actual: Stress"],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 22, "color": "white"},
                    colorscale=[[0, "#185FA5"], [1, "#E63946"]],
                    showscale=False,
                ))
                cm_fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                     template="plotly_white")

                col_cm, col_breakdown = st.columns([1, 1])
                with col_cm:
                    st.plotly_chart(cm_fig, use_container_width=True)
                with col_breakdown:
                    tn, fp, fn, tp = cm.ravel()
                    st.markdown(f"""
                    - ✅ **True Negatives** (correctly calm): `{tn}`
                    - ✅ **True Positives** (correctly stressed): `{tp}`
                    - ⚠️ **False Positives** (false alarms): `{fp}`
                    - ⚠️ **False Negatives** (missed stress): `{fn}`
                    """)

                # Timeline
                st.markdown('<div class="section-header">📈 Stress Probability Timeline</div>',
                            unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(probs))), y=probs,
                    mode="lines+markers",
                    name="Stress probability",
                    line=dict(color="#185FA5", width=2),
                    marker=dict(size=5),
                    hovertemplate="Window %{x}<br>Probability: %{y:.2%}<extra></extra>",
                ))
                fig.add_hline(
                    y=THRESHOLD, line_dash="dash", line_color="#E63946",
                    annotation_text=f"Threshold = {THRESHOLD:.2f}",
                    annotation_position="top right",
                )
                for i, true in enumerate(y_win):
                    if true == 1:
                        fig.add_vrect(x0=i-0.5, x1=i+0.5,
                                      fillcolor="rgba(230,57,70,0.10)",
                                      layer="below", line_width=0)
                fig.update_layout(
                    height=400,
                    xaxis_title="Window index (each = 60s)",
                    yaxis_title="Predicted stress probability",
                    yaxis=dict(range=[0, 1]),
                    hovermode="x unified",
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 Light red bands = ground-truth stress windows. "
                           "Blue line = model's predicted probability.")

                # Threshold sweep
                with st.expander("🔬 Threshold sweep: how metrics change with the cutoff"):
                    sweep_thrs = np.arange(0.10, 0.91, 0.05)
                    sweep_rows = []
                    for t in sweep_thrs:
                        mm = compute_metrics_at_threshold(y_win, probs, t)
                        sweep_rows.append({
                            "Threshold":  f"{t:.2f}",
                            "Accuracy":   f"{mm['accuracy']:.3f}",
                            "Precision":  f"{mm['precision']:.3f}",
                            "Recall":     f"{mm['recall']:.3f}",
                            "F1":         f"{mm['f1']:.3f}",
                        })
                    st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True,
                                 hide_index=True)

                # Single window inspection
                st.markdown('<div class="section-header">🔬 Inspect a Single Window</div>',
                            unsafe_allow_html=True)
                win_idx = st.slider("Pick a window", 0, len(X_win) - 1, 0)

                prob       = probs[win_idx]
                pred_label = preds[win_idx]
                true_label = y_win[win_idx]

                col_verdict, col_signals = st.columns([1, 2])

                with col_verdict:
                    if pred_label == 1:
                        st.markdown(f'''
                            <div class="stress-card">
                                <div class="verdict-text" style="color: #dc2626;">
                                    🚨 STRESSED
                                </div>
                                <div class="verdict-prob">
                                    Probability: <b>{prob:.1%}</b><br>
                                    Threshold: {THRESHOLD:.2f}
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                            <div class="calm-card">
                                <div class="verdict-text" style="color: #16a34a;">
                                    😌 NOT STRESSED
                                </div>
                                <div class="verdict-prob">
                                    Probability: <b>{prob:.1%}</b><br>
                                    Threshold: {THRESHOLD:.2f}
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                    st.markdown("")
                    gt_emoji = "🚨" if true_label == 1 else "😌"
                    gt_label = "STRESSED" if true_label == 1 else "NOT STRESSED"
                    correct  = "✅ Correct" if pred_label == true_label else "❌ Wrong"
                    st.markdown(f"**Ground truth:** {gt_emoji} {gt_label}")
                    st.markdown(f"**Prediction:** {correct}")

                with col_signals:
                    win = X_win[win_idx]
                    time_axis = np.arange(240) / 4
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("Heart-related (BVP)", "Skin Conductance (EDA)",
                                        "Temperature", "Movement (ACC magnitude)"),
                    )
                    fig.add_trace(go.Scatter(x=time_axis, y=win[:, 3],
                                             line=dict(color="#AB47BC", width=1.5)),
                                  row=1, col=1)
                    fig.add_trace(go.Scatter(x=time_axis, y=win[:, 4],
                                             line=dict(color="#26C6DA", width=1.5)),
                                  row=2, col=1)
                    fig.add_trace(go.Scatter(x=time_axis, y=win[:, 5],
                                             line=dict(color="#66BB6A", width=1.5)),
                                  row=3, col=1)
                    acc_mag = np.sqrt(win[:, 0]**2 + win[:, 1]**2 + win[:, 2]**2)
                    fig.add_trace(go.Scatter(x=time_axis, y=acc_mag,
                                             line=dict(color="#FFA726", width=1.5)),
                                  row=4, col=1)
                    fig.update_layout(
                        height=420, showlegend=False,
                        template="plotly_white",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)
                    st.plotly_chart(fig, use_container_width=True)

    # ─── Mode 2: Demo data ───
    else:
        st.info("🎲 Generate a **synthetic stress-like** or **calm-like** "
                "60-second window to test the model.")

        col1, col2 = st.columns(2)

        if col1.button("🚨 Generate STRESSED-like signal", use_container_width=True):
            np.random.seed(np.random.randint(0, 1000))
            t = np.linspace(0, 60, 240)
            win = np.zeros((240, 7))
            win[:, 0] = np.random.normal(0,    0.5, 240)
            win[:, 1] = np.random.normal(0,    0.5, 240)
            win[:, 2] = np.random.normal(0,    0.5, 240)
            win[:, 3] = 0.8 * np.sin(t * 6) + np.random.normal(0, 0.3, 240)
            win[:, 4] = 1.5 + 0.6 * np.sin(t * 0.5) + np.random.normal(0, 0.1, 240)
            win[:, 5] = -0.8 + np.random.normal(0, 0.2, 240)
            win[:, 6] = np.random.normal(0, 0.7, 240)
            st.session_state["demo_win"] = win
            st.session_state["demo_label"] = "Stressed-like (synthetic)"

        if col2.button("😌 Generate CALM signal", use_container_width=True):
            np.random.seed(np.random.randint(0, 1000))
            t = np.linspace(0, 60, 240)
            win = np.zeros((240, 7))
            win[:, 0] = np.random.normal(0,   0.2, 240)
            win[:, 1] = np.random.normal(0,   0.2, 240)
            win[:, 2] = np.random.normal(0,   0.2, 240)
            win[:, 3] = 0.3 * np.sin(t * 4) + np.random.normal(0, 0.1, 240)
            win[:, 4] = -0.5 + np.random.normal(0, 0.05, 240)
            win[:, 5] = 0.5 + np.random.normal(0, 0.1, 240)
            win[:, 6] = np.random.normal(0, 0.3, 240)
            st.session_state["demo_win"] = win
            st.session_state["demo_label"] = "Calm-like (synthetic)"

        if "demo_win" in st.session_state:
            win = st.session_state["demo_win"]
            prob = float(cnn_model.predict(win[None, ...], verbose=0).flatten()[0])
            pred = int(prob >= THRESHOLD)

            st.markdown("---")
            col_verdict, col_signals = st.columns([1, 2])

            with col_verdict:
                st.markdown(f"**Input:** {st.session_state['demo_label']}")
                if pred == 1:
                    st.markdown(f'''
                        <div class="stress-card">
                            <div class="verdict-text" style="color: #dc2626;">
                                🚨 STRESSED
                            </div>
                            <div class="verdict-prob">
                                Probability: <b>{prob:.1%}</b><br>
                                Threshold: {THRESHOLD:.2f}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                        <div class="calm-card">
                            <div class="verdict-text" style="color: #16a34a;">
                                😌 NOT STRESSED
                            </div>
                            <div class="verdict-prob">
                                Probability: <b>{prob:.1%}</b><br>
                                Threshold: {THRESHOLD:.2f}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

            with col_signals:
                time_axis = np.arange(240) / 4
                fig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("BVP", "EDA", "TEMP", "ACC magnitude"),
                )
                fig.add_trace(go.Scatter(x=time_axis, y=win[:, 3],
                                         line=dict(color="#AB47BC")), row=1, col=1)
                fig.add_trace(go.Scatter(x=time_axis, y=win[:, 4],
                                         line=dict(color="#26C6DA")), row=2, col=1)
                fig.add_trace(go.Scatter(x=time_axis, y=win[:, 5],
                                         line=dict(color="#66BB6A")), row=3, col=1)
                acc_mag = np.sqrt(win[:, 0]**2 + win[:, 1]**2 + win[:, 2]**2)
                fig.add_trace(go.Scatter(x=time_axis, y=acc_mag,
                                         line=dict(color="#FFA726")), row=4, col=1)
                fig.update_layout(height=420, showlegend=False,
                                  template="plotly_white",
                                  margin=dict(l=10, r=10, t=40, b=10))
                fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  TAB 2 — LOGISTIC REGRESSION (Lie Classification)
# ═════════════════════════════════════════════════════════════
with tab_clf:
    st.markdown("""
    <div class="model-info-box">
        <b>Model 2 of 3 — Logistic Regression Classifier</b><br>
        Binary classifier trained on 8 observable behavioral signals.
        Outputs <i>Truth</i> vs <i>Lie</i> with a confidence probability.
    </div>
    """, unsafe_allow_html=True)

    try:
        clf_model, _ = load_behavioral_models()
    except Exception as e:
        st.error(f"❌ Could not load classification model: {e}")
        st.info(f"Required file: `{CLF_PATH}`")
        st.stop()

    st.markdown('<div class="section-header">📋 Behavioral Signals — Mark What You Observe</div>',
                unsafe_allow_html=True)

    user_input_clf = {}
    feature_names = list(FEATURES.keys())

    col1, col2 = st.columns(2, gap="medium")
    for i, (feature, cfg) in enumerate(FEATURES.items()):
        with (col1 if i % 2 == 0 else col2):
            choice = st.radio(
                f"**{feature}**",
                cfg["options"],
                help=cfg["desc"],
                key=f"clf_{feature}",
            )
            user_input_clf[feature] = 0 if choice == cfg["options"][0] else 1

    elevated_clf = [f for f, v in user_input_clf.items() if v == 1]
    normal_clf   = [f for f, v in user_input_clf.items() if v == 0]

    badge_html = '<div style="margin-top:1rem">'
    for f in elevated_clf:
        badge_html += f'<span class="badge badge-elevated">⚠ {f}</span>'
    for f in normal_clf:
        badge_html += f'<span class="badge badge-normal">✓ {f}</span>'
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡ Analyze Subject (Classification)", use_container_width=True, key="btn_clf"):
        input_array = np.array([[user_input_clf[f] for f in feature_names]])
        proba = clf_model.predict_proba(input_array)[0]
        truth_prob = proba[0]
        lie_prob   = proba[1]

        if lie_prob >= 0.70:
            verdict, card_cls, color = "🚨 DECEPTION DETECTED", "stress-card", "#dc2626"
            note = f"{len(elevated_clf)} of 8 signals elevated · High deception likelihood."
        elif lie_prob >= 0.40:
            verdict, card_cls, color = "❓ INCONCLUSIVE", "mid-card", "#f59e0b"
            note = f"{len(elevated_clf)} of 8 signals elevated · Mixed indicators."
        else:
            verdict, card_cls, color = "✅ TRUTH INDICATED", "calm-card", "#16a34a"
            note = f"{len(elevated_clf)} of 8 signals elevated · No significant deception signals."

        st.markdown(f'''
            <div class="{card_cls}">
                <div class="verdict-text" style="color: {color};">{verdict}</div>
                <div class="verdict-prob">{note}</div>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Truth Probability", f"{truth_prob * 100:.1f}%")
        c2.metric("Lie Probability",   f"{lie_prob * 100:.1f}%")
        c3.metric("Signals Elevated",  f"{len(elevated_clf)} / 8")

        # Probability bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=["Truth", "Lie"],
                y=[truth_prob, lie_prob],
                marker_color=["#16a34a", "#dc2626"],
                text=[f"{truth_prob:.1%}", f"{lie_prob:.1%}"],
                textposition="auto",
            )
        ])
        fig.update_layout(
            height=300,
            yaxis=dict(range=[0, 1], title="Probability"),
            template="plotly_white",
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  TAB 3 — LINEAR REGRESSION (Deception Score)
# ═════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown("""
    <div class="model-info-box">
        <b>Model 3 of 3 — Linear Regression Model</b><br>
        Outputs a continuous <i>deception score</i> from the same 8 behavioral signals.
        Useful for nuanced rankings rather than yes/no decisions.
    </div>
    """, unsafe_allow_html=True)

    try:
        _, reg_model = load_behavioral_models()
    except Exception as e:
        st.error(f"❌ Could not load regression model: {e}")
        st.info(f"Required file: `{REG_PATH}`")
        st.stop()

    st.markdown('<div class="section-header">📋 Behavioral Signals — Mark What You Observe</div>',
                unsafe_allow_html=True)

    user_input_reg = {}

    col1, col2 = st.columns(2, gap="medium")
    for i, (feature, cfg) in enumerate(FEATURES.items()):
        with (col1 if i % 2 == 0 else col2):
            choice = st.radio(
                f"**{feature}**",
                cfg["options"],
                help=cfg["desc"],
                key=f"reg_{feature}",
            )
            user_input_reg[feature] = 0 if choice == cfg["options"][0] else 1

    elevated_reg = [f for f, v in user_input_reg.items() if v == 1]
    normal_reg   = [f for f, v in user_input_reg.items() if v == 0]

    badge_html = '<div style="margin-top:1rem">'
    for f in elevated_reg:
        badge_html += f'<span class="badge badge-elevated">⚠ {f}</span>'
    for f in normal_reg:
        badge_html += f'<span class="badge badge-normal">✓ {f}</span>'
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡ Analyze Subject (Regression)", use_container_width=True, key="btn_reg"):
        input_array = np.array([[user_input_reg[f] for f in list(FEATURES.keys())]])
        raw_score = float(reg_model.predict(input_array)[0])

        # Normalize using the model's actual output range
        SCORE_MIN = 0.4384
        SCORE_MAX = 0.7518
        normalized = (raw_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
        normalized = float(np.clip(normalized, 0.0, 1.0))
        display_pct = normalized * 100

        if display_pct >= 65:
            label, card_cls, color = "🚨 HIGH DECEPTION", "stress-card", "#dc2626"
        elif display_pct >= 35:
            label, card_cls, color = "⚠️ MODERATE SIGNALS", "mid-card", "#f59e0b"
        else:
            label, card_cls, color = "✅ LOW DECEPTION", "calm-card", "#16a34a"

        st.markdown(f'''
            <div class="{card_cls}">
                <div class="verdict-text" style="color: {color};">
                    {display_pct:.0f}%
                </div>
                <div class="verdict-prob">
                    {label} · {len(elevated_reg)} of 8 signals elevated<br>
                    Raw model output: <b>{raw_score:.4f}</b>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(normalized)

        c1, c2, c3 = st.columns(3)
        c1.metric("Deception Index",   f"{display_pct:.1f}%")
        c2.metric("Raw Score",         f"{raw_score:.4f}")
        c3.metric("Signals Elevated",  f"{len(elevated_reg)} / 8")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=display_pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Deception Index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35],   "color": "#dcfce7"},
                    {"range": [35, 65],  "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": display_pct,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📚 Three models: 1D CNN (WESAD physiological stress) · "
    "Logistic Regression (binary lie detection) · "
    "Linear Regression (deception scoring) · "
    "Built with Streamlit + TensorFlow + scikit-learn"
)
