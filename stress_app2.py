"""
═══════════════════════════════════════════════════════════════════════
  STRESS DETECTION — Streamlit Demo App  (FIXED + ENHANCED)
═══════════════════════════════════════════════════════════════════════

  Run with:
      streamlit run stress_app2.py

  Required files in same folder:
      - stress_cnn_model.keras
      - normalization_stats.json
      - model_config.json
═══════════════════════════════════════════════════════════════════════
"""

import io
import json
import pickle
from pathlib import Path

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
    page_title="Stress Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH  = "stress_cnn_model.keras"
NORM_PATH   = "normalization_stats.json"
CONFIG_PATH = "model_config.json"

CHANNELS = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp", "ecg"]


# ─────────────────────────────────────────────────────────────
#  LOAD MODEL & CONFIG (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_config():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(NORM_PATH) as f:
        norm = json.load(f)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return model, norm, cfg


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING — same as training
# ─────────────────────────────────────────────────────────────
def resample_signal(signal, orig_fs, target_fs):
    signal = np.asarray(signal).squeeze()
    n = int(len(signal) * target_fs / orig_fs)
    return resample(signal, n)


def preprocess_pkl(pkl_data, target_fs=4):
    """Replicates the training preprocessing for WESAD .pkl files."""
    signals = pkl_data["signal"]
    labels  = np.asarray(pkl_data["label"]).squeeze()

    # Resample labels (700 Hz → 4 Hz)
    n_target  = int(len(labels) * target_fs / 700)
    indices   = np.linspace(0, len(labels) - 1, n_target).astype(int)
    labels_rs = labels[indices]
    mask      = np.isin(labels_rs, [1, 2])

    # Resample all signals to 4 Hz
    acc_raw = np.asarray(signals["wrist"]["ACC"])  # shape (N, 3)
    bvp_raw = np.asarray(signals["wrist"]["BVP"]).squeeze()
    eda_raw = np.asarray(signals["wrist"]["EDA"]).squeeze()
    tmp_raw = np.asarray(signals["wrist"]["TEMP"]).squeeze()
    ecg_raw = np.asarray(signals["chest"]["ECG"]).squeeze()

    # ACC has 3 axes — resample each axis
    acc = np.column_stack([
        resample_signal(acc_raw[:, i], 32, target_fs) for i in range(3)
    ])
    bvp  = resample_signal(bvp_raw, 64, target_fs)
    eda  = eda_raw  # already at 4 Hz
    temp = tmp_raw  # already at 4 Hz
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

    # Per-subject z-score normalization
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


# ─────────────────────────────────────────────────────────────
#  CACHED INFERENCE — runs once per uploaded file
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_inference(file_bytes: bytes):
    """Process a .pkl file and run the model. Cached on file content."""
    data = pickle.loads(file_bytes, encoding="latin1")
    X_proc, y_true = preprocess_pkl(data)
    X_win, y_win = make_windows(X_proc, y_true)

    if len(X_win) == 0:
        return None, None, None

    _model, _, _ = load_model_and_config()
    probs = _model.predict(X_win, verbose=0).flatten()
    return X_win, y_win, probs


def compute_metrics_at_threshold(y_true, probs, thr):
    """Compute classification metrics at a given threshold."""
    preds = (probs >= thr).astype(int)
    out = {
        "accuracy":  accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall":    recall_score(y_true, preds, zero_division=0),
        "f1":        f1_score(y_true, preds, zero_division=0),
        "preds":     preds,
    }
    # AUC doesn't depend on threshold, but include it for context
    try:
        out["auc"] = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        out["auc"] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS — modernized
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default streamlit padding at top */
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
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #185FA5;
        margin: 0.3rem 0;
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
    .verdict-text  { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }
    .verdict-prob  { font-size: 1.05rem; color: #4b5563; margin-top: 0.5rem; line-height: 1.6; }

    /* Better metric styling */
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6b7280; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e7eb;
    }

    /* Pill badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-stress { background: #fee2e2; color: #b91c1c; }
    .badge-calm   { background: #dcfce7; color: #15803d; }
    .badge-info   { background: #dbeafe; color: #1e40af; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 Stress Detection AI</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">1D CNN trained on the WESAD dataset — '
            'detects stress from wearable sensor signals in real time</div>',
            unsafe_allow_html=True)

# Load the model
try:
    model, norm_stats, config = load_model_and_config()
    DEFAULT_THRESHOLD = float(config.get("threshold", 0.30))
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")
    st.info("Make sure these files are in the same folder as this app:\n\n"
            "- `stress_cnn_model.keras`\n"
            "- `normalization_stats.json`\n"
            "- `model_config.json`")
    st.stop()


# ─────────────────────────────────────────────────────────────
#  SIDEBAR — model info + settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Training Metrics")
    st.caption("Held-out test set (subject-independent)")

    metrics = config.get("test_metrics", {})
    col1, col2 = st.columns(2)
    col1.metric("AUC",      f"{metrics.get('auc',      0):.3f}")
    col2.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
    col1.metric("F1",       f"{metrics.get('f1',       0):.3f}")
    col2.metric("Recall",   f"{metrics.get('recall',   0)*100:.1f}%")

    st.markdown("---")
    st.markdown("### ⚙️ Decision Threshold")
    THRESHOLD = st.slider(
        "Probability cutoff",
        min_value=0.10, max_value=0.90,
        value=DEFAULT_THRESHOLD, step=0.01,
        help="Predictions above this value → STRESSED. "
             "Lower it to catch more stress (higher recall); "
             "raise it for fewer false alarms (higher precision)."
    )
    st.caption(f"Default from training: **{DEFAULT_THRESHOLD:.2f}**")

    st.markdown("---")
    st.markdown("### ℹ️ Model Specs")
    st.markdown(
        "- Window: **60 s**\n"
        "- Sampling rate: **4 Hz**\n"
        "- Channels: **7** (ACC×3, BVP, EDA, TEMP, ECG)\n"
        "- Architecture: **1D CNN, ~25K params**\n"
        "- Overlap: **75%** between windows"
    )


# ─────────────────────────────────────────────────────────────
#  MAIN — input mode selector
# ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Choose input mode:",
    ["📁 Upload WESAD subject (.pkl)", "🎲 Try with demo data"],
    horizontal=True,
    label_visibility="visible",
)


# ═════════════════════════════════════════════════════════════
#  MODE 1 — UPLOAD PKL
# ═════════════════════════════════════════════════════════════
if mode == "📁 Upload WESAD subject (.pkl)":

    uploaded = st.file_uploader(
        "Upload a WESAD subject .pkl file (e.g. S2.pkl, S3.pkl ...)",
        type=["pkl"],
        help="The file should follow the standard WESAD format with "
             "'signal' and 'label' keys."
    )

    if uploaded is None:
        st.info("👆 Drag & drop a `.pkl` file from a WESAD subject folder to begin.")
        st.stop()

    # Read bytes ONCE (so we can hash for caching)
    file_bytes = uploaded.getvalue()
    st.markdown(f"<span class='badge badge-info'>📦 {uploaded.name}</span> "
                f"<span class='badge badge-info'>{len(file_bytes)/1e6:.1f} MB</span>",
                unsafe_allow_html=True)

    # Run inference (cached)
    with st.spinner("📥 Loading, preprocessing, and running inference..."):
        try:
            X_win, y_win, probs = run_inference(file_bytes)
        except KeyError as e:
            st.error(f"❌ Missing key in PKL file: {e}. "
                     "This doesn't look like a WESAD-format file.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Could not process file: {type(e).__name__}: {e}")
            st.exception(e)
            st.stop()

    if X_win is None or len(X_win) == 0:
        st.warning("⚠️ No usable windows found in this file. "
                   "Make sure it contains baseline (label=1) and stress (label=2) segments.")
        st.stop()

    # ── Recompute metrics at the CURRENT threshold (this is the fix!) ──
    m = compute_metrics_at_threshold(y_win, probs, THRESHOLD)
    preds = m["preds"]

    st.success(f"✅ Processed **{len(X_win)} windows** ({len(X_win)*60/60:.0f} minutes of data)")

    # ── Live metrics @ current threshold ──
    st.markdown('<div class="section-header">📊 Performance at threshold = '
                f'{THRESHOLD:.2f}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
    c2.metric("Precision", f"{m['precision']*100:.1f}%")
    c3.metric("Recall",    f"{m['recall']*100:.1f}%")
    c4.metric("F1",        f"{m['f1']:.3f}")
    c5.metric("AUC",       f"{m['auc']:.3f}" if not np.isnan(m['auc']) else "—")

    st.caption("💡 Move the threshold slider in the sidebar — these metrics update live.")

    # ── Window-count summary ──
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Total Windows", f"{len(X_win):,}")
    cc2.metric("Predicted Stress", f"{preds.sum():,}", f"{100*preds.mean():.1f}%")
    cc3.metric("Actual Stress",    f"{y_win.sum():,}", f"{100*y_win.mean():.1f}%")
    cc4.metric("Match Rate",       f"{(preds == y_win).mean()*100:.1f}%")

    # ── Confusion matrix ──
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

    # ── Window-level timeline ──
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
    # Highlight true stress windows
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
               "Blue line = model's predicted probability. "
               "Red dashed line = current threshold.")

    # ── Threshold sweep table ──
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

    # ── Window picker for detailed inspection ──
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
        time_axis = np.arange(240) / 4   # 60 seconds @ 4Hz

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


# ═════════════════════════════════════════════════════════════
#  MODE 2 — DEMO DATA
# ═════════════════════════════════════════════════════════════
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
        prob = float(model.predict(win[None, ...], verbose=0).flatten()[0])
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


st.markdown("---")
st.caption(
    "Model trained on WESAD dataset · "
    "1D CNN with 60s windows, 75% overlap, per-subject normalization · "
    f"AUC {metrics.get('auc', 0):.3f} on subject-independent test set"
)