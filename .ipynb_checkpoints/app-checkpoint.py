import os
import json
import io
from typing import Any, Optional, Tuple

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

APP_TITLE = "Monkeypox Detection Demo"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DEFAULT_LABELS = ["Monkeypox", "Non-Monkeypox"]

@st.cache_resource(show_spinner=False)
def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _safe_import(module: str):
    try:
        return __import__(module)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[Optional[Any], dict, Optional[list]]:
    cfg = load_config()
    framework = cfg.get("framework")  # one of: sklearn, keras, tensorflow, pytorch, onnx
    model_path = cfg.get("model_path")
    labels = cfg.get("labels") or DEFAULT_LABELS

    if not framework or not model_path or not os.path.exists(model_path):
        return None, cfg, labels

    model = None
    if framework == "sklearn":
        joblib = _safe_import("joblib") or _safe_import("sklearn.externals.joblib")
        if joblib:
            try:
                model = joblib.load(model_path)
            except Exception:
                model = None
    elif framework in ("keras", "tensorflow"):
        tf = _safe_import("tensorflow")
        if tf:
            try:
                model = tf.keras.models.load_model(model_path)
            except Exception:
                model = None
    elif framework == "pytorch":
        torch = _safe_import("torch")
        if torch:
            try:
                model = torch.jit.load(model_path) if model_path.endswith(".pt") or model_path.endswith(".pth") else None
            except Exception:
                model = None
    elif framework == "onnx":
        ort = _safe_import("onnxruntime")
        if ort:
            try:
                model = ort.InferenceSession(model_path)
            except Exception:
                model = None

    return model, cfg, labels


def preprocess_image(img: Image.Image, cfg: dict) -> np.ndarray:
    # Read preprocessing from config; provide safe defaults
    size = cfg.get("input_size", [224, 224])
    to_rgb = cfg.get("to_rgb", True)
    normalize = cfg.get("normalize", True)
    mean = np.array(cfg.get("mean", [0.485, 0.456, 0.406]))
    std = np.array(cfg.get("std", [0.229, 0.224, 0.225]))

    if to_rgb:
        img = img.convert("RGB")
    img = img.resize((size[1], size[0]))  # (H, W)
    arr = np.array(img).astype("float32") / 255.0
    if normalize:
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = (arr - mean) / std
        else:
            arr = (arr - mean.mean()) / std.mean()
    arr = np.transpose(arr, (2, 0, 1)) if arr.ndim == 3 else arr  # CHW if 3-ch
    arr = np.expand_dims(arr, 0)
    return arr


def predict(model: Any, x: np.ndarray, framework: Optional[str], labels: list) -> Tuple[str, float]:
    try:
        if model is None:
            return "Model not configured", 0.0

        if framework == "sklearn":
            # Expecting vector input; flatten as a baseline
            x2 = x.reshape((x.shape[0], -1))
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x2)[0]
                idx = int(np.argmax(probs))
                return labels[idx] if idx < len(labels) else str(idx), float(np.max(probs))
            pred = model.predict(x2)[0]
            return str(pred), 1.0

        if framework in ("keras", "tensorflow"):
            probs = model.predict(x, verbose=0)[0]
            if probs.ndim == 0:
                conf = float(probs)
                label = labels[1 if conf < 0.5 else 0]
                return label, float(conf if conf >= 0.5 else 1.0 - conf)
            idx = int(np.argmax(probs))
            return labels[idx] if idx < len(labels) else str(idx), float(np.max(probs))

        if framework == "pytorch":
            import torch  # local import
            with torch.no_grad():
                xt = torch.from_numpy(x)
                out = model(xt)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                return labels[idx] if idx < len(labels) else str(idx), float(np.max(probs))

        if framework == "onnx":
            import onnxruntime as ort
            input_name = model.get_inputs()[0].name
            out = model.run(None, {input_name: x})[0][0]
            if out.ndim == 0:
                conf = float(out)
                label = labels[1 if conf < 0.5 else 0]
                return label, float(conf if conf >= 0.5 else 1.0 - conf)
            idx = int(np.argmax(out))
            return labels[idx] if idx < len(labels) else str(idx), float(np.max(out))

        return "Unsupported framework", 0.0
    except Exception as e:
        return f"Prediction error: {e}", 0.0


def sidebar_info():
    st.sidebar.title("Configuration")
    cfg = load_config()
    st.sidebar.markdown("- Framework: **{}**".format(cfg.get("framework", "not set")))
    st.sidebar.markdown("- Model path: `{}`".format(cfg.get("model_path", "not set")))
    st.sidebar.markdown("- Input size: `{}`".format(cfg.get("input_size", [224, 224])))
    st.sidebar.markdown("- Labels: `{}`".format(", ".join(cfg.get("labels", DEFAULT_LABELS))))
    st.sidebar.info("To enable real predictions, add a `config.json` and your model under `models/`. See README.")


def page_overview():
    st.header("Overview")
    st.write("This app demonstrates a Monkeypox detection model. Upload an image to see a prediction. Use the tabs to explore methodology, results, and limitations.")
    st.markdown("""
    - Problem: Early detection of Monkeypox from skin lesion images.
    - Approach: Supervised classification with image preprocessing and CNN/ML model.
    - Note: This demo is for educational purposes and not for clinical use.
    """)


def page_demo():
    st.header("Interactive Prediction")
    model, cfg, labels = load_model()
    framework = cfg.get("framework") if cfg else None

    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Uploaded image", use_column_width=True)

        with st.spinner("Preprocessing..."):
            x = preprocess_image(image, cfg or {})
        st.caption(f"Preprocessed shape: {x.shape}")

        with st.spinner("Predicting..."):
            label, conf = predict(model, x, framework, labels)
        st.success(f"Prediction: {label}")
        st.progress(min(max(conf, 0.0), 1.0))
        st.caption(f"Confidence: {conf:.3f}")

        if model is None:
            st.warning("Model not configured. Add `config.json` and your model file to enable real predictions.")


def page_results():
    st.header("Results & Metrics")
    st.write("Drop plots (confusion matrix, ROC, PR) into `assets/` and reference them below.")
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    imgs = [f for f in os.listdir(assets_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(assets_dir) else []
    if not imgs:
        st.info("No images found in `assets/`. Place your metrics plots there to display.")
    else:
        for name in imgs:
            st.image(os.path.join(assets_dir, name), caption=name)


def page_methodology():
    st.header("Methodology")
    st.markdown("""
    - Data: Briefly describe dataset, splits, and preprocessing.
    - Model: Architecture/algorithm and training setup.
    - Evaluation: Metrics used and validation procedure.
    - Limitations: Potential biases, dataset coverage, and failure modes.
    - Ethics: Appropriate use, disclaimers, and data privacy.
    """)


def _parse_series_from_wide(df: pd.DataFrame) -> pd.Series:
    if 'Country' in df.columns:
        df2 = df.drop(columns=['Country'])
        totals = df2.sum(axis=0)
        s = totals.copy()
        s.index = pd.to_datetime(s.index, errors='coerce')
        s = s.dropna().sort_index()
        return s
    return pd.Series(dtype=float)


def _parse_series_from_two_col(df: pd.DataFrame) -> pd.Series:
    cols = [c.lower() for c in df.columns]
    if len(df.columns) >= 2:
        date_col = df.columns[0]
        val_col = df.columns[1]
        s = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col], errors='coerce'))
        s = s.dropna().astype(float).sort_index()
        return s
    return pd.Series(dtype=float)


def _detect_and_parse_series(df: pd.DataFrame) -> pd.Series:
    if 'Country' in df.columns:
        return _parse_series_from_wide(df)
    return _parse_series_from_two_col(df)


def _parse_series_from_timeline(df: pd.DataFrame) -> pd.Series:
    # For Worldwide_Case_Detection_Timeline: aggregate by Date_confirmation
    date_col_candidates = [c for c in df.columns if str(c).lower() in ("date_confirmation", "date_confirmation", "date")]
    if date_col_candidates:
        dc = date_col_candidates[0]
        s = df[[dc]].copy()
        s[dc] = pd.to_datetime(s[dc], errors='coerce')
        s = s.dropna()
        agg = s.groupby(dc).size().sort_index()
        agg = agg.asfreq('D', fill_value=0)
        return agg
    return pd.Series(dtype=float)


def _plot_history_forecast(s: pd.Series, fc_df: pd.DataFrame):
    x_hist = s.index.strftime('%Y-%m-%d').tolist()
    y_hist = s.values.tolist()
    x_for = fc_df['date'].dt.strftime('%Y-%m-%d').tolist()
    y_for = fc_df['mean'].tolist()
    y_lo = fc_df['lo'].tolist()
    y_hi = fc_df['hi'].tolist()

    hist = go.Scatter(x=x_hist, y=y_hist, mode='lines+markers', name='Total cases', line=dict(color='#2563eb'))
    band = go.Scatter(x=x_for + x_for[::-1], y=y_hi + y_lo[::-1], fill='toself', name='95% interval', line=dict(color='rgba(16,185,129,0.2)'), fillcolor='rgba(16,185,129,0.15)')
    fore = go.Scatter(x=x_for, y=y_for, mode='lines+markers', name='Forecast', line=dict(color='#16a34a'))

    fig = go.Figure(data=[hist, band, fore])
    fig.update_layout(title='Daily Total Cases and ARIMA Forecast', xaxis_title='Date', yaxis_title='Cases', margin=dict(t=40, r=20, b=40, l=50), legend=dict(orientation='h'))
    st.plotly_chart(fig, use_container_width=True)


def page_forecast():
    st.header("Time-Series Forecasting (ARIMA)")
    st.write("Upload a CSV and configure ARIMA to forecast Monkeypox daily total cases.")

    src = st.radio("Data source", ["Upload CSV/Excel", "Sample from assets"], horizontal=True)
    uploaded = None
    if src == "Upload CSV/Excel":
        uploaded = st.file_uploader("Upload CSV/Excel. Formats: Wide (Country + date cols), Two-column (Date,Total), or Timeline (Date_confirmation per row)", type=["csv", "xlsx"]) 

    p = st.number_input("p (AR order)", min_value=0, max_value=20, value=12)
    d = st.number_input("d (diff order)", min_value=0, max_value=3, value=2)
    q = st.number_input("q (MA order)", min_value=0, max_value=20, value=9)
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=90, value=14)

    if st.button("Run Forecast", type="primary"):
        try:
            if src == "Sample from assets":
                sample_path = os.path.join(os.path.dirname(__file__), 'assets', 'Daily_Country_Wise_Confirmed_Cases.csv')
                if not os.path.exists(sample_path):
                    st.error("Sample CSV not found under assets/.")
                    return
                df = pd.read_csv(sample_path)
            else:
                if uploaded is None:
                    st.warning("Please upload a CSV file.")
                    return
                if uploaded.name.lower().endswith('.xlsx'):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)

            # Try multiple parsers
            s = _detect_and_parse_series(df)
            if s.empty:
                s = _parse_series_from_timeline(df)
            if s.empty:
                st.error("Could not parse CSV. Ensure format matches Wide or Two-column examples.")
                return

            st.caption(f"Loaded {len(s)} dates from {s.index.min().date()} to {s.index.max().date()}")

            result = adfuller(s.dropna())
            st.write({"adf_stat": float(result[0]), "p_value": float(result[1]), "n_lags": int(result[2]), "n_obs": int(result[3])})

            try:
                decomp = seasonal_decompose(s, model='additive', period=7)
                st.line_chart(pd.DataFrame({"Observed": s}))
            except Exception:
                pass

            model = ARIMA(s, order=(int(p), int(d), int(q)))
            fitted = model.fit()
            forecast_res = fitted.get_forecast(steps=int(horizon))
            mean = forecast_res.predicted_mean
            conf = forecast_res.conf_int(alpha=0.05)
            fc_df = pd.DataFrame({
                'date': pd.date_range(start=s.index.max() + pd.Timedelta(days=1), periods=int(horizon), freq='D'),
                'mean': mean.values,
                'lo': conf.iloc[:, 0].values,
                'hi': conf.iloc[:, 1].values
            })

            _plot_history_forecast(s, fc_df)

            st.subheader("Model Summary (truncated)")
            st.text(str(fitted.summary())[:2000])
        except Exception as e:
            st.error(f"Error: {e}")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
    st.title(APP_TITLE)

    sidebar_info()

    tab_forecast, tab_overview, tab_demo, tab_results, tab_methodology = st.tabs([
        "Forecast", "Overview", "Interactive Demo", "Results", "Methodology"
    ])

    with tab_forecast:
        page_forecast()
    with tab_overview:
        page_overview()
    with tab_demo:
        page_demo()
    with tab_results:
        page_results()
    with tab_methodology:
        page_methodology()


