import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from groq import Groq
import os

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Forecasting Engine",
    page_icon="📈",
    layout="centered"
)

# ── STYLES ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
  background-color: #0a0a0a;
  color: #e8e2d8;
}
.title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.6rem; font-weight: 600;
  color: #e8e2d8; text-align: center;
  letter-spacing: .04em; margin-bottom: .25rem;
}
.subtitle {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .65rem; letter-spacing: .18em;
  text-transform: uppercase; color: #666;
  text-align: center; margin-bottom: 2rem;
}
.divider { border-top: 1px solid #1a1a1a; margin: 1.5rem 0; }
.section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .6rem; letter-spacing: .2em;
  text-transform: uppercase; color: #c8a96e;
  margin-bottom: .75rem; margin-top: 1rem;
}
.stat-card {
  background: #111; border: 1px solid #1e1e1e;
  border-radius: 6px; padding: 1rem;
  text-align: center; position: relative; overflow: hidden;
}
.stat-card::before {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #c8a96e, transparent);
}
.stat-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.4rem; font-weight: 600; color: #e8e2d8;
}
.stat-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .48rem; letter-spacing: .14em;
  text-transform: uppercase; color: #555; margin-top: .25rem;
}
.model-card {
  background: #111; border: 1px solid #1e1e1e;
  border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: .75rem;
}
.model-name {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .7rem; font-weight: 600; color: #c8a96e;
  letter-spacing: .1em; text-transform: uppercase;
}
.model-desc {
  font-size: .82rem; color: #888; margin-top: .2rem; line-height: 1.6;
}
.model-metric {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .75rem; color: #e8e2d8; margin-top: .4rem;
}
.ai-box {
  background: #0d0d1a; border: 1px solid #1e1e2e;
  border-radius: 6px; padding: 1.5rem;
  font-size: .92rem; line-height: 1.85; color: #c8c0d8;
}
.footer {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .48rem; letter-spacing: .14em;
  text-transform: uppercase; color: #333;
  text-align: center; margin-top: 3rem;
}
.mode-tag {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .5rem; letter-spacing: .12em; text-transform: uppercase;
  background: #1a1a0a; color: #c8a96e;
  border: 1px solid #2e2e0a;
  padding: .15rem .6rem; border-radius: 3px; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="title">📈 Forecasting Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Moving Average · Linear Trend · Exponential Smoothing · AI Commentary</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── GROQ ─────────────────────────────────────────────────────
API_KEY = os.environ.get("animeama", "")
if API_KEY:
    os.environ["GROQ_API_KEY"] = API_KEY
client = Groq()

# ── CHART THEME ──────────────────────────────────────────────
def dark_layout(fig, height=400):
    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#111",
        font=dict(color="#888", family="IBM Plex Mono", size=11),
        margin=dict(t=40, b=40, l=50, r=20),
        height=height,
        legend=dict(bgcolor="#111", bordercolor="#222",
                    font=dict(color="#888", size=10)),
        xaxis=dict(gridcolor="#1a1a1a", linecolor="#222",
                   tickfont=dict(color="#555")),
        yaxis=dict(gridcolor="#1a1a1a", linecolor="#222",
                   tickfont=dict(color="#555")),
    )
    return fig

# ── SAMPLE DATA ──────────────────────────────────────────────
def make_sample():
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    trend = np.linspace(1000, 1800, 36)
    seasonal = 120 * np.sin(np.arange(36) * 2 * np.pi / 12)
    noise = np.random.normal(0, 60, 36)
    return pd.DataFrame({
        "Date": dates,
        "Revenue": (trend + seasonal + noise).round(0)
    })

# ── FORECASTING MODELS ───────────────────────────────────────

def moving_average_forecast(series, window, periods):
    """Simple moving average — smoothed last N values extrapolated flat."""
    w = min(window, len(series))
    last_avg = series[-w:].mean()
    return np.full(periods, last_avg)

def linear_trend_forecast(series, periods):
    """OLS linear regression on index — extrapolate the trend line."""
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series, 1)
    future_x = np.arange(len(series), len(series) + periods)
    return np.polyval(coeffs, future_x)

def exp_smoothing_forecast(series, periods, alpha=0.3):
    """Exponential smoothing — recent values weighted more heavily."""
    s = series.copy().astype(float)
    smoothed = [s[0]]
    for v in s[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    level = smoothed[-1]
    # Simple trend from last 6 points
    if len(smoothed) >= 6:
        recent_trend = np.mean(np.diff(smoothed[-6:]))
    else:
        recent_trend = np.mean(np.diff(smoothed))
    forecast = [level + recent_trend * (i + 1) for i in range(periods)]
    return np.array(forecast)

def rmse(actual, predicted):
    n = min(len(actual), len(predicted))
    return np.sqrt(np.mean((actual[:n] - predicted[:n]) ** 2))

def mape(actual, predicted):
    n = min(len(actual), len(predicted))
    actual = actual[:n]
    predicted = predicted[:n]
    mask = actual != 0
    if mask.sum() == 0:
        return None
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# ── UPLOAD ───────────────────────────────────────────────────
st.markdown('<div class="section-label">01 · Upload your data</div>', unsafe_allow_html=True)

use_sample = st.checkbox("Use sample dataset (36-month revenue time series)", value=False)

if use_sample:
    df = make_sample()
    st.success(f"Sample loaded — {len(df)} rows · {len(df.columns)} columns")
else:
    uploaded = st.file_uploader(
        "Upload CSV or Excel — any dataset with a numeric column to forecast",
        type=["csv", "xlsx", "xls"]
    )
    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
                 else pd.read_excel(uploaded)
            st.success(f"Loaded — {len(df)} rows · {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        st.info("Upload a file or use the sample dataset to begin.")
        st.stop()

with st.expander("Preview data"):
    st.dataframe(df.head(10), use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── AUTO-DETECT COLUMNS ──────────────────────────────────────
num_cols  = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
date_cols = []
for c in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        date_cols.append(c)
    elif df[c].dtype == object:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.7:
            df[c] = parsed
            date_cols.append(c)

has_dates = len(date_cols) > 0

# ── CONFIGURE ────────────────────────────────────────────────
st.markdown('<div class="section-label">02 · Configure forecast</div>', unsafe_allow_html=True)

if has_dates:
    st.markdown('<div class="mode-tag">⏱ Time Series Mode — date column detected</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="mode-tag">🔢 Index Mode — no date column, using row index</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    metric_col = st.selectbox(
        "Column to forecast",
        num_cols,
        help="The numeric metric you want to project forward"
    )
with col2:
    if has_dates:
        date_col = st.selectbox("Date column", date_cols)
    else:
        date_col = None

col3, col4 = st.columns(2)
with col3:
    forecast_periods = st.slider(
        "Periods to forecast",
        min_value=3, max_value=24, value=6,
        help="How many future periods to predict"
    )
with col4:
    ma_window = st.slider(
        "Moving average window",
        min_value=2, max_value=12, value=3,
        help="Number of past periods to average for MA model"
    )

alpha_es = st.slider(
    "Exponential smoothing alpha (α)",
    min_value=0.05, max_value=0.95, value=0.30, step=0.05,
    help="Higher = more weight on recent values (0.1 = smooth, 0.9 = reactive)"
)

# ── RUN FORECAST ─────────────────────────────────────────────
if st.button("📈 Run Forecast", use_container_width=True):

    # Prepare series
    if date_col:
        df_sorted = df[[date_col, metric_col]].dropna().sort_values(date_col)
        series    = df_sorted[metric_col].values.astype(float)
        dates_hist = df_sorted[date_col].values
    else:
        df_sorted  = df[[metric_col]].dropna()
        series     = df_sorted[metric_col].values.astype(float)
        dates_hist = np.arange(len(series))

    if len(series) < 4:
        st.error("Need at least 4 data points to forecast.")
        st.stop()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── SUMMARY STATS ─────────────────────────────────────────
    st.markdown('<div class="section-label">03 · Historical summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    trend_dir = "▲" if series[-1] > series[0] else "▼"
    total_chg  = (series[-1] - series[0]) / series[0] * 100 if series[0] != 0 else 0

    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{len(series)}</div><div class="stat-label">Data points</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{series.mean():.1f}</div><div class="stat-label">Historical mean</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{series[-1]:.1f}</div><div class="stat-label">Last value</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{trend_dir} {abs(total_chg):.1f}%</div><div class="stat-label">Total change</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RUN THE 3 MODELS ──────────────────────────────────────
    ma_fc  = moving_average_forecast(series, ma_window, forecast_periods)
    lt_fc  = linear_trend_forecast(series, forecast_periods)
    es_fc  = exp_smoothing_forecast(series, forecast_periods, alpha_es)

    # Back-test on last 20% of history to get error metrics
    split    = max(int(len(series) * 0.8), len(series) - 6)
    train    = series[:split]
    test     = series[split:]
    n_test   = len(test)

    ma_bt  = moving_average_forecast(train, ma_window, n_test)
    lt_bt  = linear_trend_forecast(train, n_test)
    es_bt  = exp_smoothing_forecast(train, n_test, alpha_es)

    ma_rmse  = rmse(test, ma_bt);  ma_mape  = mape(test, ma_bt)
    lt_rmse  = rmse(test, lt_bt);  lt_mape  = mape(test, lt_bt)
    es_rmse  = rmse(test, es_bt);  es_mape  = mape(test, es_bt)

    # Best model by RMSE
    scores = {"Moving Average": ma_rmse, "Linear Trend": lt_rmse,
              "Exp. Smoothing": es_rmse}
    best_model = min(scores, key=scores.get)

    # ── BUILD FUTURE INDEX ────────────────────────────────────
    if date_col and len(dates_hist) > 1:
        last_date = pd.Timestamp(dates_hist[-1])
        # Infer frequency
        try:
            diffs = np.diff(pd.to_datetime(dates_hist).asi8)
            avg_diff_days = int(np.median(diffs) / 1e9 / 86400)
            if avg_diff_days <= 1:    freq = "D"
            elif avg_diff_days <= 8:  freq = "W"
            elif avg_diff_days <= 32: freq = "MS"
            elif avg_diff_days <= 95: freq = "QS"
            else:                     freq = "YS"
            future_dates = pd.date_range(
                last_date + pd.tseries.frequencies.to_offset(freq),
                periods=forecast_periods, freq=freq
            )
        except Exception:
            future_dates = pd.date_range(last_date, periods=forecast_periods+1,
                                         freq="MS")[1:]
        future_labels = future_dates
    else:
        future_labels = np.arange(len(series), len(series) + forecast_periods)
        dates_hist    = np.arange(len(series))

    # ── MAIN FORECAST CHART ───────────────────────────────────
    st.markdown('<div class="section-label">04 · Forecast comparison — all 3 models</div>', unsafe_allow_html=True)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=list(dates_hist), y=series,
        name="Historical", mode="lines+markers",
        line=dict(color="#e8e2d8", width=2),
        marker=dict(size=4, color="#e8e2d8")
    ))

    # Connector dots (last historical → first forecast)
    last_x = dates_hist[-1]
    last_y = series[-1]

    # Moving Average
    fig.add_trace(go.Scatter(
        x=[last_x] + list(future_labels),
        y=[last_y] + list(ma_fc),
        name=f"Moving Avg (w={ma_window})",
        mode="lines+markers",
        line=dict(color="#6a8fc8", width=2, dash="dot"),
        marker=dict(size=5)
    ))

    # Linear Trend
    fig.add_trace(go.Scatter(
        x=[last_x] + list(future_labels),
        y=[last_y] + list(lt_fc),
        name="Linear Trend",
        mode="lines+markers",
        line=dict(color="#c8a96e", width=2, dash="dash"),
        marker=dict(size=5)
    ))

    # Exponential Smoothing
    fig.add_trace(go.Scatter(
        x=[last_x] + list(future_labels),
        y=[last_y] + list(es_fc),
        name=f"Exp. Smoothing (α={alpha_es})",
        mode="lines+markers",
        line=dict(color="#6ab87a", width=2, dash="longdash"),
        marker=dict(size=5)
    ))

    # Forecast zone shading
    fig.add_vrect(
        x0=last_x, x1=future_labels[-1],
        fillcolor="rgba(200,169,110,.04)",
        layer="below", line_width=0
    )
    fig.add_vline(x=last_x, line_dash="dash",
                  line_color="rgba(200,169,110,.3)", line_width=1)

    fig = dark_layout(fig, height=420)
    fig.update_layout(
        xaxis_title="Period",
        yaxis_title=metric_col,
        title=dict(
            text=f"Forecast — {metric_col} · Next {forecast_periods} periods",
            font=dict(color="#c8a96e", size=13), x=0
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── MODEL SCORECARDS ──────────────────────────────────────
    st.markdown('<div class="section-label">05 · Model accuracy (back-test on last 20%)</div>', unsafe_allow_html=True)

    for name, fc_vals, rmse_val, mape_val, color in [
        (f"Moving Average (window={ma_window})", ma_fc, ma_rmse, ma_mape, "#6a8fc8"),
        ("Linear Trend",                         lt_fc, lt_rmse, lt_mape, "#c8a96e"),
        (f"Exponential Smoothing (α={alpha_es})", es_fc, es_rmse, es_mape, "#6ab87a"),
    ]:
        is_best = name.split(" (")[0].strip() == best_model or \
                  best_model in name
        badge   = " ✦ BEST FIT" if is_best else ""
        mape_str = f"{mape_val:.1f}%" if mape_val is not None else "N/A"
        next_val = fc_vals[0]
        st.markdown(f"""
        <div class="model-card" style="border-color:{'#c8a96e' if is_best else '#1e1e1e'};">
          <div class="model-name" style="color:{color};">{name}{badge}</div>
          <div class="model-metric">
            RMSE: {rmse_val:.2f} &nbsp;·&nbsp; MAPE: {mape_str} &nbsp;·&nbsp;
            Next period forecast: <strong style="color:#e8e2d8;">{next_val:.2f}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── FORECAST TABLE ────────────────────────────────────────
    st.markdown('<div class="section-label">06 · Forecast values</div>', unsafe_allow_html=True)

    forecast_df = pd.DataFrame({
        "Period":             [str(x)[:10] if hasattr(x, 'date') else str(x)
                               for x in future_labels],
        "Moving_Average":     ma_fc.round(2),
        "Linear_Trend":       lt_fc.round(2),
        "Exp_Smoothing":      es_fc.round(2),
        "Ensemble_Average":   ((ma_fc + lt_fc + es_fc) / 3).round(2)
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    # ── ENSEMBLE CHART ────────────────────────────────────────
    st.markdown('<div class="section-label">07 · Ensemble forecast (average of 3 models)</div>', unsafe_allow_html=True)

    ensemble = ((ma_fc + lt_fc + es_fc) / 3)
    spread   = np.std([ma_fc, lt_fc, es_fc], axis=0)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(dates_hist), y=series,
        name="Historical", mode="lines",
        line=dict(color="#888", width=1.5)
    ))
    # Confidence band
    fig2.add_trace(go.Scatter(
        x=list(future_labels) + list(future_labels)[::-1],
        y=list(ensemble + spread) + list(ensemble - spread)[::-1],
        fill="toself",
        fillcolor="rgba(200,169,110,.08)",
        line=dict(color="transparent"),
        name="Spread band", showlegend=True
    ))
    fig2.add_trace(go.Scatter(
        x=[last_x] + list(future_labels),
        y=[last_y] + list(ensemble),
        name="Ensemble forecast",
        mode="lines+markers",
        line=dict(color="#c8a96e", width=2.5),
        marker=dict(size=6, color="#c8a96e",
                    line=dict(color="#0a0a0a", width=1))
    ))
    fig2 = dark_layout(fig2, height=340)
    fig2.update_layout(
        xaxis_title="Period", yaxis_title=metric_col,
        title=dict(text="Ensemble forecast with spread band",
                   font=dict(color="#c8a96e", size=13), x=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── AI COMMENTARY ─────────────────────────────────────────
    st.markdown('<div class="section-label">08 · AI analyst commentary</div>', unsafe_allow_html=True)

    prompt = f"""You are a senior data analyst reviewing a forecasting result.

METRIC: {metric_col}
HISTORICAL DATA: {len(series)} periods, mean={series.mean():.2f}, 
  first={series[0]:.2f}, last={series[-1]:.2f}, 
  total change={total_chg:.1f}%

MODEL BACK-TEST RESULTS (lower RMSE = better):
- Moving Average (window={ma_window}): RMSE={ma_rmse:.2f}, MAPE={f'{ma_mape:.1f}%' if ma_mape else 'N/A'}
- Linear Trend: RMSE={lt_rmse:.2f}, MAPE={f'{lt_mape:.1f}%' if lt_mape else 'N/A'}
- Exponential Smoothing (α={alpha_es}): RMSE={es_rmse:.2f}, MAPE={f'{es_mape:.1f}%' if es_mape else 'N/A'}
- Best fit model: {best_model}

FORECAST (next {forecast_periods} periods):
- Moving Average next: {ma_fc[0]:.2f} → {ma_fc[-1]:.2f}
- Linear Trend next: {lt_fc[0]:.2f} → {lt_fc[-1]:.2f}
- Exp Smoothing next: {es_fc[0]:.2f} → {es_fc[-1]:.2f}
- Ensemble average next: {ensemble[0]:.2f} → {ensemble[-1]:.2f}

Provide a concise commentary covering:
1. What the historical trend shows (one sentence, specific numbers)
2. Which model to trust most and why
3. What the forecast implies for the next period
4. One key risk or assumption to watch
Be direct. No fluff. Max 150 words."""

    with st.spinner("Generating commentary..."):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            ai_text = response.choices[0].message.content
        except Exception:
            ai_text = "AI commentary unavailable. Check your Groq API key in Space secrets."

    st.markdown(f'<div class="ai-box">{ai_text}</div>', unsafe_allow_html=True)

    # ── DOWNLOAD ──────────────────────────────────────────────
    st.markdown('<div class="section-label">09 · Download forecast CSV</div>', unsafe_allow_html=True)

    st.download_button(
        "⬇ Download forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name=f"{metric_col}_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown('<div class="footer">Forecasting Engine · Moving Average + Linear Trend + Exp. Smoothing · Built by Venkatesh Srinivasan</div>', unsafe_allow_html=True)
