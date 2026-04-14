"""
RainSense — Streamlit App
Built from: pbl_predictions.ipynb
Models: Ridge Regression | ARIMA | Prophet
Horizons: 7-Day | 30-Day
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RainSense · Rainfall Forecasting",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0b1120; }
section[data-testid="stSidebar"] {
    background: #0d1626;
    border-right: 1px solid rgba(56,138,221,0.18);
}
[data-testid="metric-container"] {
    background: rgba(16,28,50,0.9);
    border: 1px solid rgba(56,138,221,0.22);
    border-radius: 10px; padding: 0.9rem 1rem;
}
[data-testid="stMetricValue"]  { color: #63b3ed; font-size: 1.5rem; font-weight: 600; }
[data-testid="stMetricLabel"]  { color: rgba(180,205,240,0.6); font-size: 0.78rem;
                                  letter-spacing: 0.06em; text-transform: uppercase; }
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,22,38,0.8); border-radius: 10px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: rgba(180,205,240,0.5); font-weight: 500; }
.stTabs [aria-selected="true"] {
    background: rgba(56,138,221,0.22) !important; color: #63b3ed !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a6ea8, #0d4f7c);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif; font-weight: 500; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2089d4, #1a6ea8);
    transform: translateY(-1px); box-shadow: 0 4px 18px rgba(56,138,221,0.35);
}
.hero {
    background: linear-gradient(135deg, rgba(26,110,168,0.14) 0%, rgba(13,79,124,0.07) 100%);
    border: 1px solid rgba(99,179,237,0.18); border-radius: 14px;
    padding: 1.8rem 2.2rem; margin-bottom: 1.6rem;
}
.hero h1 { font-size: 1.85rem; font-weight: 600; color: #e2e8f0; margin: 0; }
.hero p  { color: rgba(180,205,240,0.6); margin: 0.35rem 0 0; font-size: 0.92rem; }
.divider { border-top: 1px solid rgba(99,179,237,0.1); margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette (matches your notebook exactly) ────────────────────────────
BLUE   = "#378ADD"
ORANGE = "#D85A30"
GREEN  = "#1D9E75"
PURPLE = "#7F77DD"
GRAY   = "#888780"
RED    = "#D84040"

PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,22,38,0.7)",
    font=dict(family="Space Grotesk", color="#c4d4e8"),
    xaxis=dict(gridcolor="rgba(56,138,221,0.08)", linecolor="rgba(56,138,221,0.15)", zeroline=False),
    yaxis=dict(gridcolor="rgba(56,138,221,0.08)", linecolor="rgba(56,138,221,0.15)", zeroline=False),
    margin=dict(l=20, r=20, t=44, b=24),
    legend=dict(bgcolor="rgba(13,22,38,0.8)", bordercolor="rgba(56,138,221,0.2)", borderwidth=1),
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA  — identical generation logic from your Cell 0
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def build_dataset():
    np.random.seed(42)
    data = []
    for year in range(1981, 2018):
        for day in range(1, 366):
            rainfall = (np.random.gamma(shape=2.5, scale=20)
                        if 150 <= day <= 273
                        else np.random.gamma(shape=0.8, scale=5))
            data.append([year, day, round(rainfall, 2)])
    df = pd.DataFrame(data, columns=["Year", "Day_of_Year", "Rainfall_mm"])
    df["Date"] = pd.date_range(start="1981-01-01", periods=len(df), freq="D")
    df = df.set_index("Date")
    df["Decade"] = (df["Year"] // 10) * 10
    return df


@st.cache_data
def compute_stats(df):
    rainfall          = df["Rainfall_mm"].values
    rainfall_positive = rainfall[rainfall > 0]
    dists = {"Normal": stats.norm, "Lognormal": stats.lognorm,
             "Gamma": stats.gamma, "Weibull": stats.weibull_min}
    results, fitted_params = [], {}
    for name, dist in dists.items():
        try:
            fit_data = rainfall if name == "Normal" else rainfall_positive
            if name == "Weibull":
                params = dist.fit(fit_data, f0=1.5, floc=0)
            elif name in ("Lognormal", "Gamma"):
                params = dist.fit(fit_data, floc=0)
            else:
                params = dist.fit(fit_data)
            fitted_params[name] = params
            ks_stat, _ = stats.kstest(fit_data, lambda x: dist.cdf(x, *params))
            ad_stat    = (stats.anderson(fit_data, dist="norm").statistic
                          if name == "Normal" else np.nan)
            observed, bins = np.histogram(fit_data, bins=10)
            expected = np.clip(len(fit_data) * np.diff(dist.cdf(bins, *params)), 1e-6, 1e6)
            chi_sq   = np.sum((observed - expected)**2 / expected)
            if np.isfinite(ks_stat) and np.isfinite(chi_sq):
                results.append([name, ks_stat, ad_stat, chi_sq])
        except Exception:
            pass
    df_r = pd.DataFrame(results, columns=["Distribution","KS","AD","Chi"])
    ad_fill = float(df_r["AD"].dropna().max()) if len(df_r["AD"].dropna()) > 0 else 999.0
    df_r["AD"] = df_r["AD"].fillna(ad_fill)
    for col in ["KS","AD","Chi"]:
        df_r[f"{col}_rank"] = df_r[col].rank()
    df_r["Total_score"] = df_r[["KS_rank","AD_rank","Chi_rank"]].sum(axis=1)
    best_name   = df_r.loc[df_r["Total_score"].idxmin(), "Distribution"]
    return {
        "mean": np.mean(rainfall), "std": np.std(rainfall),
        "skewness": stats.skew(rainfall), "cv": (np.std(rainfall)/np.mean(rainfall))*100,
        "max": np.max(rainfall), "min": np.min(rainfall),
        "rankings": df_r, "best_name": best_name,
        "best_dist": dists[best_name], "best_params": fitted_params[best_name],
        "rainfall_positive": rainfall_positive,
        "dists": dists, "fitted_params": fitted_params,
    }


# ── Model fitting — your Cell 1 / Cell 2 logic, generalised by horizon ───────
@st.cache_data
def fit_ridge(df, horizon):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import Ridge as _Ridge
    series    = df["Rainfall_mm"]
    SEQ_LEN   = 30
    TEST_DAYS = 90
    scaler    = MinMaxScaler()
    scaled    = scaler.fit_transform(series.values.reshape(-1,1)).flatten()

    def make_seq(s, q):
        X, y = [], []
        for i in range(len(s)-q):
            X.append(s[i:i+q]); y.append(s[i+q])
        return np.array(X), np.array(y)

    Xa, ya = make_seq(scaled, SEQ_LEN)
    sp     = len(Xa) - TEST_DAYS
    ridge  = _Ridge(alpha=1.0)
    ridge.fit(Xa[:sp], ya[:sp])
    y_pred_te = scaler.inverse_transform(ridge.predict(Xa[sp:]).reshape(-1,1)).flatten()
    y_true_te = scaler.inverse_transform(ya[sp:].reshape(-1,1)).flatten()

    window = scaled[-SEQ_LEN:].tolist()
    fc = []
    for _ in range(horizon):
        xin  = np.array(window[-SEQ_LEN:]).reshape(1,-1)
        pred = np.clip(ridge.predict(xin)[0], 0, 1)
        fc.append(pred); window.append(pred)
    fc = np.clip(scaler.inverse_transform(np.array(fc).reshape(-1,1)).flatten(), 0, None)
    mae  = np.mean(np.abs(y_pred_te - y_true_te))
    rmse = np.sqrt(np.mean((y_pred_te - y_true_te)**2))
    return fc, y_pred_te, y_true_te, mae, rmse


@st.cache_data
def fit_arima(df, horizon):
    try:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        series    = df["Rainfall_mm"]
        TEST_DAYS = 90
        train, test = series.iloc[:-TEST_DAYS], series.iloc[-TEST_DAYS:]
        fit_te = _ARIMA(train.values, order=(2,0,2)).fit()
        tp     = np.clip(fit_te.forecast(steps=TEST_DAYS), 0, None)
        mae    = np.mean(np.abs(tp - test.values))
        rmse   = np.sqrt(np.mean((tp - test.values)**2))
        fit_full = _ARIMA(series.values, order=(2,0,2)).fit()
        fc_obj   = fit_full.get_forecast(steps=horizon)
        fc       = np.clip(fc_obj.predicted_mean, 0, None)
        ci       = np.clip(fc_obj.conf_int(), 0, None)
        return fc, ci, mae, rmse, True
    except ImportError:
        return np.full(horizon, np.nan), None, np.nan, np.nan, False


@st.cache_data
def fit_prophet(df, horizon):
    try:
        from prophet import Prophet
        series    = df["Rainfall_mm"]
        TEST_DAYS = 90
        pdf = series.reset_index().rename(columns={"Date":"ds","Rainfall_mm":"y"})
        pdf["y"] = pdf["y"].clip(lower=0)
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.1,
                    seasonality_mode="multiplicative")
        m.fit(pdf.iloc[:-TEST_DAYS])
        ft = m.predict(m.make_future_dataframe(periods=TEST_DAYS))
        yh = np.clip(ft["yhat"].iloc[-TEST_DAYS:].values, 0, None)
        mae  = np.mean(np.abs(yh - series.iloc[-TEST_DAYS:].values))
        rmse = np.sqrt(np.mean((yh - series.iloc[-TEST_DAYS:].values)**2))
        m2 = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                     daily_seasonality=False, changepoint_prior_scale=0.1,
                     seasonality_mode="multiplicative")
        m2.fit(pdf)
        ff   = m2.predict(m2.make_future_dataframe(periods=horizon))
        fc   = np.clip(ff["yhat"].iloc[-horizon:].values, 0, None)
        lo   = np.clip(ff["yhat_lower"].iloc[-horizon:].values, 0, None)
        hi   = np.clip(ff["yhat_upper"].iloc[-horizon:].values, 0, None)
        return fc, lo, hi, mae, rmse, True
    except ImportError:
        return np.full(horizon, np.nan), None, None, np.nan, np.nan, False


# ── Load data & stats ─────────────────────────────────────────────────────────
df   = build_dataset()
stat = compute_stats(df)
series    = df["Rainfall_mm"]
last_date = series.index[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌧️ RainSense")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    horizon_label = st.radio("Forecast Horizon", ["7-Day","30-Day"], horizontal=True)
    HORIZON = 7 if horizon_label == "7-Day" else 30
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=HORIZON, freq="D")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**Models**")
    use_ridge   = st.checkbox("Ridge Regression", value=True)
    use_arima   = st.checkbox("ARIMA(2,0,2)",     value=True)
    use_prophet = st.checkbox("Prophet",           value=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    LOOKBACK = st.slider("History window (days)", 30, 180,
                         90 if HORIZON == 30 else 60, 15)
    st.caption(f"Dataset: 1981-01-01 → {last_date.date()}")
    st.caption(f"{len(df):,} daily records · seed=42")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN MODELS
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Fitting models…"):
    ridge_fc, y_pred_te_r, y_true_te_r, mae_r, rmse_r = fit_ridge(df, HORIZON)

    if use_arima:
        arima_fc, arima_ci, mae_a, rmse_a, arima_ok = fit_arima(df, HORIZON)
    else:
        arima_fc = np.full(HORIZON, np.nan); arima_ci = None
        mae_a = rmse_a = np.nan; arima_ok = False

    if use_prophet:
        prophet_fc, prophet_lo, prophet_hi, mae_p, rmse_p, prophet_ok = fit_prophet(df, HORIZON)
    else:
        prophet_fc = np.full(HORIZON, np.nan); prophet_lo = prophet_hi = None
        mae_p = rmse_p = np.nan; prophet_ok = False

stack    = np.stack([f for f in [ridge_fc, arima_fc, prophet_fc] if not np.all(np.isnan(f))])
ensemble = np.nanmean(stack, axis=0)

pred_df = pd.DataFrame({
    "Date":        future_dates.strftime("%Y-%m-%d"),
    "Ridge_mm":    np.round(ridge_fc,   2),
    "ARIMA_mm":    np.round(arima_fc,   2),
    "Prophet_mm":  np.round(prophet_fc, 2),
    "Ensemble_mm": np.round(ensemble,   2),
})


# ═══════════════════════════════════════════════════════════════════════════════
# HERO + KPIs
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <h1>🌧️ RainSense — Rainfall Forecasting</h1>
  <p>Dataset 1981–2017 &nbsp;·&nbsp; Ridge Regression, ARIMA(2,0,2), Prophet
  &nbsp;·&nbsp; Currently: <strong style="color:#63b3ed">{horizon_label} Forecast</strong></p>
</div>
""", unsafe_allow_html=True)

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Records",        f"{len(df):,}")
k2.metric("Mean Rainfall",  f"{stat['mean']:.2f} mm")
k3.metric("Skewness",       f"{stat['skewness']:.3f}")
k4.metric("Best Fit",       stat["best_name"])
k5.metric("Ridge MAE",      f"{mae_r:.2f} mm")
k6.metric("Ensemble Peak",  f"{ensemble.max():.1f} mm",
          delta=f"Day {int(np.argmax(ensemble))+1}")

st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Forecast", "📊  Statistical Analysis", "🤖  Model Performance", "📋  Forecast Table"])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — FORECAST  (your Cell 1 / Cell 2 charts, rebuilt in Plotly)
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    hist_dates = series.index[-LOOKBACK:]
    hist_vals  = series.values[-LOOKBACK:]
    fut_x      = future_dates
    test_dates = series.index[-90:]

    # Main combined chart
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(
        x=hist_dates, y=hist_vals, name="Historical",
        fill="tozeroy", fillcolor="rgba(56,138,221,0.07)",
        line=dict(color=BLUE, width=1.5), opacity=0.85))
    fig_main.add_trace(go.Scatter(
        x=test_dates, y=y_pred_te_r, name="Ridge (test fit)",
        line=dict(color=ORANGE, width=1.2, dash="dot"), opacity=0.7))
    fig_main.add_vline(x=str(last_date.date()), line_dash="dash",
                       line_color="rgba(212,64,64,0.5)", line_width=1.5)
    fig_main.add_annotation(x=str(last_date.date()), y=1, yref="paper",
        text="▸ forecast", showarrow=False,
        font=dict(color=RED, size=10), xanchor="left", yanchor="top")

    if use_ridge:
        fig_main.add_trace(go.Scatter(
            x=fut_x, y=ridge_fc, name="Ridge",
            mode="lines+markers", marker=dict(size=5),
            line=dict(color=GREEN, width=2)))
    if use_arima and arima_ok:
        fig_main.add_trace(go.Scatter(
            x=fut_x, y=arima_fc, name="ARIMA",
            mode="lines+markers", marker=dict(size=5),
            line=dict(color=ORANGE, width=2)))
        fig_main.add_trace(go.Scatter(
            x=list(fut_x)+list(fut_x[::-1]),
            y=list(arima_ci[:,1])+list(arima_ci[:,0][::-1]),
            fill="toself", fillcolor="rgba(216,90,48,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="ARIMA 95% CI"))
    if use_prophet and prophet_ok:
        fig_main.add_trace(go.Scatter(
            x=fut_x, y=prophet_fc, name="Prophet",
            mode="lines+markers", marker=dict(size=5),
            line=dict(color=PURPLE, width=2)))
        fig_main.add_trace(go.Scatter(
            x=list(fut_x)+list(fut_x[::-1]),
            y=list(prophet_hi)+list(prophet_lo[::-1]),
            fill="toself", fillcolor="rgba(127,119,221,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Prophet CI"))
    fig_main.add_trace(go.Scatter(
        x=fut_x, y=ensemble, name="Ensemble avg",
        mode="lines+markers",
        line=dict(color=RED, width=2.5, dash="dash"),
        marker=dict(symbol="diamond", size=7, color=RED)))

    fig_main.update_layout(**PLOT_BG, height=420,
        title=f"{horizon_label} Rainfall Forecast — All Models",
        xaxis_title="Date", yaxis_title="Rainfall (mm)")
    st.plotly_chart(fig_main, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        # Side-by-side bar comparison (your Cell 1 Chart 4 equivalent)
        day_labels = [d.strftime("%d %b") for d in fut_x]
        x_idx = list(range(HORIZON))
        fig_bars = go.Figure()
        for name, fc, col, offset in [
                ("Ridge",   ridge_fc,   GREEN,  -0.25),
                ("ARIMA",   arima_fc,   ORANGE,  0.0),
                ("Prophet", prophet_fc, PURPLE,  0.25)]:
            if not np.all(np.isnan(fc)):
                fig_bars.add_trace(go.Bar(
                    x=[i + offset for i in x_idx], y=fc,
                    name=name, width=0.22,
                    marker_color=col, opacity=0.82))
        fig_bars.add_trace(go.Scatter(
            x=x_idx, y=ensemble, name="Ensemble",
            mode="lines+markers",
            line=dict(color=RED, width=2.2, dash="dash"),
            marker=dict(symbol="diamond", size=6, color=RED)))
        step = max(1, HORIZON // 10)
       # fig_bars.update_layout(**PLOT_BG, height=320, barmode="overlay",
        #    title="Model Comparison",
         #   xaxis=dict(tickvals=x_idx[::step], ticktext=day_labels[::step],
          #             **PLOT_BG["xaxis"]),
           # yaxis_title="Rainfall (mm)")
        #st.plotly_chart(fig_bars, use_container_width=True)

    with col_b:
        if HORIZON == 7:
            fig_ens = go.Figure(go.Bar(
                x=day_labels, y=ensemble,
                marker=dict(color=ensemble,
                            colorscale=[[0,"#0d4f7c"],[0.5,"#1a6ea8"],[1,"#63b3ed"]]),
                text=[f"{v:.1f}" for v in ensemble],
                textposition="outside", textfont=dict(color="#c4d4e8")))
            fig_ens.update_layout(**PLOT_BG, height=320,
                title="Ensemble — Daily Forecast",
                yaxis_title="Rainfall (mm)")
        else:
            # Weekly cumulative (your Cell 2 weekly totals)
            slices = [(0,7),(7,14),(14,21),(21,30)]
            wlabels = [f"Week {i+1}\n{future_dates[s].strftime('%d %b')}–{future_dates[min(e-1,HORIZON-1)].strftime('%d %b')}"
                       for i,(s,e) in enumerate(slices)]
            wtotals = [ensemble[s:e].sum() for s,e in slices]
            fig_ens = go.Figure(go.Bar(
                x=wlabels, y=wtotals,
                marker_color=[GREEN, BLUE, PURPLE, ORANGE],
                text=[f"{v:.1f} mm" for v in wtotals],
                textposition="outside", textfont=dict(color="#c4d4e8")))
            fig_ens.update_layout(**PLOT_BG, height=320,
                title="Weekly Cumulative Ensemble",
                yaxis_title="Cumulative Rainfall (mm)")
        st.plotly_chart(fig_ens, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — STATISTICAL ANALYSIS  (your Cell 0 charts rebuilt in Plotly)
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    best_dist   = stat["best_dist"]
    best_params = stat["best_params"]
    best_name   = stat["best_name"]
    rp          = stat["rainfall_positive"]
    clip98      = np.percentile(rp, 98)
    clipped     = rp[rp <= clip98]

    c1, c2 = st.columns(2)
    with c1:
        hist_y, hist_x = np.histogram(clipped, bins=45, density=True)
        bc = (hist_x[:-1]+hist_x[1:])/2
        x_pdf = np.linspace(0, clip98, 400)
        y_pdf = best_dist.pdf(x_pdf, *best_params)
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=bc, y=hist_y, name="Observed",
            marker_color=BLUE, opacity=0.55,
            marker_line=dict(color="white", width=0.3)))
        fig_h.add_trace(go.Scatter(x=x_pdf, y=y_pdf,
            name=f"{best_name} PDF", line=dict(color=ORANGE, width=2.5)))
        fig_h.update_layout(**PLOT_BG, height=300,
            title=f"Histogram + {best_name} PDF",
            xaxis_title="Rainfall (mm)", yaxis_title="Density")
        st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        month_bounds = [0,31,59,90,120,150,181,211,242,272,303,333,365]
        month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        m_avgs = []
        for i in range(12):
            mask = ((df["Day_of_Year"] > month_bounds[i]) &
                    (df["Day_of_Year"] <= month_bounds[i+1]))
            m_avgs.append(df.loc[mask, "Rainfall_mm"].mean())
        bar_cols = [ORANGE if 5 <= i <= 8 else BLUE for i in range(12)]
        fig_m = go.Figure(go.Bar(x=month_names, y=m_avgs, marker_color=bar_cols,
            marker_line=dict(color="white", width=0.3),
            text=[f"{v:.1f}" for v in m_avgs],
            textposition="outside", textfont=dict(color="#c4d4e8")))
        fig_m.update_layout(**PLOT_BG, height=300,
            title="Monthly Average Rainfall",
            xaxis_title="Month", yaxis_title="mm / day")
        st.plotly_chart(fig_m, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        annual   = df.groupby("Year")["Rainfall_mm"].sum().reset_index()
        ann_mean = annual["Rainfall_mm"].mean()
        z = np.polyfit(annual["Year"], annual["Rainfall_mm"], 1)
        p = np.poly1d(z)
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(
            x=annual["Year"], y=annual["Rainfall_mm"],
            fill="tozeroy", fillcolor="rgba(56,138,221,0.12)",
            mode="lines+markers", name="Annual total",
            marker=dict(size=5, color="white", line=dict(color=BLUE, width=1.5)),
            line=dict(color=BLUE, width=2)))
        fig_a.add_hline(y=ann_mean, line_dash="dash", line_color=ORANGE,
                        annotation_text=f"Mean {ann_mean:.0f} mm",
                        annotation_font_color=ORANGE)
        fig_a.add_trace(go.Scatter(
            x=annual["Year"], y=p(annual["Year"]),
            name="Trend", line=dict(color=GRAY, width=1.5, dash="dot")))
        fig_a.update_layout(**PLOT_BG, height=300,
            title="Annual Total Rainfall Trend",
            xaxis_title="Year", yaxis_title="mm / year")
        st.plotly_chart(fig_a, use_container_width=True)

    with c4:
        ks_data = stat["rankings"].set_index("Distribution")["KS"]
        ks_data = ks_data.reindex(list(stat["dists"].keys())).dropna()
        bc_ks   = [GREEN if d == best_name else GRAY for d in ks_data.index]
        fig_ks  = go.Figure(go.Bar(
            x=ks_data.values, y=list(ks_data.index), orientation="h",
            marker_color=bc_ks,
            text=[f"{v:.4f}" for v in ks_data.values],
            textposition="outside", textfont=dict(color="#c4d4e8")))
        fig_ks.update_layout(**PLOT_BG, height=300,
            title="KS Statistic by Distribution (lower = better)",
            xaxis_title="KS Statistic")
        fig_ks.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_ks, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fit_data = stat["rainfall_positive"]
        sorted_d = np.sort(fit_data)
        n_pts    = min(2000, len(sorted_d))
        idx      = np.linspace(0, len(sorted_d)-1, n_pts, dtype=int)
        emp_x    = sorted_d[idx]
        emp_y    = (idx+1) / len(sorted_d)
        fit_y    = best_dist.cdf(emp_x, *best_params)
        fig_cdf  = go.Figure()
        fig_cdf.add_trace(go.Scatter(x=emp_x, y=emp_y, mode="markers",
            name="Empirical CDF",
            marker=dict(size=3, color=BLUE, opacity=0.45)))
        fig_cdf.add_trace(go.Scatter(x=emp_x, y=fit_y, mode="lines",
            name=f"{best_name} CDF", line=dict(color=ORANGE, width=2.5)))
        fig_cdf.add_trace(go.Scatter(x=[0, emp_x.max()], y=[0,1], mode="lines",
            name="Perfect fit", line=dict(color=GRAY, width=1, dash="dot")))
        fig_cdf.update_layout(**PLOT_BG, height=300,
            title="Empirical vs Fitted CDF",
            xaxis_title="Rainfall (mm)", yaxis_title="Cumulative Probability")
        st.plotly_chart(fig_cdf, use_container_width=True)

    with c6:
        decades = sorted(df["Decade"].unique())
        colors_dec = [BLUE, GREEN, PURPLE, ORANGE]
        fig_box = go.Figure()
        for i, dec in enumerate(decades):
            vals = df[df["Decade"]==dec]["Rainfall_mm"].values
            fig_box.add_trace(go.Box(
                y=vals, name=f"{dec}s",
                marker_color=colors_dec[i % len(colors_dec)],
                line_color=colors_dec[i % len(colors_dec)],
                opacity=0.75, boxmean="sd",
                marker=dict(size=2, opacity=0.3)))
        fig_box.update_layout(**PLOT_BG, height=300,
            title="Rainfall Distribution by Decade",
            yaxis_title="Rainfall (mm)")
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Distribution Ranking")
    rank_show = stat["rankings"][["Distribution","KS","Chi","Total_score"]].sort_values("Total_score")
    rank_show = rank_show.rename(columns={"KS":"KS Stat","Chi":"Chi²","Total_score":"Score"})
    st.dataframe(rank_show.style.format({"KS Stat":"{:.4f}","Chi²":"{:.2f}","Score":"{:.0f}"}),
                 hide_index=True, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE  (your Cell 1/2 error section + residuals)
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    model_names = ["Ridge","ARIMA","Prophet"]
    mae_vals    = [mae_r, mae_a, mae_p]
    rmse_vals   = [rmse_r, rmse_a, rmse_p]
    colors_m    = [GREEN, ORANGE, PURPLE]

    fig_err = make_subplots(rows=1, cols=2,
        subplot_titles=["MAE (mm)", "RMSE (mm)"])
    for name, mae, rmse, col in zip(model_names, mae_vals, rmse_vals, colors_m):
        fig_err.add_trace(go.Bar(x=[name], y=[mae], marker_color=col, opacity=0.85,
            text=[f"{mae:.3f}" if np.isfinite(mae) else "N/A"],
            textposition="outside", textfont=dict(color="#c4d4e8"),
            showlegend=False), row=1, col=1)
        fig_err.add_trace(go.Bar(x=[name], y=[rmse], marker_color=col, opacity=0.55,
            marker_line=dict(color=col, width=1.5),
            text=[f"{rmse:.3f}" if np.isfinite(rmse) else "N/A"],
            textposition="outside", textfont=dict(color="#c4d4e8"),
            showlegend=False), row=1, col=2)
    fig_err.update_layout(**PLOT_BG, height=300,
        title="Model Error on Test Set (90 days)")
    st.plotly_chart(fig_err, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        ti = np.arange(len(y_true_te_r))
        fig_av = go.Figure()
        fig_av.add_trace(go.Scatter(x=ti, y=y_true_te_r, name="Actual",
            line=dict(color=BLUE, width=1.3), opacity=0.8))
        fig_av.add_trace(go.Scatter(x=ti, y=y_pred_te_r, name="Ridge Predicted",
            line=dict(color=ORANGE, width=1.3, dash="dot"), opacity=0.85))
        fig_av.update_layout(**PLOT_BG, height=290,
            title="Ridge — Actual vs Predicted (Test Set)",
            xaxis_title="Test Day", yaxis_title="Rainfall (mm)")
        st.plotly_chart(fig_av, use_container_width=True)

    with cb:
        residuals = y_true_te_r - y_pred_te_r
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=ti, y=residuals, mode="markers",
            marker=dict(size=3, color=PURPLE, opacity=0.6), name="Residual"))
        fig_res.add_hline(y=0, line_dash="dash", line_color=GRAY, line_width=1)
        fig_res.update_layout(**PLOT_BG, height=290,
            title="Ridge — Residuals (Actual − Predicted)",
            xaxis_title="Test Day", yaxis_title="Residual (mm)")
        st.plotly_chart(fig_res, use_container_width=True)

    perf_df = pd.DataFrame({
        "Model":    model_names,
        "MAE (mm)":  [f"{v:.3f}" if np.isfinite(v) else "N/A" for v in mae_vals],
        "RMSE (mm)": [f"{v:.3f}" if np.isfinite(v) else "N/A" for v in rmse_vals],
        "Status":    [
            "✅ Active",
            "✅ Active" if arima_ok   else "⚠️  pip install statsmodels",
            "✅ Active" if prophet_ok else "⚠️  pip install prophet",
        ],
    })
    st.dataframe(perf_df, hide_index=True, use_container_width=True)

    valid = [(n, m) for n, m in zip(model_names, mae_vals) if np.isfinite(m)]
    if valid:
        best_m = min(valid, key=lambda x: x[1])
        st.success(f"🏆 Best model (lowest MAE): **{best_m[0]}** — {best_m[1]:.3f} mm")
    if not arima_ok:
        st.info("ARIMA requires statsmodels: `pip install statsmodels`")
    if not prophet_ok:
        st.info("Prophet requires: `pip install prophet`")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — FORECAST TABLE + CSV download
# ───────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"{horizon_label} Forecast Table")
    disp = pred_df.copy()
    disp.index = range(1, len(disp)+1); disp.index.name = "Day"
    st.dataframe(
        disp.style.format({
            "Ridge_mm":"{:.2f}","ARIMA_mm":"{:.2f}",
            "Prophet_mm":"{:.2f}","Ensemble_mm":"{:.2f}"}
        ).background_gradient(subset=["Ensemble_mm"], cmap="Blues"),
        use_container_width=True)
    st.download_button(
        label=f"⬇️  Download {horizon_label} CSV",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name=f"rainfall_forecast_{HORIZON}day.csv",
        mime="text/csv")

    if HORIZON == 30:
        st.markdown("---")
        st.subheader("Weekly Cumulative Totals (Ensemble)")
        ens = pred_df["Ensemble_mm"].values
        rows = []
        for i, (s, e) in enumerate([(0,7),(7,14),(14,21),(21,30)], 1):
            e2 = min(e, HORIZON)
            rows.append({
                "Week":          f"Week {i}",
                "Period":        f"{future_dates[s].strftime('%d %b')} – {future_dates[e2-1].strftime('%d %b')}",
                "Total (mm)":    f"{ens[s:e2].sum():.2f}",
                "Daily Avg":     f"{ens[s:e2].mean():.2f} mm",
                "Peak Day":      f"{ens[s:e2].max():.2f} mm",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;font-size:0.78rem;color:rgba(180,205,240,0.28);'>"
    "RainSense · Built from pbl_predictions.ipynb · "
    "Ridge Regression · ARIMA(2,0,2) · Prophet · Dataset: 1981–2017 · seed=42"
    "</p>", unsafe_allow_html=True)
