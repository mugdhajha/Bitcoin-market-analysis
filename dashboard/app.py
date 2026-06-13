from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from rag.rag_pipeline import ask_rag

st.set_page_config(
    page_title="BTC Quant Terminal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "bitcoin_dataset.csv"
FIGURES = ROOT / "figures"

GREEN = "#00ff88"
GREEN_2 = "#00c96b"
RED = "#ff4d6d"
AMBER = "#f5b942"
CYAN = "#31d7ff"
BG = "#030806"
PANEL = "#07110d"
GRID = "#163527"
MUTED = "#87a394"


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root { --green:#00ff88; --panel:#07110d; --border:#173c2b; --muted:#87a394; }
    html, body, [class*="css"] { font-family: "Inter", sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at 82% 4%, rgba(0,255,136,.08), transparent 25%),
            linear-gradient(rgba(0,255,136,.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,136,.018) 1px, transparent 1px),
            #030806;
        background-size: auto, 32px 32px, 32px 32px, auto;
        color: #dcebe3;
    }
    [data-testid="stSidebar"] {
        background: rgba(4,12,8,.97);
        border-right: 1px solid #173c2b;
    }
    [data-testid="stSidebar"] * { font-family: "IBM Plex Mono", monospace; }
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="collapsedControl"] span {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
        font-size: 1.35rem !important;
        line-height: 1 !important;
        width: 1.35rem !important;
        overflow: hidden !important;
        white-space: nowrap !important;
    }
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        width: 2.25rem !important;
        min-width: 2.25rem !important;
        overflow: hidden !important;
    }
    [data-testid="stHeader"] { background: rgba(3,8,6,.76); }
    .block-container { max-width: 1540px; padding-top: 1.8rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: "IBM Plex Mono", monospace !important; letter-spacing: -.04em; }
    h1 { color: #effff6 !important; }
    h2, h3 { color: #d8f8e7 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(9,27,18,.96), rgba(4,14,9,.96));
        border: 1px solid #173c2b;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: inset 0 1px 0 rgba(0,255,136,.06);
    }
    [data-testid="stMetricLabel"] { color: #87a394; font-family: "IBM Plex Mono", monospace; }
    [data-testid="stMetricValue"] {
        color: #f0fff7; font-family: "IBM Plex Mono", monospace;
        font-size: clamp(1.35rem, 2.25vw, 2rem);
    }
    [data-testid="stMetricDelta"] { font-family: "IBM Plex Mono", monospace; }
    .terminal-head {
        border: 1px solid #173c2b; border-left: 4px solid #00ff88; border-radius: 8px;
        background: linear-gradient(100deg, rgba(0,255,136,.09), rgba(0,255,136,.01));
        padding: 22px 24px; margin-bottom: 18px;
    }
    .terminal-kicker {
        font-family:"IBM Plex Mono",monospace; color:#00ff88; font-size:.76rem;
        letter-spacing:.18em; text-transform:uppercase; margin-bottom:8px;
    }
    .terminal-title {
        font-family:"IBM Plex Mono",monospace; font-weight:700; font-size:2.15rem;
        line-height:1.08; letter-spacing:-.06em; color:#f0fff7;
    }
    .terminal-copy { color:#9bb5a7; margin-top:8px; max-width:900px; }
    .status-dot { color:#00ff88; text-shadow:0 0 10px #00ff88; }
    .insight {
        background:rgba(7,17,13,.92); border:1px solid #173c2b; border-radius:8px;
        padding:16px 18px; min-height:118px;
    }
    .insight-label {
        color:#00ff88; font-family:"IBM Plex Mono",monospace; font-size:.72rem;
        letter-spacing:.12em; text-transform:uppercase; margin-bottom:8px;
    }
    .insight-value { color:#e8fff3; font-weight:600; font-size:1.02rem; }
    .insight-copy { color:#87a394; font-size:.86rem; margin-top:5px; }
    .section-tag {
        color:#00ff88; font-family:"IBM Plex Mono",monospace; font-size:.72rem;
        letter-spacing:.14em; text-transform:uppercase; margin:1.4rem 0 .25rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #173c2b; }
    .stTabs [data-baseweb="tab"] {
        font-family:"IBM Plex Mono",monospace; color:#87a394; background:#07110d;
        border:1px solid #173c2b; border-radius:6px 6px 0 0; padding:8px 16px;
    }
    .stTabs [aria-selected="true"] { color:#00ff88 !important; border-bottom-color:#00ff88 !important; }
    div[data-testid="stDataFrame"] { border:1px solid #173c2b; border-radius:8px; overflow:hidden; }
    .stSelectbox label, .stSlider label, .stDateInput label, .stToggle label {
        color:#a8c2b4 !important; font-family:"IBM Plex Mono",monospace !important; font-size:.78rem !important;
    }
    .stButton button { border:1px solid #00ff88; color:#00ff88; background:#07110d; }
    hr { border-color:#173c2b !important; }
    #MainMenu, footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
    data["Daily Return"] = data["Close"].pct_change()
    data["MA7"] = data["Close"].rolling(7).mean()
    data["MA30"] = data["Close"].rolling(30).mean()
    data["MA90"] = data["Close"].rolling(90).mean()
    data["Volatility"] = data["Daily Return"].rolling(30).std() * np.sqrt(365)
    data["Drawdown"] = data["Close"] / data["Close"].cummax() - 1

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD Histogram"] = data["MACD"] - data["MACD Signal"]

    std20 = data["Close"].rolling(20).std()
    data["BB Upper"] = data["Close"].rolling(20).mean() + 2 * std20
    data["BB Lower"] = data["Close"].rolling(20).mean() - 2 * std20
    data["Volume MA30"] = data["Volume"].rolling(30).mean()
    return data


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:+.2%}"


def plot_layout(fig: go.Figure, height: int = 430, legend=True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=32, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3,8,6,.35)",
        font=dict(family="IBM Plex Mono", color="#9bb5a7", size=11),
        hoverlabel=dict(bgcolor="#07110d", bordercolor=GREEN, font_color="#e8fff3"),
        legend=dict(orientation="h", y=1.09, x=0, bgcolor="rgba(0,0,0,0)"),
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="terminal-head">
            <div class="terminal-kicker"><span class="status-dot">●</span> {kicker}</div>
            <div class="terminal-title">{title}</div>
            <div class="terminal-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="insight">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_figure(filename: str, caption: str) -> None:
    path = FIGURES / filename
    if path.exists():
        st.image(str(path), use_container_width=True)
        st.caption(caption)


data = load_data()

with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 2px 14px">
            <div style="font-size:2.1rem;color:#00ff88;font-weight:700">₿//QNT</div>
            <div style="font-size:.72rem;letter-spacing:.16em;color:#87a394">BITCOIN RESEARCH TERMINAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Historical OHLCV · Daily · USD")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Terminal", "Market Structure", "Technical Lab", "Risk & Regimes", "Models", "Strategy", "Explainability","AI Analyst"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("##### GLOBAL WINDOW")
    presets = {
        "Full history": data["Date"].min(),
        "1 year": data["Date"].max() - pd.DateOffset(years=1),
        "3 years": data["Date"].max() - pd.DateOffset(years=3),
        "5 years": data["Date"].max() - pd.DateOffset(years=5),
    }
    preset = st.selectbox("Range preset", list(presets.keys()))
    default_start = max(presets[preset], data["Date"].min())
    dates = st.date_input(
        "Custom date range",
        value=(default_start.date(), data["Date"].max().date()),
        min_value=data["Date"].min().date(),
        max_value=data["Date"].max().date(),
    )
    log_scale = st.toggle("Log price scale", value=True)
    st.divider()
    st.markdown(
        f"""
        <div style="font-size:.72rem;color:#87a394;line-height:1.8">
        <span style="color:#00ff88">●</span> DATA ONLINE<br>
        {len(data):,} OBSERVATIONS<br>
        {data['Date'].min():%Y-%m-%d} → {data['Date'].max():%Y-%m-%d}
        </div>
        """,
        unsafe_allow_html=True,
    )

if isinstance(dates, (tuple, list)) and len(dates) == 2:
    start_date, end_date = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
else:
    start_date, end_date = default_start, data["Date"].max()

view = data.loc[data["Date"].between(start_date, end_date)].copy()
if view.empty:
    st.error("The selected window contains no observations.")
    st.stop()

latest = view.iloc[-1]
first = view.iloc[0]
window_return = latest["Close"] / first["Close"] - 1
latest_daily = latest["Daily Return"] if pd.notna(latest["Daily Return"]) else 0
annualized_return = (1 + window_return) ** (365 / max((latest["Date"] - first["Date"]).days, 1)) - 1
annualized_vol = view["Daily Return"].std() * np.sqrt(365)
sharpe = annualized_return / annualized_vol if annualized_vol else np.nan
max_dd = view["Close"].div(view["Close"].cummax()).sub(1).min()
high = view["Close"].max()


def price_chart(frame: pd.DataFrame, indicators=True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame["Date"], open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"],
            name="BTC-USD", increasing_line_color=GREEN, decreasing_line_color=RED,
        )
    )
    if indicators:
        for col, color in [("MA30", CYAN), ("MA90", AMBER)]:
            fig.add_trace(go.Scatter(x=frame["Date"], y=frame[col], name=col, line=dict(color=color, width=1.4)))
    fig.update_yaxes(type="log" if log_scale else "linear", tickprefix="$")
    return plot_layout(fig, 500)


if page == "Terminal":
    header(
        "SYSTEM / OVERVIEW",
        "Bitcoin Quant Research Terminal",
        "A decision-oriented view of the complete market analysis, technical research, predictive modeling, and strategy backtest.",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("LAST CLOSE", money(latest["Close"]), pct(latest_daily))
    c2.metric("WINDOW RETURN", f"{window_return:+,.0%}", f"since {first['Date']:%Y}")
    c3.metric("30D ANN. VOL", f"{latest['Volatility']:.1%}" if pd.notna(latest["Volatility"]) else "N/A")
    c4, c5 = st.columns(2)
    c4.metric("MAX DRAWDOWN", pct(max_dd))
    c5.metric("DISTANCE TO ATH", pct(latest["Close"] / high - 1))

    st.markdown('<div class="section-tag">MARKET TAPE</div>', unsafe_allow_html=True)
    st.plotly_chart(price_chart(view), use_container_width=True)

    a, b, c = st.columns(3)
    with a:
        insight("Best predictive model", "Random Forest · 52.09%", "Non-linear structure delivered a small but measurable out-of-sample edge.")
    with b:
        insight("Backtest result", "51.57% strategy return", "Outperformed the 23.50% buy-and-hold benchmark by 28.07 percentage points.")
    with c:
        insight("Core research finding", "Signal is weak, payoff can matter", "A modest hit rate still produced value by capturing larger profitable moves.")

    st.markdown('<div class="section-tag">RESEARCH SNAPSHOT</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 0.85])
    with left:
        annual = data.assign(Year=data["Date"].dt.year).groupby("Year").agg(
            Return=("Daily Return", lambda x: (1 + x.dropna()).prod() - 1),
            Volatility=("Daily Return", lambda x: x.std() * np.sqrt(365)),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=annual["Year"], y=annual["Return"], name="Annual return", marker_color=np.where(annual["Return"] >= 0, GREEN, RED)))
        fig.add_trace(go.Scatter(x=annual["Year"], y=annual["Volatility"], name="Annualized volatility", yaxis="y2", line=dict(color=AMBER, width=2)))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", gridcolor="rgba(0,0,0,0)"))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig, 390), use_container_width=True)
    with right:
        st.markdown("#### What the analysis established")
        st.markdown(
            """
            - Bitcoin delivered strong long-term growth through repeated bull and bear cycles.
            - Volatility clusters around major events and has generally moderated in recent years.
            - Volume and volatility had almost no linear relationship (`-0.0245` correlation).
            - Daily return, RSI, and volatility were the strongest Random Forest features.
            - More features and more complex models did not beat the baseline Random Forest.
            """
        )

elif page == "Market Structure":
    header("RESEARCH / MARKET STRUCTURE", "Price, Returns & Participation", "Explore Bitcoin's long-run price evolution, daily return distribution, volume behavior, and calendar effects.")
    t1, t2, t3 = st.tabs(["PRICE & VOLUME", "RETURN DISTRIBUTION", "CALENDAR MAP"])
    with t1:
        st.plotly_chart(price_chart(view), use_container_width=True)
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=view["Date"], y=view["Volume"], name="Volume", marker_color=GREEN_2, opacity=.55))
        vol_fig.add_trace(go.Scatter(x=view["Date"], y=view["Volume MA30"], name="30D volume average", line=dict(color=AMBER, width=2)))
        st.plotly_chart(plot_layout(vol_fig, 300), use_container_width=True)
    with t2:
        clean_returns = view["Daily Return"].dropna()
        q1, q2 = st.columns(2)
        q1.metric("AVG DAILY RETURN", pct(clean_returns.mean()))
        q2.metric("POSITIVE DAYS", f"{(clean_returns > 0).mean():.1%}")
        q3, q4 = st.columns(2)
        q3.metric("BEST DAY", pct(clean_returns.max()))
        q4.metric("WORST DAY", pct(clean_returns.min()))
        fig = px.histogram(clean_returns, nbins=90, marginal="box", color_discrete_sequence=[GREEN])
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(plot_layout(fig, 430, legend=False), use_container_width=True)
        best_idx, worst_idx = view["Daily Return"].idxmax(), view["Daily Return"].idxmin()
        st.caption(f"Window extremes: best {view.loc[best_idx, 'Date']:%Y-%m-%d} ({view.loc[best_idx, 'Daily Return']:+.2%}); worst {view.loc[worst_idx, 'Date']:%Y-%m-%d} ({view.loc[worst_idx, 'Daily Return']:+.2%}).")
    with t3:
        monthly = data.set_index("Date")["Close"].resample("ME").last().pct_change()
        heat = monthly.to_frame("Return")
        heat["Year"] = heat.index.year
        heat["Month"] = heat.index.strftime("%b")
        matrix = heat.pivot(index="Year", columns="Month", values="Return").reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
        fig = px.imshow(matrix, color_continuous_scale=[[0, RED], [.5, "#07110d"], [1, GREEN]], color_continuous_midpoint=0, aspect="auto", text_auto=".0%")
        fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%", title="Return"))
        st.plotly_chart(plot_layout(fig, 560, legend=False), use_container_width=True)

elif page == "Technical Lab":
    header("RESEARCH / TECHNICAL LAB", "Trend, Momentum & Volatility Signals", "Inspect the same indicator families used throughout feature engineering and predictive modeling.")
    indicator = st.selectbox("ACTIVE STUDY", ["Moving averages", "RSI (14)", "MACD", "Bollinger Bands"])
    if indicator == "Moving averages":
        fig = price_chart(view)
        st.plotly_chart(fig, use_container_width=True)
        trend = "BULLISH" if latest["MA30"] > latest["MA90"] else "BEARISH"
        a, b, c = st.columns(3)
        a.metric("MA30", money(latest["MA30"]) if pd.notna(latest["MA30"]) else "N/A")
        b.metric("MA90", money(latest["MA90"]) if pd.notna(latest["MA90"]) else "N/A")
        c.metric("TREND STATE", trend)
    elif indicator == "RSI (14)":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.68, .32], vertical_spacing=.05)
        fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], name="Close", line=dict(color=GREEN)), row=1, col=1)
        fig.add_trace(go.Scatter(x=view["Date"], y=view["RSI"], name="RSI", line=dict(color=CYAN)), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=.08, line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor=GREEN, opacity=.08, line_width=0, row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=RED, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=GREEN, row=2, col=1)
        st.plotly_chart(plot_layout(fig, 590), use_container_width=True)
        state = "OVERBOUGHT" if latest["RSI"] >= 70 else "OVERSOLD" if latest["RSI"] <= 30 else "NEUTRAL"
        st.metric("LATEST RSI STATE", f"{latest['RSI']:.1f} · {state}")
    elif indicator == "MACD":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.62, .38], vertical_spacing=.05)
        fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], name="Close", line=dict(color=GREEN)), row=1, col=1)
        fig.add_trace(go.Bar(x=view["Date"], y=view["MACD Histogram"], name="Histogram", marker_color=np.where(view["MACD Histogram"] >= 0, GREEN, RED)), row=2, col=1)
        fig.add_trace(go.Scatter(x=view["Date"], y=view["MACD"], name="MACD", line=dict(color=CYAN)), row=2, col=1)
        fig.add_trace(go.Scatter(x=view["Date"], y=view["MACD Signal"], name="Signal", line=dict(color=AMBER)), row=2, col=1)
        st.plotly_chart(plot_layout(fig, 590), use_container_width=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["Date"], y=view["BB Upper"], name="Upper band", line=dict(color=CYAN, width=1)))
        fig.add_trace(go.Scatter(x=view["Date"], y=view["BB Lower"], name="Lower band", line=dict(color=CYAN, width=1), fill="tonexty", fillcolor="rgba(49,215,255,.06)"))
        fig.add_trace(go.Scatter(x=view["Date"], y=view["Close"], name="Close", line=dict(color=GREEN, width=2)))
        fig.update_yaxes(type="log" if log_scale else "linear", tickprefix="$")
        st.plotly_chart(plot_layout(fig, 540), use_container_width=True)
    st.info("Research conclusion: these indicators capture meaningful market structure, but no added indicator improved out-of-sample accuracy beyond the baseline Random Forest.")

elif page == "Risk & Regimes":
    header("RESEARCH / RISK", "Volatility, Drawdown & Market Regimes", "Measure the changing risk profile of Bitcoin and inspect the project's two central volatility hypotheses.")
    a, b = st.columns(2)
    a.metric("ANN. VOLATILITY", f"{annualized_vol:.1%}")
    b.metric("MAX DRAWDOWN", pct(max_dd))
    c, d = st.columns(2)
    c.metric("WINDOW SHARPE", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
    d.metric("VOL / VOLUME CORR.", f"{data[['Volume','Volatility']].corr().iloc[0,1]:.4f}")
    tabs = st.tabs(["VOLATILITY TAPE", "DRAWDOWN", "YEARLY REGIMES", "HYPOTHESES"])
    with tabs[0]:
        fig = go.Figure(go.Scatter(x=view["Date"], y=view["Volatility"], fill="tozeroy", line=dict(color=AMBER), fillcolor="rgba(245,185,66,.12)"))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig, 450, legend=False), use_container_width=True)
    with tabs[1]:
        dd = view["Close"].div(view["Close"].cummax()).sub(1)
        fig = go.Figure(go.Scatter(x=view["Date"], y=dd, fill="tozeroy", line=dict(color=RED), fillcolor="rgba(255,77,109,.15)"))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig, 450, legend=False), use_container_width=True)
    with tabs[2]:
        yearly = data.assign(Year=data["Date"].dt.year).groupby("Year")["Daily Return"].std().mul(np.sqrt(365)).reset_index(name="Volatility")
        fig = px.bar(yearly, x="Year", y="Volatility", color="Volatility", color_continuous_scale=["#0b3d27", GREEN, AMBER])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig, 450, legend=False), use_container_width=True)
    with tabs[3]:
        x, y = st.columns(2)
        with x:
            insight("Hypothesis 01 · Rejected", "Volume does not explain volatility", "Observed correlation: -0.0245. There is virtually no linear relationship.")
        with y:
            insight("Hypothesis 02 · Supported", "Speculative cycles are more volatile", "2017, 2018, and 2021 were high-volatility years; recent years were calmer.")

elif page == "Models":
    header("ML / MODEL RESEARCH", "Out-of-Sample Model Performance", "Chronological 80/20 train-test evaluation. The narrow accuracy range demonstrates how difficult short-horizon direction prediction remains.")
    results = pd.DataFrame(
        {
            "Configuration": [
                "Logistic Regression", "Logistic + Momentum", "Logistic (7-Day Target)", "XGBoost",
                "RF + MACD ratio", "RF + Return 7D / 30D", "RF + BB position", "RF Top 3 Features", "Random Forest Baseline",
            ],
            "Accuracy": [49.22, 48.87, 50.66, 50.66, 51.25, 51.49, 51.61, 51.01, 52.09],
            "Family": ["Linear", "Linear", "Linear", "Boosting", "Random Forest", "Random Forest", "Random Forest", "Random Forest", "Random Forest"],
        }
    )
    a, b = st.columns(2)
    a.metric("BEST MODEL", "Random Forest")
    b.metric("BEST ACCURACY", "52.09%", "+2.87pp vs 50%")
    c, d = st.columns(2)
    c.metric("XGBOOST", "50.66%")
    d.metric("LOGISTIC", "49.22%")
    tab1, tab2, tab3 = st.tabs(["LEADERBOARD", "FEATURE IMPORTANCE", "EXPERIMENT TABLE"])
    with tab1:
        ordered = results.sort_values("Accuracy")
        fig = px.bar(ordered, x="Accuracy", y="Configuration", orientation="h", color="Family", color_discrete_map={"Linear": MUTED, "Boosting": AMBER, "Random Forest": GREEN})
        fig.add_vline(x=50, line_dash="dot", line_color=RED, annotation_text="random baseline")
        fig.update_xaxes(range=[47.5, 53], ticksuffix="%")
        st.plotly_chart(plot_layout(fig, 520), use_container_width=True)
    with tab2:
        importance = pd.DataFrame({"Feature": ["Daily Return", "RSI", "Volatility", "MA7 vs MA30 Ratio", "Close vs MA30 Ratio"], "Importance": [22.8, 20.9, 20.6, 18.8, 16.9]})
        fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#0b3d27", GREEN])
        fig.update_xaxes(ticksuffix="%")
        st.plotly_chart(plot_layout(fig, 430, legend=False), use_container_width=True)
        st.caption("Random Forest baseline feature importance. Daily Return was the strongest predictor, followed closely by RSI and Volatility.")
    with tab3:
        st.dataframe(results.sort_values("Accuracy", ascending=False), use_container_width=True, hide_index=True, column_config={"Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=45, max_value=55, format="%.2f%%")})

elif page == "Strategy":
    header("PORTFOLIO / BACKTEST", "ML Signal Strategy vs Buy & Hold", "Portfolio-level evaluation on unseen test data. A long-only position is held when the model predicts an upward move.")
    a, b = st.columns(2)
    a.metric("ML STRATEGY RETURN", "51.57%", "+28.07pp excess")
    b.metric("BUY & HOLD RETURN", "23.50%")
    c, d = st.columns(2)
    c.metric("SHARPE RATIO", "0.54")
    d.metric("MODEL ACCURACY", "52.09%")
    show_figure("strategy_backtest.png", "Documented equity curve from the completed out-of-sample backtest.")
    left, right = st.columns(2)
    with left:
        insight("The payoff lesson", "Accuracy is not profitability", "The strategy captured sufficiently large positive moves to outperform despite a modest classification hit rate.")
    with right:
        insight("Important limitation", "Research backtest, not live execution", "Transaction costs, slippage, liquidity, drift, and position sizing require production-grade validation.")
    st.markdown("#### Backtest construction")
    st.markdown(
        """
        `Prediction = 1` → hold Bitcoin for the next period  
        `Prediction = 0` → remain out of the market  
        Forward returns ensure each signal is evaluated only against future price movement.
        """
    )

elif page=="Explainability":
    header("ML / EXPLAINABILITY", "How the Random Forest Makes Decisions", "SHAP analysis connects model outputs to economically meaningful momentum, volatility, and trend features.")
    a, b, c = st.columns(3)
    with a:
        insight("Primary driver", "Daily Return", "Recent price movement had the largest average influence on predictions.")
    with b:
        insight("Risk context", "Volatility", "Market uncertainty materially changed the model's directional assessment.")
    with c:
        insight("Momentum context", "RSI", "Overbought and oversold conditions contributed meaningful signal.")
    tab1, tab2, tab3 = st.tabs(["GLOBAL IMPACT", "MEAN IMPORTANCE", "LOCAL DECISION"])
    with tab1:
        show_figure("shap_summary.png", "SHAP summary: feature values and their directional impact across observations.")
    with tab2:
        show_figure("shap_bar.png", "Mean absolute SHAP values rank global feature influence.")
    with tab3:
        show_figure("shap_waterfall.png", "Waterfall explanation: feature contributions for one individual prediction.")
    st.markdown("#### Interpretation")
    st.markdown(
        """
        The agreement between Random Forest feature importance and SHAP strengthens the interpretation that the model learned plausible market relationships rather than relying only on random correlations. Still, explanation quality does not eliminate the weak-signal and regime-change risks visible in the out-of-sample results.
        """
    )

elif page == "AI Analyst":

    header(
        "AI / RESEARCH ASSISTANT",
        "Bitcoin AI Analyst...",
        "Ask questions about current Bitcoin news. Responses are grounded in retrieved articles from the live news knowledge base."
    )

    query = st.text_input(
        "Ask the Bitcoin AI Analyst",
        placeholder="Why is Bitcoin rising today?"
    )

    if query:

        with st.spinner("Analyzing latest news..."):

            answer, docs = ask_rag(query)

        st.markdown(
            f"""
        <div style="
        background: rgba(7,17,13,.92);
        border:1px solid #173c2b;
        padding:20px;
        border-radius:10px;
        margin-bottom:20px;
        ">
        {answer}
        </div>
        """,
            unsafe_allow_html=True
        )

        st.markdown("### Sources")

        for doc in docs:

            title = doc["meta"]["title"]
            source = doc["meta"]["source"]
            url = doc["meta"]["url"]

            st.markdown(
                f"• [{title}]({url}) — {source}"
            )

st.divider()
st.markdown(
    f"<div style='font-family:IBM Plex Mono,monospace;color:#587466;font-size:.68rem'>BTC QUANT TERMINAL · SOURCE WINDOW {first['Date']:%Y-%m-%d} → {latest['Date']:%Y-%m-%d} · FOR RESEARCH PURPOSES</div>",
    unsafe_allow_html=True,
)
