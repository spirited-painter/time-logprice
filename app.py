"""
Time-LogPrice Trading System
============================
A Streamlit-based backtest system that fits a trend line (linear or log-linear)
to historical price data and executes multi-tier buy/sell rules based on
deviation from the fitted theoretical value.

Data sources: akshare (A-shares) / yfinance (universal)
"""

import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Time-LogPrice Trading System", layout="wide")
st.title("📈 Time-LogPrice Trading System")

# ---------------------------------------------------------------------------
# Preset tickers
# ---------------------------------------------------------------------------
PRESETS: dict[str, str] = {
    "创业板指 (399006.SZ)": "399006.SZ",
    "沪深300 (000300.SH)": "000300.SH",
    "上证指数 (000001.SH)": "000001.SH",
    "深证成指 (399001.SZ)": "399001.SZ",
    "中证500 (000905.SH)": "000905.SH",
    "纳斯达克 (^IXIC)": "^IXIC",
    "标普500 (^GSPC)": "^GSPC",
    "道琼斯 (^DJI)": "^DJI",
    "自定义 (Custom)": "__CUSTOM__",
}

# ---------------------------------------------------------------------------
# Helper – resolve ticker
# ---------------------------------------------------------------------------

def _resolve_ticker(raw: str) -> str:
    """Try to add .SH / .SZ suffix for bare A-share codes."""
    raw = raw.strip()
    if raw.startswith("^") or "." in raw:
        return raw
    # Pure digits → guess exchange
    if raw.isdigit():
        if raw.startswith("6"):
            return raw + ".SH"
        elif raw.startswith(("0", "3")):
            return raw + ".SZ"
    return raw


def _china_ticker_to_yahoo(ticker: str) -> str:
    """Convert A-share ticker to Yahoo Finance format.
    000300.SH -> 000300.SS    (Shanghai)
    399006.SZ -> 399006.SZ    (Shenzhen, same)
    """
    if ticker.endswith(".SH"):
        return ticker.replace(".SH", ".SS")
    return ticker


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resample_daily_to(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample a daily DataFrame (columns: date, close) to weekly or monthly."""
    if freq == "日线":
        return df
    df = df.set_index("date")
    rule = "W" if freq == "周线" else "ME"
    df_resampled = df["close"].resample(rule).last().dropna().reset_index()
    df_resampled.columns = ["date", "close"]
    return df_resampled


@st.cache_data(show_spinner="正在下载数据 …")
def load_data(ticker: str, freq: str, adjust: str, source: str) -> pd.DataFrame:
    """
    Download historical price data.
    Returns a DataFrame with columns: date, close
    """
    is_china = not ticker.startswith("^") and (
        ticker.endswith(".SH") or ticker.endswith(".SZ")
    )

    if source == "akshare - 东方财富" and is_china:
        return _load_eastmoney(ticker, freq, adjust)
    elif source == "akshare - 腾讯" and is_china:
        return _load_tencent(ticker, freq, adjust)
    elif source == "yfinance":
        if is_china:
            yahoo_ticker = _china_ticker_to_yahoo(ticker)
            return _load_yfinance(yahoo_ticker, freq)
        else:
            return _load_yfinance(ticker, freq)
    else:
        # 自动模式：东方财富 → 腾讯 → yfinance
        if is_china:
            for loader in [
                lambda: _load_eastmoney(ticker, freq, adjust),
                lambda: _load_tencent(ticker, freq, adjust),
                lambda: _load_yfinance(_china_ticker_to_yahoo(ticker), freq),
            ]:
                try:
                    return loader()
                except Exception:
                    continue
            raise ValueError("所有数据源均加载失败，请检查网络或代码")
        else:
            return _load_yfinance(ticker, freq)


def _load_eastmoney(ticker: str, freq: str, adjust: str) -> pd.DataFrame:
    """东方财富 via akshare: stock_zh_a_hist / index_zh_a_hist"""
    import akshare as ak

    code = ticker.split(".")[0]
    is_index = code.startswith(("000", "399")) and len(code) == 6

    freq_map = {"日线": "daily", "周线": "weekly", "月线": "monthly"}
    ak_period = freq_map.get(freq, "daily")

    if is_index:
        df = ak.index_zh_a_hist(
            symbol=code,
            period=ak_period,
            start_date="19900101",
            end_date=datetime.date.today().strftime("%Y%m%d"),
        )
        df = df.rename(columns={"日期": "date", "收盘": "close"})
    else:
        adj_map = {"前除权": "qfq", "后除权": "hfq"}
        ak_adjust = adj_map.get(adjust, "qfq")
        df = ak.stock_zh_a_hist(
            symbol=code,
            period=ak_period,
            start_date="19900101",
            end_date=datetime.date.today().strftime("%Y%m%d"),
            adjust=ak_adjust,
        )
        df = df.rename(columns={"日期": "date", "收盘": "close"})

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
    return df


def _load_tencent(ticker: str, freq: str, adjust: str) -> pd.DataFrame:
    """腾讯证券 via akshare: stock_zh_a_hist_tx / stock_zh_index_daily_tx
    Note: 腾讯 API only provides daily data; we resample if weekly/monthly is needed.
    """
    import akshare as ak

    code = ticker.split(".")[0]
    exchange = ticker.split(".")[-1]
    is_index = code.startswith(("000", "399")) and len(code) == 6

    # 腾讯 format: sh000300 / sz399006
    if exchange == "SH":
        tx_symbol = f"sh{code}"
    else:
        tx_symbol = f"sz{code}"

    if is_index:
        df = ak.stock_zh_index_daily_tx(symbol=tx_symbol)
        df = df.rename(columns={"date": "date", "close": "close"})
    else:
        adj_map = {"前除权": "qfq", "后除权": "hfq"}
        ak_adjust = adj_map.get(adjust, "qfq")
        df = ak.stock_zh_a_hist_tx(
            symbol=tx_symbol,
            start_date="19900101",
            end_date=datetime.date.today().strftime("%Y%m%d"),
            adjust=ak_adjust,
        )
        df = df.rename(columns={"日期": "date", "收盘": "close"})

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True)

    # 腾讯 only has daily data — resample if needed
    df = _resample_daily_to(df, freq)
    return df


def _load_yfinance(ticker: str, freq: str) -> pd.DataFrame:
    import yfinance as yf

    interval_map = {"日线": "1d", "周线": "1wk", "月线": "1mo"}
    yf_interval = interval_map.get(freq, "1d")

    tk = yf.Ticker(ticker)
    try:
        df = tk.history(period="max", interval=yf_interval)
    except Exception:
        df = tk.history(
            start="1900-01-01",
            end=datetime.date.today().strftime("%Y-%m-%d"),
            interval=yf_interval,
        )
    df = df.reset_index()

    if df.empty:
        raise ValueError(f"yfinance 未能获取到 {ticker} 的数据，请检查代码是否正确")

    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date", "Close": "close"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df.dropna(subset=["date", "close"])
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            f"yfinance 仅获取到 {len(df)} 条有效数据（需要至少 2 条）。"
            f"A 股标的建议切换数据源为 akshare。"
        )
    return df


# ===================================================================
# SIDEBAR
# ===================================================================
with st.sidebar:
    st.header("📋 数据设置")

    preset = st.selectbox("预设标的", list(PRESETS.keys()))
    if PRESETS[preset] == "__CUSTOM__":
        custom_code = st.text_input(
            "输入代码",
            placeholder="例如 600519 或 ^IXIC",
            help="A 股可以只输入数字代码，系统会自动补充 .SH/.SZ 后缀",
        )
        ticker = _resolve_ticker(custom_code) if custom_code else ""
    else:
        ticker = PRESETS[preset]

    freq = st.selectbox("数据频率", ["日线", "周线", "月线"], index=2)
    adjust = st.selectbox("除权方式 (A 股个股适用)", ["前除权", "后除权"])

    data_source = st.selectbox(
        "数据源",
        [
            "自动 (东方财富→腾讯→yfinance)",
            "akshare - 东方财富",
            "akshare - 腾讯",
            "yfinance",
        ],
        index=0,
        help="东方财富数据全但可能被限流；腾讯稳定但仅日线（自动重采样）；yfinance 适合美股",
    )

    load_btn = st.button("🔄  加载数据", type="primary", width="stretch")

# ===================================================================
# Session state init
# ===================================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "fit_done" not in st.session_state:
    st.session_state.fit_done = False

# ===================================================================
# Load data
# ===================================================================
if load_btn and ticker:
    try:
        st.session_state.df = load_data(ticker, freq, adjust, data_source)
        st.session_state.fit_done = False
        st.session_state.bt_result = None
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        st.info(
            "💡 提示：如果 akshare 加载失败，可以在左侧将数据源切换为 **yfinance** 后重试。"
        )
        st.stop()

if st.session_state.df is None:
    st.info("👈 请在左侧选择投资标的并点击 **加载数据**")
    st.stop()

df_raw: pd.DataFrame = st.session_state.df.copy()

# Data validation – drop any rows with NaT dates or NaN close
df_raw = df_raw.dropna(subset=["date", "close"]).reset_index(drop=True)
if len(df_raw) < 2:
    st.error("有效数据不足（至少需要 2 条）。请尝试切换数据源或标的。")
    st.stop()

# ===================================================================
# Section 1: Price charts
# ===================================================================
st.subheader("📊 价格走势")
st.caption(
    f"数据范围: **{df_raw['date'].min().strftime('%Y-%m-%d')}** → "
    f"**{df_raw['date'].max().strftime('%Y-%m-%d')}** "
    f"（共 {len(df_raw)} 条记录）"
)

col_lin, col_log = st.columns(2)

with col_lin:
    fig_lin = go.Figure()
    fig_lin.add_trace(
        go.Scatter(x=df_raw["date"], y=df_raw["close"], mode="lines", name="Close Price")
    )
    fig_lin.update_layout(
        title="Price (Linear Scale)",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
    )
    st.plotly_chart(fig_lin, width="stretch")

with col_log:
    fig_log = go.Figure()
    fig_log.add_trace(
        go.Scatter(x=df_raw["date"], y=df_raw["close"], mode="lines", name="Close Price")
    )
    fig_log.update_layout(
        title="Price (Log Scale)",
        xaxis_title="Date",
        yaxis_title="Price (log)",
        yaxis_type="log",
        height=400,
    )
    st.plotly_chart(fig_log, width="stretch")

# ===================================================================
# Section 2: Fitting parameters
# ===================================================================
st.divider()
st.subheader("📐 拟合参数设置")

fit_method = st.radio(
    "拟合方式",
    ["时间-价格线性拟合", "时间-价格对数线性拟合"],
    index=1,
    horizontal=True,
)

col_f1, col_f2 = st.columns(2)
with col_f1:
    fit_start = st.date_input(
        "拟合开始日期",
        value=df_raw["date"].min().date(),
        min_value=df_raw["date"].min().date(),
        max_value=df_raw["date"].max().date(),
    )
with col_f2:
    fit_end = st.date_input(
        "拟合结束日期",
        value=df_raw["date"].max().date(),
        min_value=df_raw["date"].min().date(),
        max_value=df_raw["date"].max().date(),
    )

calc_fit_btn = st.button("📏  计算拟合直线", type="primary")

# Perform fitting
if calc_fit_btn:
    mask = (df_raw["date"].dt.date >= fit_start) & (df_raw["date"].dt.date <= fit_end)
    df_fit = df_raw.loc[mask].copy()

    if len(df_fit) < 2:
        st.error("拟合区间内数据不足（至少需要 2 条）")
        st.stop()

    # Time index: 1, 2, 3, ...
    df_fit["t"] = np.arange(1, len(df_fit) + 1)

    if fit_method == "时间-价格线性拟合":
        # y = slope * t + intercept
        slope, intercept = np.polyfit(df_fit["t"].values, df_fit["close"].values, 1)
    else:
        # log(y) = slope * t + intercept
        slope, intercept = np.polyfit(
            df_fit["t"].values, np.log(df_fit["close"].values), 1
        )

    st.session_state.fit_done = True
    st.session_state.fit_slope = slope
    st.session_state.fit_intercept = intercept
    st.session_state.fit_method = fit_method
    st.session_state.fit_start = fit_start
    st.session_state.fit_end = fit_end
    st.session_state.bt_result = None

if st.session_state.fit_done:
    st.success("拟合完成！可以在下方微调斜率和截距。")
    col_s, col_i = st.columns(2)
    with col_s:
        slope_val = st.number_input(
            "斜率 (slope)",
            value=float(st.session_state.fit_slope),
            format="%.10f",
            key="slope_input",
        )
    with col_i:
        intercept_val = st.number_input(
            "截距 (intercept)",
            value=float(st.session_state.fit_intercept),
            format="%.10f",
            key="intercept_input",
        )

    # Allow user overrides to update session state
    st.session_state.fit_slope = slope_val
    st.session_state.fit_intercept = intercept_val

    # ===============================================================
    # Section 3: Trading rules
    # ===============================================================
    st.divider()
    st.subheader("⚙️ 交易规则设置")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        init_cash = st.number_input("初始资金", value=100000.0, step=10000.0, min_value=0.0)
    with col_r2:
        dca_amount = st.number_input("每期定投金额", value=5000.0, step=1000.0, min_value=0.0)
    with col_r3:
        bt_start = st.date_input(
            "回测起始日期",
            value=st.session_state.fit_start,
            min_value=df_raw["date"].min().date(),
            max_value=df_raw["date"].max().date(),
        )

    # --- Buy rules ---
    st.markdown("##### 📥 买入规则（市场低估时触发）")
    num_buy_tiers = st.number_input("买入档数", value=4, min_value=1, max_value=10, step=1)

    default_buy = [
        (20, 5.0),
        (25, 10.5),
        (30, 17.6),
        (35, 100.0),
    ]

    buy_tiers = []
    cols_buy = st.columns(int(num_buy_tiers))
    for i in range(int(num_buy_tiers)):
        with cols_buy[i]:
            d_pct = default_buy[i][0] if i < len(default_buy) else (20 + 5 * i)
            d_buy = default_buy[i][1] if i < len(default_buy) else 5.0
            pct = st.number_input(f"低估 ≥ (%)", value=float(d_pct), key=f"buy_pct_{i}", step=1.0)
            buy = st.number_input(f"买入现金 (%)", value=float(d_buy), key=f"buy_amt_{i}", step=0.5)
            buy_tiers.append((pct, buy))

    # --- Sell rules ---
    st.markdown("##### 📤 卖出规则（市场高估时触发）")
    num_sell_tiers = st.number_input("卖出档数", value=4, min_value=1, max_value=10, step=1)

    default_sell = [
        (25, 5.0),
        (30, 10.5),
        (35, 17.6),
        (40, 100.0),
    ]

    sell_tiers = []
    cols_sell = st.columns(int(num_sell_tiers))
    for i in range(int(num_sell_tiers)):
        with cols_sell[i]:
            d_pct = default_sell[i][0] if i < len(default_sell) else (25 + 5 * i)
            d_sell = default_sell[i][1] if i < len(default_sell) else 5.0
            pct = st.number_input(f"高估 ≥ (%)", value=float(d_pct), key=f"sell_pct_{i}", step=1.0)
            sell = st.number_input(f"卖出持仓 (%)", value=float(d_sell), key=f"sell_amt_{i}", step=0.5)
            sell_tiers.append((pct, sell))

    # Sort tiers
    buy_tiers.sort(key=lambda x: x[0])
    sell_tiers.sort(key=lambda x: x[0])

    run_bt_btn = st.button("🚀  运行回测分析", type="primary", width="stretch")

    # ===============================================================
    # BACKTEST ENGINE
    # ===============================================================
    if run_bt_btn:
        slope = st.session_state.fit_slope
        intercept = st.session_state.fit_intercept
        method = st.session_state.fit_method
        f_start = st.session_state.fit_start
        f_end = st.session_state.fit_end

        # Build fitting mask to assign time index
        mask_fit = (df_raw["date"].dt.date >= f_start) & (df_raw["date"].dt.date <= f_end)
        df_fit_range = df_raw.loc[mask_fit].copy().reset_index(drop=True)
        df_fit_range["t"] = np.arange(1, len(df_fit_range) + 1)

        # Create a date->t mapping
        date_to_t = dict(zip(df_fit_range["date"].dt.date, df_fit_range["t"]))

        # Full data for backtest
        mask_bt = df_raw["date"].dt.date >= bt_start
        df_bt = df_raw.loc[mask_bt].copy().reset_index(drop=True)

        if len(df_bt) == 0:
            st.error("回测区间没有数据")
            st.stop()

        # Assign t values – extend for dates beyond fit range
        # Build a full t-index for the entire dataset within [f_start, max_date]
        mask_full = df_raw["date"].dt.date >= f_start
        df_full = df_raw.loc[mask_full].copy().reset_index(drop=True)
        df_full["t"] = np.arange(1, len(df_full) + 1)
        full_date_to_t = dict(zip(df_full["date"].dt.date, df_full["t"]))

        # Now map onto df_bt
        df_bt["t"] = df_bt["date"].dt.date.map(full_date_to_t)
        # For any NaN t (bt_start might be before f_start), use position
        if df_bt["t"].isna().any():
            df_bt["t"] = np.arange(1, len(df_bt) + 1)

        is_log = method == "时间-价格对数线性拟合"

        # Theoretical value
        if is_log:
            df_bt["theo_log"] = slope * df_bt["t"] + intercept
            df_bt["theo"] = np.exp(df_bt["theo_log"])
            df_bt["close_log"] = np.log(df_bt["close"])
            df_bt["ratio"] = df_bt["close_log"] / df_bt["theo_log"]
        else:
            df_bt["theo"] = slope * df_bt["t"] + intercept
            df_bt["ratio"] = df_bt["close"] / df_bt["theo"]

        df_bt["deviation_pct"] = (df_bt["ratio"] - 1.0) * 100.0  # positive = 高估, negative = 低估

        # ---------- Simulate trading ----------
        cash = float(init_cash)
        shares = 0.0
        total_invested = float(init_cash)
        records = []
        nav_history = []
        hist_max_dd = 0.0        # Track the running historical maximum drawdown
        nav_peak = 0.0           # Track the peak portfolio value for drawdown calc

        for idx, row in df_bt.iterrows():
            period_num = idx + 1

            # Add DCA (starting from second period)
            if period_num > 1:
                cash += dca_amount
                total_invested += dca_amount

            price = row["close"]
            dev = row["deviation_pct"]

            # Determine action
            buy_pct_to_use = 0.0
            sell_pct_to_use = 0.0

            if dev < 0:
                # Undervalued → check buy tiers (highest threshold first)
                abs_dev = abs(dev)
                for threshold, pct in reversed(buy_tiers):
                    if abs_dev >= threshold:
                        buy_pct_to_use = pct
                        break
            elif dev > 0:
                # Overvalued → check sell tiers
                for threshold, pct in reversed(sell_tiers):
                    if dev >= threshold:
                        sell_pct_to_use = pct
                        break

            # Cap at 100%
            buy_pct_to_use = min(buy_pct_to_use, 100.0)
            sell_pct_to_use = min(sell_pct_to_use, 100.0)

            # Execute
            if buy_pct_to_use > 0 and cash > 0:
                buy_amount_cash = cash * (buy_pct_to_use / 100.0)
                shares_bought = buy_amount_cash / price
                cash -= buy_amount_cash
                shares += shares_bought

            if sell_pct_to_use > 0 and shares > 0:
                sell_amount_shares = shares * (sell_pct_to_use / 100.0)
                cash += sell_amount_shares * price
                shares -= sell_amount_shares

            # Portfolio value
            portfolio_value = cash + shares * price
            position = (shares * price) / portfolio_value if portfolio_value > 0 else 0.0
            profit = portfolio_value - total_invested

            nav_history.append(portfolio_value)

            # Annualized return (from bt_start)
            days_elapsed = (row["date"].date() - bt_start).days
            years_elapsed = days_elapsed / 365.25 if days_elapsed > 0 else 0
            if years_elapsed > 0 and total_invested > 0:
                ann_return = (portfolio_value / total_invested) ** (1.0 / years_elapsed) - 1.0
            else:
                ann_return = 0.0

            # Historical max drawdown (running maximum of all peak-to-trough drawdowns)
            if portfolio_value > nav_peak:
                nav_peak = portfolio_value
            current_dd = (nav_peak - portfolio_value) / nav_peak if nav_peak > 0 else 0.0
            if current_dd > hist_max_dd:
                hist_max_dd = current_dd

            # Past 3-year max drawdown
            three_yr_ago = row["date"] - pd.DateOffset(years=3)
            recent_navs = []
            for j in range(len(nav_history)):
                if df_bt.loc[j, "date"] >= three_yr_ago:
                    recent_navs.append(nav_history[j])
            if recent_navs:
                r_peak = recent_navs[0]
                r_max_dd = 0.0
                for v in recent_navs:
                    if v > r_peak:
                        r_peak = v
                    dd = (r_peak - v) / r_peak if r_peak > 0 else 0.0
                    r_max_dd = max(r_max_dd, dd)
                three_yr_dd = r_max_dd
            else:
                three_yr_dd = 0.0

            # Net value index (start = 100, based on initial capital)
            # This equals portfolio_value / init_cash * 100
            # When DCA=0 and no trades: nav_index stays at 100
            # When DCA=0 and trades happen: nav_index reflects pure investment return
            nav_index = (portfolio_value / init_cash) * 100.0 if init_cash > 0 else 100.0

            # Period return (vs previous period)
            if period_num == 1:
                monthly_return = 0.0
            else:
                prev_val = nav_history[-2] if len(nav_history) >= 2 else portfolio_value
                monthly_return = (portfolio_value - prev_val) / prev_val if prev_val > 0 else 0.0

            record = {
                "时间": row["date"].strftime("%Y-%m-%d"),
                "序号": period_num,
                "收盘价": round(price, 4),
                "理论值": round(row["theo"], 4),
            }
            if is_log:
                record["收盘价对数值"] = round(row["close_log"], 6)
                record["理论值对数值"] = round(row["theo_log"], 6)

            record.update({
                "实际理论比值": round(row["ratio"], 6),
                "高估低估%": round(dev, 2),
                "持有股票数量": round(shares, 4),
                "现金": round(cash, 2),
                "仓位": round(position * 100, 2),
                "累计投资": round(total_invested, 2),
                "收益": round(profit, 2),
                "当期收益率": f"{monthly_return * 100:.2f}%",
                "年化收益率": f"{ann_return * 100:.2f}%",
                "历史最大回撤": f"{hist_max_dd * 100:.2f}%",
                "3年最大回撤": f"{three_yr_dd * 100:.2f}%",
                "净值指数": round(nav_index, 2),
            })
            records.append(record)

        df_result = pd.DataFrame(records)
        st.session_state.bt_result = df_result
        st.session_state.bt_df = df_bt
        st.session_state.bt_nav = nav_history
        st.session_state.bt_is_log = is_log
        st.session_state.bt_start_date = bt_start

    # ===============================================================
    # OUTPUT — only shown after backtest
    # ===============================================================
    if st.session_state.get("bt_result") is not None:
        df_result = st.session_state.bt_result
        df_bt = st.session_state.bt_df
        nav_history = st.session_state.bt_nav
        is_log = st.session_state.bt_is_log

        st.divider()
        st.subheader("📋 回测结果")
        st.dataframe(df_result, width="stretch", height=500)

        # ---------- CSV Downloads ----------
        col_dl1, col_dl2 = st.columns(2)

        # CSV 1 – data
        csv1 = df_result.to_csv(index=False).encode("utf-8-sig")
        with col_dl1:
            st.download_button(
                "💾  下载数据 CSV",
                data=csv1,
                file_name="backtest_data.csv",
                mime="text/csv",
                width="stretch",
            )

        # CSV 2 – formula explanations
        formula_rows = []
        for i, _ in df_result.iterrows():
            row_num = i + 2  # Excel row (header = row 1)
            f = {
                "时间": "原始数据日期",
                "序号": f"=ROW()-1",
                "收盘价": "原始数据收盘价",
                "理论值": (
                    f"=EXP(斜率*B{row_num}+截距)"
                    if is_log
                    else f"=斜率*B{row_num}+截距"
                ),
            }
            if is_log:
                f["收盘价对数值"] = f"=LN(C{row_num})"
                f["理论值对数值"] = f"=斜率*B{row_num}+截距"

            ratio_formula = (
                f"=收盘价对数值/理论值对数值" if is_log else f"=收盘价/理论值"
            )
            f["实际理论比值"] = ratio_formula
            f["高估低估%"] = "=(实际理论比值-1)*100"
            f["持有股票数量"] = "=上期持股 + 买入股数 - 卖出股数"
            f["现金"] = "=上期现金 + 定投 - 买入金额 + 卖出金额"
            f["仓位"] = "=持股市值/(持股市值+现金)*100"
            f["累计投资"] = "=上期累计投资 + 定投金额"
            f["收益"] = "=(现金+持股市值) - 累计投资"
            f["当期收益率"] = "=(本期总资产-上期总资产)/上期总资产*100"
            f["年化收益率"] = "=(总资产/累计投资)^(1/经过年数)-1)*100"
            f["历史最大回撤"] = "=MAX(所有历史期的(峰值-当期值)/峰值)*100"
            f["3年最大回撤"] = "=过去3年内MAX((峰值-谷底)/峰值)*100"
            f["净值指数"] = "=当前总资产/初始资金*100"
            formula_rows.append(f)

        df_formula = pd.DataFrame(formula_rows)
        csv2 = df_formula.to_csv(index=False).encode("utf-8-sig")
        with col_dl2:
            st.download_button(
                "📝  下载公式说明 CSV",
                data=csv2,
                file_name="backtest_formulas.csv",
                mime="text/csv",
                width="stretch",
            )

        # ---------- Chart 1: Price + Theoretical + Position ----------
        st.divider()
        st.subheader("📈 可视化分析")

        dates = pd.to_datetime(df_result["时间"])
        closes = df_result["收盘价"]
        theos = df_result["理论值"]
        positions = df_result["仓位"]

        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(
            go.Scatter(x=dates, y=closes, name="Close Price", line=dict(color="#2196F3")),
            secondary_y=False,
        )
        fig1.add_trace(
            go.Scatter(
                x=dates,
                y=theos,
                name="Theoretical Value",
                line=dict(color="#FF9800", dash="dash"),
            ),
            secondary_y=False,
        )
        fig1.add_trace(
            go.Scatter(
                x=dates,
                y=positions,
                name="Position %",
                line=dict(color="#4CAF50", width=1),
                fill="tozeroy",
                opacity=0.3,
            ),
            secondary_y=True,
        )
        fig1.update_layout(
            title="Price vs Theoretical Value & Position",
            height=500,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        )
        fig1.update_xaxes(title_text="Date")
        fig1.update_yaxes(title_text="Price", secondary_y=False)
        fig1.update_yaxes(title_text="Position (%)", secondary_y=True, range=[0, 105])
        st.plotly_chart(fig1, width="stretch")

        # ---------- Chart 2: Strategy NAV vs Index ----------
        nav_series = np.array(nav_history)
        nav_pct = (nav_series / nav_series[0]) * 100.0
        index_pct = (closes.values / closes.values[0]) * 100.0

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=dates,
                y=nav_pct,
                name="Strategy NAV Index",
                line=dict(color="#E91E63"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=dates,
                y=index_pct,
                name="Actual Price Index",
                line=dict(color="#9E9E9E"),
            )
        )
        fig2.update_layout(
            title="Strategy NAV vs Actual Price Index (Start = 100%)",
            xaxis_title="Date",
            yaxis_title="Growth (%)",
            height=450,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        )
        st.plotly_chart(fig2, width="stretch")

        # ---------- Chart 3: Ratio distribution ----------
        ratios = df_result["实际理论比值"].values

        fig3 = go.Figure()
        fig3.add_trace(
            go.Histogram(
                x=ratios,
                nbinsx=50,
                marker_color="#7C4DFF",
                opacity=0.85,
                name="Frequency",
            )
        )
        fig3.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Fair Value (1.0)")
        fig3.update_layout(
            title="Distribution of Actual / Theoretical Ratio",
            xaxis_title="Actual / Theoretical Ratio",
            yaxis_title="Frequency",
            height=400,
        )
        st.plotly_chart(fig3, width="stretch")
