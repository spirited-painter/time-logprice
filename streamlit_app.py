# ===================================================================
# 终极防过拟合量化分析平台 — Streamlit 版
# ===================================================================
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --------------- 全局配置 ---------------
st.set_page_config(page_title="量化分析平台", page_icon="📈", layout="wide")

# 设置 matplotlib 中文字体（macOS）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

ASSET_MAP = {
    "A股指数": {
        "创业板指": {"code": "sz399006", "source": "akshare"},
        "沪深300": {"code": "sh000300", "source": "akshare"},
        "上证50": {"code": "sh000016", "source": "akshare"},
        "中证500": {"code": "sh000905", "source": "akshare"},
        "中证1000": {"code": "sh000852", "source": "akshare"},
        "科创50": {"code": "sh000688", "source": "akshare"},
        "上证综合指数": {"code": "sh000001", "source": "akshare"},
        "中证银行": {"code": "sz399986", "source": "akshare"},
        "中证券商": {"code": "sz399975", "source": "akshare"},
        "中证保险": {"code": "sz399809", "source": "akshare"},
        "中证主要消费": {"code": "sh000932", "source": "akshare"},
        "中证可选消费": {"code": "sh000931", "source": "akshare"},
        "国证食品饮料": {"code": "sz399396", "source": "akshare"},
        "中证白酒": {"code": "sz399997", "source": "akshare"},
        "中证医药卫生": {"code": "sh000933", "source": "akshare"},
        "中证房地产": {"code": "sh000952", "source": "akshare"},
        "中证基建工程": {"code": "sz399995", "source": "akshare"},
        "中证能源": {"code": "sh000928", "source": "akshare"},
        "中证材料": {"code": "sh000929", "source": "akshare"},
    },
    "美股指数": {
        "S&P 500": {"code": "^GSPC", "source": "yfinance"},
        "Nasdaq": {"code": "^IXIC", "source": "yfinance"},
        "Dow Jones": {"code": "^DJI", "source": "yfinance"},
    },
    "大宗商品": {
        "黄金 (Gold)": {"code": "GC=F", "source": "yfinance"},
        "原油 (Crude Oil)": {"code": "CL=F", "source": "yfinance"},
        "大豆 (Soybeans)": {"code": "ZS=F", "source": "yfinance"},
    },
    "债券": {
        "美国长期国债ETF (TLT)": {"code": "TLT", "source": "yfinance"},
    },
}

DEFAULT_BUY_RULES = [(-0.12, 1.0), (-0.08, 0.50), (-0.04, 0.20)]
DEFAULT_SELL_RULES = [(0.12, 1.0), (0.08, 0.50), (0.04, 0.20)]


# ===================================================================
# --- 核心引擎函数 ---
# ===================================================================


FREQ_CONFIG = {
    "monthly": {"resample": "ME", "fmt": "%Y-%m", "periods_per_year": 12},
    "weekly":  {"resample": "W",  "fmt": "%Y-%m-%d", "periods_per_year": 52},
    "daily":   {"resample": None, "fmt": "%Y-%m-%d", "periods_per_year": 252},
}


def fetch_data(asset_code, source, freq="monthly"):
    """获取 OHLCV 数据。freq: 'monthly'/'weekly'/'daily'。source: 'tushare'/'akshare'/'yfinance'。"""
    if source == "tushare":
        import tushare as ts
        pro = ts.pro_api('0fa942b3121e1e96a016f0db7bbbab30a13e2d0b8106fdb29248092c')
        # 支持 akshare 格式转换: sz399006 → 399006.SZ
        ts_code = asset_code
        if asset_code.startswith(('sh', 'sz')) and '.' not in asset_code:
            prefix = asset_code[:2].upper()
            num = asset_code[2:]
            ts_code = f"{num}.{'SH' if prefix == 'SH' else 'SZ'}"
        daily_data = pro.index_daily(ts_code=ts_code, start_date='19900101')
        if daily_data is None or daily_data.empty:
            return None
        daily_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                    'close': 'Close', 'vol': 'Volume'}, inplace=True)
        daily_data['date'] = pd.to_datetime(daily_data['trade_date'], format='%Y%m%d')
        daily_data.set_index('date', inplace=True)
        daily_data.sort_index(inplace=True)
    elif source == "akshare":
        import akshare as ak
        daily_data = ak.stock_zh_index_daily(symbol=asset_code)
        if daily_data.empty:
            return None
        daily_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                    'close': 'Close', 'volume': 'Volume'}, inplace=True)
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data.set_index('date', inplace=True)
    else:  # yfinance
        import yfinance as yf
        ticker = yf.Ticker(asset_code)
        daily_data = ticker.history(period="max", auto_adjust=True)
        if daily_data.empty:
            return None
        daily_data.index = pd.to_datetime(daily_data.index)

    cfg = FREQ_CONFIG[freq]
    if cfg["resample"]:
        result = daily_data.resample(cfg["resample"]).apply(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    else:
        result = daily_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    result.dropna(inplace=True)
    result.index = result.index.strftime(cfg["fmt"])
    result.index.name = '日期'
    return result


def compute_dynamic_regression(x, log_y):
    """扩展窗口动态回归：从最早2个点开始逐步扩展拟合。
    返回 (slopes, intercepts, theoretical_log_values) 三个与 x 等长的数组。"""
    n = len(x)
    slopes = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)
    theo_log = np.full(n, np.nan)

    for i in range(1, n):  # 从第2个点(index=1)开始，用 x[:i+1] 拟合
        s, ic = np.polyfit(x[:i + 1], log_y[:i + 1], 1)
        slopes[i] = s
        intercepts[i] = ic
        theo_log[i] = s * x[i] + ic

    # 第0个点无法拟合，复制第1个点的结果
    if n >= 2:
        slopes[0] = slopes[1]
        intercepts[0] = intercepts[1]
        theo_log[0] = slopes[0] * x[0] + intercepts[0]

    return slopes, intercepts, theo_log


def generate_monotonic_matrix(num_iters, num_tiers, min_val, max_val, min_gap):
    if max_val < min_val + num_tiers * min_gap:
        max_val = min_val + num_tiers * min_gap + 0.01
    rand_mat = np.random.uniform(min_val, max_val - (num_tiers - 1) * min_gap, (num_iters, num_tiers))
    rand_mat.sort(axis=1)
    for k in range(num_tiers):
        rand_mat[:, k] += k * min_gap
    return rand_mat


def run_full_analysis(index_name, index_code, source="akshare",
                      custom_slope=None, custom_intercept=None,
                      custom_buy_rules=None, custom_sell_rules=None,
                      backtest_start_date=None, extra_initial_cash=0,
                      regression_mode="static", freq="monthly",
                      buy_mode="tiered", npower_params=None):
    """运行完整分析，返回 (processed_data, slope, intercept, figures_list)。"""
    ppy = FREQ_CONFIG[freq]["periods_per_year"]
    try:
        data = fetch_data(index_code, source, freq=freq)
        if data is None or data.empty:
            st.error(f"未能获取到 '{index_code}' 的数据。")
            return None
    except Exception as e:
        st.error(f"获取数据时出错: {e}")
        return None

    processed_data = data[['Close']].copy()
    processed_data.rename(columns={'Close': '当月收盘价'}, inplace=True)
    processed_data.reset_index(inplace=True)
    processed_data.insert(1, '序号', range(1, len(processed_data) + 1))

    x = processed_data['序号']
    log_y = np.log(processed_data['当月收盘价'])

    if regression_mode == "dynamic":
        dyn_slopes, dyn_intercepts, dyn_theo = compute_dynamic_regression(
            x.values, log_y.values)
        processed_data['理论对数值'] = dyn_theo
        processed_data['理论值'] = np.exp(dyn_theo)
        slope, intercept = dyn_slopes[-1], dyn_intercepts[-1]
    else:
        if custom_slope is not None and custom_intercept is not None:
            slope, intercept = custom_slope, custom_intercept
        else:
            slope, intercept = np.polyfit(x, log_y, 1)
        processed_data['理论对数值'] = slope * x + intercept
        processed_data['理论值'] = np.exp(processed_data['理论对数值'])

    processed_data['百分比'] = (log_y - processed_data['理论对数值']) / processed_data['理论对数值']

    if custom_buy_rules is not None and custom_sell_rules is not None:
        buy_rules, sell_rules = custom_buy_rules, custom_sell_rules
    else:
        buy_rules, sell_rules = DEFAULT_BUY_RULES, DEFAULT_SELL_RULES

    results_lists = {k: [] for k in [
        'shares', 'cash', 'stock_value', 'total_assets', 'cumulative_investment',
        'profit', 'actual_monthly_return', 'annual_irr', 'max_drawdown',
        'volatility', 'net_value_index']}
    shares_held, cash_held, cumulative_investment = 0.0, 0.0, 0.0
    net_value_index, peak_net_value_index = 1000.0, 1000.0
    active_returns = []
    trading_started = False
    last_month_assets = 0.0
    start_idx_for_irr = processed_data[
        processed_data['日期'] == backtest_start_date].index[0] if backtest_start_date else 0

    for i, row in processed_data.iterrows():
        current_date = row['日期']
        if backtest_start_date and current_date < backtest_start_date:
            for key in results_lists:
                results_lists[key].append(
                    0 if key not in ['annual_irr', 'net_value_index']
                    else (np.nan if key == 'annual_irr' else 1000.0))
            continue

        close_price = row['当月收盘价']
        percentage = row['百分比']
        assets_before_sip = shares_held * close_price + cash_held

        if not trading_started:
            trading_started = True
            actual_monthly_return = 0.0
            cash_held += (1.0 + extra_initial_cash)
            cumulative_investment += (1.0 + extra_initial_cash)
        else:
            actual_monthly_return = (
                (assets_before_sip - last_month_assets) / last_month_assets
                if last_month_assets > 0 else 0.0)
            net_value_index *= (1.0 + actual_monthly_return)
            cash_held += 1.0
            cumulative_investment += 1.0

        active_returns.append(actual_monthly_return)

        # --- 买入逻辑 ---
        if buy_mode == "npower" and npower_params:
            np_bt = npower_params["buy_threshold"]
            np_bn = npower_params["buy_n"]
            np_min_ratio = npower_params.get("min_ratio", 0.01)
            if percentage < 0:  # 低估时每期都买入
                ratio = min(1.0, (abs(percentage) / np_bt) ** np_bn)
                if ratio >= np_min_ratio:  # 比例低于最低交易比例时不操作
                    cash_to_spend = cash_held * ratio
                    shares_held += cash_to_spend / close_price
                    cash_held -= cash_to_spend
        else:  # tiered
            buy_trigger = max([r[0] for r in buy_rules] + [-999]) if buy_rules else -999
            if percentage <= buy_trigger:
                for threshold, ratio in sorted(buy_rules, key=lambda item: item[0]):
                    if percentage <= threshold:
                        cash_to_spend = cash_held * ratio
                        shares_held += cash_to_spend / close_price
                        cash_held -= cash_to_spend
                        break

        # --- 卖出逻辑 ---
        if buy_mode == "npower" and npower_params:
            np_st = npower_params["sell_threshold"]
            np_sn = npower_params["sell_n"]
            np_min_ratio = npower_params.get("min_ratio", 0.01)
            if percentage > 0:  # 高估时每期都卖出
                ratio = min(1.0, (percentage / np_st) ** np_sn)
                if ratio >= np_min_ratio:  # 比例低于最低交易比例时不操作
                    shares_to_sell = shares_held * ratio
                    cash_gained = shares_to_sell * close_price
                    shares_held -= shares_to_sell
                    cash_held += cash_gained
        else:
            sell_trigger = min([r[0] for r in sell_rules] + [999]) if sell_rules else 999
            if percentage >= sell_trigger:
                for threshold, ratio in sorted(sell_rules, key=lambda item: item[0], reverse=True):
                    if percentage >= threshold:
                        shares_to_sell = shares_held * ratio
                        cash_gained = shares_to_sell * close_price
                        shares_held -= shares_to_sell
                        cash_held += cash_gained
                        break

        stock_value = shares_held * close_price
        total_assets = stock_value + cash_held
        profit = total_assets - cumulative_investment
        last_month_assets = total_assets

        volatility = np.std(active_returns, ddof=1) * np.sqrt(ppy) if len(active_returns) > 1 else 0.0
        peak_net_value_index = max(peak_net_value_index, net_value_index)
        drawdown = ((net_value_index - peak_net_value_index) / peak_net_value_index
                     if peak_net_value_index != 0 else 0.0)
        current_max_drawdown = min([r for r in results_lists['max_drawdown'] if r < 0] + [0.0])
        max_drawdown = min(current_max_drawdown, drawdown)

        annual_irr = np.nan
        num_periods = i - start_idx_for_irr + 1
        if num_periods >= ppy:
            cash_flows = [-1.0] * num_periods
            cash_flows[0] -= extra_initial_cash
            cash_flows.append(total_assets)
            try:
                period_irr = npf.irr(cash_flows)
                if not np.isnan(period_irr) and period_irr > -0.99:
                    annual_irr = (1 + period_irr) ** ppy - 1
            except (ValueError, TypeError):
                pass

        for key, val in zip(results_lists.keys(), [
            shares_held, cash_held, stock_value, total_assets, cumulative_investment,
            profit, actual_monthly_return, annual_irr, max_drawdown, volatility,
            net_value_index]):
            results_lists[key].append(val)

    processed_data['持有股票数量'] = results_lists['shares']
    processed_data['股票价值'] = results_lists['stock_value']
    processed_data['现金'] = results_lists['cash']
    processed_data['总资产'] = results_lists['total_assets']
    cash_percentage = np.where(
        processed_data['总资产'] > 0,
        processed_data['现金'] / processed_data['总资产'], np.nan)
    processed_data['仓位百分比'] = 1 - cash_percentage
    processed_data['累计投资'] = results_lists['cumulative_investment']
    processed_data['收益'] = results_lists['profit']
    processed_data['当月真实收益率'] = results_lists['actual_monthly_return']
    processed_data['净值指数'] = results_lists['net_value_index']
    processed_data['年化收益率(IRR)'] = results_lists['annual_irr']
    processed_data['最大回撤'] = results_lists['max_drawdown']
    processed_data['年化波动率'] = results_lists['volatility']

    # ---------- 生成图表 ----------
    dates_for_plot = pd.to_datetime(processed_data['日期'])
    figures = []

    # 图 1: 实际价格 vs 趋势 & 仓位
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(dates_for_plot, processed_data['当月收盘价'],
             label=f'{index_code} (Actual Price)', color='blue', linewidth=2)
    ax1.plot(dates_for_plot, processed_data['理论值'],
             label='Exponential Trendline (Theory)', color='red', linestyle='--', linewidth=2)
    ax1.set_title(f'{index_code} Actual Price vs Exponential Trend & Position %', fontsize=18)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Index Points', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax1_pos = ax1.twinx()
    ax1_pos.plot(dates_for_plot, processed_data['仓位百分比'],
                 label='Position %', color='purple', linestyle=':', marker='.', markersize=4, alpha=0.5)
    ax1_pos.set_ylabel('Position Percentage', color='purple', fontsize=12)
    ax1_pos.tick_params(axis='y', labelcolor='purple')
    ax1_pos.yaxis.set_major_formatter(PercentFormatter(1.0))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_pos, labels1_pos = ax1_pos.get_legend_handles_labels()
    ax1.legend(lines1 + lines1_pos, labels1 + labels1_pos, loc='upper left', fontsize=12)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    ax1.grid(True, which='minor', linestyle=':', alpha=0.5)
    ax1.grid(True, which='major', linestyle='--', alpha=0.7)
    fig1.autofmt_xdate()
    fig1.tight_layout()
    figures.append(("实际价格 vs 趋势 & 仓位", fig1))

    # 图 2: 累计收益曲线
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    ax2.plot(dates_for_plot, processed_data['收益'], label='Cumulative P/L', color='purple', linewidth=2)
    ax2.fill_between(dates_for_plot, processed_data['收益'],
                     where=(processed_data['收益'] >= 0), color='mediumpurple', alpha=0.3, interpolate=True)
    ax2.fill_between(dates_for_plot, processed_data['收益'],
                     where=(processed_data['收益'] < 0), color='lightcoral', alpha=0.3, interpolate=True)
    ax2.set_title('Cumulative Profit/Loss Curve', fontsize=18)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Profit/Loss (CNY)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator())
    ax2.grid(True, which='minor', linestyle=':', alpha=0.5)
    ax2.grid(True, which='major', linestyle='--', alpha=0.7)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    figures.append(("累计收益曲线", fig2))

    # 图 3: 策略 vs 指数对比
    fig3, ax3 = plt.subplots(figsize=(16, 8))
    color1, color2 = 'dodgerblue', 'crimson'
    ax4 = ax3.twinx()
    start_idx = processed_data[
        processed_data['日期'] == backtest_start_date].index[0] if backtest_start_date else 0

    price_series = processed_data['当月收盘价']
    net_value_series = processed_data['净值指数']

    if start_idx < len(price_series):
        base_price = price_series.iloc[start_idx]
        base_net_value = net_value_series.iloc[start_idx]

        price_pct_change = ((price_series / base_price).astype(float) - 1.0
                            if base_price > 0 else pd.Series(0, index=price_series.index))
        price_pct_change.iloc[:start_idx] = np.nan
        net_value_pct_change = ((net_value_series / base_net_value).astype(float) - 1.0
                                if base_net_value > 0 else pd.Series(0, index=net_value_series.index))
        net_value_pct_change.iloc[:start_idx] = np.nan

        ax3.plot(dates_for_plot, price_pct_change, color=color1,
                 label=f'{index_code} (Rebased)', linewidth=2, alpha=0.8)
        ax4.plot(dates_for_plot, net_value_pct_change, color=color2,
                 label='Strategy Net Value (Rebased)', linewidth=2)
        y_min = min(price_pct_change.min(), net_value_pct_change.min())
        y_max = max(price_pct_change.max(), net_value_pct_change.max())
        margin = (y_max - y_min) * 0.05
        ax3.set_ylim(y_min - margin, y_max + margin)
        ax4.set_ylim(y_min - margin, y_max + margin)
        ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax4.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Price Return Since Start (%)', color=color1, fontsize=12)
        ax4.set_ylabel('Strategy Return Since Start (%)', color=color2, fontsize=12)
        ax3.tick_params(axis='y', labelcolor=color1)
        ax4.tick_params(axis='y', labelcolor=color2)
        ax3.set_title(
            f'Performance Comparison (Rebased to '
            f'{backtest_start_date if backtest_start_date else processed_data["日期"].iloc[0]})',
            fontsize=18)
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax4.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.xaxis.set_minor_locator(mdates.MonthLocator())
        ax3.grid(True, which='minor', linestyle=':', alpha=0.5)
        ax3.grid(True, which='major', linestyle='--', alpha=0.7)
        fig3.autofmt_xdate()
    fig3.tight_layout()
    figures.append(("策略 vs 指数对比", fig3))

    # 图 4: 对数偏差分布
    fig4, ax4_hist = plt.subplots(figsize=(16, 8))
    ax4_hist.hist(processed_data['百分比'], bins=50, edgecolor='black', alpha=0.75, color='skyblue')
    ax4_hist.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax4_hist.set_title('Distribution of Logarithmic Percentage Deviation from Trend', fontsize=18)
    ax4_hist.set_xlabel('Log Percentage Deviation (%)', fontsize=12)
    ax4_hist.set_ylabel('Frequency (Number of Months)', fontsize=12)
    ax4_hist.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Trend Line (0%)')
    ax4_hist.legend()
    ax4_hist.grid(axis='y', linestyle='--', alpha=0.7)
    fig4.tight_layout()
    figures.append(("对数偏差分布", fig4))

    return processed_data, slope, intercept, figures


def run_monte_carlo_optimization(data, start_date, extra_cash, base_slope, base_icpt,
                                  num_buy_tiers, num_sell_tiers, num_iters,
                                  target_calmar, min_b_gap, min_s_gap,
                                  regression_mode="static", freq="monthly",
                                  buy_mode="tiered", npower_params=None):
    ppy = FREQ_CONFIG[freq]["periods_per_year"]
    start_idx = data[data['日期'] == start_date].index[0] if start_date else 0
    close_prices = data['当月收盘价'].values[start_idx:]
    x_arr = data['序号'].values[start_idx:]
    num_periods = len(close_prices)
    if num_periods < ppy:
        st.warning("回测周期太短，无法寻优。")
        return None

    is_dynamic = (regression_mode == "dynamic")
    is_npower = (buy_mode == "npower")

    # 计算维度数
    if is_npower:
        buy_dims = 2  # buy_N + buy_threshold
        sell_dims = 2  # sell_N + sell_threshold
    else:
        buy_dims = num_buy_tiers * 2
        sell_dims = num_sell_tiers * 2
    slope_dims = 0 if is_dynamic else 2
    total_dims = slope_dims + buy_dims + sell_dims

    mode_label = f"{'动态' if is_dynamic else '静态'}+{'N次方' if is_npower else '梯队'}"
    st.info(f"🚀 蒙特卡洛寻优 [{mode_label}] {total_dims}维 (迭代: {num_iters}, 底线卡玛: {target_calmar})")
    start_time = time.time()

    best_target_ret, best_target_params = -9999.0, None
    fallback_calmar, fallback_params = -9999.0, None
    sampled_results = []
    progress_bar = st.progress(0, text="寻优进度: 0%")

    if is_dynamic:
        fixed_pct = data['百分比'].values[start_idx:]
        valid_pct = fixed_pct[~np.isnan(fixed_pct)]
        max_b_depth = abs(valid_pct.min()) * 1.3
        max_s_height = valid_pct.max() * 1.3
    else:
        baseline_theo = base_slope * x_arr + base_icpt
        baseline_pct = (np.log(close_prices) - baseline_theo) / baseline_theo
        max_b_depth = abs(baseline_pct.min()) * 1.3
        max_s_height = baseline_pct.max() * 1.3

    np.random.seed(int(time.time()))

    if not is_dynamic:
        s_min, s_max = ((base_slope * 0.8, base_slope * 1.2) if base_slope > 0
                        else (base_slope * 1.2, base_slope * 0.8))
        i_min, i_max = ((base_icpt * 0.9, base_icpt * 1.1) if base_icpt > 0
                        else (base_icpt * 1.1, base_icpt * 0.9))
        rand_slopes = np.random.uniform(s_min, s_max, num_iters)
        rand_icpts = np.random.uniform(i_min, i_max, num_iters)

    # N次方模式：搜索买入/卖出的 N 和 threshold
    if is_npower:
        rand_np_bn = np.random.uniform(0.5, 5.0, num_iters)
        rand_np_bt = np.random.uniform(0.01, max(0.02, max_b_depth), num_iters)
        rand_np_sn = np.random.uniform(0.5, 5.0, num_iters)
        rand_np_st = np.random.uniform(0.01, max(0.02, max_s_height), num_iters)
    else:
        b_gap_val = min_b_gap / 100.0
        rand_b_t_abs = generate_monotonic_matrix(num_iters, num_buy_tiers, b_gap_val, max_b_depth, b_gap_val)
        rand_b_r = generate_monotonic_matrix(num_iters, num_buy_tiers, 0.05, 1.0, 0.05)
        rand_b_r = np.clip(rand_b_r, 0.05, 1.0)

        s_gap_val = min_s_gap / 100.0
        rand_s_t = generate_monotonic_matrix(num_iters, num_sell_tiers, s_gap_val, max_s_height, s_gap_val)
        rand_s_r = generate_monotonic_matrix(num_iters, num_sell_tiers, 0.05, 1.0, 0.05)
        rand_s_r = np.clip(rand_s_r, 0.05, 1.0)

    for i in range(num_iters):
        if i % max(1, num_iters // 100) == 0:
            pct = int(i / num_iters * 100)
            progress_bar.progress(pct / 100, text=f"寻优进度: {pct}%")

        if not is_npower:
            exec_sell = sorted(list(zip(rand_s_t[i], rand_s_r[i])), key=lambda x: x[0], reverse=True)

        if is_dynamic:
            test_pct = fixed_pct
            test_slope = base_slope
            test_icpt = base_icpt
        else:
            test_slope = rand_slopes[i]
            test_icpt = rand_icpts[i]
            test_theoretical_log = test_slope * x_arr + test_icpt
            test_theoretical_log = np.where(test_theoretical_log == 0, 1e-9, test_theoretical_log)
            test_pct = (np.log(close_prices) - test_theoretical_log) / test_theoretical_log

        if is_npower:
            cur_np_bn = rand_np_bn[i]
            cur_np_bt = rand_np_bt[i]
            cur_np_sn = rand_np_sn[i]
            cur_np_st = rand_np_st[i]
            cur_np_min_ratio = npower_params.get("min_ratio", 0.01) if npower_params else 0.01
        else:
            exec_buy = sorted(list(zip(-rand_b_t_abs[i], rand_b_r[i])), key=lambda x: x[0])
            buy_trig = exec_buy[-1][0]

        cash = 1.0 + extra_cash
        shares = 0.0
        net_val = 1.0
        peak_nv = 1.0
        max_dd = 0.0
        last_assets = 0.0
        if not is_npower:
            sell_trig = exec_sell[-1][0]
        is_first = True

        for pct_val, price in zip(test_pct, close_prices):
            assets_before_sip = shares * price + cash
            if is_first:
                is_first = False
            else:
                ret = ((assets_before_sip - last_assets) / last_assets
                       if last_assets > 0 else 0.0)
                net_val *= (1.0 + ret)
                cash += 1.0

            # 买入
            if is_npower:
                if pct_val < 0:  # 低估时每期都买入
                    ratio = min(1.0, (abs(pct_val) / cur_np_bt) ** cur_np_bn)
                    if ratio >= cur_np_min_ratio:
                        spend = cash * ratio
                        shares += spend / price
                        cash -= spend
            else:
                if pct_val <= buy_trig:
                    for thresh, ratio in exec_buy:
                        if pct_val <= thresh:
                            spend = cash * ratio
                            shares += spend / price
                            cash -= spend
                            break

            # 卖出
            if is_npower:
                if pct_val > 0:  # 高估时每期都卖出
                    ratio = min(1.0, (pct_val / cur_np_st) ** cur_np_sn)
                    if ratio >= cur_np_min_ratio:
                        sold = shares * ratio
                        cash += sold * price
                        shares -= sold
            else:
                if pct_val >= sell_trig:
                    for thresh, ratio in exec_sell:
                        if pct_val >= thresh:
                            sold = shares * ratio
                            cash += sold * price
                            shares -= sold
                            break

            assets = shares * price + cash
            last_assets = assets
            if net_val > peak_nv:
                peak_nv = net_val
            dd = (net_val - peak_nv) / peak_nv if peak_nv > 0 else 0.0
            if dd < max_dd:
                max_dd = dd

        ann_ret = net_val ** (float(ppy) / num_periods) - 1.0
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else (ann_ret * 10 if ann_ret > 0 else 0)

        if is_npower:
            buy_info = {"buy_threshold": cur_np_bt, "buy_n": cur_np_bn}
            sell_info = {"sell_threshold": cur_np_st, "sell_n": cur_np_sn}
            first_buy_val = -cur_np_bt
        else:
            buy_info = exec_buy
            sell_info = exec_sell
            first_buy_val = exec_buy[-1][0]

        params_dict = {
            'slope': test_slope, 'icpt': test_icpt,
            'buy_rules': buy_info, 'sell_rules': sell_info,
            'ret': ann_ret, 'dd': max_dd, 'calmar': calmar,
            '1st_buy': first_buy_val, 'buy_mode': buy_mode
        }
        # 展平 N-power 参数供 DataFrame 敏感度分析使用
        if is_npower:
            params_dict['buy_n'] = cur_np_bn
            params_dict['buy_threshold'] = cur_np_bt
            params_dict['sell_n'] = cur_np_sn
            params_dict['sell_threshold'] = cur_np_st

        if calmar >= target_calmar:
            if ann_ret > best_target_ret:
                best_target_ret = ann_ret
                best_target_params = params_dict

        if calmar > fallback_calmar:
            fallback_calmar = calmar
            fallback_params = params_dict

        if np.random.rand() < (5000.0 / num_iters):
            sampled_results.append(params_dict)

    progress_bar.progress(1.0, text="寻优完成！✅")
    elapsed = time.time() - start_time
    st.success(f"寻优完成！总耗时: {elapsed:.2f} 秒。")

    best = best_target_params if best_target_params is not None else fallback_params
    if best is None:
        st.error("未能找到任何有效策略参数。")
        return None

    title_msg = (f"🏆 满足卡玛底线(>={target_calmar}) 的【最高收益】组合 🏆"
                 if best_target_params
                 else f"⚠️ 未能找到卡玛>={target_calmar}的组合。退而求其次展示【最高卡玛】组合 ⚠️")

    st.markdown("---")
    st.markdown(f"### {title_msg}")
    st.markdown(
        f"- **预期年化**: {best['ret'] * 100:.2f}% | **最大回撤**: {best['dd'] * 100:.2f}% "
        f"| **卡玛得分**: {best['calmar']:.2f}")
    st.markdown(
        f"- **对数拟合线参数** → 对数斜率: `{best['slope']:.6f}`, 对数截距: `{best['icpt']:.4f}`")

    if best.get('buy_mode') == 'npower':
        bp = best['buy_rules']
        st.markdown(f"**最佳N次方买入曲线**: 阈值 = **{bp['buy_threshold']*100:.2f}%**, N = **{bp['buy_n']:.2f}**")
        st.markdown(f"  - 当偏差 ≤ -{bp['buy_threshold']*100:.2f}% 时，买入比例 = min(1, (|偏差|/{bp['buy_threshold']*100:.2f}%)^{bp['buy_n']:.2f})")
        sp = best['sell_rules']
        st.markdown(f"**最佳N次方卖出曲线**: 阈值 = **{sp['sell_threshold']*100:.2f}%**, N = **{sp['sell_n']:.2f}**")
        st.markdown(f"  - 当偏差 ≥ {sp['sell_threshold']*100:.2f}% 时，卖出比例 = min(1, (偏差/{sp['sell_threshold']*100:.2f}%)^{sp['sell_n']:.2f})")
    else:
        st.markdown("最佳买入规则集:")
        for t, r in sorted(best['buy_rules'], key=lambda x: x[0], reverse=True):
            st.markdown(f"  - 如果对数低估达到 **{-t * 100:.1f}%** → 买入剩余现金的 **{r * 100:.1f}%**")

        st.markdown("最佳卖出规则集:")
        for t, r in sorted(best['sell_rules'], key=lambda x: x[0]):
            st.markdown(f"  - 如果对数高估达到 **{t * 100:.1f}%** → 卖出持有股票的 **{r * 100:.1f}%**")

    # ---------- 散点图 & 敏感度 ----------
    if sampled_results:
        df_samp = pd.DataFrame(sampled_results)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        ax1 = axes[0]

        vmax_calmar = df_samp['calmar'].quantile(0.95)
        scatter = ax1.scatter(df_samp['dd'] * 100, df_samp['ret'] * 100,
                              c=df_samp['calmar'], cmap='viridis', alpha=0.5, s=15,
                              edgecolors='none', vmax=vmax_calmar)

        df_valid = (df_samp[df_samp['calmar'] > 0.5]
                    if len(df_samp[df_samp['calmar'] > 0.5]) > 0 else df_samp)
        df_sorted = df_valid.sort_values('dd', ascending=False)
        pareto_x, pareto_y = [], []
        max_ret_val = -999
        for _, row in df_sorted.iterrows():
            if row['ret'] > max_ret_val:
                pareto_x.append(row['dd'] * 100)
                pareto_y.append(row['ret'] * 100)
                max_ret_val = row['ret']

        ax1.plot(pareto_x, pareto_y, color='red', linewidth=2, linestyle='-',
                 marker='o', markersize=4, label='Efficient Frontier')
        ax1.scatter([best['dd'] * 100], [best['ret'] * 100], color='gold',
                    marker='*', s=800, edgecolor='black', zorder=5,
                    label='Selected Optimal Strategy')

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Calmar Score', rotation=270, labelpad=15)
        ax1.set_title(f'Monte Carlo Efficient Frontier ({num_iters} Iters)',
                      fontsize=16, fontweight='bold')
        ax1.set_xlabel('Risk: Maximum Drawdown (%)', fontsize=12)
        ax1.set_ylabel('Reward: Annualized Return (%)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        xlims = ax1.get_xlim()
        if xlims[0] < xlims[1]:
            ax1.set_xlim(xlims[::-1])
        ax1.legend()

        ax2 = axes[1]
        is_np_mode = best.get('buy_mode') == 'npower'
        if is_np_mode and 'buy_n' in df_samp.columns:
            labels_vals = []
            for col, lbl in [('buy_threshold', '买入阈值'), ('buy_n', '买入N'),
                              ('sell_threshold', '卖出阈值'), ('sell_n', '卖出N')]:
                if col in df_samp.columns:
                    c, _ = spearmanr(df_samp[col], df_samp['ret'])
                    labels_vals.append((lbl, c))
            x_labels = [lv[0] for lv in labels_vals]
            y_values = [lv[1] for lv in labels_vals]
            palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(x_labels)]
        else:
            corr_slope, _ = spearmanr(df_samp['slope'], df_samp['ret'])
            corr_buy1, _ = spearmanr(df_samp['1st_buy'], df_samp['ret'])
            x_labels = ['对数斜率', '第一档买入阈值']
            y_values = [corr_slope, corr_buy1]
            palette = ['#3498db', '#e74c3c']
        sns.barplot(x=x_labels, y=y_values, ax=ax2, hue=x_labels,
                    palette=palette, legend=False)
        ax2.set_title('参数敏感度 (与收益率 Spearman 相关)',
                      fontsize=16, fontweight='bold')
        ax2.set_ylabel('Spearman 相关系数 (-1 ~ 1)', fontsize=12)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # 有效前沿表
        if pareto_x:
            st.markdown("### ✨ 有效前沿策略参数明细库 ✨")
            # Rebuild pareto_rows for the table
            pareto_rows = []
            max_ret_val2 = -999
            for _, row in df_sorted.iterrows():
                if row['ret'] > max_ret_val2:
                    pareto_rows.append(row.to_dict())
                    max_ret_val2 = row['ret']

            frontier_df = pd.DataFrame(pareto_rows)

            def format_buy(rules):
                if isinstance(rules, dict):
                    return f"N次方: 阈值={rules['buy_threshold']*100:.1f}%, N={rules['buy_n']:.2f}"
                return " | ".join([f"跌{-t * 100:.1f}%买{r * 100:.0f}%"
                                   for t, r in sorted(rules, key=lambda x: x[0], reverse=True)])

            def format_sell(rules):
                if isinstance(rules, dict):
                    return f"N次方: 阈值={rules['sell_threshold']*100:.1f}%, N={rules['sell_n']:.2f}"
                return " | ".join([f"涨{t * 100:.1f}%卖{r * 100:.0f}%"
                                   for t, r in sorted(rules, key=lambda x: x[0])])

            base_cols = {
                '风险(最大回撤)': (frontier_df['dd'] * 100).map('{:.2f}%'.format),
                '收益(预期年化)': (frontier_df['ret'] * 100).map('{:.2f}%'.format),
                '卡玛得分': frontier_df['calmar'].map('{:.2f}'.format),
                '对数斜率': frontier_df['slope'].map('{:.6f}'.format),
                '对数截距': frontier_df['icpt'].map('{:.4f}'.format),
            }
            if is_np_mode and 'buy_threshold' in frontier_df.columns:
                base_cols['买入阈值'] = (frontier_df['buy_threshold'] * 100).map('{:.2f}%'.format)
                base_cols['买入N'] = frontier_df['buy_n'].map('{:.2f}'.format)
                base_cols['卖出阈值'] = (frontier_df['sell_threshold'] * 100).map('{:.2f}%'.format)
                base_cols['卖出N'] = frontier_df['sell_n'].map('{:.2f}'.format)
            else:
                base_cols['买入梯队'] = frontier_df['buy_rules'].apply(format_buy)
                base_cols['卖出梯队'] = frontier_df['sell_rules'].apply(format_sell)
            display_df = pd.DataFrame(base_cols)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ---- 回撤稳定区：回撤相近，收益递增 ----
        st.markdown("### 📊 回撤稳定区 — 回撤相近时哪些参数提升了收益？")
        df_samp_copy = df_samp.copy()
        df_samp_copy['dd_bin'] = (df_samp_copy['dd'] * 200).round() / 200  # ±0.5% bins
        dd_stable = df_samp_copy.loc[df_samp_copy.groupby('dd_bin')['ret'].idxmax()]
        dd_stable = dd_stable.sort_values('ret', ascending=True).tail(30)
        dd_cols = {'dd': '最大回撤', 'ret': '年化收益', 'calmar': '卡玛'}
        if is_np_mode:
            dd_cols.update({'buy_threshold': '买入阈值', 'buy_n': '买入N', 'sell_threshold': '卖出阈值', 'sell_n': '卖出N'})
        else:
            dd_cols.update({'slope': '斜率', '1st_buy': '首档买入'})
        dd_display = dd_stable[[c for c in dd_cols if c in dd_stable.columns]].copy()
        dd_display.columns = [dd_cols[c] for c in dd_display.columns]
        for c in dd_display.columns:
            if c in ('最大回撤', '年化收益'):
                dd_display[c] = dd_display[c].map(lambda v: f"{v*100:.2f}%")
            elif c == '卡玛':
                dd_display[c] = dd_display[c].map(lambda v: f"{v:.2f}")
            elif c in ('买入阈值', '卖出阈值'):
                dd_display[c] = dd_display[c].map(lambda v: f"{v*100:.2f}%")
            elif c in ('买入N', '卖出N'):
                dd_display[c] = dd_display[c].map(lambda v: f"{v:.2f}")
            elif c == '斜率':
                dd_display[c] = dd_display[c].map(lambda v: f"{v:.6f}")
            elif c == '首档买入':
                dd_display[c] = dd_display[c].map(lambda v: f"{v*100:.2f}%")
        st.dataframe(dd_display.reset_index(drop=True), use_container_width=True, hide_index=True)

        # ---- 收益稳定区：收益相近，回撤递增 ----
        st.markdown("### 📊 收益稳定区 — 收益相近时哪些参数增加了风险？")
        df_samp_copy['ret_bin'] = (df_samp_copy['ret'] * 200).round() / 200
        ret_stable = df_samp_copy.loc[df_samp_copy.groupby('ret_bin')['dd'].idxmin()]
        ret_stable = ret_stable.sort_values('dd', ascending=True).tail(30)
        ret_display = ret_stable[[c for c in dd_cols if c in ret_stable.columns]].copy()
        ret_display.columns = [dd_cols[c] for c in ret_display.columns]
        for c in ret_display.columns:
            if c in ('最大回撤', '年化收益'):
                ret_display[c] = ret_display[c].map(lambda v: f"{v*100:.2f}%")
            elif c == '卡玛':
                ret_display[c] = ret_display[c].map(lambda v: f"{v:.2f}")
            elif c in ('买入阈值', '卖出阈值'):
                ret_display[c] = ret_display[c].map(lambda v: f"{v*100:.2f}%")
            elif c in ('买入N', '卖出N'):
                ret_display[c] = ret_display[c].map(lambda v: f"{v:.2f}")
            elif c == '斜率':
                ret_display[c] = ret_display[c].map(lambda v: f"{v:.6f}")
            elif c == '首档买入':
                ret_display[c] = ret_display[c].map(lambda v: f"{v*100:.2f}%")
        st.dataframe(ret_display.reset_index(drop=True), use_container_width=True, hide_index=True)

    return best


# ===================================================================
# --- Streamlit 主界面 ---
# ===================================================================

st.title("📈 终极防过拟合量化分析平台")
st.caption("基于对数线性回归 & 蒙特卡洛寻优的指数定投策略回测")

# ---------- 初始化 session_state ----------
if 'primary_result' not in st.session_state:
    st.session_state.primary_result = None
if 'opt_params' not in st.session_state:
    st.session_state.opt_params = None
if 'buy_rules_count' not in st.session_state:
    st.session_state.buy_rules_count = len(DEFAULT_BUY_RULES)
if 'sell_rules_count' not in st.session_state:
    st.session_state.sell_rules_count = len(DEFAULT_SELL_RULES)
if 'regression_mode' not in st.session_state:
    st.session_state.regression_mode = "static"

# ---------- 侧边栏：标的选择 ----------
st.sidebar.header("标的选择")
asset_category = st.sidebar.selectbox("资产类别", list(ASSET_MAP.keys()), index=0)
asset_names = list(ASSET_MAP[asset_category].keys())
asset_name_sel = st.sidebar.selectbox("选择标的", asset_names, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("或输入自定义代码")
custom_code = st.sidebar.text_input("代码", placeholder="例如 sz399006 或 ^GSPC")
custom_source = st.sidebar.selectbox("数据源", ["tushare (A股)", "akshare (A股)", "yfinance (全球)"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("回归模式")
regression_mode = st.sidebar.radio(
    "选择对数回归方式",
    ["静态 (全局拟合)", "动态 (扩展窗口)"],
    index=0,
    help="静态：使用全部历史数据拟合一条趋势线。动态：从最早2个月数据开始逐步扩展拟合，每月使用截至当月的趋势线。"
)
st.session_state.regression_mode = "dynamic" if "动态" in regression_mode else "static"

st.sidebar.markdown("---")
st.sidebar.subheader("数据频率")
freq_choice = st.sidebar.radio(
    "K线频率", ["月线", "周线", "日线"], index=0,
    help="月线：按月汇总。周线：按周汇总。日线：使用每日数据。"
)
FREQ_LABEL_MAP = {"月线": "monthly", "周线": "weekly", "日线": "daily"}
st.session_state.freq = FREQ_LABEL_MAP[freq_choice]

run_primary = st.sidebar.button("▶️ Run Primary Analysis", type="primary", use_container_width=True)

# ---------- 主分析运行 ----------
if run_primary:
    if custom_code.strip():
        sel_name = custom_code.strip()
        sel_code = custom_code.strip()
        sel_source = "tushare" if "tushare" in custom_source else ("akshare" if "akshare" in custom_source else "yfinance")
    else:
        sel_name = asset_name_sel
        asset_info = ASSET_MAP[asset_category][asset_name_sel]
        sel_code = asset_info["code"]
        sel_source = asset_info["source"]

    with st.spinner(f"正在分析 {sel_name} ({sel_code}) [{sel_source}] [{freq_choice}] …"):
        result = run_full_analysis(sel_name, sel_code, source=sel_source,
                                   regression_mode=st.session_state.regression_mode,
                                   freq=st.session_state.freq)

    if result is not None:
        st.session_state.primary_result = result
        st.session_state.opt_params = None  # reset optimization on new primary run

# ---------- 显示主分析结果 ----------
if st.session_state.primary_result is not None:
    data, base_slope, base_intercept, figs = st.session_state.primary_result

    st.success("回测完成 ✅")

    # 数据表格
    format_dict = {
        '当月收盘价': '{:.3f}', '理论对数值': '{:.4f}', '理论值': '{:.2f}',
        '百分比': '{:.2%}', '持有股票数量': '{:.4f}', '股票价值': '{:,.2f}',
        '现金': '{:,.2f}', '仓位百分比': '{:.2%}', '总资产': '{:,.2f}',
        '累计投资': '{:.3f}', '收益': '{:,.2f}', '当月真实收益率': '{:.2%}',
        '净值指数': '{:,.2f}', '年化收益率(IRR)': '{:.2%}', '最大回撤': '{:.2%}',
        '年化波动率': '{:.2%}',
    }
    styled_df = data.style.format(format_dict, na_rep='NA')
    st.dataframe(styled_df, use_container_width=True, height=400)

    # 图表
    for title, fig in figs:
        st.subheader(title)
        st.pyplot(fig)
        plt.close(fig)

    # ============================================================
    # 策略实验室（展开面板）
    # ============================================================
    with st.expander("🧠 策略实验室：自定义回测 & 高维防过拟合寻优", expanded=False):
        unique_dates = sorted(data['日期'].unique())

        st.markdown("#### 1. 基础环境")
        col_env1, col_env2 = st.columns(2)
        backtest_start = col_env1.selectbox("回测起始年月", unique_dates, index=0)
        extra_cash = col_env2.number_input("额外初始现金", value=0.0, step=1.0, format="%.1f")

        st.markdown("---")
        cur_reg_mode = st.session_state.regression_mode
        opt = st.session_state.opt_params

        if cur_reg_mode == "static":
            st.markdown("#### 2. 拟合线微调 (静态模式)")
            col_sl, col_ic = st.columns(2)
            default_slope = opt['slope'] if opt else base_slope
            default_intercept = opt['icpt'] if opt else base_intercept
            slope_val = col_sl.number_input("对数斜率", value=default_slope, step=0.0001, format="%.6f")
            intercept_val = col_ic.number_input("对数截距", value=default_intercept, step=0.1, format="%.4f")
        else:
            st.markdown("#### 2. 拟合线 (动态模式 — 由数据自动决定)")
            st.info("动态回归模式下，斜率和截距由扩展窗口拟合自动计算，无需手动调节。")
            slope_val = None
            intercept_val = None

        st.markdown("---")
        st.markdown("#### 3. 定义规则")

        buy_mode_choice = st.radio(
            "买入模式", ["梯队法", "N次方曲线法"], index=0, horizontal=True,
            help="梯队法：多档阈值+比例。N次方曲线法：ratio = min(1, (|偏差|/阈值)^N)"
        )
        cur_buy_mode = "npower" if "N次方" in buy_mode_choice else "tiered"

        # 决定卖出规则的默认值 (仅梯队模式可复用 opt 参数)
        if opt and opt.get('buy_mode', 'tiered') == 'tiered' and isinstance(opt.get('sell_rules'), list):
            default_sell = sorted(opt['sell_rules'], key=lambda x: x[0])
            if 'sell_rules_count' in st.session_state:
                st.session_state.sell_rules_count = len(default_sell)
        else:
            default_sell = DEFAULT_SELL_RULES

        # --- N次方曲线买卖 ---
        npower_params_input = None
        buy_rules_input = None
        sell_rules_input = None
        if cur_buy_mode == "npower":
            st.markdown("**N次方曲线**: `ratio = min(1, (|偏差| / 阈值) ^ N)`")
            st.markdown("**买入参数**")
            col_np1, col_np2 = st.columns(2)
            if opt and opt.get('buy_mode') == 'npower':
                def_bt = opt['buy_rules']['buy_threshold'] * 100
                def_bn = opt['buy_rules']['buy_n']
                def_st = opt['sell_rules']['sell_threshold'] * 100
                def_sn = opt['sell_rules']['sell_n']
            else:
                def_bt, def_bn = 4.0, 2.0
                def_st, def_sn = 4.0, 2.0
            np_bt = col_np1.number_input("买入阈值 (%)", value=def_bt, step=0.5, format="%.1f")
            np_bn = col_np2.number_input("买入 N次方", value=def_bn, step=0.1, format="%.2f")
            st.markdown("**卖出参数**")
            col_np3, col_np4 = st.columns(2)
            np_st = col_np3.number_input("卖出阈值 (%)", value=def_st, step=0.5, format="%.1f")
            np_sn = col_np4.number_input("卖出 N次方", value=def_sn, step=0.1, format="%.2f")
            st.markdown("**最低交易比例**")
            np_min_ratio = st.number_input(
                "最低交易比例 (%)，低于此比例的买卖自动忽略",
                value=1.0, step=0.5, min_value=0.0, format="%.1f",
                help="当计算出的买入/卖出比例低于此值时，该期不进行操作。")
            npower_params_input = {
                "buy_threshold": np_bt / 100.0, "buy_n": np_bn,
                "sell_threshold": np_st / 100.0, "sell_n": np_sn,
                "min_ratio": np_min_ratio / 100.0
            }
        else:
            # --- 梯队买入 ---
            if opt and opt.get('buy_mode', 'tiered') == 'tiered' and isinstance(opt.get('buy_rules'), list):
                default_buy = sorted(opt['buy_rules'], key=lambda x: x[0], reverse=True)
                if 'buy_rules_count' in st.session_state:
                    st.session_state.buy_rules_count = len(default_buy)
            else:
                default_buy = DEFAULT_BUY_RULES

            st.markdown("**买入规则** (对数阈值需为负数)")
            buy_col1, buy_col2 = st.columns([4, 1])
            with buy_col2:
                if st.button("➕ 加一档买入"):
                    st.session_state.buy_rules_count += 1
                    st.rerun()
                if st.button("➖ 减一档买入") and st.session_state.buy_rules_count > 1:
                    st.session_state.buy_rules_count -= 1
                    st.rerun()

            buy_rules_input = []
            with buy_col1:
                for idx in range(st.session_state.buy_rules_count):
                    c1, c2 = st.columns(2)
                    default_t = default_buy[idx][0] * 100 if idx < len(default_buy) else -5.0
                    default_r = default_buy[idx][1] * 100 if idx < len(default_buy) else 20.0
                    t = c1.number_input(f"买入阈值 % (档{idx + 1})", value=default_t,
                                        step=0.5, format="%.1f", key=f"buy_t_{idx}")
                    r = c2.number_input(f"买入比例 % (档{idx + 1})", value=default_r,
                                        step=5.0, format="%.1f", key=f"buy_r_{idx}")
                    buy_rules_input.append((t / 100.0, r / 100.0))

            # 卖出规则 (仅梯队模式)
            st.markdown("**卖出规则** (对数阈值需为正数)")
            sell_col1, sell_col2 = st.columns([4, 1])
            with sell_col2:
                if st.button("➕ 加一档卖出"):
                    st.session_state.sell_rules_count += 1
                    st.rerun()
                if st.button("➖ 减一档卖出") and st.session_state.sell_rules_count > 1:
                    st.session_state.sell_rules_count -= 1
                    st.rerun()

            sell_rules_input = []
            with sell_col1:
                for idx in range(st.session_state.sell_rules_count):
                    c1, c2 = st.columns(2)
                    default_t = default_sell[idx][0] * 100 if idx < len(default_sell) else 5.0
                    default_r = default_sell[idx][1] * 100 if idx < len(default_sell) else 20.0
                    t = c1.number_input(f"卖出阈值 % (档{idx + 1})", value=default_t,
                                        step=0.5, format="%.1f", key=f"sell_t_{idx}")
                    r = c2.number_input(f"卖出比例 % (档{idx + 1})", value=default_r,
                                        step=5.0, format="%.1f", key=f"sell_r_{idx}")
                    sell_rules_input.append((t / 100.0, r / 100.0))

        st.markdown("---")

        # 自定义回测按钮
        run_custom = st.button("🔄 运行单次自定义回测", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 4. 蒙特卡洛寻优")

        if cur_reg_mode == "dynamic":
            st.caption("ℹ️ 动态回归模式：拟合始终从历史最早数据开始，即使回测起始日期较晚。斜率/截距无需搜索。")

        p_min = data['百分比'].dropna().min() * 100
        p_max = data['百分比'].dropna().max() * 100
        rec_b_gap = max(0.5, round(abs(p_min) / (len(DEFAULT_BUY_RULES) * 2.0), 1))
        rec_s_gap = max(0.5, round(abs(p_max) / (len(DEFAULT_SELL_RULES) * 2.0), 1))

        col_mc1, col_mc2 = st.columns(2)
        mc_iters = col_mc1.number_input("蒙特卡洛次数", value=20000, step=5000, min_value=100)
        mc_calmar = col_mc2.number_input("底线卡玛(>0)", value=1.0, step=0.1, format="%.1f")
        col_mc3, col_mc4 = st.columns(2)
        mc_b_gap = col_mc3.number_input("买档最小间距 (%)", value=rec_b_gap, step=0.1, format="%.1f",
                                         disabled=(cur_buy_mode == "npower"))
        mc_s_gap = col_mc4.number_input("卖档最小间距 (%)", value=rec_s_gap, step=0.1, format="%.1f",
                                         disabled=(cur_buy_mode == "npower"))

        # 自动计算维度
        if cur_buy_mode == "npower":
            mc_buy_dims = 2
        else:
            mc_buy_dims = st.session_state.buy_rules_count * 2
        mc_sell_dims = st.session_state.sell_rules_count * 2
        mc_slope_dims = 0 if cur_reg_mode == "dynamic" else 2
        mc_total_dims = mc_slope_dims + mc_buy_dims + mc_sell_dims

        opt_label = f"🚀 启动蒙特卡洛 {mc_total_dims}维寻优"
        run_opt = st.button(opt_label, type="secondary", use_container_width=True)

        # ---- 执行自定义回测 ----
        if run_custom:
            st.markdown("---")
            st.markdown("### ↓↓↓ 自定义参数单次回测结果 ↓↓↓")
            if custom_code.strip():
                sel_name = custom_code.strip()
                sel_code = custom_code.strip()
                sel_source = "tushare" if "tushare" in custom_source else ("akshare" if "akshare" in custom_source else "yfinance")
            else:
                sel_name = asset_name_sel
                asset_info = ASSET_MAP[asset_category][asset_name_sel]
                sel_code = asset_info["code"]
                sel_source = asset_info["source"]
            with st.spinner("正在执行自定义回测…"):
                custom_result = run_full_analysis(
                    sel_name, sel_code, source=sel_source,
                    custom_slope=slope_val, custom_intercept=intercept_val,
                    custom_buy_rules=buy_rules_input, custom_sell_rules=sell_rules_input,
                    backtest_start_date=backtest_start,
                    extra_initial_cash=extra_cash if extra_cash > 0 else 0,
                    regression_mode=cur_reg_mode, freq=st.session_state.freq,
                    buy_mode=cur_buy_mode, npower_params=npower_params_input)
            if custom_result is not None:
                c_data, _, _, c_figs = custom_result
                c_styled = c_data.style.format(format_dict, na_rep='NA')
                st.dataframe(c_styled, use_container_width=True, height=400)
                for title, fig in c_figs:
                    st.subheader(title)
                    st.pyplot(fig)
                    plt.close(fig)

        # ---- 执行蒙特卡洛寻优 ----
        if run_opt:
            st.markdown("---")
            st.markdown("### ↓↓↓ 蒙特卡洛寻优 ↓↓↓")
            best_p = run_monte_carlo_optimization(
                data, backtest_start,
                extra_cash if extra_cash > 0 else 0,
                base_slope, base_intercept,
                st.session_state.buy_rules_count,
                st.session_state.sell_rules_count,
                mc_iters, mc_calmar, mc_b_gap, mc_s_gap,
                regression_mode=cur_reg_mode, freq=st.session_state.freq,
                buy_mode=cur_buy_mode, npower_params=npower_params_input)
            if best_p:
                st.session_state.opt_params = best_p
                st.info("💡 寻优参数已保存。点击下方按钮将最优参数填入面板并执行回测。")
                if st.button("🎯 将星号策略参数填入面板并执行回测", type="primary"):
                    st.rerun()

