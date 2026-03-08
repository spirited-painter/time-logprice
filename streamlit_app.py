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
        "创业板指": {"code": "sz399006", "source": "akshare_sina"},
        "沪深300": {"code": "sh000300", "source": "akshare_sina"},
        "上证50": {"code": "sh000016", "source": "akshare_sina"},
        "中证500": {"code": "sh000905", "source": "akshare_sina"},
        "中证1000": {"code": "sh000852", "source": "akshare_sina"},
        "科创50": {"code": "sh000688", "source": "akshare_sina"},
        "上证综合指数": {"code": "sh000001", "source": "akshare_sina"},
        "中证银行": {"code": "sz399986", "source": "akshare_sina"},
        "中证券商": {"code": "sz399975", "source": "akshare_sina"},
        "中证保险": {"code": "sz399809", "source": "akshare_sina"},
        "中证主要消费": {"code": "sh000932", "source": "akshare_sina"},
        "中证可选消费": {"code": "sh000931", "source": "akshare_sina"},
        "国证食品饮料": {"code": "sz399396", "source": "akshare_sina"},
        "中证白酒": {"code": "sz399997", "source": "akshare_sina"},
        "中证医药卫生": {"code": "sh000933", "source": "akshare_sina"},
        "中证房地产": {"code": "sh000952", "source": "akshare_sina"},
        "中证基建工程": {"code": "sz399995", "source": "akshare_sina"},
        "中证能源": {"code": "sh000928", "source": "akshare_sina"},
        "中证材料": {"code": "sh000929", "source": "akshare_sina"},
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
    """获取 OHLCV 数据。freq: 'monthly'/'weekly'/'daily'。
    source: 'akshare'/'akshare_em'(东方财富) / 'akshare_sina'(新浪) / 'akshare_tx'(腾讯) / 'yfinance'。"""
    if source in ("akshare", "akshare_em"):
        import akshare as ak
        daily_data = ak.stock_zh_index_daily_em(
            symbol=asset_code, period="daily",
            start_date="19900101", end_date="20990101")
        if daily_data is None or daily_data.empty:
            return None
        daily_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                    'close': 'Close', 'volume': 'Volume'}, inplace=True)
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data.set_index('date', inplace=True)
    elif source == "akshare_sina":
        import akshare as ak
        daily_data = ak.stock_zh_index_daily(symbol=asset_code)
        if daily_data is None or daily_data.empty:
            return None
        daily_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                    'close': 'Close', 'volume': 'Volume'}, inplace=True)
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data.set_index('date', inplace=True)
    elif source == "akshare_tx":
        import akshare as ak
        daily_data = ak.stock_zh_index_daily_tx(symbol=asset_code)
        if daily_data is None or daily_data.empty:
            return None
        daily_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                    'close': 'Close', 'amount': 'Volume'}, inplace=True)
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
                      backtest_start_date=None, backtest_end_date=None,
                      initial_capital=10000.0, monthly_investment=1000.0,
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

    # 截止日期过滤
    if backtest_end_date:
        end_mask = processed_data['日期'] <= backtest_end_date
        processed_data = processed_data[end_mask].reset_index(drop=True)
        processed_data['序号'] = range(1, len(processed_data) + 1)

    x = processed_data['序号']
    log_y = np.log(processed_data['当月收盘价'])

    if regression_mode == "dynamic":
        dyn_slopes, dyn_intercepts, dyn_theo = compute_dynamic_regression(
            x.values, log_y.values)
        processed_data['理论对数值'] = dyn_theo
        processed_data['理论值'] = np.exp(dyn_theo)
        processed_data['滚动斜率'] = dyn_slopes
        processed_data['滚动截距'] = dyn_intercepts
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
        'profit', 'actual_monthly_return', 'annual_irr', 'annual_twr', 'max_drawdown',
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
                    0 if key not in ['annual_irr', 'annual_twr', 'net_value_index']
                    else (np.nan if key in ['annual_irr', 'annual_twr'] else 1000.0))
            continue

        close_price = row['当月收盘价']
        percentage = row['百分比']
        assets_before_sip = shares_held * close_price + cash_held

        if not trading_started:
            trading_started = True
            actual_monthly_return = 0.0
            cash_held += initial_capital
            cumulative_investment += initial_capital
        else:
            actual_monthly_return = (
                (assets_before_sip - last_month_assets) / last_month_assets
                if last_month_assets > 0 else 0.0)
            net_value_index *= (1.0 + actual_monthly_return)
            cash_held += monthly_investment
            cumulative_investment += monthly_investment

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
        annual_twr = np.nan
        num_periods = i - start_idx_for_irr + 1
        if num_periods >= ppy:
            cash_flows = [-monthly_investment] * num_periods
            cash_flows[0] = -initial_capital
            cash_flows.append(total_assets)
            try:
                period_irr = npf.irr(cash_flows)
                if not np.isnan(period_irr) and period_irr > -0.99:
                    annual_irr = (1 + period_irr) ** ppy - 1
            except (ValueError, TypeError):
                pass
            
            if num_periods > 0:
                annual_twr = (net_value_index / 1000.0) ** (float(ppy) / num_periods) - 1.0

        for key, val in zip(results_lists.keys(), [
            shares_held, cash_held, stock_value, total_assets, cumulative_investment,
            profit, actual_monthly_return, annual_irr, annual_twr, max_drawdown, volatility,
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
    processed_data['年化收益率(TWR)'] = results_lists['annual_twr']
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
    ax1.set_ylabel('Index Points (Log Scale)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')
    from matplotlib.ticker import ScalarFormatter, FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))

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
    start_idx = processed_data[
        processed_data['日期'] == backtest_start_date].index[0] if backtest_start_date else 0

    price_series = processed_data['当月收盘价']
    net_value_series = processed_data['净值指数']

    if start_idx < len(price_series):
        base_price = price_series.iloc[start_idx]
        base_net_value = net_value_series.iloc[start_idx]

        price_to_plot = price_series.copy()
        price_to_plot.iloc[:start_idx] = np.nan
        
        strategy_mapped_price = (net_value_series / base_net_value) * base_price
        strategy_mapped_price.iloc[:start_idx] = np.nan

        ax3.plot(dates_for_plot, price_to_plot, color=color1,
                 label=f'{index_code} (Actual Price)', linewidth=2, alpha=0.8)
        ax3.plot(dates_for_plot, strategy_mapped_price, color=color2,
                 label='Strategy Net Value (Mapped to Index)', linewidth=2)
        
        ax3.set_yscale('log')
        from matplotlib.ticker import FuncFormatter
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Index Points (Log Scale)', fontsize=12)
        ax3.set_title(
            f'Performance Comparison (Starting from {base_price:.2f} at '
            f'{backtest_start_date if backtest_start_date else processed_data["日期"].iloc[0]})',
            fontsize=18)
        ax3.legend(loc='upper left', fontsize=12)
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


def run_monte_carlo_optimization(data, start_date, end_date, initial_cap, monthly_inv, base_slope, base_icpt,
                                  num_buy_tiers, num_sell_tiers, num_iters,
                                  target_calmar, min_b_gap, min_s_gap,
                                  regression_mode="static", freq="monthly",
                                  buy_mode="tiered", npower_params=None):
    ppy = FREQ_CONFIG[freq]["periods_per_year"]
    start_idx = data[data['日期'] == start_date].index[0] if start_date else 0
    if end_date:
        end_indices = data[data['日期'] == end_date].index
        end_idx = end_indices[0] + 1 if len(end_indices) > 0 else len(data)
    else:
        end_idx = len(data)
    close_prices = data['当月收盘价'].values[start_idx:end_idx]
    x_arr = data['序号'].values[start_idx:end_idx]
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
    worst_target_ret, worst_target_params = 9999.0, None
    fallback_calmar, fallback_params = -9999.0, None
    sampled_results = []
    progress_bar = st.progress(0, text="寻优进度: 0%")

    if is_dynamic:
        fixed_pct = data['百分比'].values[start_idx:end_idx]
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

        cash = initial_cap
        shares = 0.0
        net_val = 1.0
        peak_nv = 1.0
        max_dd = 0.0
        last_assets = 0.0
        if not is_npower:
            sell_trig = exec_sell[-1][0]
        is_first = True

        # HHI Tracking variables
        buy_spends = []
        sell_solds = []

        for pct_val, price in zip(test_pct, close_prices):
            assets_before_sip = shares * price + cash
            if is_first:
                is_first = False
            else:
                ret = ((assets_before_sip - last_assets) / last_assets
                       if last_assets > 0 else 0.0)
                net_val *= (1.0 + ret)
                cash += monthly_inv

            # 买入
            if is_npower:
                if pct_val < 0:  # 低估时每期都买入
                    ratio = min(1.0, (abs(pct_val) / cur_np_bt) ** cur_np_bn)
                    if ratio >= cur_np_min_ratio:
                        spend = cash * ratio
                        buy_spends.append(spend)
                        shares += spend / price
                        cash -= spend
            else:
                if pct_val <= buy_trig:
                    for thresh, ratio in exec_buy:
                        if pct_val <= thresh:
                            spend = cash * ratio
                            buy_spends.append(spend)
                            shares += spend / price
                            cash -= spend
                            break

            # 卖出
            if is_npower:
                if pct_val > 0:  # 高估时每期都卖出
                    ratio = min(1.0, (pct_val / cur_np_st) ** cur_np_sn)
                    if ratio >= cur_np_min_ratio:
                        sold = shares * ratio
                        sell_solds.append(sold)
                        cash += sold * price
                        shares -= sold
            else:
                if pct_val >= sell_trig:
                    for thresh, ratio in exec_sell:
                        if pct_val >= thresh:
                            sold = shares * ratio
                            sell_solds.append(sold)
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
        base_calmar = ann_ret / abs(max_dd) if max_dd < 0 else (ann_ret * 10 if ann_ret > 0 else 0)

        # HHI Overfit Penalty
        enable_hhi_penalty = npower_params.get("enable_hhi", False) if npower_params else False
        if enable_hhi_penalty:
            target_months = npower_params.get("hhi_target_months", 12.0)
            penalty_strength = npower_params.get("hhi_penalty_strength", 0.5)

            target_hhi = 1.0 / max(1.0, target_months)
            
            tot_buy = sum(buy_spends)
            hhi_buy = sum((s / tot_buy) ** 2 for s in buy_spends) if tot_buy > 0 else 1.0
            
            tot_sell = sum(sell_solds)
            hhi_sell = sum((s / tot_sell) ** 2 for s in sell_solds) if tot_sell > 0 else 1.0
            
            avg_hhi = (hhi_buy + hhi_sell) / 2.0
            
            if avg_hhi > target_hhi:
                excess_ratio = min(1.0, (avg_hhi - target_hhi) / (1.0 - target_hhi))
                penalty_multiplier = 1.0 - (penalty_strength * excess_ratio)
                calmar = base_calmar * penalty_multiplier
            else:
                calmar = base_calmar
        else:
            calmar = base_calmar

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
        # 展平梯队参数：每档的阈值和比例单独作为列
        if not is_npower:
            for ti, (t_val, r_val) in enumerate(sorted(exec_buy, key=lambda x: x[0])):
                params_dict[f'buy_t{ti+1}'] = t_val
                params_dict[f'buy_r{ti+1}'] = r_val
            for ti, (t_val, r_val) in enumerate(sorted(exec_sell, key=lambda x: x[0], reverse=True)):
                params_dict[f'sell_t{ti+1}'] = t_val
                params_dict[f'sell_r{ti+1}'] = r_val

        if calmar >= target_calmar:
            if ann_ret > best_target_ret:
                best_target_ret = ann_ret
                best_target_params = params_dict
        
        if ann_ret < worst_target_ret:
            worst_target_ret = ann_ret
            worst_target_params = params_dict
        
        # We also might want to track fallback by base_calmar or calmar. Using penalized calmar for fallback.
        if calmar > fallback_calmar:
            fallback_calmar = calmar
            fallback_params = params_dict

        if np.random.rand() < (5000.0 / num_iters):
            sampled_results.append(params_dict)

    progress_bar.progress(1.0, text="寻优完成！✅")
    elapsed = time.time() - start_time
    st.success(f"寻优完成！总耗时: {elapsed:.2f} 秒。")

    best = best_target_params if best_target_params is not None else fallback_params
    worst = worst_target_params
    if best is None:
        st.error("未能找到任何有效策略参数。")
        return None, None

    title_msg = (f"🏆 满足卡玛底线(>={target_calmar}) 的【最高收益】组合 🏆"
                 if best_target_params
                 else f"⚠️ 未能找到卡玛>={target_calmar}的组合。退而求其次展示【最高卡玛】组合 ⚠️")

    st.markdown("---")
    st.markdown(f"### {title_msg}")
    st.markdown(
        f"- **预期年化**: {best['ret'] * 100:.2f}% | **最大回撤**: {best['dd'] * 100:.2f}% "
        f"| **卡玛得分**: {best['calmar']:.2f}")
    if not is_dynamic:
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

        # 收集所有参数的 Spearman 相关系数
        spearman_rows = []
        if is_np_mode and 'buy_n' in df_samp.columns:
            for col, lbl in [('buy_threshold', '买入阈值'), ('buy_n', '买入N'),
                              ('sell_threshold', '卖出阈值'), ('sell_n', '卖出N')]:
                if col in df_samp.columns:
                    rho, pval = spearmanr(df_samp[col], df_samp['ret'])
                    spearman_rows.append(({'参数': lbl, 'Spearman ρ': f'{rho:.4f}', 'p-value': f'{pval:.2e}'}))
        else:
            # 梯队模式：分析每档参数
            if not is_dynamic and 'slope' in df_samp.columns:
                rho, pval = spearmanr(df_samp['slope'], df_samp['ret'])
                spearman_rows.append({'参数': '对数斜率', 'Spearman ρ': f'{rho:.4f}', 'p-value': f'{pval:.2e}'})
            for ti in range(1, num_buy_tiers + 1):
                col_t = f'buy_t{ti}'
                col_r = f'buy_r{ti}'
                if col_t in df_samp.columns:
                    rho, pval = spearmanr(df_samp[col_t], df_samp['ret'])
                    spearman_rows.append({'参数': f'买入阈值{ti}', 'Spearman ρ': f'{rho:.4f}', 'p-value': f'{pval:.2e}'})
                if col_r in df_samp.columns:
                    rho, pval = spearmanr(df_samp[col_r], df_samp['ret'])
                    spearman_rows.append({'参数': f'买入比例{ti}', 'Spearman ρ': f'{rho:.4f}', 'p-value': f'{pval:.2e}'})
            for ti in range(1, num_sell_tiers + 1):
                col_t = f'sell_t{ti}'
                col_r = f'sell_r{ti}'
                if col_t in df_samp.columns:
                    rho, pval = spearmanr(df_samp[col_t], df_samp['ret'])
                    spearman_rows.append({'参数': f'卖出阈值{ti}', 'Spearman ρ': f'{rho:.4f}', 'p-value': f'{pval:.2e}'})
                if col_r in df_samp.columns:
                    rho, pval = spearmanr(df_samp[col_r], df_samp['ret'])
                    spearman_rows.append({'参数': f'卖出比例{ti}', 'Spearman ρ': f'{rho:.4f}', 'p-value': f'{pval:.2e}'})

        # 将 Spearman 结果显示为表格（替代原来的柱状图）
        if spearman_rows:
            spearman_df = pd.DataFrame(spearman_rows)
            spearman_df['含义说明'] = spearman_df['Spearman ρ'].apply(
                lambda x: '强正相关 (参数增大→收益增)' if float(x) > 0.3
                else ('强负相关 (参数增大→收益减)' if float(x) < -0.3
                      else '弱相关 (参数对收益影响不显著)'))
            # 仍然在第二个子图中绘制柱状图
            rho_vals = [float(r['Spearman ρ']) for r in spearman_rows]
            param_labels = [r['参数'] for r in spearman_rows]
            colors = ['#2ecc71' if v > 0.3 else ('#e74c3c' if v < -0.3 else '#95a5a6') for v in rho_vals]
            ax2.barh(param_labels, rho_vals, color=colors)
            ax2.set_title('参数敏感度 (与收益率 Spearman 相关)',
                          fontsize=16, fontweight='bold')
            ax2.set_xlabel('Spearman ρ', fontsize=12)
            ax2.axvline(0, color='black', linewidth=1)
            ax2.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # 显示 Spearman 详细表格
            st.markdown("### 📊 参数敏感度 Spearman 相关系数表")
            st.caption("ρ 值范围 -1 ~ 1。正值表示参数增大时收益倾向于增大，负值反之。|0.3| 以上为显著相关。")
            st.dataframe(spearman_df, use_container_width=True, hide_index=True)
        else:
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # 有效前沿表
        if pareto_x:
            st.markdown("### ✨ 有效前沿策略参数明细库 ✨")
            pareto_rows = []
            max_ret_val2 = -999
            for _, row in df_sorted.iterrows():
                if row['ret'] > max_ret_val2:
                    pareto_rows.append(row.to_dict())
                    max_ret_val2 = row['ret']

            frontier_df = pd.DataFrame(pareto_rows)

            base_cols = {
                '风险(最大回撤)': (frontier_df['dd'] * 100).map('{:.2f}%'.format),
                '收益(预期年化)': (frontier_df['ret'] * 100).map('{:.2f}%'.format),
                '卡玛得分': frontier_df['calmar'].map('{:.2f}'.format),
            }
            if not is_dynamic:
                base_cols['对数斜率'] = frontier_df['slope'].map('{:.6f}'.format)
                base_cols['对数截距'] = frontier_df['icpt'].map('{:.4f}'.format)

            if is_np_mode and 'buy_threshold' in frontier_df.columns:
                base_cols['买入阈值'] = (frontier_df['buy_threshold'] * 100).map('{:.2f}%'.format)
                base_cols['买入N'] = frontier_df['buy_n'].map('{:.2f}'.format)
                base_cols['卖出阈值'] = (frontier_df['sell_threshold'] * 100).map('{:.2f}%'.format)
                base_cols['卖出N'] = frontier_df['sell_n'].map('{:.2f}'.format)
            else:
                # 每档参数展开为独立列
                for ti in range(1, num_buy_tiers + 1):
                    col_t = f'buy_t{ti}'
                    col_r = f'buy_r{ti}'
                    if col_t in frontier_df.columns:
                        base_cols[f'买入阈值{ti}'] = (frontier_df[col_t] * 100).map('{:.2f}%'.format)
                    if col_r in frontier_df.columns:
                        base_cols[f'买入比例{ti}'] = (frontier_df[col_r] * 100).map('{:.1f}%'.format)
                for ti in range(1, num_sell_tiers + 1):
                    col_t = f'sell_t{ti}'
                    col_r = f'sell_r{ti}'
                    if col_t in frontier_df.columns:
                        base_cols[f'卖出阈值{ti}'] = (frontier_df[col_t] * 100).map('{:.2f}%'.format)
                    if col_r in frontier_df.columns:
                        base_cols[f'卖出比例{ti}'] = (frontier_df[col_r] * 100).map('{:.1f}%'.format)
            display_df = pd.DataFrame(base_cols)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ---- 回撤稳定区/收益稳定区公用列映射 ----
        dd_cols = {'dd': '最大回撤', 'ret': '年化收益', 'calmar': '卡玛'}
        if not is_dynamic:
            dd_cols['slope'] = '斜率'
            dd_cols['icpt'] = '截距'

        if is_np_mode:
            dd_cols.update({'buy_threshold': '买入阈值', 'buy_n': '买入N', 'sell_threshold': '卖出阈值', 'sell_n': '卖出N'})
        else:
            for ti in range(1, num_buy_tiers + 1):
                dd_cols[f'buy_t{ti}'] = f'买入阈值{ti}'
                dd_cols[f'buy_r{ti}'] = f'买入比例{ti}'
            for ti in range(1, num_sell_tiers + 1):
                dd_cols[f'sell_t{ti}'] = f'卖出阈值{ti}'
                dd_cols[f'sell_r{ti}'] = f'卖出比例{ti}'

        def _format_stability_col(display_df_in):
            for c in display_df_in.columns:
                if c in ('最大回撤', '年化收益'):
                    display_df_in[c] = display_df_in[c].map(lambda v: f"{v*100:.2f}%")
                elif c == '卡玛':
                    display_df_in[c] = display_df_in[c].map(lambda v: f"{v:.2f}")
                elif '阈值' in c:
                    display_df_in[c] = display_df_in[c].map(lambda v: f"{v*100:.2f}%")
                elif '比例' in c:
                    display_df_in[c] = display_df_in[c].map(lambda v: f"{v*100:.1f}%")
                elif c in ('买入N', '卖出N'):
                    display_df_in[c] = display_df_in[c].map(lambda v: f"{v:.2f}")
                elif c == '斜率':
                    display_df_in[c] = display_df_in[c].map(lambda v: f"{v:.6f}")
                elif c == '截距':
                    display_df_in[c] = display_df_in[c].map(lambda v: f"{v:.4f}")
            return display_df_in

        # ---- 回撤稳定区：回撤相近，收益递增 ----
        st.markdown("### 📊 回撤稳定区 — 回撤相近时哪些参数提升了收益？")
        df_samp_copy = df_samp.copy()
        df_samp_copy['dd_bin'] = (df_samp_copy['dd'] * 200).round() / 200
        dd_stable = df_samp_copy.loc[df_samp_copy.groupby('dd_bin')['ret'].idxmax()]
        dd_stable = dd_stable.sort_values('ret', ascending=True).tail(30)
        dd_display = dd_stable[[c for c in dd_cols if c in dd_stable.columns]].copy()
        dd_display.columns = [dd_cols[c] for c in dd_display.columns]
        dd_display = _format_stability_col(dd_display)
        st.dataframe(dd_display.reset_index(drop=True), use_container_width=True, hide_index=True)

        # ---- 收益稳定区：收益相近，回撤递增 ----
        st.markdown("### 📊 收益稳定区 — 收益相近时哪些参数增加了风险？")
        df_samp_copy['ret_bin'] = (df_samp_copy['ret'] * 200).round() / 200
        ret_stable = df_samp_copy.loc[df_samp_copy.groupby('ret_bin')['dd'].idxmin()]
        ret_stable = ret_stable.sort_values('dd', ascending=True).tail(30)
        ret_display = ret_stable[[c for c in dd_cols if c in ret_stable.columns]].copy()
        ret_display.columns = [dd_cols[c] for c in ret_display.columns]
        ret_display = _format_stability_col(ret_display)
        st.dataframe(ret_display.reset_index(drop=True), use_container_width=True, hide_index=True)

    return best, worst


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
if 'show_star_backtest_picker' not in st.session_state:
    st.session_state.show_star_backtest_picker = False

# ---------- 侧边栏：标的选择 ----------
st.sidebar.header("标的选择")
asset_category = st.sidebar.selectbox("资产类别", list(ASSET_MAP.keys()), index=0)
asset_names = list(ASSET_MAP[asset_category].keys())
asset_name_sel = st.sidebar.selectbox("选择标的", asset_names, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("或输入自定义代码")
custom_code = st.sidebar.text_input("代码", placeholder="例如 sz399006 或 ^GSPC")
custom_source = st.sidebar.selectbox("数据源", [
    "akshare_sina (A股-新浪)",
    "akshare (A股-东方财富)",
    "akshare_tx (A股-腾讯)",
    "yfinance (全球)",
], index=0)

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
        if "akshare_sina" in custom_source:
            sel_source = "akshare_sina"
        elif "akshare_tx" in custom_source:
            sel_source = "akshare_tx"
        elif "akshare" in custom_source:
            sel_source = "akshare"
        else:
            sel_source = "yfinance"
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
        '百分比': '{:.2%}', '滚动斜率': '{:.6f}', '滚动截距': '{:.4f}',
        '持有股票数量': '{:.4f}', '股票价值': '{:,.2f}',
        '现金': '{:,.2f}', '仓位百分比': '{:.2%}', '总资产': '{:,.2f}',
        '累计投资': '{:.3f}', '收益': '{:,.2f}', '当月真实收益率': '{:.2%}',
        '净值指数': '{:,.2f}', '年化收益率(IRR)': '{:.2%}', '年化收益率(TWR)': '{:.2%}', '最大回撤': '{:.2%}',
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
    # 策略实验室（分离版本）
    # ============================================================
    st.markdown("---")
    st.markdown("## 🧠 策略实验室")
    
    tab1, tab2 = st.tabs(["🧩 单次自定义策略回测", "🚀 蒙特卡洛参数高维寻优"])
    
    unique_dates = sorted(data['日期'].unique())
    date_series = pd.Series(unique_dates)
    parsed_dates = pd.to_datetime(date_series)
    available_years = sorted(parsed_dates.dt.year.unique())

    # ------------------------------------------------------------------
    # TAB 1: 单次自定义策略回测
    # ------------------------------------------------------------------
    with tab1:
        st.markdown("### 1. 基础环境设置")
        t1_col_sy, t1_col_sm = st.columns(2)
        start_year_1 = t1_col_sy.selectbox("回测起始年", available_years, index=0, key="t1_start_year")
        start_months_mask_1 = parsed_dates.dt.year == start_year_1
        start_month_dates_1 = date_series[start_months_mask_1].tolist()
        backtest_start_1 = t1_col_sm.selectbox("起始月", start_month_dates_1, index=0, key="t1_start_month")

        t1_col_ey, t1_col_em = st.columns(2)
        end_year_1 = t1_col_ey.selectbox("回测截止年", available_years, index=len(available_years) - 1, key="t1_end_year")
        end_months_mask_1 = parsed_dates.dt.year == end_year_1
        end_month_dates_1 = date_series[end_months_mask_1].tolist()
        backtest_end_1 = t1_col_em.selectbox("截止月", end_month_dates_1, index=len(end_month_dates_1) - 1, key="t1_end_month")

        t1_col_cap, t1_col_inv = st.columns(2)
        initial_capital_1 = t1_col_cap.number_input("初始资金", value=10000.0, step=1000.0, format="%.2f", key="t1_init_cap")
        monthly_investment_1 = t1_col_inv.number_input("每月定投资金", value=1000.0, step=100.0, format="%.2f", key="t1_monthly_inv")

        st.markdown("---")
        cur_reg_mode = st.session_state.regression_mode
        
        if cur_reg_mode == "static":
            st.markdown("### 2. 拟合线微调 (静态模式)")
            t1_col_sl, t1_col_ic = st.columns(2)
            slope_val = t1_col_sl.number_input("对数斜率", value=base_slope, step=0.0001, format="%.6f", key="t1_slope")
            intercept_val = t1_col_ic.number_input("对数截距", value=base_intercept, step=0.1, format="%.4f", key="t1_icpt")
        else:
            st.markdown("### 2. 拟合线 (动态模式 — 由数据自动决定)")
            st.info("动态回归模式下，斜率和截距由扩展窗口拟合自动计算，无需手动调节。")
            slope_val = None
            intercept_val = None

        st.markdown("---")
        st.markdown("### 3. 定义交易规则")

        buy_mode_choice_1 = st.radio(
            "买入模式", ["梯队法", "N次方曲线法"], index=0, horizontal=True,
            help="梯队法：多档阈值+比例。N次方曲线法：ratio = min(1, (|偏差|/阈值)^N)",
            key="t1_buy_mode"
        )
        cur_buy_mode_1 = "npower" if "N次方" in buy_mode_choice_1 else "tiered"

        npower_params_input_1 = None
        buy_rules_input_1 = None
        sell_rules_input_1 = None

        if cur_buy_mode_1 == "npower":
            st.markdown("**N次方曲线**: `ratio = min(1, (|偏差| / 阈值) ^ N)`")
            st.markdown("**买入参数**")
            col_np1, col_np2 = st.columns(2)
            np_bt_1 = col_np1.number_input("买入阈值 (%)", value=4.0, step=0.5, format="%.1f", key="t1_np_bt")
            np_bn_1 = col_np2.number_input("买入 N次方", value=2.0, step=0.1, format="%.2f", key="t1_np_bn")
            st.markdown("**卖出参数**")
            col_np3, col_np4 = st.columns(2)
            np_st_1 = col_np3.number_input("卖出阈值 (%)", value=4.0, step=0.5, format="%.1f", key="t1_np_st")
            np_sn_1 = col_np4.number_input("卖出 N次方", value=2.0, step=0.1, format="%.2f", key="t1_np_sn")
            st.markdown("**最低交易比例**")
            np_min_ratio_1 = st.number_input("最低交易比例 (%)，低于此比例的买卖自动忽略", value=1.0, step=0.5, min_value=0.0, format="%.1f", key="t1_np_min")
            npower_params_input_1 = {
                "buy_threshold": np_bt_1 / 100.0, "buy_n": np_bn_1,
                "sell_threshold": np_st_1 / 100.0, "sell_n": np_sn_1,
                "min_ratio": np_min_ratio_1 / 100.0
            }
        else:
            default_buy = DEFAULT_BUY_RULES
            default_sell = DEFAULT_SELL_RULES
            st.markdown("**买入规则** (对数阈值需为负数)")
            buy_col1, buy_col2 = st.columns([4, 1])
            with buy_col2:
                if st.button("➕ 加一档买入", key="t1_btn_add_buy"):
                    st.session_state.buy_rules_count += 1
                    st.rerun()
                if st.button("➖ 减一档买入", key="t1_btn_sub_buy") and st.session_state.buy_rules_count > 1:
                    st.session_state.buy_rules_count -= 1
                    st.rerun()

            buy_rules_input_1 = []
            with buy_col1:
                for idx in range(st.session_state.buy_rules_count):
                    c1, c2 = st.columns(2)
                    default_t = default_buy[idx][0] * 100 if idx < len(default_buy) else -5.0
                    default_r = default_buy[idx][1] * 100 if idx < len(default_buy) else 20.0
                    t = c1.number_input(f"买入阈值 % (档{idx + 1})", value=default_t, step=0.5, format="%.1f", key=f"t1_buy_t_{idx}")
                    r = c2.number_input(f"买入比例 % (档{idx + 1})", value=default_r, step=5.0, format="%.1f", key=f"t1_buy_r_{idx}")
                    buy_rules_input_1.append((t / 100.0, r / 100.0))

            st.markdown("**卖出规则** (对数阈值需为正数)")
            sell_col1, sell_col2 = st.columns([4, 1])
            with sell_col2:
                if st.button("➕ 加一档卖出", key="t1_btn_add_sell"):
                    st.session_state.sell_rules_count += 1
                    st.rerun()
                if st.button("➖ 减一档卖出", key="t1_btn_sub_sell") and st.session_state.sell_rules_count > 1:
                    st.session_state.sell_rules_count -= 1
                    st.rerun()

            sell_rules_input_1 = []
            with sell_col1:
                for idx in range(st.session_state.sell_rules_count):
                    c1, c2 = st.columns(2)
                    default_t = default_sell[idx][0] * 100 if idx < len(default_sell) else 5.0
                    default_r = default_sell[idx][1] * 100 if idx < len(default_sell) else 20.0
                    t = c1.number_input(f"卖出阈值 % (档{idx + 1})", value=default_t, step=0.5, format="%.1f", key=f"t1_sell_t_{idx}")
                    r = c2.number_input(f"卖出比例 % (档{idx + 1})", value=default_r, step=5.0, format="%.1f", key=f"t1_sell_r_{idx}")
                    sell_rules_input_1.append((t / 100.0, r / 100.0))

        st.markdown("---")
        run_custom = st.button("🔄 运行单次自定义回测", type="primary", use_container_width=True, key="t1_btn_run")

        if run_custom:
            st.markdown("### ↓↓↓ 回测结果 ↓↓↓")
            if custom_code.strip():
                sel_name = custom_code.strip()
                sel_code = custom_code.strip()
                if "akshare_sina" in custom_source:
                    sel_source = "akshare_sina"
                elif "akshare_tx" in custom_source:
                    sel_source = "akshare_tx"
                elif "akshare" in custom_source:
                    sel_source = "akshare"
                else:
                    sel_source = "yfinance"
            else:
                sel_name = asset_name_sel
                asset_info = ASSET_MAP[asset_category][asset_name_sel]
                sel_code = asset_info["code"]
                sel_source = asset_info["source"]
            with st.spinner("正在执行自定义回测…"):
                custom_result = run_full_analysis(
                    sel_name, sel_code, source=sel_source,
                    custom_slope=slope_val, custom_intercept=intercept_val,
                    custom_buy_rules=buy_rules_input_1, custom_sell_rules=sell_rules_input_1,
                    backtest_start_date=backtest_start_1, backtest_end_date=backtest_end_1,
                    initial_capital=initial_capital_1, monthly_investment=monthly_investment_1,
                    regression_mode=cur_reg_mode, freq=st.session_state.freq,
                    buy_mode=cur_buy_mode_1, npower_params=npower_params_input_1)
            if custom_result is not None:
                c_data, _, _, c_figs = custom_result
                c_styled = c_data.style.format(format_dict, na_rep='NA')
                st.dataframe(c_styled, use_container_width=True, height=400)
                for title, fig in c_figs:
                    st.subheader(title)
                    st.pyplot(fig)
                    plt.close(fig)

    # ------------------------------------------------------------------
    # TAB 2: 蒙特卡洛参数高维寻优
    # ------------------------------------------------------------------
    with tab2:
        st.markdown("### 1. 寻优时间范围")
        t2_col_sy, t2_col_sm = st.columns(2)
        start_year_2 = t2_col_sy.selectbox("寻优起始年", available_years, index=0, key="t2_start_year")
        start_months_mask_2 = parsed_dates.dt.year == start_year_2
        start_month_dates_2 = date_series[start_months_mask_2].tolist()
        backtest_start_2 = t2_col_sm.selectbox("起始月", start_month_dates_2, index=0, key="t2_start_month")

        t2_col_ey, t2_col_em = st.columns(2)
        end_year_2 = t2_col_ey.selectbox("寻优截止年", available_years, index=len(available_years) - 1, key="t2_end_year")
        end_months_mask_2 = parsed_dates.dt.year == end_year_2
        end_month_dates_2 = date_series[end_months_mask_2].tolist()
        backtest_end_2 = t2_col_em.selectbox("截止月", end_month_dates_2, index=len(end_month_dates_2) - 1, key="t2_end_month")

        t2_col_cap, t2_col_inv = st.columns(2)
        initial_capital_2 = t2_col_cap.number_input("初始资金", value=10000.0, step=1000.0, format="%.2f", key="t2_init_cap")
        monthly_investment_2 = t2_col_inv.number_input("每月定投资金", value=1000.0, step=100.0, format="%.2f", key="t2_monthly_inv")

        if cur_reg_mode == "dynamic":
            st.caption("ℹ️ 动态回归模式：拟合始终从历史最早数据开始，无需搜索斜率/截距。")

        st.markdown("---")
        st.markdown("### 2. 寻优配置参数")
        
        buy_mode_choice_2 = st.radio("寻优买入模式", ["梯队法", "N次方曲线法"], index=0, horizontal=True, key="t2_buy_mode")
        cur_buy_mode_2 = "npower" if "N次方" in buy_mode_choice_2 else "tiered"

        col_t2_tiers1, col_t2_tiers2 = st.columns(2)
        if cur_buy_mode_2 == "tiered":
            mc_buy_tiers = col_t2_tiers1.number_input("买入档位数量", value=st.session_state.buy_rules_count, step=1, min_value=1, key="t2_buy_tiers")
            mc_sell_tiers = col_t2_tiers2.number_input("卖出档位数量", value=st.session_state.sell_rules_count, step=1, min_value=1, key="t2_sell_tiers")
        else:
            mc_buy_tiers = 1
            mc_sell_tiers = 1

        p_min = data['百分比'].dropna().min() * 100
        p_max = data['百分比'].dropna().max() * 100
        rec_b_gap = max(0.5, round(abs(p_min) / (len(DEFAULT_BUY_RULES) * 2.0), 1))
        rec_s_gap = max(0.5, round(abs(p_max) / (len(DEFAULT_SELL_RULES) * 2.0), 1))

        col_mc1, col_mc2 = st.columns(2)
        mc_iters = col_mc1.number_input("蒙特卡洛次数", value=20000, step=5000, min_value=100, key="t2_mc_iters")
        mc_calmar = col_mc2.number_input("底线卡玛(>0)", value=1.0, step=0.1, format="%.1f", key="t2_mc_cal")
        col_mc3, col_mc4 = st.columns(2)
        mc_b_gap = col_mc3.number_input("买档最小间距 (%)", value=rec_b_gap, step=0.1, format="%.1f", disabled=(cur_buy_mode_2 == "npower"), key="t2_mc_bgap")
        mc_s_gap = col_mc4.number_input("卖档最小间距 (%)", value=rec_s_gap, step=0.1, format="%.1f", disabled=(cur_buy_mode_2 == "npower"), key="t2_mc_sgap")

        npower_params_input_2 = {}
        if cur_buy_mode_2 == "npower":
            np_min_ratio_2 = st.number_input("寻优时最低交易比例 (%)", value=1.0, step=0.5, min_value=0.0, format="%.1f", key="t2_np_min")
            npower_params_input_2["min_ratio"] = np_min_ratio_2 / 100.0
            
        st.markdown("---")
        st.markdown("### 🛡️ 防过拟合惩罚配置 (HHI集中度惩罚)")
        st.caption("避免算法找到“单月全仓抄底”的极端参数，鼓励在市场极值期平稳建仓/平仓。")
        col_hhi1, col_hhi2, col_hhi3 = st.columns(3)
        enable_hhi = col_hhi1.checkbox("启用集中度惩罚", value=True, key="t2_en_hhi", help="若开启，1-2个月内急速建仓的策略将受到卡玛降级惩罚。")
        hhi_target_months = col_hhi2.number_input("目标建仓期 (月)", value=12, step=1, min_value=1, key="t2_hhi_months", help="期望在一轮跌势中耗时几个月买满。推荐 12-18 个月。")
        hhi_penalty_strength = col_hhi3.number_input("惩罚力度 (0~1)", value=0.5, step=0.1, min_value=0.0, max_value=1.0, key="t2_hhi_str", help="0.5表示最极端情况下，卡玛得分减半。")
        
        npower_params_input_2["enable_hhi"] = enable_hhi
        npower_params_input_2["hhi_target_months"] = float(hhi_target_months)
        npower_params_input_2["hhi_penalty_strength"] = float(hhi_penalty_strength)

        if cur_buy_mode_2 == "npower":
            mc_buy_dims = 2
        else:
            mc_buy_dims = mc_buy_tiers * 2
        mc_sell_dims = mc_sell_tiers * 2
        mc_slope_dims = 0 if cur_reg_mode == "dynamic" else 2
        mc_total_dims = mc_slope_dims + mc_buy_dims + mc_sell_dims

        opt_label = f"🚀 启动蒙特卡洛 {mc_total_dims}维寻优"
        run_opt = st.button(opt_label, type="primary", use_container_width=True, key="t2_btn_run")

        if run_opt:
            st.markdown("---")
            best_p, worst_p = run_monte_carlo_optimization(
                data, backtest_start_2, backtest_end_2,
                initial_capital_2, monthly_investment_2,
                base_slope, base_intercept,
                mc_buy_tiers,
                mc_sell_tiers,
                mc_iters, mc_calmar, mc_b_gap, mc_s_gap,
                regression_mode=cur_reg_mode, freq=st.session_state.freq,
                buy_mode=cur_buy_mode_2, npower_params=npower_params_input_2)
            
            # Persist the optimization results to session state
            if best_p:
                st.session_state.opt_best_p = best_p
                st.session_state.opt_worst_p = worst_p
                st.session_state.opt_npower_params = npower_params_input_2

        # --- 渲染已缓存的寻优结果 ---
        if st.session_state.get('opt_best_p') is not None:
            best_p = st.session_state.opt_best_p
            worst_p = st.session_state.opt_worst_p
            saved_npower = st.session_state.opt_npower_params
            
            if custom_code.strip():
                sel_name_star = custom_code.strip()
                sel_code_star = custom_code.strip()
                if "akshare_sina" in custom_source:
                    sel_source_star = "akshare_sina"
                elif "akshare_tx" in custom_source:
                    sel_source_star = "akshare_tx"
                elif "akshare" in custom_source:
                    sel_source_star = "akshare"
                else:
                    sel_source_star = "yfinance"
            else:
                sel_name_star = asset_name_sel
                asset_info_star = ASSET_MAP[asset_category][asset_name_sel]
                sel_code_star = asset_info_star["code"]
                sel_source_star = asset_info_star["source"]

            # --- 1. 执行最优组参数回测 ---
            st.markdown("---")
            st.markdown("### 🏆 【最优参数组】结果验证与展示")
            
            s_buy_mode = best_p.get('buy_mode', 'tiered')
            s_buy_rules = None
            s_sell_rules = None
            s_npower_params = None
            
            if s_buy_mode == 'npower':
                s_npower_params = {
                    "buy_threshold": best_p['buy_rules']['buy_threshold'],
                    "buy_n": best_p['buy_rules']['buy_n'],
                    "sell_threshold": best_p['sell_rules']['sell_threshold'],
                    "sell_n": best_p['sell_rules']['sell_n'],
                    "min_ratio": saved_npower.get('min_ratio', 0.01) if saved_npower else 0.01
                }
            else:
                s_buy_rules = best_p['buy_rules']
                s_sell_rules = best_p['sell_rules']

            with st.spinner("正在绘制最优策略的回测曲线…"):
                star_result = run_full_analysis(
                    sel_name_star, sel_code_star, source=sel_source_star,
                    custom_slope=best_p.get('slope'), custom_intercept=best_p.get('icpt'),
                    custom_buy_rules=s_buy_rules,
                    custom_sell_rules=s_sell_rules,
                    backtest_start_date=backtest_start_2,
                    backtest_end_date=backtest_end_2,
                    initial_capital=initial_capital_2, monthly_investment=monthly_investment_2,
                    regression_mode=cur_reg_mode,
                    freq=st.session_state.freq,
                    buy_mode=s_buy_mode,
                    npower_params=s_npower_params)
            if star_result is not None:
                s_data, _, _, s_figs = star_result
                s_styled = s_data.style.format(format_dict, na_rep='NA')
                st.dataframe(s_styled, use_container_width=True, height=400)
                for title, fig in s_figs:
                    st.subheader(title)
                    st.pyplot(fig)
                    plt.close(fig)

                # --- 2. 执行最差组参数回测 ---
            if worst_p:
                st.markdown("---")
                st.markdown("### 🚫 【最差参数组】反面教材展示")
                
                w_buy_mode = worst_p.get('buy_mode', 'tiered')
                w_buy_rules = None
                w_sell_rules = None
                w_npower_params = None
                
                if w_buy_mode == 'npower':
                    w_npower_params = {
                        "buy_threshold": worst_p['buy_rules']['buy_threshold'],
                        "buy_n": worst_p['buy_rules']['buy_n'],
                        "sell_threshold": worst_p['sell_rules']['sell_threshold'],
                        "sell_n": worst_p['sell_rules']['sell_n'],
                        "min_ratio": saved_npower.get('min_ratio', 0.01) if saved_npower else 0.01
                    }
                else:
                    w_buy_rules = worst_p['buy_rules']
                    w_sell_rules = worst_p['sell_rules']

                with st.spinner("正在绘制最差策略的回测曲线…"):
                    worst_result = run_full_analysis(
                        sel_name_star, sel_code_star, source=sel_source_star,
                        custom_slope=worst_p.get('slope'), custom_intercept=worst_p.get('icpt'),
                        custom_buy_rules=w_buy_rules,
                        custom_sell_rules=w_sell_rules,
                        backtest_start_date=backtest_start_2,
                        backtest_end_date=backtest_end_2,
                        initial_capital=initial_capital_2, monthly_investment=monthly_investment_2,
                        regression_mode=cur_reg_mode,
                        freq=st.session_state.freq,
                        buy_mode=w_buy_mode,
                        npower_params=w_npower_params)
                if worst_result is not None:
                    w_data, _, _, w_figs = worst_result
                    w_styled = w_data.style.format(format_dict, na_rep='NA')
                    with st.expander("显示最差组合的收益数据和图表"):
                        st.dataframe(w_styled, use_container_width=True, height=400)
                        for title, fig in w_figs:
                            st.subheader(title)
                            st.pyplot(fig)
                            plt.close(fig)

            # --- 3. 跨期验证区域 (Out-of-sample) ---
            st.markdown("---")
            st.markdown("### 📅 跨期验证：使用最优参数在其他时间段回测")
            st.caption("选择一个不同于寻优期的时间范围，验证最优参数是否仍然有效。")
            
            oos_col_sy, oos_col_sm = st.columns(2)
            oos_start_year = oos_col_sy.selectbox("验证起始年", available_years, index=0, key="oos_start_year")
            oos_start_months_mask = parsed_dates.dt.year == oos_start_year
            oos_start_month_dates = date_series[oos_start_months_mask].tolist()
            oos_backtest_start = oos_col_sm.selectbox("起始月", oos_start_month_dates, index=0, key="oos_start_month")

            oos_col_ey, oos_col_em = st.columns(2)
            oos_end_year = oos_col_ey.selectbox("验证截止年", available_years, index=len(available_years) - 1, key="oos_end_year")
            oos_end_months_mask = parsed_dates.dt.year == oos_end_year
            oos_end_month_dates = date_series[oos_end_months_mask].tolist()
            oos_backtest_end = oos_col_em.selectbox("截止月", oos_end_month_dates, index=len(oos_end_month_dates) - 1, key="oos_end_month")
            
            oos_col_cap, oos_col_inv = st.columns(2)
            oos_initial_capital = oos_col_cap.number_input("验证期初始资金", value=initial_capital_2, step=1000.0, format="%.2f", key="oos_init_cap")
            oos_monthly_investment = oos_col_inv.number_input("验证期每月定投资金", value=monthly_investment_2, step=100.0, format="%.2f", key="oos_monthly_inv")

            if st.button("▶️ 执行跨期验证", type="primary", use_container_width=True, key="btn_run_oos"):
                st.markdown("#### ↓↓↓ 验证期回测结果 ↓↓↓")
                with st.spinner(f"正在使用最优参数在这段区间 ({oos_backtest_start} 到 {oos_backtest_end}) 进行回测…"):
                    oos_result = run_full_analysis(
                        sel_name_star, sel_code_star, source=sel_source_star,
                        custom_slope=best_p.get('slope'), custom_intercept=best_p.get('icpt'),
                        custom_buy_rules=s_buy_rules,
                        custom_sell_rules=s_sell_rules,
                        backtest_start_date=oos_backtest_start,
                        backtest_end_date=oos_backtest_end,
                        initial_capital=oos_initial_capital, monthly_investment=oos_monthly_investment,
                        regression_mode=cur_reg_mode,
                        freq=st.session_state.freq,
                        buy_mode=s_buy_mode,
                        npower_params=s_npower_params)
                if oos_result is not None:
                    o_data, _, _, o_figs = oos_result
                    o_styled = o_data.style.format(format_dict, na_rep='NA')
                    st.dataframe(o_styled, use_container_width=True, height=400)
                    for title, fig in o_figs:
                        st.subheader(title)
                        st.pyplot(fig)
                        plt.close(fig)

