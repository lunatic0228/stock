"""
回測 v2：抓低谷（策略C）+ 抓起勢（策略A），雙策略共存
─────────────────────────────────────────────
策略C（低谷反彈）：
  進場：不限 ADX，收盤 < MA10，以下 3 個條件達到 ≥ 2 個
        ① detect_oversold 分數 ≥ 3  （超跌）
        ② detect_selling_exhaustion ≥ 1  （賣壓衰竭）
        ③ 法人連買 ≥ 1 日
  出場：① 收復 MA10 → 目標達成
        ② 收盤 < 近20日低點 × 0.97 → 停損
        ③ 持倉 > 20天 且未獲利 → 時間停損
        ④ 法人連賣 ≥ 3日 → 提前出場

策略A（起勢跟蹤）：
  進場：ADX ≥ 25（真趨勢） + 近20日漲幅 ≥ 3%（非橫盤假突破）
        + MA5>MA10 + MA5向上 + 量比≥0.8，或 entry_signals 路徑B（放量突破）
  出場：① 吊燈線（進場後峰值 - 2×ATR）
        ② 雙線跌破（MA5 & MA10）→ 趨勢結束
        ③ 法人連賣 ≥ 3日 → 提前出場
        ④ 乖離 > 8% + RSI > 68 → 減碼30%
           乖離 > 12% + RSI > 75 → 強力減碼（已減碼者出清）

優先順序：C（逆勢低谷）先判斷，不觸發才檢查 A（順勢起勢）
冷卻期：3 天（MIN_HOLD_DAYS 內不出場，避免當沖誤判）

用法：python backtest_v2.py
"""

import sys
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from daily_analysis import calculate_indicators, entry_signals, FINMIND_TOKEN

TICKERS = {
    "2313.TW": "華通",
    "2367.TW": "燿華",
    "6282.TW": "康舒",
    "2049.TW": "上銀",
    "2374.TW": "佳能",
}
LOOKBACK_DAYS         = 365
INIT_CAPITAL          = 100_000
TRIM_RATIO            = 0.30
MIN_HOLD_DAYS         = 3
CHANDELIER_ATR_MULT   = 2.0   # 吊燈線倍數，與 exit_signals 一致

ADX_TREND        = 25
ADX_CHOP         = 20
ADX_INST_BOOST   = 18   # 法人有效買超時降低的 ADX 門檻
INST_BUY_DAYS    = 3    # 外資連買幾日觸發（對齊主程式：≥3日）
INST_SELL_DAYS   = 3    # 外資連賣幾日觸發提前出場

# 停利門檻（A/B 共用，對齊 exit_signals）
TRIM_DEV        = 8.0    # 乖離>8% + RSI>68 → 第一次減碼30%
TRIM_RSI        = 68.0
STRONG_TRIM_DEV = 12.0   # 乖離>12% + RSI>75 → 強力減碼（已減碼者直接出清）
STRONG_TRIM_RSI = 75.0

# 策略C 低谷反彈參數
C_OVERSOLD_MIN   = 3     # 超跌評分最低門檻（/7）
C_STOP_PCT       = 0.97  # 停損：近20日低點 × 此值
C_TIME_STOP      = 20    # 持倉超過 N 天且未獲利 → 時間停損
C_MIN_CONDITIONS = 2     # 3 個進場條件至少達到幾個

# ════════════════════════════════════════════════════════════
#  法人籌碼資料（一次下載整年，回測時查表）
# ════════════════════════════════════════════════════════════
def fetch_institutional(code):
    """從 FinMind 下載整年三大法人資料
    回傳 dict：{date_str: {'foreign': int, 'trust': int, 'dealer': int, 'total': int}}
    單位：張（股數 ÷ 1000）
    """
    start = (datetime.today() - timedelta(days=LOOKBACK_DAYS + 30)).strftime('%Y-%m-%d')
    params = {
        'dataset':    'TaiwanStockInstitutionalInvestorsBuySell',
        'data_id':    code,
        'start_date': start,
    }
    if FINMIND_TOKEN:
        params['token'] = FINMIND_TOKEN

    try:
        resp = requests.get(
            'https://api.finmindtrade.com/api/v4/data',
            params=params, timeout=20
        )
        body = resp.json()
        if not (body.get('status') == 200 and body.get('data')):
            return {}

        by_date = defaultdict(lambda: {'foreign': 0, 'trust': 0, 'dealer': 0})
        for r in body['data']:
            date = r['date']
            name = r.get('name', '')
            net  = ((r.get('buy') or 0) - (r.get('sell') or 0)) // 1000  # 張
            if name == 'Foreign_Investor':
                by_date[date]['foreign'] = net
            elif name == 'Investment_Trust':
                by_date[date]['trust'] = net
            elif name == 'Dealer_self':
                by_date[date]['dealer'] = net

        result = {}
        for date, d in by_date.items():
            result[date] = {**d, 'total': d['foreign'] + d['trust'] + d['dealer']}
        return result
    except Exception:
        return {}


def inst_streak_simple(inst_data, date_str, direction='buy'):
    """外資連續買超或賣超天數（對齊主程式：net > 0 即算買，net < 0 即算賣，無動態門檻）"""
    if not inst_data:
        return 0
    sorted_dates = sorted(d for d in inst_data if d <= date_str)
    if not sorted_dates:
        return 0
    count = 0
    for d in reversed(sorted_dates):
        foreign = inst_data[d]['foreign']
        if direction == 'buy' and foreign > 0:
            count += 1
        elif direction == 'sell' and foreign < 0:
            count += 1
        else:
            break
    return count


def inst_signal(inst_data, date_str):
    """綜合法人訊號（對齊主程式 get_institutional / inst_direction 邏輯）
    回傳 (buy_signal, sell_signal, note_str)
    buy_signal：外資連買 ≥ 3 日
    sell_signal：外資連賣 ≥ 3 日
    """
    f_buy  = inst_streak_simple(inst_data, date_str, 'buy')
    f_sell = inst_streak_simple(inst_data, date_str, 'sell')

    buy_signal  = (f_buy  >= INST_BUY_DAYS)   # INST_BUY_DAYS = 3 → 對齊主程式
    sell_signal = (f_sell >= INST_SELL_DAYS)   # INST_SELL_DAYS = 3

    parts = []
    if f_buy  > 0: parts.append(f"外資連買{f_buy}日")
    if f_sell > 0: parts.append(f"外資連賣{f_sell}日")
    note = " ".join(parts)

    return buy_signal, sell_signal, note


# ════════════════════════════════════════════════════════════
#  ADX 計算
# ════════════════════════════════════════════════════════════
def calc_adx(df, period=14):
    """計算 ADX，回傳最新值"""
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    hl  = high - low
    hpc = (high - close.shift()).abs()
    lpc = (low  - close.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

    atr14      = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di14  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean()  / atr14
    minus_di14 = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr14

    dx  = (100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return float(adx.iloc[-1]) if not adx.empty else 0.0


def classify(sub, trend_thresh=None):
    """回傳 'trend'、'chop' 或 'unclear'"""
    if trend_thresh is None:
        trend_thresh = ADX_TREND
    if len(sub) < 30:
        return 'unclear'
    adx = calc_adx(sub)
    if adx > trend_thresh:
        return 'trend'
    if adx < ADX_CHOP:
        return 'chop'
    return 'unclear'


# ════════════════════════════════════════════════════════════
#  資料下載
# ════════════════════════════════════════════════════════════
def fetch_data(ticker):
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS + 120)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return calculate_indicators(df)


# ════════════════════════════════════════════════════════════
#  單股回測
# ════════════════════════════════════════════════════════════
def backtest_one(ticker, name, df_full, inst_data=None):
    cutoff    = df_full.index[-1] - timedelta(days=LOOKBACK_DAYS)
    df_year   = df_full[df_full.index >= cutoff]
    start_pos = df_full.index.get_loc(df_year.index[0])
    if isinstance(start_pos, slice):
        start_pos = start_pos.start

    trades   = []
    position = None

    for i in range(start_pos, len(df_full)):
        sub      = df_full.iloc[:i+1].copy()
        row      = sub.iloc[-1]
        date     = sub.index[-1]
        close    = float(row['Close'])
        date_str = date.strftime('%Y-%m-%d')

        # 法人當日狀態
        inst_buying, inst_selling, inst_note = inst_signal(inst_data, date_str)

        # ── 持倉中 ──────────────────────────────────────────
        if position is not None:
            buy_price  = position['buy_price']
            shares     = position['shares']
            entry_date = position['entry_date']
            trimmed    = position.get('trimmed', False)
            strategy   = position['strategy']   # 'A' or 'B'
            hold_days  = (date - pd.Timestamp(entry_date)).days

            atr        = float(row['ATR'])
            profit_pct = (close - buy_price) / buy_price * 100
            deviation  = (close - float(row['MA5'])) / float(row['MA5']) * 100
            rsi        = float(row['RSI'])

            # 吊燈線停損：對齊主程式 exit_signals，用近22日最高收盤往下扣 2×ATR
            high22     = float(row['High22'])
            atr_stop   = high22 - CHANDELIER_ATR_MULT * atr
            stop_label = f"吊燈線(22日高{high22:.1f}-2ATR={atr_stop:.1f})"

            exit_reason = None
            in_cooldown = hold_days < MIN_HOLD_DAYS

            # ── 策略A 出場（趨勢型：對齊 exit_signals，減碼為主）────
            if strategy == 'A':
                if deviation > STRONG_TRIM_DEV and rsi > STRONG_TRIM_RSI:
                    if trimmed:
                        # 已減碼過，強力停利直接出清
                        exit_reason = f"A強力停利出清（乖離{deviation:.1f}% RSI{rsi:.1f}）"
                    else:
                        trim_shares = int(shares * TRIM_RATIO)
                        if trim_shares > 0:
                            trades.append({
                                'entry_date': entry_date, 'exit_date': date_str,
                                'buy_price': buy_price, 'exit_price': close,
                                'shares': trim_shares, 'pnl_pct': profit_pct,
                                'reason': f"A強力減碼30%（乖離{deviation:.1f}% RSI{rsi:.1f}）",
                                'hold_days': hold_days, 'strategy': 'A',
                            })
                            position['shares'] -= trim_shares
                            position['trimmed'] = True
                        continue
                elif deviation > TRIM_DEV and rsi > TRIM_RSI and not trimmed:
                    trim_shares = int(shares * TRIM_RATIO)
                    if trim_shares > 0:
                        trades.append({
                            'entry_date': entry_date, 'exit_date': date_str,
                            'buy_price': buy_price, 'exit_price': close,
                            'shares': trim_shares, 'pnl_pct': profit_pct,
                            'reason': f"A減碼30%（乖離{deviation:.1f}% RSI{rsi:.1f}）",
                            'hold_days': hold_days, 'strategy': 'A',
                        })
                        position['shares'] -= trim_shares
                        position['trimmed'] = True
                    continue
                elif close < atr_stop:
                    exit_reason = f"停損（{stop_label}）"
                elif not in_cooldown and inst_selling:
                    exit_reason = f"法人賣超提前出場（{inst_note}）"
                elif not in_cooldown and close < float(row['MA5']) and close < float(row['MA10']):
                    exit_reason = f"趨勢結束（雙線跌破 MA5={row['MA5']:.1f} MA10={row['MA10']:.1f}）"

            # ── 策略C 出場（低谷反彈：目標MA10，停損近低點）────
            elif strategy == 'C':
                low20  = float(row['Low_20'])
                c_stop = low20 * C_STOP_PCT
                target = float(row['MA10'])
                if close < c_stop:
                    exit_reason = f"C停損（低點{low20:.1f}×0.97={c_stop:.1f}）"
                elif close >= target and not in_cooldown:
                    exit_reason = f"C目標達成（收復MA10={target:.1f}）"
                elif hold_days > C_TIME_STOP and profit_pct <= 0:
                    exit_reason = f"C時間停損（持倉{hold_days}天未回升）"
                elif not in_cooldown and inst_selling:
                    exit_reason = f"法人賣超提前出場（{inst_note}）"

            # ── 出場執行 ──────────────────────────────────────
            if exit_reason:
                trades.append({
                    'entry_date': entry_date, 'exit_date': date_str,
                    'buy_price': buy_price, 'exit_price': close,
                    'shares': position['shares'],
                    'pnl_pct': profit_pct,
                    'reason': exit_reason,
                    'hold_days': hold_days,
                    'strategy': strategy,
                })
                position = None
                continue   # 出場後不在同日再進場

        # ── 無持倉：找進場 ──────────────────────────────────
        if date < df_year.index[0] or len(sub) < 30:
            continue

        ma5_rising = float(sub.iloc[-1]['MA5']) > float(sub.iloc[-2]['MA5'])
        ma5        = float(row['MA5'])
        ma10       = float(row['MA10'])
        vol_ratio  = float(row['Vol_ratio'])
        adx_val    = calc_adx(sub)

        # ── 策略C 優先：低谷反彈（不限ADX，收盤需在MA10以下）────
        from daily_analysis import detect_oversold, detect_selling_exhaustion
        try:
            o_score, _, _ = detect_oversold(sub)
            e_count, _    = detect_selling_exhaustion(sub)
        except Exception:
            o_score = e_count = 0

        c_conditions = [
            o_score >= C_OVERSOLD_MIN,   # ① 超跌
            e_count >= 1,                 # ② 賣壓衰竭
            inst_buying,                  # ③ 法人買超
        ]
        below_ma10 = close < ma10
        c_entry    = below_ma10 and (sum(c_conditions) >= C_MIN_CONDITIONS)

        if c_entry:
            shares = int(INIT_CAPITAL / close)
            if shares > 0:
                cond_str = f"超跌{o_score}/7" if c_conditions[0] else ""
                if c_conditions[1]: cond_str += f" 衰竭{e_count}"
                if c_conditions[2]: cond_str += f" {inst_note}"
                position = {
                    'entry_date': date_str, 'buy_price': close,
                    'shares': shares, 'trimmed': False,
                    'strategy': 'C',
                    'entry_mode': f'C-低谷({cond_str.strip()})',
                }
                continue   # C進場後跳過A判斷

        # ── 策略A：起勢跟蹤（ADX ≥ 25 + 20日漲幅 ≥ 3%）────────
        # 20日漲幅：確認真正在上漲，不是橫盤假突破
        close_20d_ago = float(sub['Close'].iloc[-21]) if len(sub) >= 21 else close
        roc_20d       = (close - close_20d_ago) / close_20d_ago * 100
        adx_thresh    = ADX_INST_BOOST if inst_buying else ADX_TREND
        if adx_val >= ADX_TREND and roc_20d >= 3.0:   # ADX≥25 + 近20日真的漲了
            try:
                score, _ = entry_signals(sub)
            except Exception:
                score = 0
            trend_entry = (score >= 4)
            if trend_entry:
                shares = int(INIT_CAPITAL / close)
                if shares > 0:
                    inst_tag = f"+{inst_note}" if inst_buying and inst_note else ""
                    entry_path = "突破" if score >= 4 else "趨勢"
                    position = {
                        'entry_date': date_str, 'buy_price': close,
                        'shares': shares, 'trimmed': False,
                        'strategy': 'A',
                        'entry_mode': f'A-{entry_path}(ADX={adx_val:.0f}{inst_tag})',
                    }

    # 強制平倉
    if position is not None:
        last_close = float(df_full.iloc[-1]['Close'])
        last_date  = df_full.index[-1].strftime('%Y-%m-%d')
        trades.append({
            'entry_date': position['entry_date'], 'exit_date': last_date,
            'buy_price': position['buy_price'], 'exit_price': last_close,
            'shares': position['shares'],
            'pnl_pct': (last_close - position['buy_price']) / position['buy_price'] * 100,
            'reason': '回測結束強制平倉',
            'hold_days': (df_full.index[-1] - pd.Timestamp(position['entry_date'])).days,
            'strategy': position.get('strategy', '?'),
        })

    return trades


# ════════════════════════════════════════════════════════════
#  輸出
# ════════════════════════════════════════════════════════════
def print_result(trades, ticker, name, df_full):
    cutoff  = df_full.index[-1] - timedelta(days=LOOKBACK_DAYS)
    bh_buy  = float(df_full[df_full.index >= cutoff].iloc[0]['Close'])
    bh_sell = float(df_full.iloc[-1]['Close'])
    bh_pct  = (bh_sell - bh_buy) / bh_buy * 100
    start_d = df_full[df_full.index >= cutoff].index[0].strftime('%Y-%m-%d')
    end_d   = df_full.index[-1].strftime('%Y-%m-%d')

    print(f"\n{'='*76}")
    print(f"  {ticker}  {name}   {start_d} ～ {end_d}")
    print(f"  Buy & Hold：{bh_buy:.1f} → {bh_sell:.1f}  ({bh_pct:+.1f}%)")
    print(f"{'='*76}")

    if not trades:
        print("  （無觸發）")
        return None

    print(f"\n  {'進場日':<12} {'出場日':<12} {'買入':>7} {'出場':>7} {'損益%':>7}  {'持有':>5}  {'策略':<4}  出場原因")
    print(f"  {'-'*76}")
    for t in trades:
        mark = '✅' if t['pnl_pct'] > 0 else '❌'
        print(f"  {t['entry_date']:<12} {t['exit_date']:<12} "
              f"{t['buy_price']:>7.1f} {t['exit_price']:>7.1f} "
              f"{t['pnl_pct']:>+6.1f}%  {t['hold_days']:>4}天  "
              f"{t.get('strategy','?'):<4}  {mark} {t['reason']}")

    wins   = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    total  = len(trades)
    wr     = len(wins) / total * 100
    avg    = sum(t['pnl_pct'] for t in trades) / total
    avg_w  = sum(t['pnl_pct'] for t in wins)   / len(wins)   if wins   else 0
    avg_l  = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
    avg_h  = sum(t['hold_days'] for t in trades) / total
    rr     = abs(avg_w / avg_l) if avg_l else 0

    streak = max_streak = 0
    for t in trades:
        streak = streak + 1 if t['pnl_pct'] <= 0 else 0
        max_streak = max(max_streak, streak)

    # 分策略統計
    for s_label, s_name in [('A','起勢跟蹤'), ('C','低谷反彈')]:
        s_trades = [t for t in trades if t.get('strategy') == s_label]
        if not s_trades:
            continue
        s_wins = [t for t in s_trades if t['pnl_pct'] > 0]
        s_wr   = len(s_wins) / len(s_trades) * 100
        s_avg  = sum(t['pnl_pct'] for t in s_trades) / len(s_trades)
        print(f"\n  策略{s_label}（{s_name}）：{len(s_trades)}次  勝率{s_wr:.0f}%  均損益{s_avg:+.1f}%")

    print(f"\n  ── 整體績效 ──")
    print(f"  交易{total}次  勝率{len(wins)}/{total}({wr:.0f}%)  "
          f"均損益{avg:+.2f}%  獲利均{avg_w:+.1f}%  虧損均{avg_l:+.1f}%")
    print(f"  獲虧比{rr:.2f}  平均持有{avg_h:.0f}天  最大連敗{max_streak}次")
    print(f"  策略 vs Buy&Hold：{avg:+.1f}% vs {bh_pct:+.1f}%")

    return {
        'ticker': ticker, 'name': name,
        'total': total, 'wr': wr, 'avg': avg,
        'avg_w': avg_w, 'avg_l': avg_l, 'rr': rr,
        'max_streak': max_streak, 'avg_hold': avg_h, 'bh_pct': bh_pct,
    }


# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n回測開始（低谷C + 起勢A 雙策略），下載資料中…\n")
    summaries = []

    for ticker, name in TICKERS.items():
        df = fetch_data(ticker)
        if df is None:
            print(f"  ⚠ {ticker} 下載失敗")
            continue

        # 下載法人資料
        code = ticker.replace('.TW', '').replace('.TWO', '')
        print(f"  下載 {ticker} 法人資料…", end=' ', flush=True)
        inst = fetch_institutional(code)
        print(f"{'取得 ' + str(len(inst)) + ' 筆' if inst else '無資料（跳過法人條件）'}")

        trades = backtest_one(ticker, name, df, inst_data=inst)
        r = print_result(trades, ticker, name, df)
        if r:
            summaries.append(r)

    if summaries:
        print(f"\n\n{'='*76}")
        print("  總覽")
        print(f"{'='*76}")
        print(f"  {'股票':<14} {'次數':>4} {'勝率':>6} {'均損益':>8} {'獲虧比':>6} {'連敗':>4}  結論")
        print(f"  {'-'*76}")
        for s in summaries:
            if s['wr'] >= 60 and s['avg'] > 3:
                tag = "✅ 適合"
            elif s['avg'] > 0 and s['rr'] >= 1.5:
                tag = "🔸 可操作"
            elif s['avg'] <= 0:
                tag = "❌ 不適合"
            else:
                tag = "⚠ 謹慎"
            print(f"  {s['ticker']} {s['name']:<4} "
                  f"{s['total']:>4}  {s['wr']:>5.0f}%  "
                  f"{s['avg']:>+7.1f}%  {s['rr']:>5.2f}  {s['max_streak']:>3}次  {tag}")

    print("\n回測完成")
