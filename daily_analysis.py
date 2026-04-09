"""
每日盤後分析腳本
執行方式：python daily_analysis.py
"""

import sys
import requests
import urllib3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 台灣時區（Streamlit Cloud 伺服器在 UTC，需明確指定）
try:
    from zoneinfo import ZoneInfo
    TZ_TW = ZoneInfo("Asia/Taipei")
except ImportError:
    from datetime import timezone
    TZ_TW = timezone(timedelta(hours=8))

def now_tw():
    """回傳台灣當地時間（naive datetime，保持向下相容）"""
    return datetime.now(TZ_TW).replace(tzinfo=None)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# 從設定檔讀取股票清單（要修改持股請編輯 stocks.py）
from stocks import HOLDINGS, WATCHLIST, FINMIND_TOKEN, FUGLE_API_KEY


# ============================================================
#  基本面抓取（台灣官方來源）
# ============================================================

# 快取，避免同一次執行重複打 API
_twse_cache  = None
_tpex_cache  = None


def _load_twse():
    """一次載入全部上市股票的本益比/本淨比/殖利率"""
    global _twse_cache
    if _twse_cache is not None:
        return _twse_cache
    try:
        resp = requests.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_d",
            timeout=15, verify=False
        )
        _twse_cache = {item['Code']: item for item in resp.json()}
    except Exception:
        _twse_cache = {}
    return _twse_cache


def _load_tpex():
    """一次載入全部上櫃股票的本益比/本淨比/殖利率"""
    global _tpex_cache
    if _tpex_cache is not None:
        return _tpex_cache
    try:
        resp = requests.get(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis",
            timeout=15, verify=False
        )
        _tpex_cache = {item['SecuritiesCompanyCode']: item for item in resp.json()}
    except Exception:
        _tpex_cache = {}
    return _tpex_cache


def get_valuation(code):
    """從證交所/櫃買中心取得本益比、本淨比、殖利率"""
    def safe(v):
        try: return float(v) if v not in ('', '-', None) else None
        except: return None

    # 先查上市
    twse = _load_twse()
    if code in twse:
        d = twse[code]
        return {
            'pe':  safe(d.get('PEratio')),
            'pb':  safe(d.get('PBratio')),
            'div': safe(d.get('DividendYield')),
        }

    # 再查上櫃
    tpex = _load_tpex()
    if code in tpex:
        d = tpex[code]
        return {
            'pe':  safe(d.get('PriceEarningRatio')),
            'pb':  safe(d.get('BookValueRatio')),
            'div': safe(d.get('DividendYield')),
        }

    return None


def get_revenue_trend(code):
    """從 FinMind 取得近3個月月營收，YoY/MoM 自行計算
    注意：FinMind 的 date 是公布日期，revenue_month/year 才是實際營收月份
    """
    start = (now_tw() - timedelta(days=460)).strftime('%Y-%m-%d')
    params = {
        'dataset':    'TaiwanStockMonthRevenue',
        'data_id':    code,
        'start_date': start,
    }
    if FINMIND_TOKEN:
        params['token'] = FINMIND_TOKEN

    try:
        resp = requests.get(
            'https://api.finmindtrade.com/api/v4/data',
            params=params, timeout=15
        )
        body = resp.json()
        if not (body.get('status') == 200 and body.get('data')):
            return None

        rows = sorted(body['data'], key=lambda x: x['date'])

        # 用實際營收年月建立對照表
        rev_map = {}
        for r in rows:
            ry = r.get('revenue_year')
            rm = r.get('revenue_month')
            if ry and rm:
                key = f"{ry}-{int(rm):02d}"
                rev_map[key] = r.get('revenue', 0)

        recent = rows[-3:]
        lines  = []
        for i, r in enumerate(recent):
            rev  = r.get('revenue', 0)
            ry   = r.get('revenue_year')
            rm   = r.get('revenue_month')
            if not (ry and rm):
                continue

            label = f"{ry}-{int(rm):02d}"   # 實際營收月份
            rev_m = rev / 1_000_000

            # MoM：和上一筆比
            if i > 0:
                prev_rev = recent[i-1].get('revenue', 0)
                mom_s = f"MoM {(rev-prev_rev)/prev_rev*100:+.1f}%" if prev_rev else ''
            else:
                mom_s = ''

            # YoY：同月份去年
            prev_key = f"{ry-1}-{int(rm):02d}"
            prev_r   = rev_map.get(prev_key, 0)
            yoy_s    = f"YoY {(rev-prev_r)/prev_r*100:+.1f}%" if prev_r else ''

            lines.append(f"  {label}  {rev_m:,.0f}百萬  {yoy_s}  {mom_s}".rstrip())

        return lines
    except Exception:
        pass
    return None


_fugle_cache = {}   # Fugle 即時報價快取（同一次執行）


def get_fugle_quote(code):
    """從 Fugle 取得即時報價（含內外盤、五檔）
    回傳 dict 或 None（未設定 key 或 API 失敗）
    """
    global _fugle_cache
    if code in _fugle_cache:
        return _fugle_cache[code]
    if not FUGLE_API_KEY:
        return None
    try:
        resp = requests.get(
            f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{code}",
            headers={"X-API-KEY": FUGLE_API_KEY},
            timeout=6,
        )
        if resp.status_code != 200:
            _fugle_cache[code] = None
            return None
        d = resp.json()
        _fugle_cache[code] = d
        return d
    except Exception:
        _fugle_cache[code] = None
        return None


def parse_fugle_price(d):
    """從 Fugle quote dict 取出現價、漲跌、內外盤等常用欄位"""
    if not d:
        return None
    tot      = d.get("total", {})
    vol_bid  = tot.get("tradeVolumeAtBid", 0) or 0
    vol_ask  = tot.get("tradeVolumeAtAsk", 0) or 0
    total_v  = vol_bid + vol_ask
    ask_pct  = vol_ask / total_v * 100 if total_v else None   # 外盤比

    last_p  = d.get("lastPrice")  or 0   # 盤中即時成交價
    close_p = d.get("closePrice") or 0   # 官方收盤價（盤後才有當日值）

    return {
        "price":       last_p  or close_p or None,   # 盤中用：即時成交
        "close_price": close_p or last_p  or None,   # 盤後用：官方收盤
        "open":       d.get("openPrice"),
        "high":       d.get("highPrice"),
        "low":        d.get("lowPrice"),
        "avg":        d.get("avgPrice"),
        "change":     d.get("change"),
        "change_pct": d.get("changePercent"),
        "volume":     tot.get("tradeVolume"),
        "vol_bid":    vol_bid,
        "vol_ask":    vol_ask,
        "ask_pct":    ask_pct,   # 外盤比 %
        "bids":       d.get("bids", []),
        "asks":       d.get("asks", []),
    }


def format_orderbook(q):
    """印出五檔掛單（需要 q = parse_fugle_price 回傳值）"""
    asks = list(reversed(q["asks"][:5]))   # 賣單由高到低
    bids = q["bids"][:5]                   # 買單由高到低
    lines = []
    lines.append("  ┌─ 五檔掛單 ────────────────────┐")
    for a in asks:
        lines.append(f"  │  賣  {a['price']:>7.1f}  {a['size']:>6} 張         │")
    lines.append("  │  ─────────────────────────── │")
    for b in bids:
        lines.append(f"  │  買  {b['price']:>7.1f}  {b['size']:>6} 張         │")
    lines.append("  └───────────────────────────────┘")
    return lines


_inst_cache = {}   # 避免同一次執行重複打 API


def get_institutional(code):
    """從 FinMind 取得近5日三大法人買賣超
    只在有 🔴 停損警示 或 攤平訊號就緒 時呼叫
    回傳 (明細行列表, 趨勢摘要字串 or None) 或 None
    """
    global _inst_cache
    if code in _inst_cache:
        return _inst_cache[code]

    start = (now_tw() - timedelta(days=20)).strftime('%Y-%m-%d')
    params = {
        'dataset':    'TaiwanStockInstitutionalInvestors',
        'data_id':    code,
        'start_date': start,
    }
    if FINMIND_TOKEN:
        params['token'] = FINMIND_TOKEN

    try:
        resp = requests.get(
            'https://api.finmindtrade.com/api/v4/data',
            params=params, timeout=15
        )
        body = resp.json()
        if not (body.get('status') == 200 and body.get('data')):
            _inst_cache[code] = None
            return None

        # 依日期彙整三類法人
        from collections import defaultdict
        by_date = defaultdict(dict)
        for r in body['data']:
            date = r['date']
            name = r.get('name', '')
            net  = r.get('net_buy') or 0
            if '外資' in name and '自營' not in name:
                by_date[date]['外資'] = net
            elif name == '投信':
                by_date[date]['投信'] = net
            elif '自營商' in name and '避險' not in name:
                by_date[date]['自營'] = net

        sorted_dates = sorted(by_date.keys())[-5:]
        if not sorted_dates:
            _inst_cache[code] = None
            return None

        def fmt(v):
            """股數 → 張（1張=1000股），顯示帶正負號"""
            z = int(v) // 1000
            if z > 0:  return f"+{z:,}張"
            if z < 0:  return f"{z:,}張"
            return "持平"

        lines = []
        for date in sorted_dates:
            d       = by_date[date]
            foreign = d.get('外資', 0)
            trust   = d.get('投信', 0)
            dealer  = d.get('自營', 0)
            total   = foreign + trust + dealer
            lines.append(
                f"  {date}  外資 {fmt(foreign):>8}  投信 {fmt(trust):>8}"
                f"  自營 {fmt(dealer):>8}  合計 {fmt(total):>8}"
            )

        # 外資連續方向判斷
        foreign_vals = [by_date[d].get('外資', 0) for d in sorted_dates]
        consec_sell = 0
        consec_buy  = 0
        for v in reversed(foreign_vals):
            if v < 0: consec_sell += 1
            else:     break
        for v in reversed(foreign_vals):
            if v > 0: consec_buy += 1
            else:     break

        if consec_sell >= 3:
            trend = f"  ⚠  外資連續 {consec_sell} 日賣超，出場訊號加強"
        elif consec_buy >= 3:
            trend = f"  💡 外資連續 {consec_buy} 日買超，籌碼偏多"
        else:
            trend = None

        result = (lines, trend)
        _inst_cache[code] = result
        return result

    except Exception:
        _inst_cache[code] = None
        return None


def _print_institutional(code):
    """顯示三大法人區塊（共用輸出邏輯）"""
    inst = get_institutional(code)
    if inst:
        lines, trend = inst
        print("  ── 三大法人近5日買賣超 ──")
        for l in lines:
            print(l)
        if trend:
            print(trend)
    else:
        print("  ── 三大法人：無法取得（速率限制，稍後再試）")


def get_fundamentals(ticker):
    """組合基本面資料，回傳格式化字串列表"""
    code  = ticker.replace('.TW', '')
    lines = []

    # 估值（TWSE / TPEx）
    val = get_valuation(code)
    if val:
        pe_s  = f"{val['pe']:.1f}x"  if val.get('pe')  else 'N/A'
        pb_s  = f"{val['pb']:.2f}x"  if val.get('pb')  else 'N/A'
        div_s = f"{val['div']:.2f}%" if val.get('div') else 'N/A'
        lines.append(f"  本益比 {pe_s}　本淨比 {pb_s}　殖利率 {div_s}")

        # 快速估值判斷
        pe = val.get('pe')
        if pe and pe > 0:
            if pe < 12:
                lines.append("  估值：偏低（本益比 < 12）")
            elif pe > 30:
                lines.append(f"  估值：偏高（本益比 {pe:.0f}x，需靠高成長支撐）")

    # 月營收趨勢（FinMind）
    rev = get_revenue_trend(code)
    if rev:
        lines.append("  近期月營收：")
        lines.extend(rev)
    elif code != '0050':
        lines.append("  月營收：無法取得（FinMind 速率限制，稍後再試）")

    return lines if lines else None


# ============================================================
#  指標計算
# ============================================================

def calculate_indicators(df):
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    df['MA5']  = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()

    # RSI(14)
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    # ATR(14)
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low  - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # 成交量比（今日量 / 5日均量）
    df['Vol_MA5']   = vol.rolling(5).mean()
    df['Vol_ratio'] = vol / df['Vol_MA5']

    # 近20日最低點
    df['Low_20'] = low.rolling(20).min()

    return df


def fetch(ticker, silent=False):
    """下載股價資料並計算技術指標。
    silent=True：抑制 yfinance 的 404/警告輸出（探測 .TW / .TWO 時用）
    """
    import io, contextlib
    try:
        if silent:
            # 把 yfinance 的 stdout/stderr 雜訊全部吞掉
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                df = yf.Ticker(ticker).history(period="60d")
        else:
            df = yf.Ticker(ticker).history(period="60d")

        if df.empty:
            return None
        # 移除尚未收盤的空行（開盤前或休市日可能出現 NaN）
        df = df.dropna(subset=['Close'])
        if df.empty:
            return None
        return calculate_indicators(df)
    except Exception as e:
        if not silent:
            print(f"  ⚠  無法抓取 {ticker}：{e}")
        return None


# ============================================================
#  訊號判斷
# ============================================================

def exit_signals(df, buy_price):
    """持倉警示"""
    r    = df.iloc[-1]
    prev = df.iloc[-2]
    msgs = []

    close        = r['Close']
    atr          = r['ATR']
    daily_change = (close - prev['Close']) / prev['Close'] * 100
    profit_pct   = (close - buy_price) / buy_price * 100

    # ATR 動態停損
    atr_stop = buy_price - 2 * atr
    if close < atr_stop:
        msgs.append(f"  🔴 跌破 ATR 停損線 {atr_stop:.1f}  → 建議出場")

    # 單日跌幅 > 4%
    if daily_change < -4:
        msgs.append(f"  🔴 單日跌幅 {daily_change:.1f}%  → 注意跳空")

    # 跌破 MA10
    if close < r['MA10']:
        msgs.append(f"  🟠 收盤跌破 MA10({r['MA10']:.1f})  → 注意風險")

    # 跌破 MA5，看 MACD
    if close < r['MA5']:
        if r['MACD_hist'] < 0:
            msgs.append(f"  🟠 跌破 MA5 且 MACD 翻負  → 考慮部分出場")
        else:
            msgs.append(f"  🟡 跌破 MA5 但 MACD 仍正  → 觀望")

    # ── 移動停利（乖離率 + RSI 雙重確認）──
    deviation = (close - r['MA5']) / r['MA5'] * 100   # 現價偏離 MA5 %
    rsi       = r['RSI']

    if deviation > 12 and rsi > 78:
        msgs.append(
            f"  🟢 乖離率 {deviation:.1f}% 且 RSI {rsi:.1f}  → 強力停利訊號，建議減碼 30% 或考慮出清"
        )
    elif deviation > 8 and rsi > 70:
        msgs.append(
            f"  🟢 乖離率 {deviation:.1f}% 且 RSI {rsi:.1f}  → 短線過熱，建議先減碼 30%"
        )
    elif deviation > 5 and rsi > 65 and profit_pct > 0:
        msgs.append(
            f"  🟡 乖離率 {deviation:.1f}% 且 RSI {rsi:.1f}  → 偏熱，留意是否需要減碼"
        )

    return msgs


def avg_down_signals(df):
    """
    攤平時機判斷（四個條件）
    適用於已深套、打算低接的標的
    回傳 (達成條數, 條件說明, 是否建議攤平)
    """
    r    = df.iloc[-1]
    prev = df.iloc[-2]
    score, msgs = 0, []

    close = r['Close']

    # 條件1：RSI 曾超賣（近10日曾低於30）且現在回升中
    rsi_recent_low = df['RSI'].iloc[-10:].min()
    rsi_rising     = r['RSI'] > prev['RSI']
    if rsi_recent_low < 35 and r['RSI'] > 35 and rsi_rising:
        score += 1
        msgs.append(f"  ✓ RSI 從超賣區回升（現 {r['RSI']:.1f}，近期低點 {rsi_recent_low:.1f}）")
    else:
        if rsi_recent_low >= 35:
            msgs.append(f"  ✗ RSI 尚未進入超賣區（近期低點 {rsi_recent_low:.1f}）")
        elif r['RSI'] <= 35:
            msgs.append(f"  ✗ RSI 仍在超賣區（{r['RSI']:.1f}），尚未反彈")
        else:
            msgs.append(f"  ✗ RSI 回升力道不足（{r['RSI']:.1f}）")

    # 條件2：MA5 斜率由負轉正（今日 MA5 > 昨日 MA5）
    ma5_turning_up = r['MA5'] > prev['MA5']
    if ma5_turning_up:
        score += 1
        msgs.append(f"  ✓ MA5 斜率翻正（短線止跌跡象）")
    else:
        msgs.append(f"  ✗ MA5 仍在下彎（趨勢未止跌）")

    # 條件3：今日收紅且成交量 > 5日均量 x 1.2（有量的反彈）
    price_up      = close > prev['Close']
    volume_enough = r['Vol_ratio'] >= 1.2
    if price_up and volume_enough:
        score += 1
        msgs.append(f"  ✓ 收紅且量比 {r['Vol_ratio']:.2f}（有量反彈）")
    else:
        if not price_up:
            msgs.append(f"  ✗ 今日收黑（尚未出現反彈K棒）")
        else:
            msgs.append(f"  ✗ 收紅但量比 {r['Vol_ratio']:.2f}（量能不足，可能假反彈）")

    # 條件4：現價距近20日低點反彈 > 3%（確認底部支撐）
    low_20       = r['Low_20']
    rebound_pct  = (close - low_20) / low_20 * 100
    if rebound_pct >= 3:
        score += 1
        msgs.append(f"  ✓ 距近20日低點反彈 {rebound_pct:.1f}%（底部有支撐）")
    else:
        msgs.append(f"  ✗ 距近20日低點僅 {rebound_pct:.1f}%（尚未確認底部）")

    ready = score >= 3
    return score, msgs, ready


def building_signals(df):
    """建倉加碼訊號（方案B）：回調加碼，不追高，量不能太縮
    4個條件全達成才建議加碼
    回傳 (達成條數, 訊息列表, 是否建議加碼)
    """
    r    = df.iloc[-1]
    score, msgs = 0, []

    # 條件1：MA5 > MA10（多頭排列，方向對）
    if r['MA5'] > r['MA10']:
        score += 1
        msgs.append(f"  ✓ MA5({r['MA5']:.1f}) > MA10({r['MA10']:.1f})  多頭排列")
    else:
        msgs.append(f"  ✗ MA5 < MA10，趨勢尚未轉多")

    # 條件2：RSI 在 45~65（有動能但不追高，比原版稍嚴格）
    rsi = r['RSI']
    if 45 <= rsi <= 65:
        score += 1
        msgs.append(f"  ✓ RSI {rsi:.1f}（加碼甜蜜點 45~65）")
    elif rsi > 65:
        msgs.append(f"  ✗ RSI {rsi:.1f}，過熱，等回檔再加")
    else:
        msgs.append(f"  ✗ RSI {rsi:.1f}，動能不足（需 > 45）")

    # 條件3：量比 > 0.8（至少不是極度量縮）
    vol_ratio = r['Vol_ratio']
    if vol_ratio >= 0.8:
        score += 1
        msgs.append(f"  ✓ 量比 {vol_ratio:.2f}（成交正常）")
    else:
        msgs.append(f"  ✗ 量比 {vol_ratio:.2f}，量能過度萎縮")

    # 條件4：現價不超過 MA5 的 5%（回調加碼，不追高）
    close     = r['Close']
    ma5       = r['MA5']
    above_ma5 = (close - ma5) / ma5 * 100
    if above_ma5 <= 5:
        score += 1
        msgs.append(f"  ✓ 現價距 MA5 {above_ma5:+.1f}%，位置合理")
    else:
        msgs.append(f"  ✗ 現價已高於 MA5 {above_ma5:.1f}%，追高風險高")

    return score, msgs, score >= 4


def entry_signals(df):
    """新標的進場過濾器"""
    r = df.iloc[-1]
    score, msgs = 0, []

    if r['MA5'] > r['MA10']:
        score += 1
        msgs.append(f"  ✓ MA5({r['MA5']:.1f}) > MA10({r['MA10']:.1f})  多頭排列")
    else:
        msgs.append(f"  ✗ MA5({r['MA5']:.1f}) < MA10({r['MA10']:.1f})  空頭排列")

    if r['RSI'] > 50:
        score += 1
        msgs.append(f"  ✓ RSI = {r['RSI']:.1f}  動能向上")
    else:
        msgs.append(f"  ✗ RSI = {r['RSI']:.1f}  動能不足")

    if r['Vol_ratio'] >= 1.2:
        score += 1
        msgs.append(f"  ✓ 量比 = {r['Vol_ratio']:.2f}  有量配合")
    else:
        msgs.append(f"  ✗ 量比 = {r['Vol_ratio']:.2f}  量能不足")

    return score, msgs


# ============================================================
#  報告輸出
# ============================================================

def divider():
    print("─" * 55)


def market_status():
    """判斷目前是盤中、盤後還是盤前"""
    now = now_tw()
    weekday = now.weekday()   # 0=週一 … 4=週五
    h, m = now.hour, now.minute
    minutes = h * 60 + m

    if weekday >= 5:
        return "休市", "（週末）"
    if minutes < 9 * 60:
        return "盤前", "（台股尚未開盤，顯示昨日收盤資料）"
    if minutes <= 13 * 60 + 30:
        remaining = (13 * 60 + 30) - minutes
        return "盤中", f"（台股交易中，距收盤約 {remaining} 分鐘，成交量為累計值）"
    return "盤後", "（台股已收盤，資料為今日最終收盤價）"


def run():
    _fugle_cache.clear()   # 每次執行都清除，確保 Fugle 價格是最新的
    now    = now_tw()
    status, status_note = market_status()

    print()
    print("=" * 55)
    if status == "盤中":
        print("   盤中即時檢查")
    else:
        print("   每日盤後分析報告")
    print(f"   {now.strftime('%Y-%m-%d  %H:%M')}  {status_note}")
    print("=" * 55)

    # 整理完整 ticker 清單（持倉 + 觀察名單，不重複）
    seen = set()
    all_tickers = []

    for ticker in HOLDINGS:
        if ticker not in seen:
            label = "台股" if ticker.endswith(".TW") else "美股"
            all_tickers.append((ticker, label))
            seen.add(ticker)

    for code in WATCHLIST.get("tw", []):
        # 先試上市(.TW)，抓不到再試上櫃(.TWO)，silent=True 避免印出 404 雜訊
        ticker = code + ".TW"
        if fetch(ticker, silent=True) is None:
            ticker = code + ".TWO"
        if ticker not in seen:
            all_tickers.append((ticker, "台股"))
            seen.add(ticker)
    for code in WATCHLIST.get("us", []):
        if code not in seen:
            all_tickers.append((code, "美股"))
            seen.add(code)

    # 預先抓好所有資料（避免重複下載）
    data_cache  = {}
    fund_cache  = {}
    for ticker, _ in all_tickers:
        data_cache[ticker] = fetch(ticker)
        if ticker in HOLDINGS:
            fund_cache[ticker] = get_fundamentals(ticker)

    # ── Fugle 最新收盤價（修正 yfinance 更新延遲）──────────
    # yfinance 盤後有時幾小時才更新，Fugle 收盤後仍回傳當日最終價
    # 直接把 df 的最後一行 Close 改成 Fugle 價，讓所有下游函式自動用正確價格
    fugle_price_cache = {}
    if FUGLE_API_KEY:
        for ticker, _ in all_tickers:
            if not ticker.endswith((".TW", ".TWO")):
                continue
            code = ticker.replace(".TWO", "").replace(".TW", "")
            fq   = parse_fugle_price(get_fugle_quote(code))
            if fq and fq.get("price"):
                fugle_price_cache[ticker] = fq["price"]
                df = data_cache.get(ticker)
                if df is not None:
                    # 更新 df 最後一行的 Close，讓訊號函式全部用最新價
                    df.iloc[-1, df.columns.get_loc('Close')] = fq["price"]

    status, _ = market_status()
    is_intraday = (status == "盤中")

    # 摘要收集桶
    summary_urgent   = []   # 🔴 需要立即處理（停損）
    summary_profit   = []   # 🟢 停利訊號
    summary_watch    = []   # 🟠 明天留意
    summary_avgdown  = []   # 攤平訊號
    summary_building = []   # 建倉加碼訊號
    summary_ok       = []   # 正常續抱
    summary_entry    = []   # 新進場機會

    # ── 持倉警示 ──────────────────────────────────────
    print("\n▌ 持倉警示\n")
    for ticker, label in all_tickers:
        if ticker not in HOLDINGS:
            continue
        df = data_cache.get(ticker)
        if df is None:
            continue

        h         = HOLDINGS[ticker]
        yf_close  = df.iloc[-1]['Close']
        # 優先用 Fugle 當日收盤價，修正 yfinance 更新延遲
        close     = fugle_price_cache.get(ticker, yf_close)
        price_src = "Fugle" if ticker in fugle_price_cache else "yfinance"
        buy_price  = h['buy_price']
        profit_pct = (close - buy_price) / buy_price * 100
        name    = h.get("name", "")
        tags    = []
        if h.get("avg_down"):  tags.append("攤平候選")
        if h.get("building"):  tags.append("建倉中")
        tag_str = "  【" + "｜".join(tags) + "】" if tags else ""

        print(f"  {ticker} {name}（{label}）{tag_str}")
        stale_note = f"  ⚠ yfinance 尚未更新（昨收 {yf_close:.1f}）" if close != yf_close else ""
        print(f"  現價 {close:.1f}　買入 {buy_price:.1f}　損益 {profit_pct:+.1f}%{stale_note}")

        msgs = exit_signals(df, buy_price)
        if msgs:
            for m in msgs:
                print(m)
            has_red   = any('🔴' in m for m in msgs)
            has_green = any('🟢' in m for m in msgs)
            if has_red:
                atr_stop = buy_price - 2 * df.iloc[-1]['ATR']
                summary_urgent.append(
                    f"{ticker} {name}（{profit_pct:+.1f}%）現價 {close:.1f} → 停損參考 {atr_stop:.1f}"
                )
                if ticker.endswith('.TW'):   # 三大法人資料僅台股適用
                    _print_institutional(ticker.replace('.TW', ''))
            elif has_green:
                green_msg  = next(m for m in msgs if '🟢' in m)
                trim_shares = int(h['shares'] * 0.3)
                summary_profit.append(
                    f"{ticker} {name}（{profit_pct:+.1f}%）現價 {close:.1f} → 減碼 ~{trim_shares} 股"
                )
            else:
                summary_watch.append(f"{ticker} {name}（{profit_pct:+.1f}%）")
        else:
            print("  ✅ 持倉正常，續抱")
            summary_ok.append(f"{ticker} {name}（{profit_pct:+.1f}%）")

        fund = fund_cache.get(ticker)
        if fund:
            print("  ── 基本面（近12個月滾動，僅供參考）──")
            for f in fund:
                print(f)
        print()

    divider()

    # ── 攤平時機 ──────────────────────────────────────
    avg_down_list = [(t, l) for t, l in all_tickers
                     if HOLDINGS.get(t, {}).get("avg_down")]

    print("\n▌ 攤平時機偵測\n")
    if not avg_down_list:
        print("  （無標示攤平候選的持股）\n")
    else:
        for ticker, label in avg_down_list:
            df = data_cache.get(ticker)
            if df is None:
                continue

            score, msgs, ready = avg_down_signals(df)
            close      = df.iloc[-1]['Close']
            buy_price  = HOLDINGS[ticker]['buy_price']
            profit_pct = (close - buy_price) / buy_price * 100

            name_h = HOLDINGS[ticker].get('name', ticker)
            if ready:
                header = f"  🟢 {ticker}（{label}）  現價 {close:.1f}  損益 {profit_pct:+.1f}%  ← 攤平訊號確認！({score}/4)"
                summary_avgdown.append(
                    f"{ticker} {name_h}（{score}/4）→ 可在 {close:.1f} 附近掛買單攤平"
                )
            else:
                header = f"  ⏳ {ticker}（{label}）  現價 {close:.1f}  損益 {profit_pct:+.1f}%  尚未就緒 ({score}/4)"
                summary_avgdown.append(f"{ticker} {name_h}（{score}/4）繼續等待")

            print(header)
            for m in msgs:
                print(m)
            if ready and ticker.endswith('.TW'):
                _print_institutional(ticker.replace('.TW', ''))
            print()

    divider()

    # ── 建倉加碼機會 ──────────────────────────────────────
    building_list = [(t, l) for t, l in all_tickers
                     if HOLDINGS.get(t, {}).get("building")]

    print("\n▌ 建倉加碼機會\n")
    if not building_list:
        print("  （無標示建倉中的持股）\n")
    else:
        for ticker, label in building_list:
            df = data_cache.get(ticker)
            if df is None:
                continue

            score, msgs, ready = building_signals(df)
            close     = df.iloc[-1]['Close']
            buy_price = HOLDINGS[ticker]['buy_price']
            profit_pct = (close - buy_price) / buy_price * 100

            name_b   = HOLDINGS[ticker].get('name', ticker)
            ma5_b    = df.iloc[-1]['MA5']
            shares_b = HOLDINGS[ticker].get('shares', 0)
            half_lot = max(1, int(shares_b * 0.15))  # 少量加碼 ≈ 現有倉位 15%

            # 計算加碼建議價：現價已近 MA5 → 現價可進；現價偏高 → 等回調
            above_pct = (close - ma5_b) / ma5_b * 100
            if above_pct <= 1.5:
                price_action = f"現價 {close:.1f} 已在 MA5 附近，可直接買進"
            else:
                price_action = f"等回調至 MA5 {ma5_b:.1f} 附近再買（現價 {close:.1f} 偏高 {above_pct:.1f}%）"

            if ready:                      # 4/4 正常加碼
                header = (f"  ✅ {ticker}（{label}）  現價 {close:.1f}"
                          f"  損益 {profit_pct:+.1f}%  ← 加碼時機！({score}/4)")
                summary_building.append(
                    f"{ticker} {name_b}（{profit_pct:+.1f}%）→ {price_action}"
                )
            elif score == 3:               # 3/4 少量試單
                header = (f"  🔸 {ticker}（{label}）  現價 {close:.1f}"
                          f"  損益 {profit_pct:+.1f}%  ← 少量加碼機會（{score}/4）")
                summary_building.append(
                    f"{ticker} {name_b}（{profit_pct:+.1f}%）→ 少量試單約 {half_lot} 股，{price_action}"
                )
            else:                          # < 3
                vol_ratio_b = df.iloc[-1]['Vol_ratio']
                probe_lot   = max(1, int(shares_b * 0.10))   # 量能突破試單 ≈ 10%
                if score == 2 and vol_ratio_b >= 1.5:        # 2/4 + 量能突破
                    header = (f"  💡 {ticker}（{label}）  現價 {close:.1f}"
                              f"  損益 {profit_pct:+.1f}%"
                              f"  ← 量能突破可試單（{score}/4，量比 {vol_ratio_b:.2f}）")
                    summary_building.append(
                        f"{ticker} {name_b}（{profit_pct:+.1f}%）"
                        f"→ 量能突破試單約 {probe_lot} 股，{price_action}"
                    )
                else:                                         # 等待
                    header = (f"  ⏳ {ticker}（{label}）  現價 {close:.1f}"
                              f"  損益 {profit_pct:+.1f}%  尚未就緒（{score}/4）")
                    summary_building.append(
                        f"{ticker} {name_b}（{profit_pct:+.1f}%）尚未就緒（{score}/4）"
                    )

            print(header)
            for m in msgs:
                print(m)
            print()

    divider()

    # ── 新標的進場機會 ──────────────────────────────────
    print("\n▌ 明日進場機會（觀察名單，三項條件全達成）\n")
    candidates  = []   # 3/3 全達成
    vol_breaks  = []   # 2/3 + 量比≥1.5 量能突破
    watching    = []   # 2/3 一般觀察 / 漲跌停

    for ticker, label in all_tickers:
        if ticker in HOLDINGS:
            continue
        df = data_cache.get(ticker)
        if df is None:
            continue

        score, msgs = entry_signals(df)
        close     = fugle_price_cache.get(ticker, df.iloc[-1]['Close'])
        prev_c    = df.iloc[-2]['Close']
        day_chg   = (close - prev_c) / prev_c * 100

        # 漲停過濾：台股漲幅接近 ±10%，漲停不追、跌停不抄
        if day_chg >= 9.5:
            watching.append((ticker, label, msgs, close, "漲停", 0))
            continue
        if day_chg <= -9.5:
            watching.append((ticker, label, msgs, close, "跌停", 0))
            continue

        if score == 3:
            candidates.append((ticker, label, msgs, close, f"{day_chg:+.1f}%"))
        elif score == 2:
            vol_r = df.iloc[-1]['Vol_ratio']
            if vol_r >= 1.5:
                vol_breaks.append((ticker, label, msgs, close, f"{day_chg:+.1f}%", vol_r))
            else:
                watching.append((ticker, label, msgs, close, f"{day_chg:+.1f}%", vol_r))

    if candidates:
        for ticker, label, msgs, close, chg in candidates:
            print(f"  ⭐ {ticker}（{label}）　現價 {close:.1f}（{chg}）")
            for m in msgs:
                print(m)
            print()
            summary_entry.append(f"{ticker} 三項條件全達成，可考慮進場")
    else:
        print("  今日無符合全部三項條件的標的\n")

    if vol_breaks:
        print("  💡 量能突破機會（2/3 條件 + 量比 ≥ 1.5，可小量試單）：")
        for ticker, label, msgs, close, chg, vol_r in vol_breaks:
            print(f"  💡 {ticker}（{label}）　現價 {close:.1f}（{chg}）量比 {vol_r:.2f}")
            for m in msgs:
                print(m)
            print()
            summary_entry.append(f"{ticker} 量能突破試單機會（2/3 + 量比 {vol_r:.2f}）")

    if watching:
        print("  距進場不遠（2/3 條件）或今日特殊狀況：")
        for item in watching:
            ticker, label, msgs, close, note, *_ = item
            if note == "漲停":
                print(f"  📈 {ticker}（{label}）　現價 {close:.1f}  漲停中，明日開盤再評估")
            elif note == "跌停":
                print(f"  📉 {ticker}（{label}）　現價 {close:.1f}  跌停中，等穩定後再觀察")
            else:
                print(f"  🔍 {ticker}（{label}）　現價 {close:.1f}（{note}）")
                for m in msgs:
                    print(m)
            print()

    # ── 明日重點摘要 ──────────────────────────────────────
    divider()
    print("\n▌ 明日開盤操作清單\n")

    if summary_urgent:
        print("  【需要處理】")
        for s in summary_urgent:
            print(f"  🔴 {s}")
        print()

    if summary_profit:
        print("  【停利訊號】")
        for s in summary_profit:
            print(f"  🟢 {s}")
        print()

    if summary_watch:
        print("  【留意觀察】")
        for s in summary_watch:
            print(f"  🟠 {s}")
        print()

    if summary_avgdown:
        print("  【攤平候選】")
        for s in summary_avgdown:
            ready_mark = "🟢" if "確認" in s else "⏳"
            print(f"  {ready_mark} {s}")
        print()

    if summary_building:
        print("  【建倉加碼】")
        for s in summary_building:
            if "正常量" in s:   mark = "✅"
            elif "少量試單" in s: mark = "🔸"
            else:               mark = "⏳"
            print(f"  {mark} {s}")
        print()

    if summary_entry:
        print("  【新進場機會】")
        for s in summary_entry:
            print(f"  ⭐ {s}")
        print()

    if summary_ok:
        print("  【正常續抱】")
        print(f"  ✅ {' / '.join(summary_ok)}")
        print()

    # 盤中提示
    if is_intraday:
        now = now_tw()
        remaining = (13 * 60 + 30) - (now.hour * 60 + now.minute)
        print(f"  ⏰ 盤中模式：距收盤約 {remaining} 分鐘")
        print(f"     技術指標以昨日收盤為基準，量比為累計值僅供參考")
        print(f"     建議：收盤前 15 分鐘再做最後決策")
        print()

    # 月營收提醒
    today = now_tw()
    if today.day <= 10:
        print(f"  📅 本月 10 日前為月營收公布期，留意各持股最新數字")
        print()

    print("=" * 55)
    print("  報告結束")
    print("=" * 55)
    print()


def quick_lookup(raw_code):
    """單支股票快速查詢（盤中 / 盤後皆可用）
    用法：python daily_analysis.py 2330
          python daily_analysis.py NVDA
    """
    # 自動補後綴：台股先試 .TW，找不到再試 .TWO（上櫃股）
    if raw_code.isdigit():
        label  = "台股"
        ticker = raw_code + ".TW"
        if fetch(ticker) is None:
            ticker = raw_code + ".TWO"
    else:
        ticker = raw_code.upper()
        label  = "美股"

    # Fugle cache 盤中/盤後欄位含意不同，每次查詢都清除確保拿最新資料
    code_to_clear = ticker.replace(".TWO", "").replace(".TW", "") if raw_code.isdigit() else raw_code.upper()
    _fugle_cache.pop(code_to_clear, None)

    now = now_tw()
    status, status_note = market_status()

    print()
    print("=" * 55)
    print(f"   快速查詢  {ticker}")
    print(f"   {now.strftime('%Y-%m-%d  %H:%M')}  {status_note}")
    print("=" * 55)

    df = fetch(ticker)
    if df is None:
        print(f"\n  ⚠  無法取得 {ticker} 資料，請確認代號是否正確\n")
        return

    r    = df.iloc[-1]
    prev = df.iloc[-2]

    # 盤中即時報價：優先 Fugle（真正即時），備援 Yahoo chart API
    price_note = ""
    live_price = None
    live_vol   = None
    if status == "盤中":
        # ── 優先：Fugle Market Data API（需填 FUGLE_API_KEY，真正即時）──
        if FUGLE_API_KEY and ticker.endswith((".TW", ".TWO")):
            code = ticker.replace(".TWO", "").replace(".TW", "")
            try:
                resp = requests.get(
                    f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{code}",
                    headers={"X-API-KEY": FUGLE_API_KEY},
                    timeout=6,
                )
                d = resp.json()
                lp = d.get("closePrice") or d.get("lastPrice") or d.get("price")
                lv = d.get("volume") or d.get("tradeVolume")
                if lp and float(lp) > 0:
                    live_price = float(lp)
                    price_note = "（Fugle 即時，延遲 < 3 秒）"
                if lv:
                    live_vol = float(lv)
            except Exception:
                pass

        # ── 備援：Yahoo Finance chart API（延遲約 1~3 分鐘）──
        if live_price is None:
            try:
                yf_sym = ticker if not ticker.isdigit() else ticker + ".TW"
                url    = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
                          f"{yf_sym}?interval=1m&range=1d")
                resp   = requests.get(url, timeout=8,
                                      headers={"User-Agent": "Mozilla/5.0"})
                meta   = resp.json()["chart"]["result"][0]["meta"]
                lp     = meta.get("regularMarketPrice")
                lv     = meta.get("regularMarketVolume")
                if lp and float(lp) > 0:
                    live_price = float(lp)
                    price_note = "（Yahoo chart API，延遲約 1~3 分鐘）"
                if lv:
                    live_vol = float(lv)
            except Exception:
                pass

        if live_price is None:
            price_note = "（Yahoo Finance history，延遲約 15~20 分鐘）"
    else:
        price_note = "（收盤價）"

    close        = live_price if live_price else r['Close']
    prev_close   = prev['Close']          # 昨日收盤，永遠準確
    daily_chg    = (close - prev_close) / prev_close * 100
    ma5          = r['MA5']
    ma10         = r['MA10']
    rsi          = r['RSI']
    macd_hist    = r['MACD_hist']
    # 量比：盤中如果有即時成交量就用即時的，否則用 yfinance 的
    vol_ma5      = r['Vol_MA5']
    if live_vol and vol_ma5 and vol_ma5 > 0:
        # Fugle/Yahoo volume 和 yfinance Vol_MA5 同樣是股數單位，直接相除
        vol_ratio = live_vol / vol_ma5
    else:
        vol_ratio = r['Vol_ratio']
    atr          = r['ATR']
    deviation    = (close - ma5) / ma5 * 100

    # ── Fugle 報價（盤中用 price=即時成交 / 盤後用 close_price=官方收盤）──
    fugle_q = None
    if FUGLE_API_KEY and ticker.endswith((".TW", ".TWO")):
        code_f  = ticker.replace(".TWO", "").replace(".TW", "")
        fugle_d = get_fugle_quote(code_f)
        fugle_q = parse_fugle_price(fugle_d)
        if fugle_q:
            # 盤中：用 price（lastPrice=即時）；盤後：用 close_price（closePrice=官方收盤）
            fugle_price = (fugle_q["price"] if status == "盤中"
                           else fugle_q.get("close_price") or fugle_q["price"])
            if fugle_price:
                close     = fugle_price
                daily_chg = (close - prev_close) / prev_close * 100
                deviation = (close - ma5) / ma5 * 100
                if fugle_q["volume"] and vol_ma5 and vol_ma5 > 0:
                    # Fugle volume 單位是張，vol_ma5 單位是股（1張=1000股），需乘以1000換算
                    vol_ratio = (fugle_q["volume"] * 1000) / vol_ma5
                price_note = "（Fugle 即時，延遲 < 3 秒）" if status == "盤中" else "（Fugle 收盤價）"

    # ── 基本數據 ──
    print(f"\n  現價  {close:.2f}  （{daily_chg:+.2f}%）  {price_note}")
    if fugle_q:
        avg = fugle_q.get("avg")
        avg_note = f"  均價 {avg:.2f}  {'現價在均價之上 ↑' if close >= avg else '現價在均價之下 ↓'}" if avg else ""
        print(f"  開 {fugle_q['open']:.1f}  高 {fugle_q['high']:.1f}  低 {fugle_q['low']:.1f}{avg_note}")
    print(f"  MA5   {ma5:.2f}  MA10  {ma10:.2f}")
    print(f"  RSI   {rsi:.1f}  MACD柱  {macd_hist:+.3f}  量比  {vol_ratio:.2f}")
    print(f"  ATR   {atr:.2f}  乖離率  {deviation:+.1f}%")

    # ── 內外盤（Fugle 盤中才有） ──
    if fugle_q and fugle_q.get("ask_pct") is not None:
        ask_pct = fugle_q["ask_pct"]
        bid_pct = 100 - ask_pct
        if ask_pct >= 60:
            ob_note = "買方主導 ↑  偏多"
        elif ask_pct <= 40:
            ob_note = "賣方主導 ↓  偏空"
        else:
            ob_note = "買賣平衡"
        print(f"\n  內外盤：外盤 {ask_pct:.1f}%  內盤 {bid_pct:.1f}%  →  {ob_note}")
        print(f"  （外盤=主動買，內盤=主動賣）")

    # ── 趨勢判斷 ──
    print()
    if ma5 > ma10:
        print(f"  📈 多頭排列（MA5 > MA10）")
    else:
        print(f"  📉 空頭排列（MA5 < MA10）")

    if close > ma5:
        print(f"  價格在 MA5 之上（+{deviation:.1f}%）")
    else:
        above = "仍正" if macd_hist > 0 else "已翻負"
        print(f"  價格在 MA5 之下（{deviation:.1f}%），MACD {above}")

    # ── RSI 解讀 ──
    if rsi > 70:
        print(f"  🔥 RSI {rsi:.1f}  超買區，留意回檔")
    elif rsi < 35:
        print(f"  🧊 RSI {rsi:.1f}  超賣區，留意反彈")
    else:
        print(f"  RSI {rsi:.1f}  正常區間")

    # ── 若在持倉中：顯示停損/停利/加碼訊號 ──
    holding = HOLDINGS.get(ticker)
    if holding:
        buy_price  = holding['buy_price']
        shares     = holding['shares']
        profit_pct = (close - buy_price) / buy_price * 100
        atr_stop   = buy_price - 2 * atr
        print()
        print(f"  ── 持倉資訊 ──")
        print(f"  買入 {buy_price:.2f}  持有 {shares} 股  損益 {profit_pct:+.1f}%")
        print(f"  ATR 停損線  {atr_stop:.2f}  {'⚠ 已跌破！' if close < atr_stop else '（未觸發）'}")

        # ── 出場/停利訊號（完整規則） ──
        exit_msgs = exit_signals(df, buy_price)
        print()
        print(f"  ── 出場訊號檢查 ──")
        # 移動停利規則說明
        print(f"  停利規則：乖離率 {deviation:+.1f}%  RSI {rsi:.1f}")
        print(f"    強力出清：乖離 >12% 且 RSI >78   → {'✅ 觸發' if deviation>12 and rsi>78 else '❌ 未到'}")
        print(f"    減碼30%：乖離 >8%  且 RSI >70   → {'✅ 觸發' if deviation>8 and rsi>70 else '❌ 未到'}")
        print(f"    留意偏熱：乖離 >5%  且 RSI >65   → {'✅ 觸發' if deviation>5 and rsi>65 and profit_pct>0 else '❌ 未到'}")
        if exit_msgs:
            print()
            for m in exit_msgs:
                print(m)
            # 如有停利訊號，顯示建議賣出股數
            has_strong = any('強力' in m or '出清' in m for m in exit_msgs)
            has_trim   = any('減碼 30%' in m for m in exit_msgs)
            has_watch  = any('偏熱' in m for m in exit_msgs)
            if has_strong:
                print(f"  → 建議賣出 {shares} 股（全出）或至少 {int(shares*0.5)} 股（50%）")
            elif has_trim:
                print(f"  → 建議賣出 ~{int(shares*0.3)} 股（30%），現價 {close:.1f}")
            elif has_watch:
                print(f"  → 可先賣出 ~{int(shares*0.3)} 股（30%）觀察，現價 {close:.1f}")
        else:
            print()
            print(f"  ✅ 目前未觸發任何出場訊號")

        # ── 攤平或加碼訊號 ──
        if holding.get('avg_down'):
            score, msgs, ready = avg_down_signals(df)
            print(f"\n  ── 攤平訊號 {score}/4 {'✅ 就緒' if ready else '⏳ 等待'} ──")
            for m in msgs:
                print(m)
        if holding.get('building'):
            score, msgs, ready = building_signals(df)
            shares_h = holding.get('shares', 0)
            half_h   = max(1, int(shares_h * 0.15))
            if ready:
                label_b = f"✅ 4/4 就緒 → 正常加碼"
            elif score == 3:
                label_b = f"🔸 3/4 少量試單 → 約 {half_h} 股（待 4/4 補足）"
            else:
                label_b = f"⏳ {score}/4 等待"
            print(f"\n  ── 建倉加碼  {label_b} ──")
            for m in msgs:
                print(m)
    else:
        # 不在持倉：顯示進場訊號
        print()
        print(f"  ── 進場條件 ──")
        score, msgs = entry_signals(df)
        for m in msgs:
            print(m)
        if score == 3:
            print(f"  ✅ 三項全達成，可考慮進場")
        elif score == 2:
            print(f"  🔍 {score}/3 條件達成，距進場不遠")
        else:
            print(f"  ⏳ {score}/3 條件達成，繼續觀察")

    # ── 三大法人（台股） ──
    if ticker.endswith('.TW'):
        code = ticker.replace('.TW', '')
        print()
        _print_institutional(code)

    # ── 基本面（台股持倉才顯示）──
    if holding and ticker.endswith('.TW'):
        fund = get_fundamentals(ticker)
        if fund:
            print()
            print("  ── 基本面 ──")
            for f in fund:
                print(f)

    print()
    print("=" * 55)
    print()


def intraday_scan():
    """即時掃描：掃全部持倉 + 觀察名單，盤中/盤後/盤前均可執行
    每次執行前清除 Fugle cache，確保拿到最新報價（Streamlit session 會保留舊 cache）
    用法：python daily_analysis.py scan
    """
    _fugle_cache.clear()   # 每次執行都清除，確保 Fugle 價格是最新的
    now = now_tw()
    status, status_note = market_status()
    minutes = now.hour * 60 + now.minute
    remaining = max(0, (13 * 60 + 30) - minutes)

    # 標題依狀態調整
    if status == "盤中":
        title     = "盤中掃描  即時評估"
        time_note = f"距收盤 {remaining} 分鐘"
    elif status == "盤後":
        title     = "盤後掃描  今日收盤回顧"
        time_note = "資料為今日最終收盤（Fugle）"
    else:
        title     = "盤前掃描  昨日資料回顧"
        time_note = "台股尚未開盤，顯示昨日收盤資料"

    print()
    print("=" * 55)
    print(f"   {title}")
    print(f"   {now.strftime('%Y-%m-%d  %H:%M')}  {time_note}")
    print("=" * 55)

    actions_urgent = []   # 今天要做的事
    actions_watch  = []   # 留意但不急
    actions_ok     = []   # 不動

    for ticker, h in HOLDINGS.items():
        if not ticker.endswith((".TW", ".TWO")):
            continue
        code = ticker.replace(".TWO", "").replace(".TW", "")
        name = h.get("name", ticker)

        # 取 Fugle 即時資料
        fugle_d = get_fugle_quote(code)
        fq      = parse_fugle_price(fugle_d)
        if not fq or not fq["price"]:
            continue

        price      = fq["price"]
        buy_price  = h["buy_price"]
        shares     = h["shares"]
        profit_pct = (price - buy_price) / buy_price * 100
        ask_pct    = fq.get("ask_pct")
        vol        = fq.get("volume") or 0

        # 取日線指標（用於訊號判斷）
        df = fetch(ticker)
        if df is None:
            continue
        # 將 df 的 Close 更新為 Fugle 即時價，讓 building/exit 訊號用正確價格計算
        df.iloc[-1, df.columns.get_loc('Close')] = price
        r         = df.iloc[-1]
        vol_ma5   = r["Vol_MA5"]
        # Fugle volume 單位是張，yfinance Vol_MA5 單位是股（1張=1000股），需乘以1000換算
        vol_ratio = (vol * 1000) / vol_ma5 if (vol and vol_ma5) else r["Vol_ratio"]
        # 以即時價更新乖離率
        ma5       = r["MA5"]
        deviation = (price - ma5) / ma5 * 100
        rsi       = r["RSI"]
        atr       = r["ATR"]
        atr_stop  = buy_price - 2 * atr

        # 內外盤解讀
        if ask_pct is not None:
            if ask_pct >= 60:   ob = f"外盤 {ask_pct:.0f}% 偏多↑"
            elif ask_pct <= 40: ob = f"外盤 {ask_pct:.0f}% 偏空↓"
            else:               ob = f"外盤 {ask_pct:.0f}% 平衡"
        else:
            ob = "無資料"

        # 量能計算：按當下時間推估全日量
        # 台股 09:00~13:30 = 270 分鐘；盤後直接用實際收盤量
        # vol 單位是張，vol_ma5 單位是股（1張=1000股），需乘以1000換算
        if status == "盤中" and vol:
            total_min   = 270   # 全日交易分鐘數
            elapsed_min = max(1, (now.hour * 60 + now.minute) - 9 * 60)
            progress    = min(elapsed_min / total_min, 1.0)   # 0~1
            est_full_vol  = vol / progress
            vol_est_ratio = (est_full_vol * 1000) / vol_ma5 if vol_ma5 else vol_ratio
            pct_str = f"{progress*100:.0f}%"
            vol_note = f"預估全日量比 {vol_est_ratio:.2f}（已過{pct_str}）"
        else:
            est_full_vol  = vol               # 盤後：Fugle 量就是今日完整量
            vol_est_ratio = (vol * 1000) / vol_ma5 if (vol and vol_ma5) else vol_ratio
            vol_note = f"量比 {vol_est_ratio:.2f}"

        print(f"\n  {ticker} {name}  現價 {price:.1f}  損益 {profit_pct:+.1f}%")
        print(f"  乖離 {deviation:+.1f}%  RSI {rsi:.1f}  {ob}  {vol_note}")

        # 判斷今天要做的事
        exit_msgs  = exit_signals(df, buy_price)
        has_red    = any("🔴" in m for m in exit_msgs)
        has_green  = any("🟢" in m for m in exit_msgs)
        has_yellow = any("🟡" in m for m in exit_msgs)

        if price < atr_stop:
            action = f"🔴 {name} 跌破停損線 {atr_stop:.1f}，收盤前考慮出場"
            actions_urgent.append(action)
            print(f"  ⚠  跌破 ATR 停損線 {atr_stop:.1f}")
            for m in exit_msgs: print(m)
        elif has_green:
            trim = int(shares * 0.3)
            for m in exit_msgs: print(m)
            if ask_pct is not None and ask_pct <= 40:
                action = f"🟢 {name} 停利訊號＋內盤偏重，建議收盤前賣 {trim} 股（{price:.1f}）"
                actions_urgent.append(action)
                print(f"  ✅ 停利訊號 + 內盤偏重，建議今天執行減碼 {trim} 股")
            else:
                action = f"🟢 {name} 停利訊號，可掛 {trim} 股賣單，外盤仍強可再觀察"
                actions_watch.append(action)
                print(f"  ⚡ 停利訊號，外盤仍強，可掛單等自動成交")
        elif has_yellow:
            for m in exit_msgs: print(m)
            if ask_pct is not None and ask_pct <= 40:
                action = f"🟡 {name} 偏熱 + 內盤偏重，可考慮減碼 {int(shares*0.3)} 股"
                actions_watch.append(action)
                print(f"  🟡 偏熱警示 + 內盤偏重，可考慮掛單")
            else:
                print(f"  🟡 偏熱但外盤尚可，繼續觀察")
                actions_ok.append(f"{name} 偏熱觀察中")
        elif h.get("avg_down"):
            score, s_msgs, ready = avg_down_signals(df)
            for m in s_msgs: print(m)           # ← 顯示各條件 ✓/✗
            if ready:
                if ask_pct is not None and ask_pct >= 55:
                    action = f"🟢 {name} 攤平就緒＋外盤偏多，可在 {price:.1f} 掛買單"
                    actions_urgent.append(action)
                    print(f"  ✅ 攤平訊號 {score}/4 + 外盤確認，今天可以買")
                else:
                    print(f"  ⏳ 攤平訊號 {score}/4，外盤待確認")
                    actions_watch.append(f"{name} 攤平訊號就緒，等外盤確認")
            else:
                print(f"  ⏳ 攤平尚未就緒 {score}/4")
                actions_ok.append(f"{name} 攤平等待中")
        elif h.get("building"):
            b_score, b_msgs, b_ready = building_signals(df)
            for m in b_msgs: print(m)           # ← 顯示各條件 ✓/✗
            half_s = max(1, int(shares * 0.15))
            if b_ready:
                if ask_pct is not None and ask_pct >= 50:
                    action = f"✅ {name} 加碼 4/4＋外盤確認，可正常加碼"
                    actions_urgent.append(action)
                    print(f"  ✅ 建倉加碼 4/4 + 外盤確認，今天可以買")
                else:
                    actions_watch.append(f"{name} 加碼 4/4，等外盤轉強")
                    print(f"  🔸 建倉加碼 4/4，外盤偏弱，可掛單等")
            elif b_score == 3:
                if ask_pct is not None and ask_pct >= 50:
                    action = f"🔸 {name} 少量試單（3/4）＋外盤尚可，約 {half_s} 股"
                    actions_watch.append(action)
                    print(f"  🔸 建倉 3/4 + 外盤尚可，可少量試單約 {half_s} 股")
                else:
                    print(f"  🔸 建倉 3/4，外盤偏空，今天先觀望")
                    actions_ok.append(f"{name} 建倉 3/4 待外盤確認")
            elif b_score == 2 and vol_est_ratio >= 1.5:
                probe_s = max(1, int(shares * 0.10))
                if ask_pct is not None and ask_pct >= 55:
                    action = (f"💡 {name} 量能突破（量比{vol_est_ratio:.2f}+外盤{ask_pct:.0f}%）"
                              f"，試單約 {probe_s} 股")
                    actions_watch.append(action)
                    print(f"  💡 量能突破：量比預估 {vol_est_ratio:.2f}，外盤 {ask_pct:.0f}%，可試單約 {probe_s} 股")
                else:
                    ob_str = f"{ask_pct:.0f}%" if ask_pct is not None else "N/A"
                    print(f"  💡 量能突破（量比{vol_est_ratio:.2f}）但外盤偏弱（{ob_str}），謹慎觀察")
                    actions_ok.append(f"{name} 量能突破待外盤確認")
            else:
                print(f"  ⏳ 建倉 {b_score}/4，繼續等待")
                actions_ok.append(f"{name} 建倉等待中")
        else:
            # 一般持倉：有出場訊號才顯示，否則續抱
            if exit_msgs:
                for m in exit_msgs: print(m)
            print(f"  ✅ 無訊號，續抱")
            actions_ok.append(f"{name}")

    # ── 觀察名單 量能突破掃描 ──────────────────────────────
    divider()
    print("\n▌ 觀察名單 量能突破掃描\n")
    watch_found = False
    for code in WATCHLIST.get("tw", []):
        # .TW / .TWO 自動偵測
        ticker_w = code + ".TW"
        df_w = fetch(ticker_w, silent=True)
        if df_w is None:
            ticker_w = code + ".TWO"
            df_w = fetch(ticker_w, silent=True)
        if df_w is None:
            continue

        fugle_dw = get_fugle_quote(code)
        fq_w     = parse_fugle_price(fugle_dw)
        if not fq_w or not fq_w["price"]:
            continue

        price_w   = fq_w["price"]
        ask_pct_w = fq_w.get("ask_pct")
        vol_w     = fq_w.get("volume") or 0

        rw         = df_w.iloc[-1]
        vol_ma5_w  = rw["Vol_MA5"]
        # 按當下時間動態推估全日量，盤後直接用實際收盤量
        if status == "盤中" and vol_w:
            total_min   = 270
            elapsed_min = max(1, (now.hour * 60 + now.minute) - 9 * 60)
            progress    = min(elapsed_min / total_min, 1.0)
            est_vol_w   = vol_w / progress
        else:
            est_vol_w = vol_w
        # Fugle volume 是張，Vol_MA5 是股（×1000 換算）
        est_ratio_w = (est_vol_w * 1000) / vol_ma5_w if vol_ma5_w else 0

        prev_cw   = df_w.iloc[-2]['Close']
        day_chg_w = (price_w - prev_cw) / prev_cw * 100

        if day_chg_w >= 9.5:   # 漲停不追
            continue

        score_w, msgs_w = entry_signals(df_w)

        # 量能突破：預估全日量比 ≥ 1.5 且外盤 ≥ 50%（無資料也算）
        is_vol_brk = est_ratio_w >= 1.5 and (ask_pct_w is None or ask_pct_w >= 50)

        if is_vol_brk or score_w >= 2:
            watch_found = True
            ob_w  = f"外盤{ask_pct_w:.0f}%" if ask_pct_w is not None else "外盤N/A"
            vol_m = f"💡量比{est_ratio_w:.2f}" if est_ratio_w >= 1.5 else f"量比{est_ratio_w:.2f}"
            print(f"  {ticker_w}  現價 {price_w:.1f}（{day_chg_w:+.1f}%）  {ob_w}  {vol_m}  訊號 {score_w}/3")
            if is_vol_brk and score_w >= 2:
                action_w = f"💡 {ticker_w} 量能突破+技術{score_w}/3，可小量試單"
                actions_watch.append(action_w)
                print(f"     ← 量能突破 + {score_w}/3 訊號，可考慮小量試單")
            elif is_vol_brk:
                print(f"     量能突破但技術訊號弱（{score_w}/3），謹慎觀察")
                actions_ok.append(f"{ticker_w} 量能突破但訊號弱")
            else:
                print(f"     技術訊號 {score_w}/3，量能待加強")
            print()

    if not watch_found:
        print("  今日觀察名單無量能突破或接近進場的標的\n")

    # ── 操作清單 ──
    divider()
    if status == "盤中":
        print(f"\n▌ 收盤前 {remaining} 分鐘操作清單\n")
    elif status == "盤後":
        print(f"\n▌ 今日操作回顧（收盤後）\n")
    else:
        print(f"\n▌ 昨日資料回顧（盤前）\n")

    if actions_urgent:
        print("  【今天要做】")
        for a in actions_urgent:
            print(f"  → {a}")
        print()
    if actions_watch:
        print("  【可以考慮】")
        for a in actions_watch:
            print(f"  ○ {a}")
        print()
    if actions_ok:
        print("  【不動】")
        print(f"  ✅ {' / '.join(actions_ok)}")
        print()

    print("=" * 55)
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == "scan":
            intraday_scan()
        else:
            quick_lookup(arg)
    else:
        run()
