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


def _apply_fugle_price(df, price, is_intraday=False):
    """Fugle 即時/收盤價寫入 df，依盤中/盤後採用不同策略

    盤後 (is_intraday=False)：
      - Fugle closePrice = 今日官方收盤價（最終值）
      - 更新 Close 後重算 MA5/MA10，逼近 Yahoo 顯示的均線值
        （yfinance 對台股有約 1 日延遲，重算後誤差從 >6 點縮小到 <2 點）
      - Vol_MA5 也以今日完整量重算

    盤中 (is_intraday=True)：
      - Fugle lastPrice = 即時波動，不代表最終收盤
      - MA5/MA10 以即時價重算（今日即時價作為第5/10根），反映即時均線位置
      - Vol_MA5 改用前一日穩定值，避免盤中部分量壓低分母使量比虛高
    """
    import datetime, pandas as pd
    df = df.copy()

    if is_intraday:
        # yfinance 對台股有 1 日延遲，盤中時最後一行可能是前一交易日
        # 若最後一行不是今天，先 append 今日 row（用前一日收盤暫填），避免覆蓋前一日收盤
        today = datetime.date.today()
        last_date = df.index[-1].date()
        if last_date < today:
            new_row = df.iloc[-1:].copy()
            tz = df.index.tz
            new_idx = pd.DatetimeIndex(
                [pd.Timestamp(today, tz=tz) if tz else pd.Timestamp(today)]
            )
            new_row.index = new_idx
            prev_close = float(df.iloc[-1]['Close'])
            new_row.iloc[0, new_row.columns.get_loc('Close')] = prev_close
            for col in ('Open', 'High', 'Low'):
                new_row.iloc[0, new_row.columns.get_loc(col)] = prev_close
            new_row.iloc[0, new_row.columns.get_loc('Volume')] = 0
            df = pd.concat([df, new_row])

    df.iloc[-1, df.columns.get_loc('Close')] = price

    if is_intraday:
        # 盤中：MA5/MA10 用即時價重算（今日即時價作為第5/10根，前幾日收盤均保留）
        # Vol_MA5 用前一日穩定值，避免盤中部分量壓低分母使量比虛高
        df['MA5']  = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        if len(df) >= 2 and df.iloc[-2]['Vol_MA5'] > 0:
            df.iloc[-1, df.columns.get_loc('Vol_MA5')] = df.iloc[-2]['Vol_MA5']
        # RSI 反映當下動能，也需即時重算
        c = df['Close']
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    else:
        # 盤後：重算 MA5/MA10（今日收盤已是最終值）
        df['MA5']    = df['Close'].rolling(5).mean()
        df['MA10']   = df['Close'].rolling(10).mean()
        df['Vol_MA5'] = df['Volume'].rolling(5).mean()

    return df


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


_inst_cache    = {}   # 避免同一次執行重複打 API
_index_df_cache = None  # 台灣加權指數快取（相對強度計算用）


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
        'dataset':    'TaiwanStockInstitutionalInvestorsBuySell',
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

        # 依日期彙整三類法人（name 欄位已改為英文）
        from collections import defaultdict
        by_date = defaultdict(dict)
        for r in body['data']:
            date = r['date']
            name = r.get('name', '')
            net  = (r.get('buy') or 0) - (r.get('sell') or 0)
            if name == 'Foreign_Investor':
                by_date[date]['外資'] = net
            elif name == 'Investment_Trust':
                by_date[date]['投信'] = net
            elif name == 'Dealer_self':
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


def _print_institutional_brief(code):
    """只顯示三大法人趨勢摘要（1行），一般持倉用"""
    inst = get_institutional(code)
    if inst:
        _, trend = inst
        if trend:
            print(f"  法人：{trend.strip()}")


def _get_twii_df():
    """取得台灣加權指數日線（全模組共用快取，避免重複下載）"""
    global _index_df_cache
    if _index_df_cache is not None:
        return _index_df_cache
    try:
        df = yf.Ticker('^TWII').history(period='60d')
        _index_df_cache = df if not df.empty else pd.DataFrame()
    except Exception:
        _index_df_cache = pd.DataFrame()
    return _index_df_cache


def calc_relative_strength(df):
    """計算個股相對台灣加權指數強度（5日、20日）
    回傳 (rs5, rs20) 皆為百分點，正值代表跑贏大盤
    任一無法計算則回傳 None
    """
    idx = _get_twii_df()
    if idx.empty:
        return None, None

    def pct_return(df_, n):
        if len(df_) < n + 1:
            return None
        return (df_['Close'].iloc[-1] - df_['Close'].iloc[-n - 1]) / df_['Close'].iloc[-n - 1] * 100

    stock_5  = pct_return(df,  5)
    stock_20 = pct_return(df, 20)
    idx_5    = pct_return(idx,  5)
    idx_20   = pct_return(idx, 20)

    rs5  = (stock_5  - idx_5)  if (stock_5  is not None and idx_5  is not None) else None
    rs20 = (stock_20 - idx_20) if (stock_20 is not None and idx_20 is not None) else None
    return rs5, rs20


def inst_direction(code):
    """取得三大法人連續方向（使用 get_institutional 快取，不重複打 API）
    回傳 ('buy'|'sell'|'neutral'|None, days: int, note: str|None)
    """
    inst = get_institutional(code)
    if not inst:
        return None, 0, None
    _, trend = inst
    if not trend:
        return 'neutral', 0, None

    try:
        if '買超' in trend:
            days = int(trend.split('連續')[1].split('日')[0]) if '連續' in trend else 1
            return 'buy', days, trend.strip()
        if '賣超' in trend:
            days = int(trend.split('連續')[1].split('日')[0]) if '連續' in trend else 1
            return 'sell', days, trend.strip()
    except Exception:
        pass
    return 'neutral', 0, None


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

    # RSI(14)：Wilder's SMMA（與 Yahoo Finance / TradingView 等圖表軟體一致）
    # 注意：rolling(14).mean() 是簡單平均，會與圖表差 10~20 點，不可用
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
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

    # 近22日最高收盤（吊燈出場用）
    df['High22'] = close.rolling(22).max()

    # 成交量比（今日量 / 5日均量）
    df['Vol_MA5']   = vol.rolling(5).mean()
    df['Vol_ratio'] = vol / df['Vol_MA5']

    # 近20日最低點
    df['Low_20'] = low.rolling(20).min()

    # Bollinger Bands (20日, 2σ)
    df['BB_mid']   = close.rolling(20).mean()
    df['BB_upper'] = df['BB_mid'] + 2 * close.rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - 2 * close.rolling(20).std()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']

    # KD 隨機指標（9日）
    low9  = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv   = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    df['KD_K'] = rsv.ewm(com=2, adjust=False).mean()   # K = 3日平滑RSV
    df['KD_D'] = df['KD_K'].ewm(com=2, adjust=False).mean()  # D = 3日平滑K

    # OBV（On-Balance Volume）
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

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
        # 最後一行 Close = NaN 但有成交量 → 今日已開盤但 yfinance 收盤尚未寫入
        # 保留此行並先用前一日收盤暫填，_apply_fugle_price 之後會用官方收盤覆蓋
        # 這樣 MA5/MA10 才能正確包含今日（否則整行被 dropna 丟棄，MA5 少算一天）
        import pandas as pd
        if (len(df) >= 2
                and pd.isna(df.iloc[-1]['Close'])
                and pd.notna(df.iloc[-1]['Volume'])
                and df.iloc[-1]['Volume'] > 0):
            df = df.copy()
            prev_close = df.iloc[-2]['Close']
            df.iloc[-1, df.columns.get_loc('Close')] = prev_close
            # High/Low/Open 也填上去，避免 ATR 計算出 NaN
            for col in ('Open', 'High', 'Low'):
                if pd.isna(df.iloc[-1][col]):
                    df.iloc[-1, df.columns.get_loc(col)] = prev_close
        # 移除其他 NaN（真正的空行）
        df = df.dropna(subset=['Close'])
        if df.empty:
            return None
        return calculate_indicators(df)
    except Exception as e:
        if not silent:
            print(f"  ⚠  無法抓取 {ticker}：{e}")
        return None


# ============================================================
#  底部偵測輔助函數
# ============================================================

def detect_rsi_divergence(df):
    """RSI 底背離：價格創新低但 RSI 不破低（賣壓衰退訊號）
    在近 35 根 K 棒中，找「前段低點」（-35 到 -8 根）vs「近段低點」（近 8 根），
    只要近段價格更低、RSI 更高（差距 > 1）即觸發。
    回傳 (bool, msg_str or None)
    """
    valid = df.dropna(subset=['RSI', 'Close'])
    if len(valid) < 16:
        return False, None

    older  = valid.iloc[-35:-8] if len(valid) >= 35 else valid.iloc[:-8]
    recent = valid.iloc[-8:]
    if len(older) < 5 or recent.empty:
        return False, None

    older_low_idx  = older['Close'].idxmin()
    recent_low_idx = recent['Close'].idxmin()
    older_price    = older.loc[older_low_idx,  'Close']
    recent_price   = recent.loc[recent_low_idx, 'Close']
    older_rsi      = older.loc[older_low_idx,  'RSI']
    recent_rsi     = recent.loc[recent_low_idx, 'RSI']

    if pd.isna(older_rsi) or pd.isna(recent_rsi):
        return False, None

    # 底背離：近期 price 更低，但 RSI 高出至少 1 點
    if recent_price < older_price and recent_rsi > older_rsi + 1:
        return True, (
            f"  💡 RSI 底背離（價格新低 {older_price:.1f}→{recent_price:.1f}，"
            f"RSI 不破低 {older_rsi:.1f}→{recent_rsi:.1f}），賣壓衰退中"
        )
    return False, None


def detect_macd_convergence(df):
    """MACD histogram 收斂方向偵測（連續3根方向）
    負值收斂 → 空頭動能衰退（底部訊號）
    正值收斂 → 多頭動能衰退（出場預警）
    回傳 ('bullish'|'bearish'|None, msg_str or None)
    """
    if len(df) < 4:
        return None, None

    h = df['MACD_hist'].dropna().values
    if len(h) < 3:
        return None, None
    h1, h2, h3 = h[-3], h[-2], h[-1]

    # 負值收斂（空頭動能衰退 → 底部前兆）
    if h1 < 0 and h2 < 0 and h3 < 0 and abs(h3) < abs(h2) < abs(h1):
        return 'bullish', (
            f"  💡 MACD 負值收斂（{h1:+.3f}→{h2:+.3f}→{h3:+.3f}），"
            f"空頭動能持續衰退，留意反轉底部"
        )

    # 正值收斂（多頭動能衰退 → 出場預警）
    if h1 > 0 and h2 > 0 and h3 > 0 and h3 < h2 < h1:
        return 'bearish', (
            f"  ⚠  MACD 正值收斂（{h1:+.3f}→{h2:+.3f}→{h3:+.3f}），"
            f"多頭動能衰退，留意提前出場"
        )

    return None, None


def detect_long_lower_shadow(df):
    """長下影線偵測：代表日內有強力接盤（錘子線、蜻蜓十字）
    條件：下影線 > 全日振幅 60%，且收盤高於日中點
    回傳 (bool, msg_str or None)
    """
    if len(df) < 1:
        return False, None

    r = df.iloc[-1]
    c = r['Close']
    o = r.get('Open', c)
    h, l = r['High'], r['Low']
    candle_range = h - l
    if candle_range <= 0:
        return False, None

    lower_shadow = min(o, c) - l
    mid_price    = (h + l) / 2

    if lower_shadow > candle_range * 0.6 and c > mid_price:
        ratio = lower_shadow / candle_range * 100
        return True, f"  💡 長下影線（下影占振幅 {ratio:.0f}%），日內買盤強力支撐"
    return False, None


def detect_volume_shrinkage(df):
    """跌勢量縮偵測：跌破 MA10 後成交量萎縮 → 拋壓耗盡
    回傳 (bool, msg_str or None)
    """
    r = df.iloc[-1]
    below_ma10 = r['Close'] < r['MA10']
    vol_shrink  = r['Vol_ratio'] < 0.6

    if below_ma10 and vol_shrink:
        return True, (
            f"  💡 跌破 MA10 + 量縮（量比 {r['Vol_ratio']:.2f} < 0.6），"
            f"拋壓明顯減弱，留意底部"
        )
    return False, None


def detect_bb_squeeze(df):
    """Bollinger Band 帶寬收縮偵測（大波動前兆）
    帶寬在近60日最窄的 120% 以內 → 蓄勢待發
    回傳 (bool, msg_str or None)
    """
    if 'BB_width' not in df.columns:
        return False, None

    widths = df['BB_width'].dropna()
    if len(widths) < 20:
        return False, None

    current = widths.iloc[-1]
    min_60  = widths.tail(60).min()

    if pd.isna(current) or pd.isna(min_60) or min_60 <= 0:
        return False, None

    if current <= min_60 * 1.2:
        macd_hist = df['MACD_hist'].iloc[-1]
        dir_str   = "MACD 偏多，偏向向上突破↑" if macd_hist > 0 else "MACD 偏空，偏向向下突破↓"
        return True, f"  💡 BB 帶寬收縮至近期最窄，蓄勢待發，{dir_str}"
    return False, None


def detect_sharp_drop_bounce(df):
    """急跌反彈型底部偵測
    情況A：今日大跌 > 5%，但尾盤收在日內高低點中段以上（有人接）
    情況B：前日大跌 > 5%，今日收紅（隔日確認反彈）
    回傳 (bool, msg_str or None)
    """
    if len(df) < 3:
        return False, None

    r    = df.iloc[-1]
    prev = df.iloc[-2]
    c    = r['Close']
    h    = r['High']
    l    = r['Low']
    prev_close = prev['Close']
    day_chg    = (c - prev_close) / prev_close * 100
    mid        = (h + l) / 2

    # 情況A：今日急跌但尾盤守住中點
    if day_chg <= -5 and c > mid:
        return True, f"  💡 急跌 {day_chg:.1f}% 但尾盤守住日內中點（{mid:.1f}），有接盤"

    # 情況B：前日大跌，今日收紅
    prev2_close = float(df.iloc[-3]['Close'])
    prev_chg    = (prev_close - prev2_close) / prev2_close * 100
    if prev_chg <= -5 and c > prev_close:
        return True, f"  💡 前日急跌 {prev_chg:.1f}%，今日收紅 {day_chg:+.1f}%，反彈確認"

    return False, None


def get_ma_state(df):
    """判斷目前均線結構

    回傳 (state_str, ma5, ma10, days_below)
      'golden'    : MA5 > MA10（多頭排列）
      'death_new' : MA5 < MA10，持續 < 5 根（死叉初期）
      'death_old' : MA5 < MA10，持續 >= 5 根（空頭持續）
    days_below = 0 表示黃金叉狀態
    """
    ma5_arr  = df['MA5'].values
    ma10_arr = df['MA10'].values
    ma5_v    = float(ma5_arr[-1])
    ma10_v   = float(ma10_arr[-1])

    if ma5_v > ma10_v:
        return 'golden', ma5_v, ma10_v, 0

    # 從末端往前數連續 MA5 < MA10 的根數
    days_below = 0
    for i in range(len(ma5_arr) - 1, -1, -1):
        if ma5_arr[i] < ma10_arr[i]:
            days_below += 1
        else:
            break

    state = 'death_new' if days_below < 5 else 'death_old'
    return state, ma5_v, ma10_v, days_below


def detect_oversold(df):
    """超跌程度評分（0-7 分，分數越高越超跌）

    回傳 (score, detail_list, level_str)
    """
    r      = df.iloc[-1]
    score  = 0
    detail = []

    # ── RSI ──
    rsi = float(r['RSI'])
    if rsi < 30:
        score += 3
        detail.append(f"RSI {rsi:.1f}  嚴重超賣（+3）")
    elif rsi < 40:
        score += 2
        detail.append(f"RSI {rsi:.1f}  超賣（+2）")
    elif rsi < 50:
        score += 1
        detail.append(f"RSI {rsi:.1f}  偏弱（+1）")
    else:
        detail.append(f"RSI {rsi:.1f}  正常")

    # ── 偏離 BB 中軌（MA20）──
    bb_mid = float(r['BB_mid']) if 'BB_mid' in df.columns else float(r.get('MA20', r['Close']))
    close  = float(r['Close'])
    dev_bb = (close - bb_mid) / bb_mid * 100
    if dev_bb < -10:
        score += 2
        detail.append(f"距 BB 中軌偏離 {dev_bb:.1f}%  嚴重偏低（+2）")
    elif dev_bb < -5:
        score += 1
        detail.append(f"距 BB 中軌偏離 {dev_bb:.1f}%  偏低（+1）")
    else:
        detail.append(f"距 BB 中軌偏離 {dev_bb:.1f}%  正常")

    # ── 從 20 日高點回撤 ──
    high20 = float(df['High'].rolling(20).max().iloc[-1])
    pullback = (close - high20) / high20 * 100
    if pullback < -20:
        score += 2
        detail.append(f"從 20 日高點回撤 {pullback:.1f}%  深度回撤（+2）")
    elif pullback < -10:
        score += 1
        detail.append(f"從 20 日高點回撤 {pullback:.1f}%  回撤中（+1）")
    else:
        detail.append(f"從 20 日高點回撤 {pullback:.1f}%  正常")

    if score >= 5:
        level = "嚴重超跌"
    elif score >= 3:
        level = "明顯超跌"
    elif score >= 1:
        level = "輕度回檔"
    else:
        level = "正常區間"

    return score, detail, level


def calc_probe_stoploss(df):
    """計算探底進場的建議停損價與風險報酬比

    停損邏輯：近20日最低點再往下 3%（底部若破就認錯出場）
    目標：MA10（中期均線回歸）

    回傳 dict:
      stop_price   : 建議停損價
      stop_pct     : 距現價跌幅（負數）
      target_price : 目標價（MA10）
      target_pct   : 距現價漲幅
      rr_ratio     : 風險報酬比（target / stop）
      low20        : 近20日最低點
    """
    r     = df.iloc[-1]
    close = float(r['Close'])
    low20 = float(r['Low_20'])
    ma10  = float(r['MA10'])

    stop_price  = round(low20 * 0.97, 1)   # 近低點下 3%
    stop_pct    = (stop_price - close) / close * 100
    target_price = round(ma10, 1)
    target_pct  = (target_price - close) / close * 100
    rr_ratio    = abs(target_pct / stop_pct) if stop_pct != 0 else 0

    return {
        'stop_price':   stop_price,
        'stop_pct':     stop_pct,
        'target_price': target_price,
        'target_pct':   target_pct,
        'rr_ratio':     rr_ratio,
        'low20':        low20,
    }


def format_probe_stoploss(df):
    """回傳探底停損的顯示字串（兩行）"""
    info = calc_probe_stoploss(df)
    rr_str = f"{info['rr_ratio']:.1f}" if info['rr_ratio'] > 0 else "N/A"
    lines = [
        f"   停損建議：近20日低點 {info['low20']:.1f} × 0.97 = {info['stop_price']:.1f}"
        f"（跌破即出場，風險 {info['stop_pct']:.1f}%）",
        f"   目標參考：MA10 {info['target_price']:.1f}"
        f"（+{info['target_pct']:.1f}%），風險報酬比 1:{rr_str}",
    ]
    return lines


def detect_selling_exhaustion(df):
    """偵測賣壓衰竭訊號

    回傳 (count, signal_list)
    """
    signals = []
    r       = df.iloc[-1]
    prev    = df.iloc[-2]

    close     = float(r['Close'])
    open_     = float(r['Open'])
    high      = float(r['High'])
    low       = float(r['Low'])
    vol_ratio = float(r['Vol_ratio']) if 'Vol_ratio' in df.columns else 1.0
    prev_close = float(prev['Close'])
    prev_open  = float(prev['Open'])

    # 1. 量縮
    if vol_ratio < 0.7:
        signals.append(f"量縮（量比 {vol_ratio:.2f}），拋壓減弱")

    # 2. 長下影線：下影線 > 60% 全振幅 且 收盤 > 日內中點
    candle_range = high - low
    if candle_range > 0:
        lower_shadow = min(open_, close) - low
        shadow_pct   = lower_shadow / candle_range * 100
        midpoint     = (high + low) / 2
        if shadow_pct > 60 and close > midpoint:
            signals.append(f"長下影線（{shadow_pct:.0f}%），日內強力接盤")

    # 3. MACD 空頭收斂：最後 3 根 MACD_hist 均為負且絕對值遞減
    if 'MACD_hist' in df.columns and len(df) >= 3:
        h1 = float(df['MACD_hist'].iloc[-3])
        h2 = float(df['MACD_hist'].iloc[-2])
        h3 = float(df['MACD_hist'].iloc[-1])
        if h1 < 0 and h2 < 0 and h3 < 0 and abs(h3) < abs(h2) < abs(h1):
            signals.append("MACD 空頭收斂，下跌動能衰退")

    # 4. 急跌隔日收紅：前日跌幅 <= -4% 且今日收紅
    prev_chg  = (prev_close - float(prev['Open'])) / float(prev['Open']) * 100 \
                if float(prev['Open']) else 0
    # 用前日漲跌幅
    prev2_close = float(df.iloc[-3]['Close']) if len(df) >= 3 else prev_close
    prev_day_chg = (prev_close - prev2_close) / prev2_close * 100 if prev2_close else 0
    today_chg    = (close - prev_close) / prev_close * 100 if prev_close else 0
    if prev_day_chg <= -4 and today_chg > 0:
        signals.append(f"前日急跌 {prev_day_chg:.1f}%，今日收紅，賣壓消化")

    # 5. 跌幅縮小：前日跌 > 2%，今日跌幅比前日小（仍為負）
    if prev_day_chg < -2 and -2 < today_chg < 0 and today_chg > prev_day_chg:
        signals.append(f"跌幅縮小（昨 {prev_day_chg:.1f}% → 今 {today_chg:.1f}%），動能減弱")

    return len(signals), signals


def bottom_score(df):
    """加權底部評分
    每個訊號依歷史勝率給予不同權重，滿分 10 分
    權重依據：3隻台股一年回測的10日勝率與平均報酬

    回傳 (weighted_score, details)
    details = [(mark, label, weight, msg), ...]
    """
    checks = [
        # (label, fn, weight)  權重依回測勝率：BB=2.0, 長下/MACD/KD/RSI=1.5, 急跌/OBV=1.0
        ('BB下緣反彈',   detect_bb_lower_bounce,   2.0),
        ('長下影線',     detect_long_lower_shadow,  1.5),
        ('MACD空頭收斂', lambda df: (detect_macd_convergence(df)[0] == 'bullish',
                                     detect_macd_convergence(df)[1]),           1.5),
        ('KD低檔黃金叉', detect_kd_golden_cross,    1.5),
        ('RSI底背離',   detect_rsi_divergence,      1.5),
        ('急跌反彈型',   detect_sharp_drop_bounce,   1.0),
        ('OBV底背離',   detect_obv_divergence,       1.0),
    ]

    score   = 0.0
    details = []
    for label, fn, w in checks:
        fired, msg = fn(df)
        if fired:
            score += w
            details.append(('✓', label, w, msg))
        else:
            details.append(('✗', label, w, None))

    return round(score, 1), details


def buy_opportunity(df, buy_price=None, shares=0):
    """統一買進機會評估——自動判斷狀態，給出一個結論

    狀態由程式自動判斷：
      watch  → 沒有持倉，找進場點
      profit → 持倉獲利中，找加碼點
      loss   → 持倉虧損中，找攤平點

    回傳 (action_str, detail_msgs, score)
    """
    r     = df.iloc[-1]
    close = float(r['Close'])
    ma5   = float(r['MA5'])
    ma10  = float(r['MA10'])
    rsi   = float(r['RSI'])

    trend_up = ma5 > ma10

    # 底部評分
    score, details = bottom_score(df)

    # 狀態判斷
    if buy_price is None:
        state      = 'watch'
        profit_pct = None
    else:
        profit_pct = (close - buy_price) / buy_price * 100
        state      = 'profit' if profit_pct >= 0 else 'loss'

    # 加權門檻：滿分10，虧損需3.0，獲利/觀察需2.0
    threshold = 3.0 if state == 'loss' else 2.0

    msgs = []

    # 底部評分 — 逐條顯示 ✓/✗ 含權重
    msgs.append(f"  底部評分：{score}/10.0")
    for mark, label, w, detail in details:
        w_str = f"[×{w}]"
        if mark == '✓':
            d_str = f"  {detail.strip()}" if detail else ""
            msgs.append(f"    ✓ {label} {w_str}{d_str}")
        else:
            msgs.append(f"    ✗ {label} {w_str}")

    # 趨勢補充
    if trend_up:
        msgs.append(f"  趨勢：多頭排列（MA5 {ma5:.1f} > MA10 {ma10:.1f}）")
    else:
        msgs.append(f"  趨勢：空頭排列（MA5 {ma5:.1f} < MA10 {ma10:.1f}）")

    # 操作建議
    if score == 0:
        action = "⏸ 無底部訊號，不動"

    elif score < threshold:
        if state == 'loss':
            action = f"⏳ 底部訊號弱（{score}/10），今天不動"
            msgs.append(f"  虧損 {profit_pct:+.1f}%，攤平需加權分 ≥3.0 才考慮")
            msgs.append(f"  明日留意：{'若出現KD交叉或下影線再評估' if not trend_up else 'MA5回升後再考慮'}")
        else:
            action = f"⏳ 底部訊號弱（{score}/10），繼續觀察"

    else:
        # 訊號足夠，根據狀態給具體建議
        if state == 'watch':
            if trend_up:
                action = f"⭐ 底部訊號 {score}/10，可考慮進場（趨勢向上）"
                msgs.append(f"  建議：小量試單，跌破 MA10（{ma10:.1f}）停損")
            else:
                action = f"💡 底部訊號 {score}/10，趨勢仍空，小量試單"
                msgs.append(f"  建議：不超過正常部位 30%，等 MA5 翻正再加")

        elif state == 'profit':
            size = "20%" if score >= 5.0 else "10%"
            action = f"✅ 底部訊號 {score}/10，回調加碼時機"
            msgs.append(f"  獲利 {profit_pct:+.1f}%，建議加碼約 {size} 部位")
            if not trend_up:
                msgs.append(f"  ⚠ 趨勢轉弱，加碼後需設停損 MA10（{ma10:.1f}）")

        else:  # loss
            size = "10%" if score >= 5.0 else "5%"
            if trend_up:
                action = f"⚠ 底部訊號 {score}/10，可小量攤平（趨勢仍多頭）"
            else:
                action = f"⚠ 底部訊號 {score}/10，謹慎攤平（趨勢向下）"
            msgs.append(f"  虧損 {profit_pct:+.1f}%，攤平不超過原部位 {size}")
            msgs.append(f"  若再跌破今日低點（{float(r['Low']):.1f}），停止攤平")

    return action, msgs, score


def detect_kd_golden_cross(df):
    """KD 低檔黃金交叉：近期 K 曾進入超賣（< 30），且 K 剛從下往上穿越 D
    放寬為：近 5 根內 K 曾 < 30，且目前 D < 35，今天 K 穿越 D
    回傳 (bool, msg_str or None)
    """
    if 'KD_K' not in df.columns or len(df) < 3:
        return False, None

    k_now  = df['KD_K'].iloc[-1]
    d_now  = df['KD_D'].iloc[-1]
    k_prev = df['KD_K'].iloc[-2]
    d_prev = df['KD_D'].iloc[-2]

    if pd.isna(k_now) or pd.isna(d_now):
        return False, None

    # 近 5 根內 K 曾進入超賣（< 30），目前 D 仍在低檔（< 35）
    k_recent_low = df['KD_K'].iloc[-5:].min()
    low_zone     = k_recent_low < 30 and d_now < 35
    golden_cross = k_prev < d_prev and k_now >= d_now

    if low_zone and golden_cross:
        return True, f"  💡 KD 低檔黃金交叉（K={k_now:.1f} D={d_now:.1f}，近期低點K={k_recent_low:.1f}），超賣區反轉訊號"
    return False, None


def detect_bb_lower_bounce(df):
    """布林通道下緣反彈：近期觸碰下緣後今日收回中線以下但高於下緣
    放寬為：近 3 根內有碰觸下緣（在下緣 1% 以內），今日收盤高於下緣
    回傳 (bool, msg_str or None)
    """
    if 'BB_lower' not in df.columns or len(df) < 4:
        return False, None

    c_now  = float(df['Close'].iloc[-1])
    bl_now = float(df['BB_lower'].iloc[-1])
    if pd.isna(bl_now):
        return False, None

    # 近 3 根內有觸碰下緣（收盤或低點在下緣 1% 以內）
    touched = False
    for i in range(-4, -1):
        c_i  = float(df['Close'].iloc[i])
        l_i  = float(df['Low'].iloc[i])
        bl_i = float(df['BB_lower'].iloc[i])
        if pd.isna(bl_i):
            continue
        if c_i <= bl_i * 1.01 or l_i <= bl_i:
            touched = True
            bl_ref = bl_i
            break

    # 今日收回下緣之上
    if touched and c_now > bl_now:
        return True, f"  💡 BB 下緣反彈（近期觸碰 {bl_ref:.1f}，今日收回 {c_now:.1f}），下緣支撐有效"
    return False, None


def detect_obv_divergence(df):
    """OBV 底背離：價格創新低但 OBV 不創新低
    下跌時有人悄悄買進，資金沒有跟著出逃
    回傳 (bool, msg_str or None)
    """
    if 'OBV' not in df.columns or len(df) < 15:
        return False, None

    older  = df.iloc[-20:-10] if len(df) >= 20 else df.iloc[:-10]
    recent = df.iloc[-5:]

    if older.empty or recent.empty:
        return False, None

    older_price_low  = older['Close'].min()
    recent_price_low = recent['Close'].min()
    older_obv_low    = older['OBV'].min()
    recent_obv_low   = recent['OBV'].min()

    # 價格創新低，但 OBV 沒有創新低（底背離）
    if recent_price_low < older_price_low and recent_obv_low > older_obv_low:
        return True, f"  💡 OBV 底背離（價格新低但成交量能未跟跌），有資金悄悄進場"
    return False, None


# ============================================================
#  訊號判斷
# ============================================================

def exit_signals(df, buy_price):
    """持倉警示與出場訊號

    停損層級（由重到輕）：
      🔴 ATR 移動停損 / 雙破均線 / 單日跳空 → 建議出場或減碼
      🟠 跌破 MA5+MACD翻負 / 單獨跌破 MA10  → 考慮減碼
      🟡 跌破 MA5+MACD仍正                  → 觀望

    停利層級（乖離率 + RSI 雙重確認）：
      🟢 強力停利 / 減碼 30% / 偏熱留意
    """
    r    = df.iloc[-1]
    prev = df.iloc[-2]
    msgs = []

    close        = r['Close']
    atr          = r['ATR']
    daily_change = (close - prev['Close']) / prev['Close'] * 100
    profit_pct   = (close - buy_price) / buy_price * 100

    # ── 吊燈出場（Chandelier Exit 2x ATR）────────────────────────────────────
    # 用近22日最高收盤往下扣 2×ATR，不受買入成本與歷史舊高影響
    # 解決「分批進場無法界定峰值起點」與「舊高點造成停損線虛高」的問題
    high22   = r['High22']
    atr_stop = high22 - 2 * atr
    stop_label = f"吊燈停損（22日高{high22:.1f}-2ATR）"

    if close < atr_stop:
        msgs.append(f"  🔴 跌破 {stop_label} {atr_stop:.1f}  → 建議出場")

    # ── 單日跳空 ──────────────────────────────────────────
    if daily_change < -4:
        msgs.append(f"  🔴 單日跌幅 {daily_change:.1f}%  → 注意跳空，考慮減碼")

    # ── 雙線同時跌破（最強出場訊號）────────────────────────
    below_ma5  = close < r['MA5']
    below_ma10 = close < r['MA10']
    if below_ma5 and below_ma10:
        msgs.append(
            f"  🔴 同時跌破 MA5({r['MA5']:.1f}) + MA10({r['MA10']:.1f})"
            f"  → 趨勢轉弱，建議減碼或出場"
        )
    else:
        # 跌破 MA10（單獨）
        if below_ma10:
            msgs.append(
                f"  🟠 跌破 MA10({r['MA10']:.1f})  → 中期支撐失守，考慮減碼 30%"
            )
        # 跌破 MA5（單獨）
        if below_ma5:
            if r['MACD_hist'] < 0:
                msgs.append(f"  🟠 跌破 MA5({r['MA5']:.1f}) 且 MACD 翻負  → 短線轉弱，考慮部分出場")
            else:
                msgs.append(f"  🟡 跌破 MA5({r['MA5']:.1f}) 但 MACD 仍正  → 觀望，留意明日走勢")

    # ── MACD 動能衰退預警（正值收斂 → 多頭快結束）─────────────
    # 只在無更強出場訊號時顯示，避免訊號噪音
    if not any('🔴' in m or '🟠' in m for m in msgs):
        conv_dir, conv_msg = detect_macd_convergence(df)
        if conv_dir == 'bearish' and conv_msg:
            msgs.append(conv_msg)

    # ── 移動停利（乖離率 + RSI 雙重確認）────────────────────
    deviation = (close - r['MA5']) / r['MA5'] * 100
    rsi       = r['RSI']

    if deviation > 12 and rsi > 75:
        msgs.append(
            f"  🟢 乖離率 {deviation:.1f}% 且 RSI {rsi:.1f}  → 強力停利，建議減碼 30%~50%"
        )
    elif deviation > 8 and rsi > 68:
        msgs.append(
            f"  🟢 乖離率 {deviation:.1f}% 且 RSI {rsi:.1f}  → 過熱，建議先減碼 30%"
        )
    elif deviation > 5 and rsi > 62 and profit_pct > 0:
        msgs.append(
            f"  🟡 乖離率 {deviation:.1f}% 且 RSI {rsi:.1f}  → 偏熱，留意是否需要獲利了結"
        )

    return msgs


def avg_down_signals(df):
    """
    反彈確認條件（第二階段，在超跌+衰竭出現後用來確認正式反彈）

    底部形成的兩個階段：
      第一階段（底部形成中）：超跌 + 賣壓衰竭（量縮/下影線）→ 少量探底 5%
      第二階段（反彈確認） ：超跌 + 本函數兩條都達到         → 正常加碼

    反彈確認兩條（都要）：
      A. 收紅 + 量比 ≥ 1.2  → 有量的真實反彈（不是死貓彈）
      B. 距近20日低點反彈 > 3% → 底部守住，不再創新低

    加分條件（影響倉位積極程度）：
      +1. RSI < 60 且正在回升  → 動能轉向
      +2. MA5 斜率翻正         → 短線趨勢改善

    回傳 (反彈確認達成數 0~2, 加分數 0~2, 說明, 是否確認反彈)
    """
    r    = df.iloc[-1]
    prev = df.iloc[-2]
    msgs = []
    close = r['Close']

    # ── 反彈確認 A：收紅 + 量比 ≥ 1.2（有量，買盤真實）─────────
    price_up      = close > prev['Close']
    volume_enough = r['Vol_ratio'] >= 1.2
    conf_a = price_up and volume_enough
    if conf_a:
        msgs.append(f"  ✅ [確認A] 收紅且量比 {r['Vol_ratio']:.2f}（放量反彈，買盤真實）")
    elif not price_up:
        msgs.append(f"  ❌ [確認A] 今日收黑（等收紅K棒確認）")
    else:
        msgs.append(f"  ❌ [確認A] 收紅但量比 {r['Vol_ratio']:.2f}（需≥1.2，量能不足可能假反彈）")

    # ── 反彈確認 B：距近20日低點反彈 > 3%（底部守住）───────────
    low_20      = r['Low_20']
    rebound_pct = (close - low_20) / low_20 * 100
    conf_b = rebound_pct >= 3
    if conf_b:
        msgs.append(f"  ✅ [確認B] 距近20日低點反彈 {rebound_pct:.1f}%（底部有支撐）")
    else:
        msgs.append(f"  ❌ [確認B] 距近20日低點僅 {rebound_pct:.1f}%（需>3%，尚未確認底部）")

    # ── 加分 1：RSI < 60 且回升（動能轉向）─────────────────────
    rsi_recent_low = df['RSI'].iloc[-10:].min()
    rsi_now        = r['RSI']
    rsi_rising     = rsi_now > prev['RSI']
    bonus_rsi = (rsi_now < 60 and rsi_rising)
    if bonus_rsi:
        msgs.append(f"  ➕ [加分1] RSI {rsi_now:.1f} 回升中（近期低 {rsi_recent_low:.1f}），動能轉向")
    elif rsi_now >= 60:
        msgs.append(f"  ➖ [加分1] RSI {rsi_now:.1f} 偏高（需<60 才算超跌回升）")
    else:
        msgs.append(f"  ➖ [加分1] RSI {rsi_now:.1f} 仍下滑，動能未止")

    # ── 加分 2：MA5 斜率翻正（短線止跌）────────────────────────
    ma5_turning_up = r['MA5'] > prev['MA5']
    bonus_ma5 = ma5_turning_up
    if bonus_ma5:
        msgs.append(f"  ➕ [加分2] MA5 斜率翻正（{prev['MA5']:.1f}→{r['MA5']:.1f}），短線止跌")
    else:
        msgs.append(f"  ➖ [加分2] MA5 仍下彎（{prev['MA5']:.1f}→{r['MA5']:.1f}）")

    conf_count  = int(conf_a) + int(conf_b)
    bonus_count = int(bonus_rsi) + int(bonus_ma5)
    confirmed   = conf_a and conf_b   # 兩條都到才算反彈確認

    return conf_count, bonus_count, msgs, confirmed


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

    # 條件4：現價在 MA5 ±5% 內（回調加碼，不追高，也不追跌）
    close     = r['Close']
    ma5       = r['MA5']
    above_ma5 = (close - ma5) / ma5 * 100
    if -5 <= above_ma5 <= 5:
        score += 1
        msgs.append(f"  ✓ 現價距 MA5 {above_ma5:+.1f}%，位置合理")
    elif above_ma5 > 5:
        msgs.append(f"  ✗ 現價高於 MA5 {above_ma5:.1f}%，追高風險高")
    else:  # above_ma5 < -5，已明顯跌破 MA5
        msgs.append(f"  ✗ 現價已跌破 MA5 {above_ma5:.1f}%，趨勢轉弱不宜追跌加碼")

    return score, msgs, score >= 4


def entry_signals(df):
    """新標的進場過濾器，支援兩種路徑：

    路徑A（回調進場）：在多頭趨勢中等回調至 MA5 附近再進
      → 買在「量縮綠K」的支撐位，風險低
      → 4 條件：MA5>MA10 + RSI 40~65 + 量比≥0.5 + 乖離≤5%

    路徑B（突破確認）：整理後放量突破，當天就確認進場
      → 買在「放量紅K」突破當日，搭上初升段
      → 4 條件：MA5>MA10 + RSI 50~75 + 量比≥1.5 + 漲幅≥3%

    回傳 (score, msgs)：
      score = 路徑A分數（若路徑B完全達標，score 強制為 4）
    """
    r    = df.iloc[-1]
    prev = df.iloc[-2]

    close      = r['Close']
    ma5        = r['MA5']
    rsi        = r['RSI']
    vr         = r['Vol_ratio']
    above_ma5  = (close - ma5) / ma5 * 100
    day_chg    = (close - prev['Close']) / prev['Close'] * 100

    # ── 路徑 B：突破確認（優先判斷，達標直接回傳）────────────
    b1 = r['MA5'] > r['MA10']
    b2 = 50 <= rsi <= 75
    b3 = vr >= 1.5
    b4 = day_chg >= 3.0
    if b1 and b2 and b3 and b4:
        msgs = [
            f"  🚀 突破確認進場（路徑B）",
            f"  ✓ MA5({r['MA5']:.1f}) > MA10({r['MA10']:.1f})  多頭排列",
            f"  ✓ RSI = {rsi:.1f}  突破動能健康（50~75）",
            f"  ✓ 量比 = {vr:.2f}  放量突破",
            f"  ✓ 今日漲幅 {day_chg:+.1f}%  啟動訊號",
        ]
        return 4, msgs

    # ── 路徑 A：回調進場（4 條件逐一評分）───────────────────
    score, msgs = 0, []
    msgs.append(f"  📋 回調進場（路徑A）")

    if r['MA5'] > r['MA10']:
        score += 1
        msgs.append(f"  ✓ MA5({r['MA5']:.1f}) > MA10({r['MA10']:.1f})  多頭排列")
    else:
        msgs.append(f"  ✗ MA5({r['MA5']:.1f}) < MA10({r['MA10']:.1f})  空頭排列，暫不進場")

    if 40 <= rsi <= 65:
        score += 1
        msgs.append(f"  ✓ RSI = {rsi:.1f}  健康進場區間（40~65）")
    elif rsi > 65:
        msgs.append(f"  ✗ RSI = {rsi:.1f}  過熱，等回調（需 ≤ 65）")
    else:
        msgs.append(f"  ✗ RSI = {rsi:.1f}  動能偏弱（需 ≥ 40）")

    if vr >= 0.5:
        score += 1
        msgs.append(f"  ✓ 量比 = {vr:.2f}  {'放量' if vr >= 1.2 else '成交正常（回調量縮可接受）'}")
    else:
        msgs.append(f"  ✗ 量比 = {vr:.2f}  成交過度低迷（需 ≥ 0.5）")

    if -2.0 <= above_ma5 <= 5.0 and day_chg >= -3.0:
        score += 1
        msgs.append(f"  ✓ 現價距 MA5 {above_ma5:+.1f}%，靠近支撐位（今日 {day_chg:+.1f}%）")
    elif above_ma5 < -2.0:
        msgs.append(f"  ✗ 現價跌破 MA5 {above_ma5:.1f}%，支撐已失守（需 > -2%）")
    elif day_chg < -3.0:
        msgs.append(f"  ✗ 今日跌幅 {day_chg:.1f}%，抓飛刀風險高（需 > -3%）")
    else:
        msgs.append(f"  ✗ 現價高於 MA5 {above_ma5:.1f}%，等回調至 MA5 附近（需 ≤ 5%）")
        # 提示路徑B還差多少
        if r['MA5'] > r['MA10'] and 50 <= rsi <= 75:
            missing = []
            if vr < 1.5:  missing.append(f"量比需 ≥ 1.5（現 {vr:.2f}）")
            if day_chg < 3: missing.append(f"漲幅需 ≥ 3%（現 {day_chg:+.1f}%）")
            if missing:
                msgs.append(f"  💡 突破路徑B還差：{' / '.join(missing)}")

    # ── BB 帶寬收縮（突破前兆，提示用）──────────────────────────
    squeeze, squeeze_msg = detect_bb_squeeze(df)
    if squeeze and squeeze_msg:
        msgs.append(squeeze_msg)

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

    # 盤中時改用快速掃描（分工明確：run=盤後詳細，intraday_scan=盤中快速）
    if status == "盤中":
        print("\n  ⏰ 目前為盤中，自動切換至快速盤中掃描")
        print("  （盤後分析報告請收盤後再執行）\n")
        intraday_scan()
        return

    print()
    print("=" * 55)
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

    # 預先抓好所有資料（避免重複下載）
    data_cache  = {}
    fund_cache  = {}
    for ticker, _ in all_tickers:
        data_cache[ticker] = fetch(ticker)
        if ticker in HOLDINGS:
            fund_cache[ticker] = get_fundamentals(ticker)

    status, _ = market_status()
    is_intraday = (status == "盤中")

    # ── Fugle 最新收盤價（修正 yfinance 更新延遲）──────────
    # yfinance 盤後有時幾小時才更新，Fugle 收盤後仍回傳當日最終價
    # 盤後：重算 MA5/MA10（今日收盤已是最終值，逼近 Yahoo 均線）
    # 盤中：只更新 Close，MA5/MA10 保持 yfinance 前幾日值（避免即時波動造成誤判）
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
                    data_cache[ticker] = _apply_fugle_price(df, fq["price"], is_intraday=is_intraday)

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
        tag_str = ""

        print(f"  {ticker} {name}（{label}）{tag_str}")
        stale_note = f"  ⚠ yfinance 尚未更新（昨收 {yf_close:.1f}）" if close != yf_close else ""
        print(f"  現價 {close:.1f}　買入 {buy_price:.1f}　損益 {profit_pct:+.1f}%{stale_note}")

        # 漲停偵測（台股 +10%）
        daily_chg_r  = (close - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100
        is_limit_up_r = daily_chg_r >= 9.5

        msgs = exit_signals(df, buy_price)
        showed_institutional = False
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
                    showed_institutional = True
            elif has_green and is_limit_up_r:
                # 漲停收盤：停利訊號暫緩，提示明日策略
                trim_shares = int(h['shares'] * 0.3)
                atr_r = df.iloc[-1]['ATR']
                print(f"  🚀 今日漲停收盤（{daily_chg_r:+.1f}%），停利訊號暫緩")
                print(f"  💡 明日開盤策略：")
                print(f"     高開 3% 以上 → 減碼 {trim_shares} 股鎖利")
                print(f"     平開 / 低開  → 持有，移動停損設於 {close - 2*atr_r:.1f}（現價-2ATR）")
                summary_watch.append(
                    f"{ticker} {name}（{profit_pct:+.1f}%）漲停 🚀 明日高開3%↑減碼，否則續抱"
                )
            elif has_green:
                trim_shares = int(h['shares'] * 0.3)
                summary_profit.append(
                    f"{ticker} {name}（{profit_pct:+.1f}%）現價 {close:.1f} → 減碼 ~{trim_shares} 股"
                )
            else:
                summary_watch.append(f"{ticker} {name}（{profit_pct:+.1f}%）")
        else:
            print("  ✅ 持倉正常，續抱")
            summary_ok.append(f"{ticker} {name}（{profit_pct:+.1f}%）")

        # ── 買進機會評估（無出場紅燈才顯示）──────────────────────
        if not any('🔴' in m for m in msgs):
            bp_action, bp_msgs, bp_score = buy_opportunity(df, buy_price, h.get('shares', 0))
            if bp_score > 0:
                print(f"\n  ── 買進機會 ──")
                print(f"  {bp_action}")
                for m in bp_msgs:
                    print(m)
                if bp_score >= 2:
                    if profit_pct < 0:
                        summary_avgdown.append(f"{ticker} {name}（{profit_pct:+.1f}%）{bp_action}")
                    else:
                        summary_building.append(f"{ticker} {name}（{profit_pct:+.1f}%）{bp_action}")

        fund = fund_cache.get(ticker)
        if fund:
            print("  ── 基本面（近12個月滾動，僅供參考）──")
            for f in fund:
                print(f)

        # ── 三大法人趨勢摘要（台股，有紅燈時已顯示完整表，其餘只顯示1行）──
        if ticker.endswith('.TW') and not showed_institutional:
            _print_institutional_brief(ticker.replace('.TW', ''))

        # ── 相對強度 vs 大盤 ──
        if ticker.endswith('.TW'):
            rs5, rs20 = calc_relative_strength(df)
            if rs5 is not None:
                rs_note  = "跑贏大盤↑" if rs5 > 0 else "跑輸大盤↓"
                rs20_str = f"  20日 {rs20:+.1f}%" if rs20 is not None else ""
                print(f"  相對強度：5日 {rs5:+.1f}%{rs20_str}  {rs_note}")

        print()

    divider()

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

    # ── 用最終 close 重算 MA5/MA10/Vol_MA5 ──────────────────
    # 盤後：今日收盤已是最終值，重算後 MA5/MA10 逼近 Yahoo 顯示值
    # 盤中：只更新 Close，MA5/MA10 保持前幾日收盤值（避免波動誤判 cross 方向）
    df      = _apply_fugle_price(df, close, is_intraday=(status == "盤中"))
    r       = df.iloc[-1]
    ma5     = r['MA5']
    ma10    = r['MA10']
    vol_ma5 = r['Vol_MA5']
    deviation = (close - ma5) / ma5 * 100
    if fugle_q and fugle_q.get("volume") and vol_ma5 > 0:
        vol_ratio = (fugle_q["volume"] * 1000) / vol_ma5

    # ── 盤中量能預估（按已過時間比例推算全日量比）──
    fugle_vol = fugle_q["volume"] if fugle_q and fugle_q.get("volume") else None
    if status == "盤中" and fugle_vol and vol_ma5 and vol_ma5 > 0:
        total_min   = 270
        elapsed_min = max(1, (now.hour * 60 + now.minute) - 9 * 60)
        progress    = min(elapsed_min / total_min, 1.0)
        est_full_vol   = fugle_vol / progress
        vol_est_ratio  = (est_full_vol * 1000) / vol_ma5
        vol_display    = f"{vol_est_ratio:.2f}（預估全日，已過{progress*100:.0f}%）"
    else:
        vol_est_ratio = vol_ratio
        vol_display   = f"{vol_ratio:.2f}"

    # ── 基本數據 ──
    print(f"\n  現價  {close:.2f}  （{daily_chg:+.2f}%）  {price_note}")
    if fugle_q:
        avg = fugle_q.get("avg")
        avg_note = f"  均價 {avg:.2f}  {'現價在均價之上 ↑' if close >= avg else '現價在均價之下 ↓'}" if avg else ""
        print(f"  開 {fugle_q['open']:.1f}  高 {fugle_q['high']:.1f}  低 {fugle_q['low']:.1f}{avg_note}")
    print(f"  MA5   {ma5:.2f}  MA10  {ma10:.2f}")
    print(f"  RSI   {rsi:.1f}  MACD柱  {macd_hist:+.3f}  量比  {vol_display}")
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

    # ── BB Squeeze（帶寬收縮偵測）──
    squeeze, squeeze_msg = detect_bb_squeeze(df)
    if squeeze and squeeze_msg:
        print(squeeze_msg)

    # ── 若在持倉中：顯示停損/停利/加碼訊號 ──
    holding = HOLDINGS.get(ticker)
    if holding:
        buy_price  = holding['buy_price']
        shares     = holding['shares']
        profit_pct = (close - buy_price) / buy_price * 100
        # ATR 停損線：依獲利分三段（與 exit_signals 邏輯一致）
        if profit_pct >= 20:
            atr_stop   = close - 2 * atr
            stop_label = "移動停損（現價-2ATR）"
        elif profit_pct >= 10:
            atr_stop   = buy_price + (close - buy_price) * 0.4
            stop_label = "保利停損（鎖住40%獲利）"
        else:
            atr_stop   = buy_price - 2 * atr
            stop_label = "初始停損"
        print()
        print(f"  ── 持倉資訊 ──")
        print(f"  買入 {buy_price:.2f}  持有 {shares} 股  損益 {profit_pct:+.1f}%")
        print(f"  ATR {stop_label}  {atr_stop:.2f}  {'⚠ 已跌破！' if close < atr_stop else '（未觸發）'}")

        # ── 出場/停利訊號（完整規則） ──
        exit_msgs = exit_signals(df, buy_price)
        print()
        print(f"  ── 出場訊號檢查 ──")
        # 移動停利規則說明
        print(f"  停利規則：乖離率 {deviation:+.1f}%  RSI {rsi:.1f}")
        print(f"    強力出清：乖離 >12% 且 RSI >75   → {'✅ 觸發' if deviation>12 and rsi>75 else '❌ 未到'}")
        print(f"    減碼30%：乖離 >8%  且 RSI >68   → {'✅ 觸發' if deviation>8 and rsi>68 else '❌ 未到'}")
        print(f"    留意偏熱：乖離 >5%  且 RSI >62   → {'✅ 觸發' if deviation>5 and rsi>62 and profit_pct>0 else '❌ 未到'}")
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

        # ── 持倉虧損：加碼條件檢查（對齊機械訊號 v2 規則）──────
        if profit_pct < 0:
            # ADX 計算（局部）
            def _adx_local(df_, period=14):
                h_, l_, c_ = df_['High'], df_['Low'], df_['Close']
                pdm = h_.diff(); mdm = -l_.diff()
                pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
                mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
                tr  = pd.concat([h_-l_, (h_-c_.shift()).abs(), (l_-c_.shift()).abs()], axis=1).max(axis=1)
                a   = 1 / period
                atr_ = tr.ewm(alpha=a, adjust=False).mean()
                pdi  = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr_
                mdi  = 100 * mdm.ewm(alpha=a, adjust=False).mean() / atr_
                dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
                adx_ = dx.ewm(alpha=a, adjust=False).mean()
                return float(adx_.iloc[-1]) if not adx_.empty else 0.0

            adx_local     = _adx_local(df)
            close_20d_ago = float(df['Close'].iloc[-21]) if len(df) >= 21 else close
            roc_20d_local = (close - close_20d_ago) / close_20d_ago * 100
            try:
                addon_score, _ = entry_signals(df)
            except Exception:
                addon_score = 0

            if adx_local >= 25 and roc_20d_local >= 3.0 and addon_score >= 4:
                print()
                print(f"  ── 加碼訊號 ──  ADX={adx_local:.0f}  20日{roc_20d_local:+.1f}%  score={addon_score}/4")
                print(f"  📈 進場條件回復（與觀察名單標準相同），可考慮加碼攤低成本")
            else:
                reasons = []
                if adx_local < 25:       reasons.append(f"ADX={adx_local:.0f}<25")
                if roc_20d_local < 3.0:  reasons.append(f"20日{roc_20d_local:+.1f}%<3%")
                if addon_score < 4:      reasons.append(f"score={addon_score}/4")
                print()
                print(f"  ── 加碼條件 ──  {'、'.join(reasons)}  ❌ 尚不符合")

        # ── 底部偵測 / 加碼條件（依均線結構切換）──
        ma_state, ma5_v, ma10_v, days_below = get_ma_state(df)
        strong_mode = (ma_state == 'golden' and close > ma5_v and close > ma10_v)

        if strong_mode:
            # 強勢格局 → 加碼進場條件（路徑A/B）
            e_score, e_msgs = entry_signals(df)
            print(f"\n  ── 加碼進場條件（路徑A/B）──")
            for m in e_msgs:
                print(m)
        else:
            # 非強勢（回檔/死叉/價跌均線下）→ 底部偵測
            if ma_state == 'golden':
                ma_str  = f"多頭排列（MA5 {ma5_v:.1f} > MA10 {ma10_v:.1f}），價格回落至均線下"
                caution = "回檔格局，留意支撐"
            elif ma_state == 'death_new':
                ma_str  = f"死亡交叉（MA5 {ma5_v:.1f} < MA10 {ma10_v:.1f}，已持續 {days_below} 天）"
                caution = "趨勢轉弱初期，謹慎"
            else:
                ma_str  = f"空頭排列（MA5 {ma5_v:.1f} < MA10 {ma10_v:.1f}，已持續 {days_below} 天）"
                caution = "趨勢偏空，需等反轉確認"

            print(f"\n  ── 底部偵測 ──  {ma_str}")

            # 超跌程度（永遠顯示，有分數就有意義）
            o_score, o_details, o_level = detect_oversold(df)
            print(f"  超跌程度：{o_level}（{o_score}/7 分）")
            for d in o_details:
                print(f"    {d}")

            # 賣壓衰竭訊號
            e_count, e_sigs = detect_selling_exhaustion(df)
            print(f"  賣壓衰竭：{'有訊號' if e_count > 0 else '尚無訊號'}（{e_count} 個）")
            for s in e_sigs:
                print(f"    ✓ {s}")
            if e_count == 0:
                print(f"    — 量能、K 線尚未出現止跌跡象")

            # ── 反彈確認條件 + 綜合建議 ─────────────────────────
            conf_cnt, bonus_cnt, a_msgs, confirmed = avg_down_signals(df)

            print()
            profit_pct_h = (close - buy_price) / buy_price * 100
            is_loss = profit_pct_h < 0
            pos_word = "攤平" if is_loss else "逢低加碼"

            if ma_state == 'death_old' and o_score < 4:
                # 空頭趨勢 + 超跌不足 → 等
                print(f"  ⏳ 空頭趨勢持續，超跌不足（{o_score}/7），暫不建議加碼")
            elif o_score >= 3 and confirmed:
                # 反彈已確認（罕見但強力訊號）
                print(f"  ── 反彈確認（{conf_cnt}/2，加分{bonus_cnt}/2）──")
                for m in a_msgs:
                    print(m)
                size = "積極（30%部位）" if bonus_cnt == 2 else "正常（20%部位）"
                warn = "  ⚠ 注意：仍在空頭格局，停損設嚴格" if ma_state == 'death_old' else ""
                print(f"  ✅ 超跌+反彈確認 → 可{pos_word}，{size}{warn}")
                for line in format_probe_stoploss(df):
                    print(line)
            elif o_score >= 3 and e_count >= 1:
                # 超跌+衰竭 → 少量探底，顯示確認進度 + 停損建議
                print(f"  ── 反彈確認（等待中，目前{conf_cnt}/2）──")
                for m in a_msgs:
                    print(m)
                print(f"  💡 超跌+賣壓衰竭，等收紅有量確認反彈 → 可先少量探底（5%部位）")
                for line in format_probe_stoploss(df):
                    print(line)
            elif o_score >= 3:
                # 超跌但尚無衰竭 → 觀察，不顯示確認細節
                print(f"  👀 超跌觀察中（{o_level} {o_score}/7），等賣壓衰竭訊號（量縮或長下影線）")
            else:
                print(f"  ⏳ {caution}，尚未進入超跌區（{o_score}/7），繼續觀察")

    else:
        # 不在持倉：依均線結構決定顯示內容
        ma_state, ma5_v, ma10_v, days_below = get_ma_state(df)
        strong_mode = (ma_state == 'golden' and close > ma5_v and close > ma10_v)

        print()
        if strong_mode:
            print(f"  ── 進場條件（路徑A/B）──")
            score, msgs = entry_signals(df)
            for m in msgs:
                print(m)
            # 量能突破：用預估量比判斷
            vol_brk = vol_est_ratio >= 1.5
            if score == 4:
                print(f"  ⭐ 四項全達成，可考慮進場")
            elif score == 3 and vol_brk:
                print(f"  💡 3/4 + 量比預估 {vol_est_ratio:.2f}（量能突破），可小量試單")
            elif score == 3:
                print(f"  🔍 3/4 條件達成，量能尚不足（量比 {vol_est_ratio:.2f}），繼續觀察")
            else:
                print(f"  ⏳ {score}/4 條件達成，繼續觀察")
        else:
            # 非強勢 → 底部偵測（觀察名單版）
            if ma_state == 'golden':
                ma_str  = f"多頭排列（MA5 {ma5_v:.1f} > MA10 {ma10_v:.1f}），價格回落至均線下"
                caution = "回檔格局，留意支撐"
            elif ma_state == 'death_new':
                ma_str  = f"死亡交叉（MA5 {ma5_v:.1f} < MA10 {ma10_v:.1f}，已持續 {days_below} 天）"
                caution = "趨勢轉弱初期，謹慎"
            else:
                ma_str  = f"空頭排列（MA5 {ma5_v:.1f} < MA10 {ma10_v:.1f}，已持續 {days_below} 天）"
                caution = "趨勢偏空，需等反轉確認"

            print(f"  ── 底部偵測 ──  {ma_str}")

            o_score, o_details, o_level = detect_oversold(df)
            print(f"  超跌程度：{o_level}（{o_score}/7 分）")
            for d in o_details:
                print(f"    {d}")

            e_count, e_sigs = detect_selling_exhaustion(df)
            print(f"  賣壓衰竭：{'有訊號' if e_count > 0 else '尚無訊號'}（{e_count} 個）")
            for s in e_sigs:
                print(f"    ✓ {s}")
            if e_count == 0:
                print(f"    — 量能、K 線尚未出現止跌跡象")

            conf_cnt, bonus_cnt, a_msgs, confirmed = avg_down_signals(df)

            print()
            if ma_state == 'death_old' and o_score < 4:
                print(f"  ⏳ 空頭趨勢持續，超跌不足（{o_score}/7），暫不建議進場")
            elif o_score >= 3 and confirmed:
                # 反彈已確認（罕見但強力訊號）
                print(f"  ── 反彈確認（{conf_cnt}/2，加分{bonus_cnt}/2）──")
                for m in a_msgs:
                    print(m)
                size = "積極（30%部位）" if bonus_cnt == 2 else "正常（20%部位）"
                warn = "  ⚠ 注意：仍在空頭格局，停損設嚴格" if ma_state == 'death_old' else ""
                print(f"  ✅ 超跌+反彈確認 → 可進場，{size}{warn}")
                for line in format_probe_stoploss(df):
                    print(line)
            elif o_score >= 3 and e_count >= 1:
                # 超跌+衰竭 → 少量探底，顯示確認進度 + 停損建議
                print(f"  ── 反彈確認（等待中，目前{conf_cnt}/2）──")
                for m in a_msgs:
                    print(m)
                print(f"  💡 超跌+賣壓衰竭，等收紅有量確認反彈 → 可先少量試單（5%部位）")
                for line in format_probe_stoploss(df):
                    print(line)
            elif o_score >= 3:
                # 超跌但尚無衰竭 → 觀察，不顯示確認細節
                print(f"  👀 超跌觀察中（{o_level} {o_score}/7），等賣壓衰竭訊號（量縮或長下影線）")
            else:
                print(f"  ⏳ {caution}，尚未進入超跌區（{o_score}/7），繼續觀察")

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
        # 覆蓋前先保存最後一筆原始收盤與日期（用於漲停次日日期判斷）
        orig_last_close = float(df.iloc[-1]['Close'])
        last_df_date    = df.index[-1].date()
        # 盤中：MA5/MA10 不重算；盤前/盤後：重算以取得準確均線值
        df = _apply_fugle_price(df, price, is_intraday=(status == "盤中"))
        r  = df.iloc[-1]
        vol_ma5   = r["Vol_MA5"]
        # Fugle volume 單位是張，yfinance Vol_MA5 單位是股（1張=1000股），需乘以1000換算
        vol_ratio = (vol * 1000) / vol_ma5 if (vol and vol_ma5) else r["Vol_ratio"]
        # 以即時價更新乖離率
        ma5       = r["MA5"]
        deviation = (price - ma5) / ma5 * 100
        rsi       = r["RSI"]
        atr       = r["ATR"]
        # 吊燈出場（Chandelier Exit 2x ATR，與 exit_signals 邏輯一致）
        high22   = max(float(r['High22']), price)
        atr_stop = high22 - 2 * atr

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

        # 把 Fugle 計算的 vol_est_ratio 寫回 df（yfinance Volume 與 Fugle 有 1~4% 誤差，統一用 Fugle 為準）
        if vol_ma5 and vol_ma5 > 0:
            df = df.copy()
            df.iloc[-1, df.columns.get_loc('Vol_ratio')] = vol_est_ratio

        # 判斷今天要做的事
        exit_msgs  = exit_signals(df, buy_price)
        has_red    = any("🔴" in m for m in exit_msgs)
        has_green  = any("🟢" in m for m in exit_msgs)
        has_yellow = any("🟡" in m for m in exit_msgs)

        # 法人方向修飾（台股，連續3日才算有意圖）
        idir = idays = inote = None
        if ticker.endswith('.TW'):
            idir, idays, inote = inst_direction(code)

        # 漲停偵測（台股 ±10%）
        # yfinance 對台股有 0~1 日延遲：若今日資料尚未入庫，
        # df.iloc[-1] 為昨日 row（Close 已被 Fugle 現價覆蓋），df.iloc[-2] 為前日
        # 需根據實際日期判斷，避免「漲停次日」偵測錯位
        today_date    = now_tw().date()
        yf_has_today  = (last_df_date >= today_date)

        if yf_has_today:
            # df[-1]=今日, df[-2]=昨日實際收盤, df[-3]=前日
            yest_close  = float(df.iloc[-2]['Close'])
            prev2_close = float(df.iloc[-3]['Close']) if len(df) >= 3 else yest_close
        else:
            # df[-1]=昨日（Close 已被 Fugle 覆蓋），需用覆蓋前的原始值
            yest_close  = orig_last_close
            prev2_close = float(df.iloc[-2]['Close'])

        prev_close_s      = yest_close
        day_chg_s         = (price - yest_close) / yest_close * 100
        is_limit_up       = day_chg_s >= 9.5
        prev_day_chg_s    = (yest_close - prev2_close) / prev2_close * 100
        was_limit_up_yest = prev_day_chg_s >= 9.5

        # ── 階段一：出場訊號（停損/停利）────────────────────
        show_bottom = True   # 預設顯示底部評分
        has_exit_action = False  # 是否已加入操作清單

        if price < atr_stop:
            action = f"🔴 {name} 跌破停損線 {atr_stop:.1f}，收盤前考慮出場"
            if idir == 'sell' and idays and idays >= 3:
                action += f"  ⚠ 法人同步連賣{idays}日，確認出場"
            elif idir == 'buy' and idays and idays >= 3:
                action += f"  （法人仍連買{idays}日，可設停損觀察）"
            actions_urgent.append(action)
            has_exit_action = True
            print(f"  ⚠  跌破 ATR 停損線 {atr_stop:.1f}")
            for m in exit_msgs: print(m)
            if inote:
                print(f"  法人：{inote}")

        elif has_red:
            for m in exit_msgs: print(m)
            trim = int(shares * 0.3)
            action = f"🔴 {name} 跌破雙均線，趨勢轉弱，考慮減碼 {trim} 股"
            print(f"  ⛔ 趨勢轉弱，建議考慮減碼約 {trim} 股")
            if idir == 'sell' and idays and idays >= 3:
                print(f"  ⚠  法人連賣{idays}日，出場訊號加強")
                action += f"  ⚠ 法人連賣{idays}日"
            elif idir == 'buy' and idays and idays >= 3:
                print(f"  💡 法人仍連買{idays}日，破線但籌碼未鬆，可設停損觀察")
            actions_urgent.append(action)
            has_exit_action = True

        elif is_limit_up and has_green:
            trim = int(shares * 0.3)
            print(f"  🚀 漲停板（{day_chg_s:+.1f}%）強勢！技術面偏熱但漲停代表買盤積極")
            for m in exit_msgs: print(m)
            print(f"  ⏸  停利訊號暫緩：漲停板當日不追賣，高機率明日仍強")
            print(f"  💡 明日策略：")
            print(f"     開盤高開 3% 以上 → 可減碼 {trim} 股（{int(trim*price):,} 元）鎖利")
            print(f"     開盤平開或低開   → 持有觀察，設移動停損（現價 - 2ATR = {price - 2*atr:.1f}）")
            action = f"🚀 {name} 漲停強勢，停利暫緩，明日視開盤再決定"
            actions_watch.append(action)
            show_bottom = False   # 漲停獲利，不顯示底部評分

        elif has_green:
            trim = int(shares * 0.3)
            for m in exit_msgs: print(m)
            if idir == 'sell' and idays and idays >= 3:
                action = f"🟢 {name} 停利訊號＋法人連賣{idays}日，建議收盤前賣 {trim} 股（{price:.1f}）"
                actions_urgent.append(action)
                print(f"  ✅ 停利訊號 + 法人連賣{idays}日，建議今天執行減碼 {trim} 股")
            elif ask_pct is not None and ask_pct <= 40:
                action = f"🟢 {name} 停利訊號＋內盤偏重，建議收盤前賣 {trim} 股（{price:.1f}）"
                actions_urgent.append(action)
                print(f"  ✅ 停利訊號 + 內盤偏重，建議今天執行減碼 {trim} 股")
            else:
                action = f"🟢 {name} 停利訊號，可掛 {trim} 股賣單，外盤仍強可再觀察"
                actions_watch.append(action)
                print(f"  ⚡ 停利訊號，外盤仍強，可掛單等自動成交")
            show_bottom = False   # 停利模式，不顯示底部評分

        elif has_yellow:
            for m in exit_msgs: print(m)
            if idir == 'buy' and idays and idays >= 3:
                print(f"  💡 法人連買{idays}日，偏熱但籌碼穩，降低緊迫性，繼續觀察")
                actions_ok.append(f"{name} 偏熱但法人撐盤")
            elif ask_pct is not None and ask_pct <= 40:
                action = f"🟡 {name} 偏熱 + 內盤偏重，可考慮減碼 {int(shares*0.3)} 股"
                actions_watch.append(action)
                print(f"  🟡 偏熱警示 + 內盤偏重，可考慮掛單")
            else:
                print(f"  🟡 偏熱但外盤尚可，繼續觀察")
                actions_ok.append(f"{name} 偏熱觀察中")
            show_bottom = False   # 偏熱獲利，不顯示底部評分

        elif was_limit_up_yest and not is_limit_up and day_chg_s < 3.0:
            trim = int(shares * 0.3)
            print(f"  🟠 漲停次日（昨 {prev_day_chg_s:+.1f}%），今未延續（{day_chg_s:+.1f}%）")
            if ask_pct is not None and ask_pct <= 40:
                action = f"🟠 {name} 漲停次日＋內盤偏重，買盤退潮，建議減碼 {trim} 股（{price:.1f}）"
                actions_urgent.append(action)
                print(f"  ⚠  內盤偏重（外盤 {ask_pct:.0f}%），建議減碼約 {trim} 股")
            else:
                ob_str = f"{ask_pct:.0f}%" if ask_pct is not None else "N/A"
                action = f"🟡 {name} 漲停次日未延續，外盤 {ob_str}，留意是否減碼"
                actions_watch.append(action)
                print(f"  👀 外盤 {ob_str} 尚可，繼續觀察，若轉內盤則減碼")
            show_bottom = False

        else:
            print(f"  ✅ 無出場訊號")

        # ── 階段二：底部偵測 / 加碼條件（依均線結構切換）──────────────────
        if show_bottom:
            ma_state, ma5_v, ma10_v, days_below = get_ma_state(df)
            strong_mode = (ma_state == 'golden' and price > ma5_v and price > ma10_v)
            ob_str = f"{ask_pct:.0f}%" if ask_pct is not None else "N/A"

            if strong_mode:
                # 強勢格局 → 加碼進場條件（路徑A/B）
                e_score, e_msgs = entry_signals(df)
                for m in e_msgs: print(m)
                if not has_exit_action:
                    if e_score >= 4:
                        if ask_pct is not None and ask_pct >= 55:
                            actions_urgent.append(f"✅ {name} 加碼訊號 4/4，外盤 {ask_pct:.0f}% 確認，可進場")
                            print(f"  ✅ 加碼條件 4/4 + 外盤確認，可掛單")
                        else:
                            actions_watch.append(f"{name} 加碼 4/4，外盤 {ob_str} 待確認")
                            print(f"  ⏳ 加碼條件 4/4，外盤待確認（{ob_str}）")
                    elif e_score == 3:
                        half_s = max(1, int(shares * 0.15))
                        if ask_pct is not None and ask_pct >= 50:
                            actions_watch.append(f"{name} 加碼 3/4，少量試單約 {half_s} 股（外盤 {ob_str}）")
                            print(f"  🔸 加碼 3/4 + 外盤尚可，可少量試單約 {half_s} 股")
                        else:
                            print(f"  🔸 加碼 3/4，外盤偏弱，繼續等待")
                            actions_ok.append(f"{name} 加碼 3/4 觀察中")
                    else:
                        print(f"  ✅ 無加碼訊號，續抱")
                        actions_ok.append(f"{name}")

            else:
                # 非強勢 → 底部偵測
                if ma_state == 'golden':
                    ma_str  = f"多頭排列（MA5 {ma5_v:.1f} > MA10 {ma10_v:.1f}），價格回落至均線下"
                    caution = "回檔格局，留意支撐"
                elif ma_state == 'death_new':
                    ma_str  = f"死亡交叉（MA5 {ma5_v:.1f} < MA10 {ma10_v:.1f}，已持續 {days_below} 天）"
                    caution = "趨勢轉弱初期，謹慎"
                else:
                    ma_str  = f"空頭排列（MA5 {ma5_v:.1f} < MA10 {ma10_v:.1f}，已持續 {days_below} 天）"
                    caution = "趨勢偏空，需等反轉確認"

                # ── 持倉虧損：加碼條件檢查（對齊機械訊號 v2 規則）──
                is_loss = bool(buy_price and (price - buy_price) / buy_price * 100 < 0)
                if is_loss:
                    def _adx_intra(df_, period=14):
                        h_, l_, c_ = df_['High'], df_['Low'], df_['Close']
                        pdm = h_.diff(); mdm = -l_.diff()
                        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
                        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
                        tr  = pd.concat([h_-l_, (h_-c_.shift()).abs(), (l_-c_.shift()).abs()], axis=1).max(axis=1)
                        a_  = 1 / period
                        atr_ = tr.ewm(alpha=a_, adjust=False).mean()
                        pdi_ = 100 * pdm.ewm(alpha=a_, adjust=False).mean() / atr_
                        mdi_ = 100 * mdm.ewm(alpha=a_, adjust=False).mean() / atr_
                        dx_  = 100 * (pdi_ - mdi_).abs() / (pdi_ + mdi_).replace(0, np.nan)
                        adx_ = dx_.ewm(alpha=a_, adjust=False).mean()
                        return float(adx_.iloc[-1]) if not adx_.empty else 0.0

                    adx_i         = _adx_intra(df)
                    c20_ago       = float(df['Close'].iloc[-21]) if len(df) >= 21 else price
                    roc_20_i      = (price - c20_ago) / c20_ago * 100
                    try:
                        addon_sc, _ = entry_signals(df)
                    except Exception:
                        addon_sc = 0

                    if adx_i >= 25 and roc_20_i >= 3.0 and addon_sc >= 4:
                        print(f"  ── 加碼訊號 ──  ADX={adx_i:.0f}  20日{roc_20_i:+.1f}%  score={addon_sc}/4")
                        print(f"  📈 進場條件回復（與觀察名單標準相同），可考慮加碼")
                        actions_watch.append(f"📈 {name} 加碼條件回復 ADX={adx_i:.0f} 20日{roc_20_i:+.1f}%")
                    else:
                        reasons_i = []
                        if adx_i < 25:      reasons_i.append(f"ADX={adx_i:.0f}<25")
                        if roc_20_i < 3.0:  reasons_i.append(f"20日{roc_20_i:+.1f}%<3%")
                        if addon_sc < 4:    reasons_i.append(f"score={addon_sc}/4")
                        print(f"  ── 加碼條件 ──  {'、'.join(reasons_i)}  ❌ 尚不符合")

                # ── 簡潔底部結論 ──────────────────────────────────
                is_loss  = bool(buy_price and (price - buy_price) / buy_price * 100 < 0)
                re_word  = "買回" if has_exit_action else ("攤平" if is_loss else "進場")
                section  = "買回時機" if has_exit_action else "底部時機"

                o_score, _, o_level   = detect_oversold(df)
                ec_count, ec_sigs     = detect_selling_exhaustion(df)
                _, _, _, confirmed    = avg_down_signals(df)
                sl_info               = calc_probe_stoploss(df)

                # 狀態一行描述
                if o_score >= 5:
                    state_str = f"嚴重超跌（{o_score}/7）"
                elif o_score >= 3:
                    state_str = f"明顯超跌（{o_score}/7）"
                else:
                    state_str = f"尚未超跌（{o_score}/7）"

                if ec_count >= 1:
                    exhaust_str = f"，{'、'.join(ec_sigs[:2][:30])}"
                else:
                    exhaust_str = "，尚無止跌訊號"

                print(f"  ── {section} ──  現況：{state_str}{exhaust_str}")

                # ── 策略C低谷反彈（與回測相同規則：收<MA10 + 3條件中≥2）──
                # C3：盤中看外盤比，盤後/盤前看法人（對齊機械訊號邏輯）
                inst_buying_c = (idir == 'buy' and idays is not None and idays >= 1)
                if status == '盤中':
                    c3_ok    = ask_pct is not None and ask_pct > 55
                    c3_label = f"外盤{ask_pct:.0f}%>55%" if ask_pct is not None else "外盤?"
                else:
                    c3_ok    = inst_buying_c
                    c3_label = f"法人買超{idays}日" if inst_buying_c else "法人未買"
                c_conds = [o_score >= 3, ec_count >= 1, c3_ok]
                below_ma10_c = price < ma10_v
                c_signal = below_ma10_c and sum(c_conds) >= 2
                if c_signal:
                    _cp = []
                    if c_conds[0]: _cp.append(f"超跌{o_score}/7")
                    if c_conds[1]: _cp.append(f"衰竭{ec_count}次")
                    if c_conds[2]: _cp.append(c3_label)
                    _cstr = " ".join(_cp)
                    print(f"  🔻 策略C低谷訊號（{_cstr}）  目標 MA10 {ma10_v:.1f}")
                    actions_watch.append(f"🔻 {name} 策略C低谷訊號（{_cstr}）")

                # 一句建議 + 停損
                if o_score >= 3 and confirmed:
                    # 最強：超跌+反彈確認
                    print(f"  🔻 低谷反彈確認  停損 {sl_info['stop_price']:.1f}（破了出場）  目標 MA10 {sl_info['target_price']:.1f}（風報比 1:{sl_info['rr_ratio']:.1f}）")
                    if ask_pct is not None and ask_pct >= 55:
                        if not has_exit_action:
                            actions_urgent.append(f"🔻 {name} 低谷反彈確認，可{re_word}，停損 {sl_info['stop_price']:.1f}")
                        else:
                            actions_watch.append(f"🔻 {name} 出場後可考慮{re_word}，低谷已確認，停損 {sl_info['stop_price']:.1f}")
                    else:
                        print(f"  （外盤偏弱 {ob_str}，可等外盤 >55% 再買）")
                        if not has_exit_action:
                            actions_watch.append(f"🔻 {name} 低谷確認，外盤待確認（停損 {sl_info['stop_price']:.1f}）")
                        else:
                            actions_watch.append(f"🔻 {name} 低谷確認，出場後留意{re_word}機會")

                elif o_score >= 3 and ec_count >= 1:
                    # 超跌+衰竭，等確認
                    print(f"  🔻 低谷訊號（超跌{o_score}/7 衰竭{ec_count}次）  可少量試水（5%部位）  停損 {sl_info['stop_price']:.1f}  等收紅有量確認")
                    if ask_pct is not None and ask_pct >= 55:
                        if not has_exit_action:
                            actions_watch.append(f"🔻 {name} 低谷訊號，可少量{re_word}（5%），停損 {sl_info['stop_price']:.1f}")
                        else:
                            actions_watch.append(f"🔻 {name} 低谷訊號，出場後可少量試買，停損 {sl_info['stop_price']:.1f}")
                    else:
                        print(f"  （外盤偏弱 {ob_str}，可再等等）")
                        # 不論外盤強弱都進摘要，讓使用者知道有低谷機會
                        if not has_exit_action:
                            actions_watch.append(f"🔻 {name} 低谷訊號（超跌{o_score}/7 衰竭{ec_count}），外盤偏弱等確認")
                        else:
                            actions_watch.append(f"🔻 {name} 低谷訊號，出場後留意{re_word}（停損 {sl_info['stop_price']:.1f}）")

                elif o_score >= 3:
                    # 超跌但尚無止跌
                    print(f"  ⏳ 還不要買  等量縮或長下影線出現  停損參考 {sl_info['stop_price']:.1f}")
                    actions_ok.append(f"{name} 超跌{o_score}/7 等止跌訊號")

                elif c_signal:
                    # 3選2達標但超跌不足3（衰竭/外盤/法人兩項成立）
                    _met = [c3_label if c_conds[2] else "", f"衰竭{ec_count}次" if c_conds[1] else ""]
                    _met_str = " ".join(x for x in _met if x)
                    print(f"  💡 低谷待確認（{_met_str}，超跌{o_score}/7 尚不足3）  停損參考 {sl_info['stop_price']:.1f}")
                    if not has_exit_action:
                        actions_ok.append(f"{name} 低谷待確認（{_met_str}）")

                else:
                    # 未超跌
                    print(f"  ⏳ 還不是時候  {caution}")
                    if not has_exit_action:
                        actions_ok.append(f"{name} 等待時機")

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


def watchlist_scan():
    """觀察名單專屬掃描：列出所有觀察名單的即時狀態與進場條件
    不混入持倉資訊，方便盤中快速判斷有無新進場機會
    """
    _fugle_cache.clear()
    now    = now_tw()
    status, status_note = market_status()
    minutes = now.hour * 60 + now.minute

    if status == "盤中":
        title     = "觀察名單掃描  盤中即時"
        time_note = f"距收盤 {max(0,(13*60+30)-minutes)} 分鐘"
    elif status == "盤後":
        title     = "觀察名單掃描  今日收盤回顧"
        time_note = "資料為今日最終收盤（Fugle）"
    else:
        title     = "觀察名單掃描  昨日收盤回顧"
        time_note = "台股尚未開盤，顯示昨日收盤資料"

    print()
    print("=" * 55)
    print(f"   {title}")
    print(f"   {now.strftime('%Y-%m-%d  %H:%M')}  {time_note}")
    print(f"   共 {len(WATCHLIST.get('tw', []))} 支觀察標的")
    print("=" * 55)

    entry_candidates = []   # 4/4 或量能突破
    watching_list    = []   # 其他

    for code in WATCHLIST.get("tw", []):
        ticker = code + ".TW"
        df = fetch(ticker, silent=True)
        if df is None:
            ticker = code + ".TWO"
            df = fetch(ticker, silent=True)
        if df is None:
            print(f"\n  ⚠  {code}：無法取得資料")
            continue

        # Fugle 即時報價
        fugle_d = get_fugle_quote(code)
        fq      = parse_fugle_price(fugle_d)

        if fq and fq.get("price" if status == "盤中" else "close_price"):
            price_raw = fq["price"] if status == "盤中" else (fq.get("close_price") or fq["price"])
        else:
            price_raw = df.iloc[-1]['Close']   # fallback yfinance

        # 盤後：重算 MA5/MA10；盤中：只更新 Close
        df = _apply_fugle_price(df, price_raw, is_intraday=(status == "盤中"))

        prev_close = df.iloc[-2]['Close']
        day_chg    = (price_raw - prev_close) / prev_close * 100
        ma5        = df.iloc[-1]['MA5']
        deviation  = (price_raw - ma5) / ma5 * 100
        rsi        = df.iloc[-1]['RSI']
        vol_ma5    = df.iloc[-1]['Vol_MA5']

        # 量能（盤中推估 / 盤後直接用）
        vol_raw = fq.get("volume") or 0 if fq else 0
        if status == "盤中" and vol_raw and vol_ma5:
            elapsed_min  = max(1, minutes - 9 * 60)
            progress     = min(elapsed_min / 270, 1.0)
            est_ratio    = (vol_raw / progress * 1000) / vol_ma5
            vol_note     = f"預估量比 {est_ratio:.2f}（已過{progress*100:.0f}%）"
        else:
            est_ratio    = (vol_raw * 1000) / vol_ma5 if (vol_raw and vol_ma5) else df.iloc[-1]['Vol_ratio']
            vol_note     = f"量比 {est_ratio:.2f}"

        # 內外盤
        ask_pct = fq.get("ask_pct") if fq else None
        if ask_pct is not None:
            if ask_pct >= 60:   ob = f"外盤 {ask_pct:.0f}% 偏多↑"
            elif ask_pct <= 40: ob = f"外盤 {ask_pct:.0f}% 偏空↓"
            else:               ob = f"外盤 {ask_pct:.0f}% 平衡"
        else:
            ob = ""

        # 漲跌停過濾
        limit_tag = ""
        if day_chg >= 9.5:
            limit_tag = "  🔴 漲停"
        elif day_chg <= -9.5:
            limit_tag = "  🟢 跌停"

        # ── 起勢訊號（entry_signals 路徑A/B）────────────────────
        score, msgs = entry_signals(df)
        is_vol_brk  = est_ratio >= 1.5 and (ask_pct is None or ask_pct >= 50)

        # ── 低谷反彈訊號（策略C：超跌 + 賣壓衰竭，收盤需在MA10以下）──
        try:
            o_score, _, _ = detect_oversold(df)
            e_count, _    = detect_selling_exhaustion(df)
        except Exception:
            o_score = e_count = 0
        ma10       = float(df.iloc[-1]['MA10'])
        below_ma10 = price_raw < ma10
        c_signal   = below_ma10 and o_score >= 3 and e_count >= 1

        # 輸出標頭
        src = "Fugle" if fq else "yfinance"
        print(f"\n  {ticker}  現價 {price_raw:.1f}（{day_chg:+.1f}%）  {ob}  {vol_note}{limit_tag}")
        print(f"  乖離率 {deviation:+.1f}%  RSI {rsi:.1f}  MA10 {ma10:.1f}  訊號 {score}/4  （{src}）")
        for m in msgs:
            print(m)

        # 低谷訊號補充輸出
        if c_signal:
            gap_pct = (ma10 - price_raw) / ma10 * 100
            print(f"  🔻 低谷反彈訊號：超跌 {o_score}/7，賣壓衰竭 {e_count} 次，距 MA10 {gap_pct:.1f}%")

        # 結論
        if limit_tag:
            print(f"  ⏸  今日{limit_tag.strip()}，明日再評估")
        elif c_signal and score >= 3:
            print(f"  🔥 低谷 + 起勢雙訊號，優先關注")
            entry_candidates.append(f"🔥 {ticker} 低谷+起勢雙訊號（超跌{o_score}/7 衰竭{e_count} 進場{score}/4）")
        elif c_signal:
            c_str = f"超跌{o_score}/7 衰竭{e_count}次"
            print(f"  🔻 低谷反彈機會，可小量佈局")
            entry_candidates.append(f"🔻 {ticker} 低谷反彈（{c_str}）")
        elif score == 4:
            print(f"  ⭐ 四項全達成，可考慮進場")
            entry_candidates.append(f"⭐ {ticker} 四項全達成（{day_chg:+.1f}%）")
        elif score == 3 and is_vol_brk:
            print(f"  💡 量能突破 + 3/4，可小量試單")
            entry_candidates.append(f"💡 {ticker} 量能突破試單（量比{est_ratio:.2f}，{day_chg:+.1f}%）")
        elif score == 3:
            print(f"  🔍 3/4 條件達成，量能待加強")
            watching_list.append(f"{ticker} 3/4（量比{est_ratio:.2f}）")
        elif score == 2 and is_vol_brk:
            print(f"  👀 量能突破但訊號弱（2/4），謹慎觀察")
            watching_list.append(f"{ticker} 量能突破但2/4")
        else:
            print(f"  ⏳ {score}/4，持續觀察")

    # ── 摘要 ──
    divider()
    print()
    if entry_candidates:
        print("  【今日可考慮進場】")
        for a in entry_candidates:
            print(f"  → {a}")
        print()
    if watching_list:
        print("  【接近但尚未就緒】")
        for a in watching_list:
            print(f"  ○ {a}")
        print()
    if not entry_candidates and not watching_list:
        print("  今日觀察名單無明確進場訊號，繼續等待")
        print()
    print("=" * 55)
    print()


# ============================================================
#  機械訊號掃描（v2 邏輯，非黑即白）
# ============================================================
def mechanical_scan():
    """按照 backtest_v2 的機械化規則掃描持倉與觀察名單。

    進場條件（無持倉）：
      ADX ≥ 25 + 近20日漲幅 ≥ 3% + entry_signals score ≥ 4（路徑A或B）

    減碼 / 出場條件（有持倉，按優先序）：
      🔴 出場：吊燈線（High22-2×ATR） / 單日跳空<-4% / 法人連賣≥3日 / MA5+MA10雙線跌破
      🟡 減碼：乖離>12%+RSI>75（強力）/ 乖離>8%+RSI>68（第一次）

    結果只顯示訊號，不給建議文字。
    """
    import numpy as np
    from collections import defaultdict
    import requests as _req

    CHANDELIER_MULT = 2.0
    ADX_TREND       = 25
    ROC_20_MIN      = 3.0
    INST_SELL_DAYS  = 3
    TRIM_DEV        = 8.0
    TRIM_RSI        = 68.0
    STRONG_DEV      = 12.0
    STRONG_RSI      = 75.0

    # ── ADX 計算 ──────────────────────────────────────────────
    def _calc_adx(df, period=14):
        high, low, close = df['High'], df['Low'], df['Close']
        pdm = high.diff()
        mdm = -low.diff()
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
        tr  = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        a   = 1 / period
        atr = tr.ewm(alpha=a, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr
        mdi = 100 * mdm.ewm(alpha=a, adjust=False).mean() / atr
        dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        adx = dx.ewm(alpha=a, adjust=False).mean()
        return float(adx.iloc[-1]) if not adx.empty else 0.0

    # ── 法人最近一日是否淨買超（盤後/盤前用）─────────────────
    def _inst_buy_latest(code):
        """回傳 (bool, str)：最近一個交易日外資是否淨買超"""
        start = (now_tw() - timedelta(days=10)).strftime('%Y-%m-%d')
        params = {
            'dataset':    'TaiwanStockInstitutionalInvestorsBuySell',
            'data_id':    code,
            'start_date': start,
        }
        if FINMIND_TOKEN:
            params['token'] = FINMIND_TOKEN
        try:
            resp = _req.get('https://api.finmindtrade.com/api/v4/data',
                            params=params, timeout=15)
            body = resp.json()
            if not (body.get('status') == 200 and body.get('data')):
                return False, '法人?'
            by_date = defaultdict(int)
            for r in body['data']:
                if r.get('name') == 'Foreign_Investor':
                    net = ((r.get('buy') or 0) - (r.get('sell') or 0)) // 1000
                    by_date[r['date']] += net
            if not by_date:
                return False, '法人?'
            latest_date = sorted(by_date.keys())[-1]
            net = by_date[latest_date]
            buying = net > 0
            label = f"法人{latest_date[5:]} {'買+' if buying else '賣-'}{abs(net)}張"
            return buying, label
        except Exception:
            return False, '法人?'

    # ── 法人連賣天數（簡單計數，對齊 v2）─────────────────────
    def _inst_sell_streak(code):
        start = (now_tw() - timedelta(days=20)).strftime('%Y-%m-%d')
        params = {
            'dataset':    'TaiwanStockInstitutionalInvestorsBuySell',
            'data_id':    code,
            'start_date': start,
        }
        if FINMIND_TOKEN:
            params['token'] = FINMIND_TOKEN
        try:
            resp = _req.get('https://api.finmindtrade.com/api/v4/data',
                            params=params, timeout=15)
            body = resp.json()
            if not (body.get('status') == 200 and body.get('data')):
                return 0
            by_date = defaultdict(int)
            for r in body['data']:
                if r.get('name') == 'Foreign_Investor':
                    net = ((r.get('buy') or 0) - (r.get('sell') or 0)) // 1000
                    by_date[r['date']] = net
            streak = 0
            for d in sorted(by_date.keys(), reverse=True):
                if by_date[d] < 0:
                    streak += 1
                else:
                    break
            return streak
        except Exception:
            return 0

    # ── 掃描 ──────────────────────────────────────────────────
    now    = now_tw()
    status, _ = market_status()

    title = {
        '盤中': '機械訊號  盤中即時',
        '盤後': '機械訊號  今日收盤',
    }.get(status, '機械訊號  昨日收盤')

    print()
    print("=" * 60)
    print(f"  🤖 {title}")
    print(f"  規則：v2 機械化邏輯（ADX≥25 + score≥4 進場，嚴格出場）")
    print("=" * 60)

    entry_hits  = []   # 觀察名單新進場
    addon_hits  = []   # 持倉虧損回復進場條件（加碼）
    exit_hits   = []
    trim_hits   = []
    hold_ok     = []
    watch_miss  = []

    holdings  = HOLDINGS
    watchlist = WATCHLIST

    # ── 資料取得輔助（依盤前/盤中/盤後套用不同策略）──────────
    def _get_df(ticker):
        code_  = ticker.replace('.TW', '').replace('.TWO', '')
        df_    = fetch(ticker, silent=True)
        if df_ is None:
            return None, None

        # 先清除 yfinance 可能塞入的假資料列（四價完全相同）
        last = df_.iloc[-1]
        is_fake = (float(last['Open']) == float(last['High']) ==
                   float(last['Low'])  == float(last['Close']))
        if is_fake:
            df_ = df_.iloc[:-1].copy()
            df_ = calculate_indicators(df_)

        fq_ = parse_fugle_price(get_fugle_quote(code_))

        def _patch_ohlcv(df__, fq__):
            """補 Fugle 的 Open/High/Low/Volume 並重算指標"""
            df__ = df__.copy()
            if fq__.get('open')  : df__.iloc[-1, df__.columns.get_loc('Open')]   = float(fq__['open'])
            if fq__.get('high')  : df__.iloc[-1, df__.columns.get_loc('High')]   = float(fq__['high'])
            if fq__.get('low')   : df__.iloc[-1, df__.columns.get_loc('Low')]    = float(fq__['low'])
            if fq__.get('volume'): df__.iloc[-1, df__.columns.get_loc('Volume')] = float(fq__['volume']) * 1000
            return calculate_indicators(df__)

        if status == '盤前':
            # 盤前：用 Fugle closePrice 補昨日（或最近一個交易日）收盤 + OHLV
            if fq_ and fq_.get('close_price'):
                df_ = _apply_fugle_price(df_, fq_['close_price'], is_intraday=False)
                df_ = _patch_ohlcv(df_, fq_)
            return df_, fq_

        elif status == '盤中':
            # 盤中：Fugle 即時報價覆蓋，High/Low/Open 也補入，量比依進度估算
            if fq_ and fq_.get('price'):
                df_ = _apply_fugle_price(df_, fq_['price'], is_intraday=True)
                # 補入即時 Open/High/Low（_apply_fugle_price 只填 prev_close）
                df_ = df_.copy()
                if fq_.get('open'): df_.iloc[-1, df_.columns.get_loc('Open')] = float(fq_['open'])
                if fq_.get('high'): df_.iloc[-1, df_.columns.get_loc('High')] = float(fq_['high'])
                if fq_.get('low') : df_.iloc[-1, df_.columns.get_loc('Low')]  = float(fq_['low'])
                # 重算 ATR/High22/BB/KD 等依賴 H/L 的指標
                # 先保留 Vol_MA5（_apply_fugle_price 已設為前一日穩定值）
                vol_ma5_stable = float(df_.iloc[-1]['Vol_MA5']) if df_.iloc[-1]['Vol_MA5'] else 0
                df_ = calculate_indicators(df_)
                # calculate_indicators 用當日不完整量算的 Vol_ratio 不可信，改用估算值
                vol_raw = fq_.get('volume') or 0
                if vol_raw and vol_ma5_stable:
                    minutes_ = now.hour * 60 + now.minute
                    elapsed  = max(1, minutes_ - 9 * 60)
                    progress = min(elapsed / 270, 1.0)
                    est_vr   = (vol_raw / progress * 1000) / vol_ma5_stable
                    df_.iloc[-1, df_.columns.get_loc('Vol_ratio')]  = est_vr
                    df_.iloc[-1, df_.columns.get_loc('Vol_MA5')]    = vol_ma5_stable
            return df_, fq_

        else:  # 盤後
            if fq_ and fq_.get('close_price'):
                df_ = _apply_fugle_price(df_, fq_['close_price'], is_intraday=False)
                df_ = _patch_ohlcv(df_, fq_)
            return df_, fq_

    # ── 持倉：出場 / 減碼掃描 ────────────────────────────────
    if holdings:
        print(f"\n  ── 持倉出場檢查 ──────────────────────────────")
        for ticker, h in holdings.items():
            buy_price = float(h.get('buy_price', 0))
            name      = h.get('name', ticker)
            code      = ticker.replace('.TW', '').replace('.TWO', '')

            df, fq = _get_df(ticker)
            if df is None or len(df) < 10:
                print(f"  {ticker} {name:<6}  ⚠ 無法取得資料")
                continue

            row      = df.iloc[-1]
            prev     = df.iloc[-2]
            close    = float(row['Close'])
            atr      = float(row['ATR'])
            high22   = float(row['High22'])
            atr_stop = high22 - CHANDELIER_MULT * atr
            ma5      = float(row['MA5'])
            ma10     = float(row['MA10'])
            rsi      = float(row['RSI'])
            deviation = (close - ma5) / ma5 * 100
            daily_chg = (close - float(prev['Close'])) / float(prev['Close']) * 100
            below_ma5  = close < ma5
            below_ma10 = close < ma10
            profit_pct = (close - buy_price) / buy_price * 100 if buy_price else 0

            # 20日漲幅 & ADX（進場條件用，持倉回復判斷）
            close_20d_ago = float(df['Close'].iloc[-21]) if len(df) >= 21 else close
            roc_20d_h     = (close - close_20d_ago) / close_20d_ago * 100
            adx_h         = _calc_adx(df)

            # 判斷訊號
            signal = None
            detail = ""
            recovery_detail = ""

            # 低谷訊號（策略C邏輯）：虧損且在MA10下方才偵測
            valley_detail = ""
            valley_lines  = []   # 逐項明細，稍後換行印出
            if profit_pct < 0 and below_ma10:
                try:
                    o_score, o_detail, _ = detect_oversold(df)
                    e_count, e_signals   = detect_selling_exhaustion(df)
                except Exception:
                    o_score = e_count = 0
                    o_detail = e_signals = []

                # 超跌逐項
                o_mark = "✓" if o_score >= 3 else "✗"
                valley_lines.append(f"       超跌評分 {o_mark} {o_score}/7（需≥3）")
                for d in o_detail:
                    valley_lines.append(f"         {d.strip()}")

                # 衰竭逐項
                e_mark = "✓" if e_count >= 1 else "✗"
                valley_lines.append(f"       賣壓衰竭 {e_mark} {e_count}項（需≥1）")

                # 衰竭各條件詳細
                r2   = df.iloc[-1]
                pr2  = df.iloc[-2]
                pr3  = df.iloc[-3] if len(df) >= 3 else pr2
                c2   = float(r2['Close']); o2 = float(r2['Open'])
                h2   = float(r2['High']);  l2 = float(r2['Low'])
                vr2  = float(r2['Vol_ratio'])
                pc2  = float(pr2['Close']); pc3 = float(pr3['Close'])

                ok_vol = vr2 < 0.7
                valley_lines.append(f"         ① 量縮（量比<0.7）：{vr2:.2f}  {'✓' if ok_vol else '✗'}")

                cr = h2 - l2
                ls = min(o2, c2) - l2
                sp = ls / cr * 100 if cr > 0 else 0
                mp = (h2 + l2) / 2
                ok_shadow = sp > 60 and c2 > mp
                valley_lines.append(f"         ② 長下影線：下影{sp:.0f}%  收{c2:.1f} vs 中點{mp:.1f}  {'✓' if ok_shadow else '✗'}")

                h1m = float(df['MACD_hist'].iloc[-3])
                h2m = float(df['MACD_hist'].iloc[-2])
                h3m = float(df['MACD_hist'].iloc[-1])
                ok_macd = h1m < 0 and h2m < 0 and h3m < 0 and abs(h3m) < abs(h2m) < abs(h1m)
                valley_lines.append(f"         ③ MACD空頭收斂：{h1m:.3f}→{h2m:.3f}→{h3m:.3f}  {'✓' if ok_macd else '✗'}")

                pd_chg = (pc2 - pc3) / pc3 * 100 if pc3 else 0
                td_chg = (c2 - pc2) / pc2 * 100 if pc2 else 0
                ok_bounce = pd_chg <= -4 and td_chg > 0
                valley_lines.append(f"         ④ 急跌隔日收紅：前日{pd_chg:.1f}%  今日{td_chg:.1f}%  {'✓' if ok_bounce else '✗（需前日≤-4%今收紅）'}")

                ok_narrow = pd_chg < -2 and -2 < td_chg < 0 and td_chg > pd_chg
                valley_lines.append(f"         ⑤ 跌幅縮小：前日{pd_chg:.1f}%  今日{td_chg:.1f}%  {'✓' if ok_narrow else '✗（需前日<-2%今日跌幅縮小）'}")

                # ── 第③條件：盤中看外盤比，盤後/盤前看法人 ──
                if status == '盤中':
                    ask_pct_val = fq.get('ask_pct') if fq else None
                    if ask_pct_val is not None:
                        ok_c3   = ask_pct_val > 55
                        c3_label = f"外盤比{ask_pct_val:.0f}%（需>55%）"
                    else:
                        ok_c3   = False
                        c3_label = "外盤比 無資料"
                else:
                    ok_c3, c3_label = _inst_buy_latest(code)

                c3_mark = "✓" if ok_c3 else "✗"
                valley_lines.append(f"       法人/外盤 {c3_mark} {c3_label}")

                c_met = sum([o_score >= 3, e_count >= 1, ok_c3])
                if c_met >= 2:
                    valley_detail = "  🟢 低谷訊號"
                elif c_met == 1:
                    valley_detail = f"  💡 低谷待確認（{c_met}/3，差1項）"
                else:
                    valley_detail = f"  ⏳ 低谷條件不足（{c_met}/3）"

            if deviation > STRONG_DEV and rsi > STRONG_RSI:
                signal = "🟡 強力減碼"
                detail = f"乖離{deviation:.1f}% RSI{rsi:.1f}"
                trim_hits.append((ticker, name, signal, detail, close, profit_pct))
            elif deviation > TRIM_DEV and rsi > TRIM_RSI:
                signal = "🟡 減碼30%"
                detail = f"乖離{deviation:.1f}% RSI{rsi:.1f}"
                trim_hits.append((ticker, name, signal, detail, close, profit_pct))
            elif close < atr_stop:
                signal = "🔴 出場"
                detail = f"吊燈線（22日高{high22:.1f}-2ATR={atr_stop:.1f}）"
                exit_hits.append((ticker, name, signal, detail, close, profit_pct))
            elif daily_chg < -4.0:
                signal = "🔴 出場"
                detail = f"單日跳空（{daily_chg:.1f}%）"
                exit_hits.append((ticker, name, signal, detail, close, profit_pct))
            elif below_ma5 and below_ma10:
                signal = "🔴 出場"
                detail = f"雙線跌破（MA5={ma5:.1f} MA10={ma10:.1f}）"
                exit_hits.append((ticker, name, signal, detail, close, profit_pct))
            else:
                # 法人連賣（只在無其他訊號時才查，節省 API）
                f_sell = _inst_sell_streak(code)
                if f_sell >= INST_SELL_DAYS:
                    signal = "🔴 出場"
                    detail = f"外資連賣{f_sell}日"
                    exit_hits.append((ticker, name, signal, detail, close, profit_pct))
                else:
                    # ── 持倉回復進場條件（對齊觀察名單標準）──────────
                    if profit_pct < 0 and adx_h >= ADX_TREND and roc_20d_h >= ROC_20_MIN:
                        try:
                            h_score, _ = entry_signals(df)
                        except Exception:
                            h_score = 0
                        if h_score >= 4:
                            recovery_detail = (
                                f"  📈 進場條件回復"
                                f"（ADX={adx_h:.0f} 20日{roc_20d_h:+.1f}% score={h_score}）"
                            )
                            addon_hits.append((ticker, name, close, profit_pct,
                                               f"ADX={adx_h:.0f} 20日{roc_20d_h:+.1f}% score={h_score}"))
                    hold_ok.append((ticker, name, close, profit_pct, deviation, rsi))

            # 印出即時行
            pnl_str = f"{profit_pct:+.1f}%" if buy_price else "?"
            if signal:
                print(f"  {ticker} {name:<6}  {signal}  {detail}  現價{close:.1f}  損益{pnl_str}{valley_detail}")
            else:
                print(f"  {ticker} {name:<6}  ✅ 持有  現價{close:.1f}  損益{pnl_str}  乖離{deviation:.1f}% RSI{rsi:.1f}{valley_detail}{recovery_detail}")
            for vl in valley_lines:
                print(vl)

    # ── 觀察名單：進場掃描 ───────────────────────────────────
    all_watch = []
    for group, tickers in watchlist.items():
        for ticker in tickers:
            if ticker not in (holdings or {}):
                all_watch.append(ticker)

    if all_watch:
        print(f"\n  ── 觀察名單進場掃描 ──────────────────────────")
        for code in all_watch:
            name   = code
            ticker = code + ".TW"
            df, _  = _get_df(ticker)
            if df is None:
                ticker = code + ".TWO"
                df, _  = _get_df(ticker)
            if df is None or len(df) < 25:
                print(f"  {code:<14}  ⚠ 資料不足")
                continue

            row   = df.iloc[-1]
            close = float(row['Close'])
            adx   = _calc_adx(df)

            close_20d_ago = float(df['Close'].iloc[-21]) if len(df) >= 21 else close
            roc_20d       = (close - close_20d_ago) / close_20d_ago * 100

            if adx >= ADX_TREND and roc_20d >= ROC_20_MIN:
                try:
                    score, _ = entry_signals(df)
                except Exception:
                    score = 0
                if score >= 4:
                    ma5  = float(row['MA5'])
                    rsi  = float(row['RSI'])
                    detail = f"ADX={adx:.0f} 20日+{roc_20d:.1f}% score={score}"
                    entry_hits.append((ticker, name, close, detail))
                    print(f"  {ticker:<14}  ✅ 進場  現價{close:.1f}  {detail}")
                else:
                    watch_miss.append((ticker, f"ADX={adx:.0f} score={score}/4"))
                    print(f"  {ticker:<14}  ─ 觀望   ADX={adx:.0f} 20日+{roc_20d:.1f}% score={score}/4（未達）")
            else:
                reasons = []
                if adx < ADX_TREND:    reasons.append(f"ADX={adx:.0f}<25")
                if roc_20d < ROC_20_MIN: reasons.append(f"20日漲幅{roc_20d:.1f}%<3%")
                watch_miss.append((ticker, " ".join(reasons)))
                print(f"  {ticker:<14}  ─ 觀望   {' / '.join(reasons)}")

    # ── 總結 ─────────────────────────────────────────────────
    # 從 hold_ok / exit / trim 裡面撈出有低谷訊號的持倉
    valley_hits = []
    for ticker, h in (holdings or {}).items():
        buy_price = float(h.get('buy_price', 0))
        name      = h.get('name', ticker)
        df2, _  = _get_df(ticker)
        if df2 is None or len(df2) < 10:
            continue
        row2       = df2.iloc[-1]
        close2     = float(row2['Close'])
        profit2    = (close2 - buy_price) / buy_price * 100 if buy_price else 0
        below_ma10_2 = close2 < float(row2['MA10'])
        if profit2 < 0 and below_ma10_2:
            try:
                o2, _, _ = detect_oversold(df2)
                e2, _    = detect_selling_exhaustion(df2)
            except Exception:
                o2 = e2 = 0
            if sum([o2 >= 3, e2 >= 1]) >= 2:
                valley_hits.append((ticker, name, close2, profit2,
                                    f"超跌{o2}/7" if o2 >= 3 else "", f"衰竭{e2}項" if e2 >= 1 else ""))

    print()
    print("=" * 60)
    if entry_hits:
        print(f"  【新進場訊號】{len(entry_hits)} 支（觀察名單）")
        for t, n, c, d in entry_hits:
            print(f"    ✅ {t} {n}  現價{c:.1f}  {d}")
    if addon_hits:
        print(f"  【加碼訊號】{len(addon_hits)} 支（持倉虧損回復進場條件）")
        for t, n, c, pnl, d in addon_hits:
            print(f"    📈 {t} {n}  現價{c:.1f}  損益{pnl:+.1f}%  {d}")
    if exit_hits:
        print(f"  【出場訊號】{len(exit_hits)} 支")
        for t, n, sig, d, c, pnl in exit_hits:
            print(f"    🔴 {t} {n}  {d}  現價{c:.1f}  損益{pnl:+.1f}%")
    if trim_hits:
        print(f"  【減碼訊號】{len(trim_hits)} 支")
        for t, n, sig, d, c, pnl in trim_hits:
            print(f"    🟡 {t} {n}  {d}  現價{c:.1f}  損益{pnl:+.1f}%")
    if valley_hits:
        print(f"  【低谷訊號】{len(valley_hits)} 支（虧損但出現底部特徵）")
        for t, n, c, pnl, os, ec in valley_hits:
            conds = " ".join(x for x in [os, ec] if x)
            print(f"    🟢 {t} {n}  {conds}  現價{c:.1f}  損益{pnl:+.1f}%")
    if not entry_hits and not addon_hits and not exit_hits and not trim_hits and not valley_hits:
        print("  今日無任何機械訊號，繼續持有 / 觀望")
    print("=" * 60)
    print()


def intraday_v2_scan():
    """V2 策略盤中分析：策略C（低谷反彈）優先 + 策略A（趨勢跟蹤）

    完整對齊 backtest_v2.py 的進出場邏輯：
      策略C：close < MA10 + ≥2/3 條件（超跌≥3, 衰竭≥1, 法人/外盤）
      策略A：ADX≥25 + 20日漲幅≥3% + entry_signals score≥4
    輸出比機械訊號更完整的進出場說明。
    """
    import numpy as np
    from collections import defaultdict
    import requests as _req

    # ── 常數（對齊 backtest_v2.py）───────────────────────────
    CHANDELIER_MULT = 2.0
    ADX_TREND       = 25
    ROC_20_MIN      = 3.0
    INST_SELL_DAYS  = 3
    TRIM_DEV        = 8.0;   TRIM_RSI    = 68.0
    STRONG_DEV      = 12.0;  STRONG_RSI  = 75.0
    C_OVERSOLD_MIN  = 3
    C_STOP_PCT      = 0.97
    C_MIN_COND      = 2      # 3 條件至少達到幾個

    # ── ADX 計算（對齊 backtest_v2）─────────────────────────
    def _calc_adx(df, period=14):
        high, low, close = df['High'], df['Low'], df['Close']
        pdm = high.diff()
        mdm = -low.diff()
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
        tr  = pd.concat([high-low, (high-close.shift()).abs(),
                         (low-close.shift()).abs()], axis=1).max(axis=1)
        a   = 1 / period
        atr = tr.ewm(alpha=a, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr
        mdi = 100 * mdm.ewm(alpha=a, adjust=False).mean() / atr
        dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        adx = dx.ewm(alpha=a, adjust=False).mean()
        return float(adx.iloc[-1]) if not adx.empty else 0.0

    # ── 法人連賣天數（對齊 backtest_v2：net<0即算賣）─────────
    def _inst_sell_streak(code):
        start = (now_tw() - timedelta(days=20)).strftime('%Y-%m-%d')
        params = {'dataset': 'TaiwanStockInstitutionalInvestorsBuySell',
                  'data_id': code, 'start_date': start}
        if FINMIND_TOKEN:
            params['token'] = FINMIND_TOKEN
        try:
            body = _req.get('https://api.finmindtrade.com/api/v4/data',
                            params=params, timeout=15).json()
            if not (body.get('status') == 200 and body.get('data')):
                return 0, '法人資料不可用'
            by_date = defaultdict(int)
            for r in body['data']:
                if r.get('name') == 'Foreign_Investor':
                    by_date[r['date']] += ((r.get('buy') or 0) - (r.get('sell') or 0)) // 1000
            streak = 0
            for d in sorted(by_date.keys(), reverse=True):
                if by_date[d] < 0:
                    streak += 1
                else:
                    break
            note = f"外資連賣{streak}日" if streak else "外資無持續賣超"
            return streak, note
        except Exception:
            return 0, '法人?'

    # ── 法人最新是否買超（用於C條件③，盤後/盤前用）──────────
    def _inst_buy_latest(code):
        start = (now_tw() - timedelta(days=10)).strftime('%Y-%m-%d')
        params = {'dataset': 'TaiwanStockInstitutionalInvestorsBuySell',
                  'data_id': code, 'start_date': start}
        if FINMIND_TOKEN:
            params['token'] = FINMIND_TOKEN
        try:
            body = _req.get('https://api.finmindtrade.com/api/v4/data',
                            params=params, timeout=15).json()
            if not (body.get('status') == 200 and body.get('data')):
                return False, '法人?'
            by_date = defaultdict(int)
            for r in body['data']:
                if r.get('name') == 'Foreign_Investor':
                    by_date[r['date']] += ((r.get('buy') or 0) - (r.get('sell') or 0)) // 1000
            if not by_date:
                return False, '法人?'
            latest = sorted(by_date.keys())[-1]
            net    = by_date[latest]
            buying = net > 0
            label  = f"外資{latest[5:]} {'買+' if buying else '賣-'}{abs(net)}張"
            return buying, label
        except Exception:
            return False, '法人?'

    # ── 資料取得（含 Fugle 補丁，對齊 mechanical_scan._get_df）─
    def _get_df_v2(ticker):
        code_ = ticker.replace('.TW', '').replace('.TWO', '')
        df_   = fetch(ticker, silent=True)
        if df_ is None:
            return None, None
        last = df_.iloc[-1]
        if (float(last['Open']) == float(last['High']) ==
                float(last['Low']) == float(last['Close'])):
            df_ = df_.iloc[:-1].copy()
            df_ = calculate_indicators(df_)

        fq_ = parse_fugle_price(get_fugle_quote(code_))

        def _patch_ohlcv(df__, fq__):
            df__ = df__.copy()
            for col, key in [('Open', 'open'), ('High', 'high'), ('Low', 'low')]:
                if fq__.get(key):
                    df__.iloc[-1, df__.columns.get_loc(col)] = float(fq__[key])
            if fq__.get('volume'):
                df__.iloc[-1, df__.columns.get_loc('Volume')] = float(fq__['volume']) * 1000
            return calculate_indicators(df__)

        if status == '盤前':
            if fq_ and fq_.get('close_price'):
                df_ = _apply_fugle_price(df_, fq_['close_price'], is_intraday=False)
                df_ = _patch_ohlcv(df_, fq_)
        elif status == '盤中':
            if fq_ and fq_.get('price'):
                df_ = _apply_fugle_price(df_, fq_['price'], is_intraday=True)
                df_ = df_.copy()
                for col, key in [('Open', 'open'), ('High', 'high'), ('Low', 'low')]:
                    if fq_.get(key):
                        df_.iloc[-1, df_.columns.get_loc(col)] = float(fq_[key])
                vol_ma5_stable = float(df_.iloc[-1]['Vol_MA5']) if df_.iloc[-1]['Vol_MA5'] else 0
                df_ = calculate_indicators(df_)
                vol_raw = fq_.get('volume') or 0
                if vol_raw and vol_ma5_stable:
                    elapsed  = max(1, now.hour * 60 + now.minute - 9 * 60)
                    progress = min(elapsed / 270, 1.0)
                    est_vr   = (vol_raw / progress * 1000) / vol_ma5_stable
                    df_.iloc[-1, df_.columns.get_loc('Vol_ratio')] = est_vr
                    df_.iloc[-1, df_.columns.get_loc('Vol_MA5')]   = vol_ma5_stable
        else:  # 盤後
            if fq_ and fq_.get('close_price'):
                df_ = _apply_fugle_price(df_, fq_['close_price'], is_intraday=False)
                df_ = _patch_ohlcv(df_, fq_)
        return df_, fq_

    # ══════════════════════════════════════════════════════════
    _fugle_cache.clear()
    now    = now_tw()
    status, _ = market_status()

    title = {'盤中': 'V2 策略盤中分析  即時評估',
             '盤後': 'V2 策略盤後分析  收盤回顧'}.get(status, 'V2 策略盤前分析  昨日資料')
    if status == '盤中':
        remaining = max(0, (13 * 60 + 30) - now.hour * 60 - now.minute)
        time_note = f"距收盤 {remaining} 分鐘"
    elif status == '盤後':
        time_note = "今日最終收盤"
    else:
        time_note = "台股尚未開盤"

    print()
    print("=" * 64)
    print(f"  🤖 {title}")
    print(f"  規則：策略C（低谷反彈）優先  ▶  策略A（趨勢跟蹤）次之")
    print(f"  {now.strftime('%Y-%m-%d  %H:%M')}  {time_note}")
    print("=" * 64)

    exits   = []   # (ticker, name, reason)
    trims   = []
    c_hits  = []
    a_hits  = []

    holdings  = HOLDINGS
    watchlist = WATCHLIST

    # ══════════════════════════════════════════════════════════
    #  持倉：出場 / 減碼判斷
    # ══════════════════════════════════════════════════════════
    if holdings:
        print()
        print("  ── 持倉出場判斷 ─────────────────────────────────────")

        for ticker, h in holdings.items():
            if not ticker.endswith(('.TW', '.TWO')):
                continue
            code      = ticker.replace('.TW', '').replace('.TWO', '')
            name      = h.get('name', ticker)
            buy_price = float(h.get('buy_price', 0))
            shares    = h.get('shares', 0)

            df, fq = _get_df_v2(ticker)
            if df is None or len(df) < 10:
                print(f"\n  {ticker} {name}  ⚠ 無法取得資料")
                continue

            row        = df.iloc[-1]
            close      = float(row['Close'])
            atr        = float(row['ATR'])
            high22     = float(row['High22'])
            ma5        = float(row['MA5'])
            ma10       = float(row['MA10'])
            rsi        = float(row['RSI'])
            low20      = float(row['Low_20'])
            deviation  = (close - ma5) / ma5 * 100
            profit_pct = (close - buy_price) / buy_price * 100 if buy_price else 0
            atr_stop   = high22 - CHANDELIER_MULT * atr
            c_stop     = low20 * C_STOP_PCT
            below_ma10 = close < ma10
            below_ma5  = close < ma5
            ask_pct    = fq.get('ask_pct') if fq else None
            ob_str     = f"外盤{ask_pct:.0f}%  " if ask_pct is not None else ""

            ma10_tag = "⬇ 在MA10下方" if below_ma10 else "在MA10上方"
            print(f"\n  {ticker} {name}  現價 {close:.1f}  損益 {profit_pct:+.1f}%  {ma10_tag}")
            print(f"  乖離 {deviation:+.1f}%  RSI {rsi:.1f}  {ob_str}ATR={atr:.2f}")

            # 策略判斷：虧損且在MA10下方 → 優先用C出場規則（底部回撤）
            use_c = (profit_pct < 0 and below_ma10)

            if use_c:
                # ── 策略C 出場邏輯 ────────────────────────────
                print(f"  📋 出場判斷（策略C 低谷反彈模式，虧損在MA10下方）")

                # ① C停損：近20日低點 × 0.97
                if close < c_stop:
                    msg = f"🔴 停損觸發！現價{close:.1f} < 低點{low20:.1f}×0.97={c_stop:.1f}"
                    print(f"  ├ {msg}")
                    print(f"  │   → 底部確認失敗，建議立即出場（不要等）")
                    exits.append((ticker, name, f"C停損({c_stop:.1f})"))
                else:
                    dist_s = (close - c_stop) / close * 100
                    print(f"  ├ 停損線：低點{low20:.1f}×0.97={c_stop:.1f}  未跌破✓  距停損{dist_s:.1f}%")

                # ② MA10 目標
                if close >= ma10:
                    print(f"  ├ 🟢 目標達成！收復MA10={ma10:.1f}，底部反彈完成")
                    print(f"  │   → 可考慮獲利了結，或切換至策略A繼續持有")
                    trims.append((ticker, name, "C目標達成(收復MA10)"))
                else:
                    dist_t = (ma10 - close) / close * 100
                    print(f"  ├ 目標：收復MA10={ma10:.1f}  → 尚差+{dist_t:.1f}%，繼續等待")

                # ④ 法人連賣（C版：≥3日提前出場）
                f_sell, f_sell_note = _inst_sell_streak(code)
                if f_sell >= INST_SELL_DAYS:
                    print(f"  ├ ⚠ 法人提前出場！{f_sell_note}（≥3日）")
                    print(f"  │   → 籌碼持續流失，不等目標，建議出場")
                    exits.append((ticker, name, f"C法人({f_sell_note})"))
                else:
                    print(f"  └ 法人：{f_sell_note}  ✓")

                # 補充A吊燈線（輔助確認）
                print(f"  ── A策略吊燈線（輔助參考）：")
                if close < atr_stop:
                    print(f"  ⚠  High22({high22:.1f})-2×ATR({atr:.2f})={atr_stop:.1f}  現價跌破，雙重出場訊號！")
                else:
                    print(f"     High22({high22:.1f})-2×ATR({atr:.2f})={atr_stop:.1f}  未跌破✓")

            else:
                # ── 策略A 出場邏輯（趨勢跟蹤）────────────────────
                print(f"  📋 出場判斷（策略A 趨勢跟蹤模式）")

                # 減碼判斷（策略A優先序第一位）
                if deviation > STRONG_DEV and rsi > STRONG_RSI:
                    trim_shares = int(shares * 0.3)
                    print(f"  ├ 🟡 強力減碼！乖離{deviation:.1f}%>12% 且 RSI{rsi:.1f}>75")
                    print(f"  │   → 若已曾減碼：直接出清剩餘")
                    print(f"  │   → 若首次：減碼30%（{trim_shares}股），保留主倉")
                    trims.append((ticker, name, f"A強力減碼(乖離{deviation:.1f}% RSI{rsi:.1f})"))
                elif deviation > TRIM_DEV and rsi > TRIM_RSI:
                    trim_shares = int(shares * 0.3)
                    print(f"  ├ 🟡 減碼訊號！乖離{deviation:.1f}%>8% 且 RSI{rsi:.1f}>68")
                    print(f"  │   → 建議減碼30%（{trim_shares}股），鎖定部分獲利")
                    print(f"  │   強力減碼門檻：乖離>12%+RSI>75（現尚未達標）")
                    trims.append((ticker, name, f"A減碼30%(乖離{deviation:.1f}% RSI{rsi:.1f})"))
                else:
                    print(f"  ├ 乖離保護：{deviation:+.1f}%  RSI:{rsi:.1f}  未達減碼門檻✓")

                # 吊燈線停損（策略A核心停損）
                if close < atr_stop:
                    dist_atr = (atr_stop - close) / close * 100
                    print(f"  ├ 🔴 吊燈線觸發！High22({high22:.1f})-2×ATR({atr:.2f})={atr_stop:.1f}")
                    print(f"  │   現價{close:.1f}已跌破停損線（跌破{dist_atr:.1f}%）→ 建議出場")
                    exits.append((ticker, name, f"A吊燈線({atr_stop:.1f})"))
                else:
                    dist_atr = (close - atr_stop) / close * 100
                    print(f"  ├ 吊燈停損：{atr_stop:.1f}  未跌破✓  距停損+{dist_atr:.1f}%")

                # 雙線跌破（趨勢結束）
                if below_ma5 and below_ma10:
                    print(f"  ├ 🔴 雙線跌破！MA5({ma5:.1f}) MA10({ma10:.1f})  趨勢已轉弱")
                    print(f"  │   → 若未觸發吊燈線，可設停損觀察一日確認")
                    exits.append((ticker, name, "A雙線跌破"))
                elif below_ma5:
                    print(f"  ├ ⚠  跌破MA5({ma5:.1f})，但MA10({ma10:.1f})仍支撐，觀察")
                else:
                    print(f"  ├ 均線：MA5({ma5:.1f}) MA10({ma10:.1f})  在雙線上方✓")

                # 法人連賣
                f_sell, f_sell_note = _inst_sell_streak(code)
                if f_sell >= INST_SELL_DAYS:
                    print(f"  └ ⚠  法人提前出場！{f_sell_note}  籌碼流失，建議出場")
                    exits.append((ticker, name, f"A法人({f_sell_note})"))
                else:
                    print(f"  └ 法人：{f_sell_note}  ✓")

    # ══════════════════════════════════════════════════════════
    #  觀察名單：進場掃描（策略C優先，不觸發才看A）
    # ══════════════════════════════════════════════════════════
    all_watch = []
    for group, tickers in watchlist.items():
        for t in tickers:
            if t not in (holdings or {}):
                all_watch.append(t)

    if all_watch:
        print()
        print("  ── 觀察名單進場掃描 ─────────────────────────────────")
        print("  策略C優先（MA10下方偵測底部）▶ C不觸發才看策略A趨勢")

        for code in all_watch:
            ticker = code + ".TW"
            df, fq = _get_df_v2(ticker)
            if df is None:
                ticker = code + ".TWO"
                df, fq = _get_df_v2(ticker)
            if df is None or len(df) < 25:
                print(f"\n  {code}  ⚠ 資料不足，跳過")
                continue

            row        = df.iloc[-1]
            close      = float(row['Close'])
            ma10       = float(row['MA10'])
            ma5        = float(row['MA5'])
            low20      = float(row['Low_20'])
            below_ma10 = close < ma10
            ask_pct    = fq.get('ask_pct') if fq else None

            ma10_tag = "⬇ MA10下方" if below_ma10 else "MA10上方"
            print(f"\n  {ticker}  現價{close:.1f}  MA10={ma10:.1f}  ({ma10_tag})")

            c_triggered = False

            # ── 策略C：低谷反彈（必須在MA10下方）────────────────
            if below_ma10:
                print(f"  【策略C 低谷反彈】前提：{close:.1f} < MA10({ma10:.1f}) ✓")

                # 條件① 超跌評分
                try:
                    o_score, o_detail, o_level = detect_oversold(df)
                except Exception:
                    o_score, o_detail, o_level = 0, [], "?"
                ok1  = (o_score >= C_OVERSOLD_MIN)
                m1   = "✓" if ok1 else "✗"
                print(f"  ① 超跌評分  {m1}  {o_score}/7（需≥3）→ {o_level}")
                for d in o_detail:
                    print(f"       {d.strip()}")

                # 條件② 賣壓衰竭
                try:
                    e_count, e_list = detect_selling_exhaustion(df)
                except Exception:
                    e_count, e_list = 0, []
                ok2 = (e_count >= 1)
                m2  = "✓" if ok2 else "✗"
                print(f"  ② 賣壓衰竭  {m2}  {e_count}項（需≥1）")
                for s in e_list:
                    print(f"       {s}")

                # 條件③ 法人/外盤
                if status == '盤中' and ask_pct is not None:
                    ok3     = (ask_pct > 55)
                    m3      = "✓" if ok3 else "✗"
                    c3_note = f"外盤比{ask_pct:.0f}%（盤中，需>55%）"
                else:
                    ok3, c3_note = _inst_buy_latest(code)
                    m3 = "✓" if ok3 else "✗"
                    c3_note = f"{c3_note}（盤後/盤前用法人）"
                print(f"  ③ 法人/外盤 {m3}  {c3_note}")

                c_met  = sum([ok1, ok2, ok3])
                c_stop = low20 * C_STOP_PCT
                target = ma10
                d_stop = (close - c_stop) / close * 100
                d_tgt  = (target - close) / close * 100
                rr     = abs(d_tgt / d_stop) if d_stop else 0

                if c_met >= C_MIN_COND:
                    met_names = []
                    if ok1: met_names.append("超跌")
                    if ok2: met_names.append("衰竭")
                    if ok3: met_names.append("法人/外盤")
                    print(f"  → ✅ 策略C進場訊號！{c_met}/3 達成（需≥{C_MIN_COND}）")
                    print(f"  → 訊號說明：{'+'.join(met_names)} 底部特徵同步出現，低谷反彈機率提升")
                    print(f"  → 停損：低點{low20:.1f}×0.97={c_stop:.1f}（風險{d_stop:.1f}%，跌破即認錯出場）")
                    print(f"  → 目標：收復MA10={target:.1f}（潛在+{d_tgt:.1f}%，風險報酬比 1:{rr:.2f}）")
                    if rr < 1.0:
                        print(f"     ⚠  報酬比偏低，建議等法人/外盤訊號再確認後進場")
                    c_hits.append((ticker, code, close, c_met))
                    c_triggered = True
                elif c_met == C_MIN_COND - 1:
                    missing = []
                    if not ok1: missing.append("超跌不足")
                    if not ok2: missing.append("無衰竭訊號")
                    if not ok3: missing.append("法人/外盤未確認")
                    print(f"  → 💡 低谷待確認（{c_met}/3，差1項：{'、'.join(missing)}）")
                    print(f"  → 尚未達進場標準，持續觀察，勿搶先進場")
                else:
                    print(f"  → ⏳ 底部特徵不足（{c_met}/3），暫不進場")

            # ── 策略A：趨勢跟蹤（C不觸發才判斷）────────────────
            if not c_triggered:
                if below_ma10:
                    print(f"  【策略A 趨勢跟蹤】（C未觸發，補充趨勢掃描）")
                else:
                    print(f"  【策略A 趨勢跟蹤】")

                adx     = _calc_adx(df)
                c20_ago = float(df['Close'].iloc[-21]) if len(df) >= 21 else close
                roc_20d = (close - c20_ago) / c20_ago * 100
                adx_ok  = (adx >= ADX_TREND)
                roc_ok  = (roc_20d >= ROC_20_MIN)
                adx_m   = "✓" if adx_ok else "✗"
                roc_m   = "✓" if roc_ok else "✗"
                print(f"  前提① ADX={adx:.1f}  {adx_m}（需≥25，{'趨勢確認' if adx_ok else '趨勢不足'}）")
                print(f"  前提② 20日漲幅={roc_20d:+.1f}%  {roc_m}（需≥3%，{'非橫盤' if roc_ok else '近期上漲不足'}）")

                if adx_ok and roc_ok:
                    try:
                        score, score_msgs = entry_signals(df)
                    except Exception:
                        score, score_msgs = 0, []
                    for msg in score_msgs:
                        print(f"  {msg}")
                    if score >= 4:
                        print(f"  → ✅ 策略A進場訊號！ADX={adx:.0f} + 20日+{roc_20d:.1f}% + score={score}")
                        print(f"  → 訊號說明：趨勢強勁（ADX≥25）且近20日真實上漲，進場條件全達標")
                        a_hits.append((ticker, code, close, adx, roc_20d, score))
                    else:
                        print(f"  → ─ 策略A 暫不進場（score={score}/4，需≥4）")
                else:
                    reasons = []
                    if not adx_ok: reasons.append(f"ADX={adx:.0f} 趨勢不夠強")
                    if not roc_ok: reasons.append(f"20日漲幅{roc_20d:.1f}% 上漲動能不足")
                    print(f"  → ─ 前提未達：{' / '.join(reasons)}")

    # ══════════════════════════════════════════════════════════
    #  總結
    # ══════════════════════════════════════════════════════════
    print()
    print("=" * 64)
    print("  📊 V2 訊號總結")
    if exits:
        print(f"  🔴 出場訊號（{len(exits)} 支）：")
        for t, n, r in exits:
            print(f"      {t} {n}  ← {r}")
    if trims:
        print(f"  🟡 減碼訊號（{len(trims)} 支）：")
        for t, n, r in trims:
            print(f"      {t} {n}  ← {r}")
    if c_hits:
        print(f"  🟢 策略C 進場（{len(c_hits)} 支）：")
        for t, code, c, met in c_hits:
            print(f"      {t}  現價{c:.1f}  條件{met}/3 達成")
    if a_hits:
        print(f"  🟢 策略A 進場（{len(a_hits)} 支）：")
        for t, code, c, adx, roc, sc in a_hits:
            print(f"      {t}  現價{c:.1f}  ADX={adx:.0f} 20日+{roc:.1f}% score={sc}")
    if not exits and not trims and not c_hits and not a_hits:
        print("  今日無任何 V2 訊號 — 持倉繼續持有，觀察名單繼續等待")
    print("=" * 64)
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == "scan":
            intraday_scan()
        elif arg.lower() == "watch":
            watchlist_scan()
        elif arg.lower() == "v2":
            intraday_v2_scan()
        else:
            quick_lookup(arg)
    else:
        run()
