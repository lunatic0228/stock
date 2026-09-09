"""K 線趨勢診斷 — 資料層

負責長期日 K 下載、額外指標欄位、快取、Fugle 即時價接合。

與 daily_analysis.py 的關係：
  - 單向依賴：本模組 import daily_analysis，daily_analysis 絕不 import 本模組
  - 重用 da.calculate_indicators() 產生基礎指標，再補上它沒有的欄位
  - da.fetch() 只抓 60 天，波段結構／趨勢階段／歷史類比需要 2~4 年，故自行下載

快取三層：
  1. 行程內 dict（_long_cache）
  2. st.cache_data（Streamlit Cloud 的正解；CLI/回測環境自動降級為無快取）
  3. 本機 parquet（預設關閉，僅在 KLINE_CACHE_DIR 環境變數存在時啟用，供回測反覆跑）
"""

import contextlib
import io
import logging
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

import daily_analysis as da

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']

# 指標暖機根數：MA60 需 60 根、BBW 百分位需 120 根、滾動 z-score 需 250 根
# 分析可用性以 120 根為界（z-score 在 kline_analysis 內另有 min_periods 處理）
WARMUP_BARS = 120

_long_cache = {}          # {(ticker, days): DataFrame}

_BATCH_SIZE  = 10         # yfinance 多檔批次大小（10 檔 = 1 個 HTTP 請求）
_BATCH_SLEEP = 0.6        # 批次之間的間隔秒數，避免 429


# ════════════════════════════════════════════════════════════
#  ADX（向量化版；最後一根必須與 backtest_v4.calc_adx 相同）
# ════════════════════════════════════════════════════════════
def calc_adx_series(df, period=14):
    """回傳 (adx, di_plus, di_minus) 三個 Series。

    計算方式與 backtest_v4.calc_adx() 完全一致（Wilder ewm alpha=1/period），
    差異僅在該函式只回傳最後一根的 float。kline_data 需要整條序列，
    因為趨勢階段分類器要在每一根 K 棒上判定。
    """
    high, low, close = df['High'], df['Low'], df['Close']

    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    hl  = high - low
    hpc = (high - close.shift()).abs()
    lpc = (low - close.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

    atr14      = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di14  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    minus_di14 = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14

    dx  = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx, plus_di14, minus_di14


# ════════════════════════════════════════════════════════════
#  額外指標欄位
# ════════════════════════════════════════════════════════════
def _ols_slope(y):
    """最小平方法斜率（單位：每根 K 棒的變化量）。y 為 numpy array。"""
    n = len(y)
    if n < 2 or np.isnan(y).any():
        return np.nan
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - xm) * (y - ym)).sum() / denom)


def _bars_since_max(a):
    """視窗內最高值距今幾根（0 = 最後一根就是最高）。a 為 numpy array。"""
    if np.isnan(a).all():
        return np.nan
    return float(len(a) - 1 - int(np.nanargmax(a)))


def add_kline_indicators(df, intraday_last=False):
    """在 da.calculate_indicators() 的輸出上補齊 K 線診斷所需欄位。

    intraday_last=True 時，最後一根的量能均線／量能斜率沿用前一根的值 ——
    盤中的 Volume 只是累計到目前的部分量，直接算會讓量能趨勢嚴重失真
    （這與 da._apply_fugle_price 對 Vol_MA5 的處理原則相同）。
    """
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']

    # MA20 直接沿用 BB_mid（同為 20 日均線），只是換個可讀的名字
    df['MA20'] = df['BB_mid'] if 'BB_mid' in df.columns else close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()

    # 均線斜率：換算成「每 N 根 K 棒的百分比變化」，跨股跨價位可比
    df['MA20_slope'] = (df['MA20'] - df['MA20'].shift(10)) / df['MA20'].shift(10) * 100
    df['MA60_slope'] = (df['MA60'] - df['MA60'].shift(20)) / df['MA60'].shift(20) * 100
    df['MA5_slope']  = (df['MA5'] - df['MA5'].shift(5)) / df['MA5'].shift(5) * 100

    # ADX / DI
    adx, di_p, di_m = calc_adx_series(df)
    df['ADX'], df['DI_plus'], df['DI_minus'] = adx, di_p, di_m

    # 區間位階：用 High/Low 的真實極值，而非 da.High22 的收盤極值
    df['High60']  = high.rolling(60).max()
    df['Low60']   = low.rolling(60).min()
    df['High252'] = high.rolling(252, min_periods=120).max()
    df['Low252']  = low.rolling(252, min_periods=120).min()

    rng60 = (df['High60'] - df['Low60']).replace(0, np.nan)
    df['Range_pos'] = (close - df['Low60']) / rng60
    rng252 = (df['High252'] - df['Low252']).replace(0, np.nan)
    df['Range_pos252'] = (close - df['Low252']) / rng252

    df['dd_252h'] = (close / df['High252'] - 1) * 100     # 距年高（負值）
    df['up_252l'] = (close / df['Low252'] - 1) * 100      # 距年低（正值）
    df['dev_ma20'] = (close / df['MA20'] - 1) * 100
    df['dev_ma60'] = (close / df['MA60'] - 1) * 100

    df['bars_since_60d_high'] = high.rolling(60).apply(_bars_since_max, raw=True)
    df['bars_since_60d_low'] = low.rolling(60).apply(
        lambda a: np.nan if np.isnan(a).all() else float(len(a) - 1 - int(np.nanargmin(a))),
        raw=True)

    # 量能：Vol_MA5 由 da 提供，這裡補 10/20 日與量能斜率
    df['Vol_MA10'] = vol.rolling(10).mean()
    df['Vol_MA20'] = vol.rolling(20).mean()
    vol_slope_abs  = vol.rolling(20).apply(_ols_slope, raw=True)
    df['Vol_slope20'] = vol_slope_abs / df['Vol_MA20'].replace(0, np.nan) * 100  # %/根

    # 波動：把價格斜率標準化成「幾個 ATR」用
    df['ATR_pct'] = df['ATR'] / close * 100

    # 布林帶寬的 120 日百分位（判斷是否處於壓縮狀態；rolling rank 為因果運算）
    df['BBW_pct'] = df['BB_width'].rolling(120, min_periods=60).rank(pct=True)

    # 動能：策略 A 閘門用（ADX≥25 且 ROC_20≥3%）
    df['ROC_20'] = (close / close.shift(20) - 1) * 100

    # 成交金額（薄量防護用；台股 Volume 單位為股）
    df['Turnover'] = close * vol

    if intraday_last and len(df) >= 2:
        # 盤中：最後一根的量相關欄位沿用前一根，避免部分量造成誤判
        for col in ('Vol_MA10', 'Vol_MA20', 'Vol_slope20', 'Turnover'):
            df.iloc[-1, df.columns.get_loc(col)] = df.iloc[-2][col]

    return df


# ════════════════════════════════════════════════════════════
#  下載
# ════════════════════════════════════════════════════════════
def _parquet_path(ticker, days):
    cache_dir = os.environ.get("KLINE_CACHE_DIR")
    if not cache_dir:
        return None
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        return None
    safe = ticker.replace(".", "_").replace("^", "idx_")
    return os.path.join(cache_dir, f"{safe}_{days}.parquet")


def _parquet_is_fresh(path):
    """parquet 是否仍新鮮：mtime 需晚於今天 14:00（台北）收盤後的寫入時點。"""
    if not path or not os.path.exists(path):
        return False
    try:
        now = da.now_tw()
        mtime = pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC').tz_convert(now.tzinfo)
        today_close = now.replace(hour=14, minute=0, second=0, microsecond=0)
        if now < today_close:
            # 今天還沒收盤 → 只要是今天寫的就算新鮮
            return mtime.date() == now.date()
        return mtime >= today_close
    except Exception:
        return False


def _raw_download(tickers, days):
    """yfinance 下載 → {ticker: OHLCV DataFrame}。tickers 可為單檔字串或 list。"""
    single = isinstance(tickers, str)
    tlist  = [tickers] if single else list(tickers)
    if not tlist:
        return {}

    end   = da.now_tw().date() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(days * 1.48) + 60)

    try:
        raw = yf.download(
            tlist, start=start, end=end,
            progress=False, auto_adjust=True, threads=False,
            group_by='ticker' if len(tlist) > 1 else 'column',
        )
    except Exception as e:
        print(f"  ⚠  下載失敗 {tlist}：{e}")
        return {}

    if raw is None or raw.empty:
        return {}

    out = {}
    for t in tlist:
        try:
            if len(tlist) > 1:
                sub = raw[t] if t in raw.columns.get_level_values(0) else None
                if sub is None:
                    continue
            else:
                sub = raw
                if isinstance(sub.columns, pd.MultiIndex):
                    sub.columns = sub.columns.get_level_values(0)
            sub = sub[OHLCV].dropna()
            if len(sub) >= WARMUP_BARS:
                out[t] = sub
        except Exception:
            continue
    return out


def _prepare(sub):
    """OHLCV → 完整指標 DataFrame。

    刻意「不」裁切長度：滾動視窗（MA60/High252/BBW百分位/z-score）的值取決於
    起點，若先算完再裁掉前面，回傳 df 的前 250 根就帶著裁切前歷史的資訊，
    回測的 verify_causality 用 df.iloc[:t+1] 重算時會對不上。
    裁切交給 fetch_long(trim=True) 在最後做，回測則用 trim=False 拿完整序列。
    """
    if sub is None or len(sub) < WARMUP_BARS:
        return None
    df = sub.copy()
    df = da.calculate_indicators(df)
    df = add_kline_indicators(df)
    return df


def _download_prepared(ticker, days):
    """單檔：parquet → yfinance → 指標。回傳 DataFrame 或 None。"""
    path = _parquet_path(ticker, days)
    if _parquet_is_fresh(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    got = _raw_download(ticker, days)
    df  = _prepare(got.get(ticker))
    if df is not None and path:
        try:
            df.to_parquet(path)
        except Exception:
            pass
    return df


# Streamlit Cloud 的本地磁碟是暫時性的，自建檔案快取只會多一個 stale 失敗模式，
# 所以雲端一律交給 st.cache_data；CLI / 回測環境沒有 streamlit 就自動降級。
try:
    import streamlit as _st

    @_st.cache_data(ttl=3600, show_spinner=False)
    def _cached_download(ticker, days):
        return _download_prepared(ticker, days)
except Exception:
    _cached_download = _download_prepared


def fetch_long(ticker, days=750, use_cache=True, trim=True):
    """長期日 K（含全部指標）。回傳 DataFrame 或 None。

    trim=True  → 裁到最後 days 根（報告用，畫面上不需要更早的歷史）
    trim=False → 回傳完整下載序列（回測用，讓滾動視窗的起點與重算時一致）
    """
    key = (ticker, days)
    if use_cache and key in _long_cache:
        df = _long_cache[key]
    else:
        df = _cached_download(ticker, days) if use_cache else _download_prepared(ticker, days)
        if df is not None:
            _long_cache[key] = df
    if df is None:
        return None
    return df.iloc[-days:] if (trim and len(df) > days) else df


def fetch_long_batch(tickers, days=750, use_cache=True, trim=True):
    """多檔長期日 K（同儕池用）。以 10 檔為一批，1 個請求抓 10 檔。"""
    out, todo = {}, []
    for t in tickers:
        key = (t, days)
        if use_cache and key in _long_cache:
            out[t] = _long_cache[key]
        else:
            todo.append(t)

    for i in range(0, len(todo), _BATCH_SIZE):
        chunk = todo[i:i + _BATCH_SIZE]
        got   = _raw_download(chunk, days)
        for t in chunk:
            df = _prepare(got.get(t))
            if df is not None:
                out[t] = df
                _long_cache[(t, days)] = df
        if i + _BATCH_SIZE < len(todo):
            time.sleep(_BATCH_SLEEP)

    if trim:
        out = {t: (d.iloc[-days:] if len(d) > days else d) for t, d in out.items()}
    return out


# ════════════════════════════════════════════════════════════
#  法人資料（結構化，同時供即時報告與回測 as-of 使用）
# ════════════════════════════════════════════════════════════
_inst_series_cache = {}


def fetch_inst_series(code, days=400):
    """FinMind 三大法人買賣超 → {date_str: {foreign, trust, dealer, total}}（單位：張）

    daily_analysis 內有 3 份幾乎相同的 _inst_buy_latest 區域函式，各自只回傳
    一個布林值；回測還需要 as-of 過濾與連續天數。這裡抓一次就回傳結構化序列，
    讓「最近一日淨買超」（策略C 條件③）與「連續買賣超天數」（本模組法人分項）
    都從同一份資料推導，不會出現兩邊定義不一致。
    """
    key = (code, days)
    if key in _inst_series_cache:
        return _inst_series_cache[key]

    token = getattr(da, 'FINMIND_TOKEN', None)
    start = (da.now_tw() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    params = {'dataset': 'TaiwanStockInstitutionalInvestorsBuySell',
              'data_id': code, 'start_date': start}
    if token:
        params['token'] = token

    out = {}
    try:
        import requests
        body = requests.get('https://api.finmindtrade.com/api/v4/data',
                            params=params, timeout=20).json()
        if body.get('status') == 200 and body.get('data'):
            field = {'Foreign_Investor': 'foreign', 'Investment_Trust': 'trust',
                     'Dealer_self': 'dealer'}
            for r in body['data']:
                k = field.get(r.get('name'))
                if not k:
                    continue
                d = out.setdefault(r['date'], {'foreign': 0, 'trust': 0, 'dealer': 0})
                d[k] += ((r.get('buy') or 0) - (r.get('sell') or 0)) // 1000
            for d in out.values():
                d['total'] = d['foreign'] + d['trust'] + d['dealer']
    except Exception:
        pass

    _inst_series_cache[key] = out
    return out


def inst_snapshot(series, as_of=None, streak_days=5):
    """由法人序列導出判定所需的快照。as_of 給定時只用 ≤ as_of 的資料（回測用）。

    回傳 dict:
      latest_date / latest_foreign / latest_buy  ← 策略C 條件③（最近一日淨買超）
      dir / days                                 ← 連續買賣超（本模組法人分項）
      rows                                       ← 最近數日明細，供報告顯示
    """
    if not series:
        return None
    dates = sorted(series.keys())
    if as_of is not None:
        cutoff = str(as_of)[:10]
        dates = [d for d in dates if d <= cutoff]
    if not dates:
        return None

    latest = dates[-1]
    lf = series[latest]['foreign']

    direction, days = 'neutral', 0
    if lf != 0:
        direction = 'buy' if lf > 0 else 'sell'
        for d in reversed(dates):
            net = series[d]['foreign']
            if (net > 0) == (lf > 0) and net != 0:
                days += 1
            else:
                break

    return {
        'latest_date': latest, 'latest_foreign': lf, 'latest_buy': lf > 0,
        'dir': direction, 'days': days,
        'rows': [(d, series[d]) for d in dates[-streak_days:]],
    }


# ════════════════════════════════════════════════════════════
#  代號解析 / 即時價
# ════════════════════════════════════════════════════════════
@contextlib.contextmanager
def _quiet_yf():
    """吞掉 yfinance 的下載錯誤輸出（探測 .TW / .TWO 時用）。

    同時處理兩條路徑：直接寫 stdout/stderr 的訊息，以及 yfinance 自己的
    logger（它的 StreamHandler 在建立時就綁定了當時的 sys.stderr，
    單靠 redirect_stderr 攔不到，必須改 logger 等級）。
    """
    loggers = [logging.getLogger(n) for n in ('yfinance', 'yfinance.data', 'peewee')]
    saved = [(lg, lg.level, lg.propagate) for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        for lg, lvl, prop in saved:
            lg.setLevel(lvl)
            lg.propagate = prop
def resolve_ticker(raw, days=750):
    """代號補後綴。台股先試 .TW 再試 .TWO（與 da.quick_lookup 相同慣例）。

    回傳 (ticker, label, df)；找不到則 (None, None, None)。
    """
    raw = str(raw).strip().upper()
    if not raw:
        return None, None, None

    if raw.replace('.TW', '').replace('.TWO', '').isdigit():
        code = raw.replace('.TWO', '').replace('.TW', '')
        for suffix in ('.TW', '.TWO'):
            # 上櫃股探測 .TW 必然 404，yfinance 的錯誤訊息會被 app.py 的
            # redirect_stdout 抓進 UI（每查一次上櫃股就跳一次），必須靜音。
            # 與 da.fetch(silent=True) 的處理原則相同。
            with _quiet_yf():
                df = fetch_long(code + suffix, days)
            if df is not None:
                return code + suffix, "台股", df
        return None, None, None

    df = fetch_long(raw, days)
    return (raw, "美股", df) if df is not None else (None, None, None)


def attach_live_price(df, ticker, status):
    """把 Fugle 即時／官方收盤價接到 df，並重算受影響的指標。

    回傳 (df, note, fq)：
      note — 資料新鮮度提示字串（無提示則為空字串）
      fq   — parse_fugle_price 的結果（含開高低、均價、內外盤、五檔），可能為 None

    刻意不使用 da._apply_fugle_price：它只在 is_intraday=True 時才會補今日新列，
    盤前／盤後遇到 yfinance 落後一日就會把新價蓋到「前一個交易日」那一根上，
    製造出超過 ±10% 漲跌停的不可能數字（實測 2313 被寫成 +17.46%）。
    這裡改用 Fugle 回傳的 date 欄位判斷該附加新列還是覆蓋最後一列。

    回測絕不呼叫本函式：它會引入「現在」的資料。
    """
    note, fq = "", None
    if not ticker.endswith(('.TW', '.TWO')) or status == "休市":
        return df, note, fq

    code = ticker.replace('.TWO', '').replace('.TW', '')
    da._fugle_cache.pop(code, None)          # 盤中/盤後欄位含意不同，每次重取
    raw = da.get_fugle_quote(code)
    fq  = da.parse_fugle_price(raw)

    last_date = df.index[-1].date()
    if not fq:
        if last_date < da.now_tw().date() and status in ("盤中", "盤後"):
            note = f"⚠ 無 Fugle 報價，資料來自 yfinance 可能延遲（最後一根 {last_date}）"
        return df, note, fq

    is_intraday = (status == "盤中")
    price = fq["price"] if is_intraday else fq["close_price"]
    if not price:
        return df, note, fq

    fugle_date = None
    if raw and raw.get("date"):
        try:
            fugle_date = pd.to_datetime(raw["date"]).date()
        except Exception:
            fugle_date = None

    if fugle_date and fugle_date < last_date:
        # yfinance 已比 Fugle 新（少見），不要往回蓋
        return df, note, fq

    df = df.copy()
    if fugle_date and fugle_date > last_date:
        # Fugle 有更新的一個交易日 → 附加新列，絕不覆蓋既有的歷史 K 棒
        new = df.iloc[-1:].copy()
        tz  = df.index.tz
        new.index = pd.DatetimeIndex(
            [pd.Timestamp(fugle_date, tz=tz) if tz else pd.Timestamp(fugle_date)])
        new.iloc[0, new.columns.get_loc('Volume')] = 0.0
        df = pd.concat([df, new])
        note = (f"ℹ yfinance 最後一根為 {last_date}，已用 Fugle 補上 {fugle_date} 的"
                f"{'即時' if is_intraday else '收盤'}資料")

    i = len(df) - 1
    df.iloc[i, df.columns.get_loc('Close')] = float(price)
    for col, key in (('Open', 'open'), ('High', 'high'), ('Low', 'low')):
        if fq.get(key):
            df.iloc[i, df.columns.get_loc(col)] = float(fq[key])
    if fq.get("volume"):
        # Fugle volume 單位為張，yfinance Volume 為股
        df.iloc[i, df.columns.get_loc('Volume')] = float(fq["volume"]) * 1000

    # 高低必須包住開收，否則影線比例會算出負值
    o, c = df.iloc[i]['Open'], df.iloc[i]['Close']
    df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i]['High'], o, c)
    df.iloc[i, df.columns.get_loc('Low')]  = min(df.iloc[i]['Low'], o, c)

    df = da.calculate_indicators(df)
    df = add_kline_indicators(df, intraday_last=is_intraday)
    if is_intraday and len(df) >= 2 and df.iloc[-2]['Vol_MA5'] > 0:
        # 盤中的量只是累計到目前的部分量，均量分母要沿用前一日的穩定值
        # （與 da._apply_fugle_price 的處理原則一致）
        df.iloc[-1, df.columns.get_loc('Vol_MA5')] = df.iloc[-2]['Vol_MA5']
        df.iloc[-1, df.columns.get_loc('Vol_ratio')] = (
            df.iloc[-1]['Volume'] / df.iloc[-1]['Vol_MA5'])

    return df, note, fq


# ════════════════════════════════════════════════════════════
#  自我檢查（步驟 1 的驗證：python kline_data.py 3037）
# ════════════════════════════════════════════════════════════
def _self_check(code="3037"):
    ticker, label, df = resolve_ticker(code)
    if df is None:
        print(f"  ⚠  無法取得 {code}")
        return

    print(f"\n  {ticker}（{label}）  {len(df)} 根  "
          f"{df.index[0].date()} ~ {df.index[-1].date()}")

    cols = ['MA20', 'MA60', 'MA20_slope', 'MA60_slope', 'ADX', 'DI_plus', 'DI_minus',
            'Range_pos', 'Range_pos252', 'dd_252h', 'Vol_MA20', 'Vol_slope20',
            'ATR_pct', 'BBW_pct', 'ROC_20', 'bars_since_60d_high', 'bars_since_60d_low']
    tail = df.iloc[WARMUP_BARS:]
    bad  = {c: int(tail[c].isna().sum()) for c in cols if tail[c].isna().any()}
    print(f"  暖機後（第 {WARMUP_BARS} 根起，共 {len(tail)} 根）NaN 檢查："
          f"{'✓ 全部欄位無 NaN' if not bad else f'✗ {bad}'}")

    # ADX 忠實度：向量化版最後一根必須等於 backtest_v4 的 float 版
    try:
        import backtest_v4
        ref = backtest_v4.calc_adx(df)
        mine = float(df['ADX'].iloc[-1])
        diff = abs(ref - mine)
        flag = "✓" if diff < 1e-6 else "✗"
        print(f"  ADX 忠實度：{flag} 本模組 {mine:.6f} vs backtest_v4 {ref:.6f}"
              f"（差 {diff:.2e}）")
    except Exception as e:
        print(f"  ADX 忠實度：⚠ 無法比對（{e}）")

    r = df.iloc[-1]
    print(f"\n  最後一根 {df.index[-1].date()}  收 {r['Close']:.2f}")
    print(f"    MA5 {r['MA5']:.2f}  MA10 {r['MA10']:.2f}  "
          f"MA20 {r['MA20']:.2f}  MA60 {r['MA60']:.2f}")
    print(f"    MA20斜率 {r['MA20_slope']:+.2f}%/10根   MA60斜率 {r['MA60_slope']:+.2f}%/20根")
    print(f"    ADX {r['ADX']:.1f}（+DI {r['DI_plus']:.1f} / -DI {r['DI_minus']:.1f}）")
    print(f"    60日位階 {r['Range_pos']*100:.0f}%（{r['Low60']:.1f}~{r['High60']:.1f}）"
          f"  距年高 {r['dd_252h']:+.1f}%  距年低 {r['up_252l']:+.1f}%")
    print(f"    量比 {r['Vol_ratio']:.2f}  量能斜率 {r['Vol_slope20']:+.2f}%/根"
          f"  ATR {r['ATR']:.2f}（{r['ATR_pct']:.1f}%）  BBW百分位 {r['BBW_pct']:.2f}")
    print(f"    ROC20 {r['ROC_20']:+.1f}%  距60日高 {r['bars_since_60d_high']:.0f} 根\n")


if __name__ == "__main__":
    import sys
    _self_check(sys.argv[1] if len(sys.argv) > 1 else "3037")
