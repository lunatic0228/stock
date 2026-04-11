"""
大股東增持掃描 + 進場分析
資料來源：MOPS t93sb06_1「持股10%以上大股東最近異動情形」
分析：借用 daily_analysis 的技術指標 + TWSE/TPEX PE/PB（全免費）

流程：
  ① 抓上市 + 上櫃本月大股東異動
  ② 篩選「異動數 > 0」（本月比上月多）
  ③ 每支股票跑 yfinance 技術分析 + entry_signals
  ④ 輸出進場評分 + 現價位置 + 停損線
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import urllib3
import warnings

warnings.filterwarnings("ignore")
urllib3.disable_warnings()

_MOPS_BASE = "https://mopsov.twse.com.tw/mops/web"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-TW,zh;q=0.9",
}


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(f"{_MOPS_BASE}/t93sb06_1", timeout=15, verify=False)
    except Exception:
        pass
    return s


def _fetch_t93sb06_1(session: requests.Session, typek: str, roc_year: int, month: int) -> pd.DataFrame:
    """抓單一市場單月的大股東異動資料"""
    payload = (
        f"encodeURIComponent=1&step=1&firstin=1&off=1"
        f"&TYPEK={typek}&year={roc_year}&month={month:02d}"
    )
    try:
        resp = session.post(
            f"{_MOPS_BASE}/ajax_t93sb06_1",
            data=payload,
            headers={**_HEADERS, "Content-Type": "application/x-www-form-urlencoded",
                     "Referer": f"{_MOPS_BASE}/t93sb06_1"},
            timeout=25, verify=False,
        )
        resp.encoding = "utf-8"
        html = resp.text
    except Exception:
        return pd.DataFrame()

    if "頁面無法執行" in html or "CANNOT BE ACCESSED" in html:
        return pd.DataFrame()

    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table", {"class": "hasBorder"})
    if not tbl:
        return pd.DataFrame()

    rows = []
    for tr in tbl.find_all("tr")[1:]:
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) >= 6:
            rows.append(tds[:6])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["代號", "公司名稱", "大股東名稱", "上月持股", "本月持股", "異動數"])
    df["市場"] = "上市" if typek == "sii" else "上櫃"
    return df


def _parse_shares(s: str) -> int:
    try:
        return int(str(s).replace(",", "").strip())
    except Exception:
        return 0


def fetch_increasing_shareholders(days_back: int = 60) -> tuple[pd.DataFrame, str | None]:
    """
    抓近幾個月大股東異動，篩出「增持」的公司。
    回傳去重後的 DataFrame（一公司可能多個大股東增持，只保留最大增持那筆）
    """
    now = datetime.now()
    months = set()
    cur = (now - timedelta(days=days_back)).replace(day=1)
    while cur <= now:
        roc_year = cur.year - 1911
        months.add((roc_year, cur.month))
        cur = cur.replace(month=cur.month % 12 + 1,
                          year=cur.year + (1 if cur.month == 12 else 0))

    session = _make_session()
    frames = []
    for roc_year, month in sorted(months):
        for typek in ("sii", "otc"):
            df = _fetch_t93sb06_1(session, typek, roc_year, month)
            if not df.empty:
                df["查詢年月"] = f"{roc_year + 1911}-{month:02d}"
                frames.append(df)

    if not frames:
        return pd.DataFrame(), "無法取得 MOPS 資料（可能需要台灣 IP 或服務暫時不可用）"

    df = pd.concat(frames, ignore_index=True)

    # 數值化
    df["上月持股_num"] = df["上月持股"].apply(_parse_shares)
    df["本月持股_num"] = df["本月持股"].apply(_parse_shares)
    df["異動數_num"]   = df["異動數"].apply(_parse_shares)

    # 只留增持（異動數 > 0）
    df = df[df["異動數_num"] > 0].copy()
    if df.empty:
        return pd.DataFrame(), None

    # 代號只取數字4碼
    df = df[df["代號"].str.match(r"^\d{4}$", na=False)]

    # 同公司取異動最大那筆（可能多個大股東同時增持）
    df = (
        df.sort_values("異動數_num", ascending=False)
          .drop_duplicates(subset=["代號", "查詢年月"])
          .sort_values(["查詢年月", "異動數_num"], ascending=[False, False])
          .reset_index(drop=True)
    )

    # 增持張數
    df["增持張數"] = (df["異動數_num"] // 1000).astype(int)

    keep = ["查詢年月", "代號", "公司名稱", "大股東名稱", "增持張數", "市場"]
    return df[[c for c in keep if c in df.columns]], None


# ────────────────────────────────────────────────────────────
#  技術分析 + 進場評分（借用 daily_analysis）
# ────────────────────────────────────────────────────────────

def _analyze_one(code: str) -> dict | None:
    """
    對單一股票代號跑技術分析。
    回傳 dict 或 None（資料不足）
    """
    import sys, io, contextlib
    sys.path.insert(0, ".")

    try:
        import daily_analysis as da
    except Exception:
        return None

    ticker = f"{code}.TW"

    # 靜默抓資料（先試上市，失敗再試上櫃）
    df = da.fetch(ticker, silent=True)
    if df is None:
        df = da.fetch(f"{code}.TWO", silent=True)
        if df is None:
            return None
        ticker = f"{code}.TWO"

    r = df.iloc[-1]
    close   = r["Close"]
    ma5     = r["MA5"]
    ma10    = r["MA10"]
    rsi     = r["RSI"]
    atr     = r["ATR"]
    vr      = r["Vol_ratio"]
    macd_h  = r["MACD_hist"]

    # 進場訊號
    entry_score, entry_msgs = da.entry_signals(df)

    # 估值（TWSE/TPEX OpenAPI，免費）
    val = da.get_valuation(code)
    pe  = val.get("pe")  if val else None
    pb  = val.get("pb")  if val else None
    div = val.get("div") if val else None

    # ATR 停損線
    stop = close - 2 * atr

    return {
        "ticker":       ticker,
        "close":        close,
        "ma5":          ma5,
        "ma10":         ma10,
        "rsi":          rsi,
        "vol_ratio":    vr,
        "macd_hist":    macd_h,
        "atr":          atr,
        "stop":         stop,
        "entry_score":  entry_score,
        "entry_msgs":   entry_msgs,
        "pe":           pe,
        "pb":           pb,
        "div":          div,
    }


def run_insider_scan(days_back: int = 60, min_lots: int = 0) -> None:
    """
    主入口：掃描增持名單 → 逐支分析 → 印出報告
    供 app.py 透過 redirect_stdout 捕捉輸出
    """
    # ① 取得增持名單
    df_inc, err = fetch_increasing_shareholders(days_back=days_back)
    if err:
        print(f"  ⚠  {err}")
        return
    if df_inc.empty:
        print(f"  近 {days_back} 天內找不到持股增加的大股東紀錄。")
        return

    # 篩選最小張數
    if min_lots > 0 and "增持張數" in df_inc.columns:
        df_inc = df_inc[df_inc["增持張數"] >= min_lots]
    if df_inc.empty:
        print(f"  篩選後（≥{min_lots}張）無符合資料。")
        return

    print(f"  共找到 {len(df_inc)} 筆增持紀錄，開始逐支分析...\n")
    print("=" * 68)

    for _, row in df_inc.iterrows():
        code    = row["代號"]
        name    = row.get("公司名稱", "")
        holder  = row.get("大股東名稱", "")
        lots    = row.get("增持張數", 0)
        month   = row.get("查詢年月", "")
        market  = row.get("市場", "")

        print(f"\n▌ {code} {name}  [{market}]  {month}")
        print(f"  大股東：{holder}")
        print(f"  本月增持：{lots:,} 張")
        print()

        res = _analyze_one(code)
        if res is None:
            print("  ⚠  無法取得股價資料（yfinance 無此股票）\n")
            print("-" * 68)
            continue

        # 現價 & 技術狀態
        trend = "多頭" if res["ma5"] > res["ma10"] else "空頭"
        macd_dir = "↑擴張" if res["macd_hist"] > 0 else "↓收縮"
        print(f"  現價  {res['close']:.1f}　MA5 {res['ma5']:.1f}　MA10 {res['ma10']:.1f}　{trend}排列")
        print(f"  RSI   {res['rsi']:.1f}　量比 {res['vol_ratio']:.2f}　MACD {macd_dir}")
        print(f"  ATR停損線  {res['stop']:.1f}　（現價 - 2×ATR）")

        # 估值
        val_parts = []
        if res["pe"]:  val_parts.append(f"本益比 {res['pe']:.1f}x")
        if res["pb"]:  val_parts.append(f"本淨比 {res['pb']:.2f}x")
        if res["div"]: val_parts.append(f"殖利率 {res['div']:.2f}%")
        if val_parts:
            print(f"  估值  " + "　".join(val_parts))

        # 進場評分
        score = res["entry_score"]
        bar   = "■" * score + "□" * (4 - score)
        verdict = (
            "⭐ 可考慮進場" if score >= 4 else
            "🔶 條件接近，可追蹤" if score == 3 else
            "⏳ 尚未到位，觀望"
        )
        print()
        print(f"  進場評分  {bar}  {score}/4   {verdict}")
        for msg in res["entry_msgs"]:
            print(msg)

        print("-" * 68)

    print("\n✅ 掃描完成")
    print("  資料來源：MOPS t93sb06_1（持股10%以上大股東異動）+ yfinance + TWSE OpenAPI")
    print("  提醒：本分析僅供參考，月報資料有 1~2 個月時間差。")
