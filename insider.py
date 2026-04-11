"""
內部人持股異動申報掃描
資料來源：證交所靜態報表 siis.twse.com.tw
  - IRB140：董監事、經理人、大股東轉讓達 100 萬股以上彙總表（每月公告）
  - 資料採月為單位，通常在次月中旬公告

URL 格式：
  上市：https://siis.twse.com.tw/publish/sii/{roc_year}IRB140_{month:02d}.HTM
  上櫃：https://siis.twse.com.tw/publish/otc/{roc_year}IRB140_{month:02d}.HTM
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import re
import urllib3
import warnings

warnings.filterwarnings("ignore")
urllib3.disable_warnings()

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

_BASE_URL = "https://siis.twse.com.tw/publish/{market}/{roc_year}IRB140_{month:02d}.HTM"
_HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


def _ad_to_roc_ym(year: int, month: int) -> tuple[int, int]:
    return year - 1911, month


def _fetch_one(market: str, roc_year: int, month: int) -> pd.DataFrame:
    """
    抓一個月的 IRB140 靜態 HTML，回傳 DataFrame。
    market: "sii"（上市）或 "otc"（上櫃）
    """
    url = _BASE_URL.format(market=market, roc_year=roc_year, month=month)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15, verify=False)
        if resp.status_code != 200:
            return pd.DataFrame()
        resp.encoding = "big5"
        html = resp.text
    except Exception:
        return pd.DataFrame()

    # 解析 HTML table（格式為固定 4 欄）
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    rows = []
    for tbl in tables:
        # 找到有資料的 table（有 <TR><TD>）
        trs = tbl.find_all("tr")
        for tr in trs:
            tds = tr.find_all("td")
            if len(tds) == 4:
                col0 = tds[0].get_text(strip=True)
                col1 = tds[1].get_text(strip=True)
                col2 = tds[2].get_text(strip=True)
                col3 = tds[3].get_text(strip=True)
                rows.append([col0, col1, col2, col3])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["公司名稱_原始", "身份別", "姓名", "轉讓股數_原始"])

    # ── 拆出股票代號（前4碼）與公司名稱 ──────────────────────
    df["代號"] = df["公司名稱_原始"].str[:4].str.strip()
    df["公司名稱"] = df["公司名稱_原始"].str[4:].str.strip()

    # ── 轉讓股數數值化 ────────────────────────────────────────
    df["轉讓股數"] = (
        df["轉讓股數_原始"]
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # ── 轉讓張數 ─────────────────────────────────────────────
    df["轉讓張數"] = df["轉讓股數"] // 1000

    # ── 月份欄位 ─────────────────────────────────────────────
    ad_year = roc_year + 1911
    df["月份"] = f"{ad_year}-{month:02d}"
    df["市場"] = "上市" if market == "sii" else "上櫃"

    # 清理多餘欄位
    df = df.drop(columns=["公司名稱_原始", "轉讓股數_原始"])
    return df


def fetch_insider_changes(
    token: str = "",       # 保留相容性，此函式不使用
    days: int = 90,        # IRB140 是月報，約 90 天 ≈ 3 個月
    min_lots: int = 1000,  # 預設 1000 張（IRB140 本身已是 100 萬股以上）
    event_filter: str = "全部",  # IRB140 只有事後申報，此參數保留相容性
) -> tuple[pd.DataFrame, str | None]:
    """
    全市場掃描董監大股東轉讓達 100 萬股以上申報。

    資料為月報，約次月中旬公告，最近 2-3 個月資料最穩定。

    Parameters
    ----------
    days     : 往前追溯幾天（轉換為月數）
    min_lots : 最小轉讓張數篩選（IRB140 本身已限 1000 張以上）

    Returns
    -------
    (DataFrame, error_msg_or_None)
    """
    if not _BS4_OK:
        return pd.DataFrame(), "缺少套件 beautifulsoup4，請執行：pip install beautifulsoup4"

    # 計算需要抓的月份（從現在往前 N 個月）
    now   = datetime.now()
    start = now - timedelta(days=days)

    months_needed: list[tuple[int, int]] = []
    cur = start.replace(day=1)
    while cur <= now:
        roc_year, month = _ad_to_roc_ym(cur.year, cur.month)
        months_needed.append((roc_year, month))
        # 往後一個月
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    frames = []
    for roc_year, month in months_needed:
        for market in ("sii", "otc"):
            df = _fetch_one(market, roc_year, month)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame(), (
            "查無資料（可能是最近月份尚未公告，"
            "IRB140 通常在次月中旬才公告）"
        )

    df = pd.concat(frames, ignore_index=True)

    # ── 篩選最小張數 ─────────────────────────────────────────
    df = df[df["轉讓張數"] >= min_lots]
    if df.empty:
        return df, None

    # ── 去除空的代號列 ───────────────────────────────────────
    df = df[df["代號"].str.match(r"^\d{4}$", na=False)]

    # ── 整理輸出欄位 ─────────────────────────────────────────
    keep = ["月份", "代號", "公司名稱", "身份別", "姓名", "轉讓張數", "市場"]
    df = (
        df[keep]
        .drop_duplicates()
        .sort_values(["月份", "轉讓張數"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return df, None
