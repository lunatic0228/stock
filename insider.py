"""
內部人持股異動申報掃描
資料來源：公開資訊觀測站 (MOPS)
- 事後申報：t05st09（董監大股東轉讓申報）
- 事前申請：t05st10（內部人申請轉讓）

注意：MOPS 僅允許台灣 IP 存取，本功能需在台灣網路環境下執行。
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import urllib3
import warnings

warnings.filterwarnings("ignore")
urllib3.disable_warnings()

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

_MOPS_BASE  = "https://mops.twse.com.tw/mops/web"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}


def _make_session(endpoint: str) -> requests.Session:
    """建立帶有 session cookie 的 requests session"""
    session = requests.Session()
    session.headers.update(_HEADERS)
    try:
        # 先 GET 主頁以取得 session cookie
        session.get(
            f"{_MOPS_BASE}/{endpoint}",
            timeout=15,
            verify=False,
            allow_redirects=True,
        )
    except Exception:
        pass
    return session


def _ad_to_roc(dt: datetime) -> tuple[int, int]:
    return dt.year - 1911, dt.month


def _parse_mops_table(html: str) -> pd.DataFrame:
    """解析 MOPS 回傳的 HTML，抽出所有 hasBorder 資料表"""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", {"class": "hasBorder"})
    if not tables:
        # 備援：抓所有有 <th> 的 table
        tables = [t for t in soup.find_all("table") if t.find("th")]

    frames = []
    for tbl in tables:
        rows = tbl.find_all("tr")
        if len(rows) < 2:
            continue
        headers = [th.get_text(" ", strip=True) for th in rows[0].find_all(["th", "td"])]
        if not headers:
            continue
        data = []
        for row in rows[1:]:
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if cells and len(cells) == len(headers):
                data.append(cells)
        if data:
            frames.append(pd.DataFrame(data, columns=headers))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fetch_one_month(
    session: requests.Session,
    endpoint: str,
    typek: str,
    roc_year: int,
    month: int,
) -> pd.DataFrame:
    payload = (
        f"encodeURIComponent=1&step=1&firstin=1&off=1"
        f"&TYPEK={typek}&year={roc_year}&month={month:02d}"
    )
    try:
        resp = session.post(
            f"{_MOPS_BASE}/ajax_{endpoint}",
            data=payload,
            headers={**_HEADERS, "Content-Type": "application/x-www-form-urlencoded",
                     "Referer": f"{_MOPS_BASE}/{endpoint}"},
            timeout=25,
            verify=False,
        )
        resp.encoding = "utf-8"
        if "頁面無法執行" in resp.text or "CANNOT BE ACCESSED" in resp.text:
            return pd.DataFrame()
        return _parse_mops_table(resp.text)
    except Exception:
        return pd.DataFrame()


def _fetch_range(endpoint: str, days: int) -> pd.DataFrame:
    """跨月份抓取，合併上市（sii）＋上櫃（otc）"""
    now   = datetime.now()
    start = now - timedelta(days=days)

    months_needed: set[tuple] = set()
    cur = start.replace(day=1)
    while cur <= now:
        months_needed.add(_ad_to_roc(cur))
        cur = cur.replace(month=cur.month % 12 + 1,
                          year=cur.year + (1 if cur.month == 12 else 0))

    session = _make_session(endpoint)

    frames = []
    for roc_year, month in sorted(months_needed):
        for typek in ("sii", "otc"):
            df = _fetch_one_month(session, endpoint, typek, roc_year, month)
            if not df.empty:
                df["市場"] = "上市" if typek == "sii" else "上櫃"
                frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _normalize(df: pd.DataFrame, event_label: str) -> pd.DataFrame:
    """統一欄位名稱（兩個 endpoint 欄名略有差異）"""
    if df.empty:
        return df

    rename_map = {}
    for col in df.columns:
        c = col.replace(" ", "")
        if any(k in c for k in ("申報日", "申請日", "異動日", "轉讓日")):
            rename_map[col] = "申報日"
        elif any(k in c for k in ("股票代號", "代號")):
            rename_map[col] = "代號"
        elif any(k in c for k in ("公司名稱", "公司")):
            rename_map[col] = "公司名稱"
        elif any(k in c for k in ("姓名", "董監事姓名", "申請人")):
            rename_map[col] = "姓名"
        elif any(k in c for k in ("職稱", "身份別", "身分")):
            rename_map[col] = "職稱"
        elif any(k in c for k in ("異動股數", "申請股數", "轉讓股數", "買賣股數")):
            rename_map[col] = "異動股數"
        elif any(k in c for k in ("異動後持股", "轉讓後持股", "持有股數")):
            rename_map[col] = "持股總數"
        elif any(k in c for k in ("原因", "事由", "異動原因")):
            rename_map[col] = "事由"

    df = df.rename(columns=rename_map)
    df["類型"] = event_label
    return df


def _roc_to_iso(s: str) -> str | None:
    """民國日期字串（115/04/11 或 1150411）→ ISO 格式（2026-04-11）"""
    try:
        s = str(s).strip().replace("/", "-")
        parts = s.split("-")
        if len(parts) == 3:
            y = int(parts[0])
            if y < 1000:
                y += 1911
            return f"{y}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        # 純數字格式：1150411
        if len(s) == 7 and s.isdigit():
            y = int(s[:3]) + 1911
            return f"{y}-{s[3:5]}-{s[5:7]}"
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────
#  公開介面
# ────────────────────────────────────────────────────────────

def fetch_insider_changes(
    token: str = "",          # 保留相容性，此函式不使用（資料來自 MOPS）
    days: int = 30,
    min_lots: int = 100,
    event_filter: str = "全部",
) -> tuple[pd.DataFrame, str | None]:
    """
    全市場掃描內部人持股異動申報（MOPS）

    Parameters
    ----------
    days        : 往前追溯幾天
    min_lots    : 最小異動張數（1張=1000股）
    event_filter: "全部" / "事前申請" / "事後申報"

    Returns
    -------
    (DataFrame, error_msg_or_None)
    """
    if not _BS4_OK:
        return pd.DataFrame(), "缺少套件 beautifulsoup4，請執行：pip install beautifulsoup4"

    frames = []

    if event_filter in ("全部", "事後申報"):
        df_post = _fetch_range("t05st09", days)
        df_post = _normalize(df_post, "事後申報")
        if not df_post.empty:
            frames.append(df_post)

    if event_filter in ("全部", "事前申請"):
        df_pre = _fetch_range("t05st10", days)
        df_pre = _normalize(df_pre, "事前申請")
        if not df_pre.empty:
            frames.append(df_pre)

    if not frames:
        return pd.DataFrame(), None

    df = pd.concat(frames, ignore_index=True)

    # ── 數值化異動股數 ────────────────────────────────────────
    if "異動股數" in df.columns:
        df["異動股數_num"] = (
            df["異動股數"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+)")[0]
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        df["異動股數_num"] = 0

    # ── 篩選最小張數 ─────────────────────────────────────────
    df = df[df["異動股數_num"] >= min_lots * 1000]
    if df.empty:
        return df, None

    df["異動張數"] = (df["異動股數_num"] // 1000).astype(int)

    # ── 日期轉換並篩選範圍 ───────────────────────────────────
    if "申報日" in df.columns:
        df["申報日_iso"] = df["申報日"].apply(_roc_to_iso)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = df[df["申報日_iso"].notna() & (df["申報日_iso"] >= cutoff)]
        df["申報日"] = df["申報日_iso"]
        df = df.drop(columns=["申報日_iso"])

    # ── 整理輸出欄位 ─────────────────────────────────────────
    keep = ["申報日", "代號", "公司名稱", "姓名", "職稱", "類型", "異動張數", "事由", "市場"]
    existing = [c for c in keep if c in df.columns]
    df = (
        df[existing]
        .drop_duplicates()
        .sort_values("申報日", ascending=False)
        .reset_index(drop=True)
    )

    return df, None
