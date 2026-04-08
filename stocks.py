# ============================================================
#  股票設定檔
#  本地執行：直接修改下方 HOLDINGS / WATCHLIST / API Keys
#  雲端部署：資料從 Streamlit Secrets 讀取，不需要修改這裡
# ============================================================

def _load_from_secrets():
    """雲端部署時從 Streamlit Secrets 讀取設定，失敗回傳 None"""
    try:
        import streamlit as st
        if "FUGLE_API_KEY" not in st.secrets:
            return None
        fugle_key     = st.secrets.get("FUGLE_API_KEY", "")
        finmind_token = st.secrets.get("FINMIND_TOKEN", "")
        holdings = {}
        for ticker, h in st.secrets.get("holdings", {}).items():
            holdings[ticker] = {
                "name":      str(h.get("name", ticker)),
                "buy_price": float(h.get("buy_price", 0)),
                "shares":    int(h.get("shares", 0)),
                "avg_down":  bool(h.get("avg_down", False)),
                "building":  bool(h.get("building", False)),
            }
        tw_list   = list(st.secrets.get("watchlist", {}).get("tw", []))
        watchlist = {"tw": tw_list, "us": []}
        return fugle_key, finmind_token, holdings, watchlist
    except Exception:
        return None

_secrets = _load_from_secrets()
if _secrets:
    FUGLE_API_KEY, FINMIND_TOKEN, HOLDINGS, WATCHLIST = _secrets
else:
    # ── 以下為本地設定，雲端部署時不會用到 ──────────────────

# 觀察名單（還沒持有、想追蹤的股票）
# 台股直接填代號，美股填美股代號
#
# ── 低軌衛星概念股 ──────────────────────────────────────────
# 3491 昇達科   : 最純低軌衛星股，Ka/Ku Band 射頻元件，已打入 SpaceX/Kuiper/OneWeb
# 6285 啟碁     : 相位陣列天線（終端設備），供 SpaceX + Kuiper，從電信轉型衛星終端整合
# 3105 穩懋     : GaAs 功率放大器晶片代工龍頭，衛星射頻晶片必用
#
# ── AI 伺服器散熱 ────────────────────────────────────────────
# 3017 奇鋐     : 液冷散熱指標龍頭，AI 伺服器耗電爆增，液冷需求暴增
#                 （台達電已持有做電源，奇鋐做散熱，兩者互補）
#
# ── AI 伺服器組裝 ────────────────────────────────────────────
# 6669 緯穎     : AI 伺服器純度最高的 ODM 廠，GB300 訂單為主力，AI 佔比 7 成+
#
# ── 人形機器人 ───────────────────────────────────────────────
# 2049 上銀     : 線性滑軌 + 滾珠螺桿，機器人底層關鍵零件，台灣機器人指標股
WATCHLIST = {
    "tw": ["3491", "6285", "3105", "3017", "6669", "2049"],
    "us": [],
}

# 目前持倉
# buy_price : 平均買入價
# shares    : 持有股數
# avg_down  : True = 深套，找低點攤平（超賣反彈訊號）
# building  : True = 建倉中，持續買進（順勢加碼訊號）
# 兩個可以同時為 True
HOLDINGS = {
    "2308.TW": {
        "name":      "台達電",
        "buy_price": 1296.25,
        "shares":    80,
        "avg_down":  False,
        "building":  True,
    },
    "2344.TW": {
        "name":      "華邦電",
        "buy_price": 122.0,
        "shares":    1000,
        "avg_down":  True,
        "building":  False,
    },
    "2367.TW": {
        "name":      "燿華",
        "buy_price": 74.33,
        "shares":    1150,
        "avg_down":  False,
        "building":  True,
    },
    "6282.TW": {
        "name":      "康舒",
        "buy_price": 47.16,
        "shares":    1200,
        "avg_down":  False,
        "building":  True,
    },
    "2607.TW": {
        "name":      "榮運",
        "buy_price": 57.8,
        "shares":    2000,
        "avg_down":  True,
        "building":  False,
    },
    "6919.TW": {
        "name":      "康霈生技",
        "buy_price": 160.28,
        "shares":    3000,
        "avg_down":  True,
        "building":  False,
    },
    "2313.TW": {
        "name":      "華通",
        "buy_price": 240.67,
        "shares":    450,
        "avg_down":  False,
        "building":  True,
    },
    "0050.TW": {
        "name":      "元大台灣50",
        "buy_price": 69.23,
        "shares":    6000,
        "avg_down":  False,
        "building":  True,
    },
}

    # FinMind API Token（選填）
    FINMIND_TOKEN = ""

    # Fugle Market Data API Key（選填，強烈建議申請）
    FUGLE_API_KEY = "NTZmNWJmNzYtZjBjYi00ZTI2LTgxMTItOTg4ZTdjNjE3OTY0IDI0ZjZhYzlhLWQxOGYtNDMwMS04ZDI4LWU0YWYyMDU1NWUyMg=="
