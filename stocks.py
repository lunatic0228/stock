# ============================================================
#  股票設定檔
#  觀察名單：watchlist.py（公開，可直接在 GitHub 修改）
#  持倉 / API Key：Streamlit Secrets（私密，不進 GitHub）
# ============================================================

# 觀察名單從公開的 watchlist.py 讀取
from watchlist import WATCHLIST


def _load_from_secrets():
    """雲端部署時從 Streamlit Secrets 讀取持倉與 API Key，失敗回傳 None"""
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
        return fugle_key, finmind_token, holdings
    except Exception:
        return None


_secrets = _load_from_secrets()

if _secrets:
    # ── 雲端：從 Streamlit Secrets 讀取 ──────────────────────
    FUGLE_API_KEY, FINMIND_TOKEN, HOLDINGS = _secrets

else:
    # ── 本地開發用備援（不含真實資料）────────────────────────
    # 真實持倉、API Key 請放在 .streamlit/secrets.toml（已被 .gitignore 排除）
    HOLDINGS      = {}
    FINMIND_TOKEN = ""
    FUGLE_API_KEY = ""
