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
    # ── 雲端：從 Streamlit Secrets 讀取 ──────────────────────
    FUGLE_API_KEY, FINMIND_TOKEN, HOLDINGS, WATCHLIST = _secrets

else:
    # ── 本地開發用備援（不含真實資料）────────────────────────
    # 真實持倉、API Key 請放在 .streamlit/secrets.toml（已被 .gitignore 排除）
    # 格式範例請參考 secrets_example.toml
    WATCHLIST     = {"tw": [], "us": []}
    HOLDINGS      = {}
    FINMIND_TOKEN = ""
    FUGLE_API_KEY = ""
