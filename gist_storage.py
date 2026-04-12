"""
Gist-based persistent storage for holdings and watchlist.
Reads/writes a single JSON file (stock_data.json) in a private GitHub Gist.
"""

import json
import requests
import streamlit as st

_GIST_FILE = "stock_data.json"
_TIMEOUT   = 6


def _creds():
    try:
        token   = st.secrets.get("GITHUB_TOKEN", "")
        gist_id = st.secrets.get("GIST_ID", "")
        return token, gist_id
    except Exception:
        return "", ""


def load_from_gist():
    """
    從 Gist 讀取持股和觀察名單。
    成功回傳 {"holdings": {...}, "watchlist": {...}}，失敗或空資料回傳 None。
    """
    token, gist_id = _creds()
    if not token or not gist_id:
        return None
    try:
        resp = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        files = resp.json().get("files", {})
        if _GIST_FILE not in files:
            return None
        content = files[_GIST_FILE]["content"]
        data = json.loads(content)
        if "holdings" in data and "watchlist" in data:
            return data
        return None
    except Exception:
        return None


def save_to_gist(holdings: dict, watchlist: dict) -> bool:
    """
    把持股和觀察名單存到 Gist。成功回傳 True，失敗回傳 False。
    """
    token, gist_id = _creds()
    if not token or not gist_id:
        return False
    try:
        payload = {
            "files": {
                _GIST_FILE: {
                    "content": json.dumps(
                        {"holdings": holdings, "watchlist": watchlist},
                        ensure_ascii=False,
                        indent=2,
                    )
                }
            }
        }
        resp = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            json=payload,
            timeout=_TIMEOUT,
        )
        return resp.status_code == 200
    except Exception:
        return False
