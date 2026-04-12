"""
Beta 功能頁面：大股東增持掃描（含流動性過濾）
Streamlit multi-page app — 放在 pages/ 資料夾，自動出現在側邊欄
"""

import streamlit as st
import sys
import io
import contextlib

# ── 頁面設定 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🧪 Beta – 大股東增持掃描",
    page_icon="🧪",
    layout="wide",
)

# ── PIN 驗證（沿用主程式的 session_state） ───────────────────
APP_PIN = st.secrets.get("APP_PIN", "")

if APP_PIN:
    if not st.session_state.get("authenticated", False):
        st.title("🔒 請輸入 PIN 碼")
        pin_input = st.text_input("PIN", type="password", key="beta_pin")
        if st.button("確認"):
            if pin_input == APP_PIN:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("PIN 碼錯誤")
        st.stop()

# ── 主畫面 ───────────────────────────────────────────────────
st.title("🧪 Beta：大股東增持掃描")
st.caption(
    "資料來源：MOPS t93sb06_1（持股10%以上大股東最近異動情形）＋ yfinance＋TWSE OpenAPI  |  "
    "月報資料有 1~2 個月時間差，僅供參考。"
)

st.info(
    "**Beta 新功能**\n"
    "- 顯示近20日日均交易量（張）\n"
    "- 自動過濾日均量不足門檻的標的（流動性過濾）"
)

st.divider()

# ── 參數設定 ─────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    days_back = st.selectbox(
        "回溯天數",
        options=[30, 60, 90],
        index=1,
        help="往前追溯幾天的大股東異動資料",
    )

with col2:
    min_lots = st.number_input(
        "最小增持張數",
        min_value=0,
        max_value=100000,
        value=500,
        step=100,
        help="增持張數低於此值的紀錄不列入分析（0 = 不過濾）",
    )

with col3:
    min_avg_vol = st.number_input(
        "最低日均量（張）",
        min_value=0,
        max_value=100000,
        value=500,
        step=100,
        help="近20日日均成交張數低於此值的標的將略過（流動性過濾）",
    )

st.divider()

# ── 掃描按鈕 ─────────────────────────────────────────────────
if st.button("🔍 開始掃描", type="primary", use_container_width=True):
    try:
        import insider_scan_beta as isb
    except ImportError as e:
        st.error(f"無法載入 insider_scan_beta 模組：{e}")
        st.stop()

    with st.spinner("正在抓取 MOPS 資料並逐支分析，請稍候..."):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            isb.run_insider_scan_beta(
                days_back=int(days_back),
                min_lots=int(min_lots),
                min_avg_vol=int(min_avg_vol),
            )
        output = buf.getvalue()

    st.code(output, language="")

    st.caption(
        "⚠️ 本頁為 Beta 測試版，功能尚在驗證中。"
        "正式版請回到主頁面使用。"
    )
