"""
台股個人分析系統 - Streamlit Web UI
用法：streamlit run app.py
"""

import streamlit as st
import io
import contextlib
from datetime import datetime

st.set_page_config(
    page_title="台股分析系統",
    page_icon="📈",
    layout="wide",
)

# ── 頁面標題 ──────────────────────────────────────────────────
st.title("📈 台股個人分析系統")

now = datetime.now()
weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
st.caption(f"{now.strftime('%Y-%m-%d')}（週{weekday_map[now.weekday()]}）  {now.strftime('%H:%M')} 更新")

# ── 側邊欄 ────────────────────────────────────────────────────
with st.sidebar:
    st.header("功能選單")

    mode = st.radio(
        "選擇功能",
        ["📊 盤後分析", "🔎 盤中掃描", "⚡ 快速查詢"],
    )

    stock_code = None
    if "快速查詢" in mode:
        stock_code = st.text_input(
            "股票代號",
            placeholder="例：2313　或　NVDA",
        ).strip()

    st.divider()
    run_btn = st.button("▶ 執行分析", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
**使用說明**
- 📊 盤後分析：持倉警示 + 進場機會
- 🔎 盤中掃描：即時量能 + 內外盤
- ⚡ 快速查詢：單股深度分析
""")

# ── 主畫面 ────────────────────────────────────────────────────
if not run_btn:
    st.info("👈 請在左側選擇功能後點擊「執行分析」")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 盤後分析")
        st.write("每日收盤後執行，顯示持倉警示、攤平／加碼訊號、觀察名單進場機會。")
    with col2:
        st.markdown("### 🔎 盤中掃描")
        st.write("盤中隨時執行，根據 Fugle 即時量能與內外盤給出當日操作建議。")
    with col3:
        st.markdown("### ⚡ 快速查詢")
        st.write("輸入股票代號，即時顯示技術指標、持倉狀態、進出場訊號。")

else:
    # 執行分析
    with st.spinner("分析中，請稍候..."):
        output_buf = io.StringIO()
        error_msg  = None

        try:
            # redirect_stdout 才能捕捉 daily_analysis 裡的 print()
            with contextlib.redirect_stdout(output_buf):
                from daily_analysis import run, quick_lookup, intraday_scan

                if "盤後分析" in mode:
                    run()
                elif "盤中掃描" in mode:
                    intraday_scan()
                elif "快速查詢" in mode:
                    if stock_code:
                        quick_lookup(stock_code)
                    else:
                        print("  ⚠  請先輸入股票代號")

        except Exception as e:
            error_msg = str(e)

    output = output_buf.getvalue()

    if error_msg:
        st.error(f"執行錯誤：{error_msg}")

    if output:
        # 用等寬字體顯示，保留原本的對齊格式
        st.code(output, language=None)
    elif not error_msg:
        st.warning("沒有輸出，請確認設定是否正確。")
