"""
台股個人分析系統 - Streamlit Web UI
"""

import streamlit as st
import pandas as pd
import io
import contextlib
from datetime import datetime

st.set_page_config(
    page_title="台股分析系統",
    page_icon="📈",
    layout="wide",
)

# ── Session State 初始化（第一次載入從 stocks.py 讀） ────────
def _init():
    if "holdings" not in st.session_state:
        from stocks import HOLDINGS, WATCHLIST
        st.session_state.holdings  = {k: dict(v) for k, v in HOLDINGS.items()}
        st.session_state.watchlist = {k: list(v) for k, v in WATCHLIST.items()}

_init()


# ── 工具：把 session_state 注入 daily_analysis 模組 ─────────
def _inject_holdings():
    """讓 daily_analysis 使用 session_state 中最新的持股資料"""
    import daily_analysis
    daily_analysis.HOLDINGS  = st.session_state.holdings
    daily_analysis.WATCHLIST = st.session_state.watchlist


# ── 工具：產生 secrets.toml 內容 ────────────────────────────
def _gen_secrets_toml():
    from stocks import FUGLE_API_KEY, FINMIND_TOKEN
    lines = [
        f'FUGLE_API_KEY  = "{FUGLE_API_KEY}"',
        f'FINMIND_TOKEN  = "{FINMIND_TOKEN}"',
        "",
        "[watchlist]",
        f'tw = {st.session_state.watchlist.get("tw", [])}',
        "",
    ]
    for ticker, h in st.session_state.holdings.items():
        lines.append(f'[holdings."{ticker}"]')
        lines.append(f'name      = "{h.get("name","")}"')
        lines.append(f'buy_price = {h["buy_price"]}')
        lines.append(f'shares    = {h["shares"]}')
        lines.append(f'avg_down  = {"true" if h.get("avg_down") else "false"}')
        lines.append(f'building  = {"true" if h.get("building") else "false"}')
        lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  頁面標題
# ════════════════════════════════════════════════════════════
now = datetime.now()
weekday_map = ["一","二","三","四","五","六","日"]
st.title("📈 台股個人分析系統")
st.caption(f"{now.strftime('%Y-%m-%d')}（週{weekday_map[now.weekday()]}）  {now.strftime('%H:%M')}")

tab_analysis, tab_holdings, tab_watchlist = st.tabs(["📊 分析", "💼 持股管理", "👁 觀察名單"])


# ════════════════════════════════════════════════════════════
#  Tab 1：分析
# ════════════════════════════════════════════════════════════
with tab_analysis:
    with st.sidebar:
        st.header("功能選單")
        mode = st.radio(
            "選擇功能",
            ["📊 盤後分析", "🔎 盤中掃描", "⚡ 快速查詢"],
        )
        stock_code = None
        if "快速查詢" in mode:
            stock_code = st.text_input("股票代號", placeholder="例：2313　或　NVDA").strip()

        st.divider()
        run_btn = st.button("▶ 執行分析", type="primary", use_container_width=True)
        st.divider()
        st.markdown("""
**說明**
- 📊 盤後分析：持倉警示＋進場機會
- 🔎 盤中掃描：即時量能＋內外盤
- ⚡ 快速查詢：單股深度分析
        """)

    if not run_btn:
        st.info("👈 左側選擇功能後點擊「執行分析」")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 📊 盤後分析")
            st.write("收盤後執行，持倉警示、攤平／加碼訊號、觀察名單進場機會。")
        with c2:
            st.markdown("### 🔎 盤中掃描")
            st.write("盤中隨時執行，Fugle 即時量能與內外盤，當日操作建議。")
        with c3:
            st.markdown("### ⚡ 快速查詢")
            st.write("輸入股票代號，即時技術指標、持倉狀態、進出場訊號。")
    else:
        _inject_holdings()
        with st.spinner("分析中，請稍候..."):
            buf = io.StringIO()
            err = None
            try:
                with contextlib.redirect_stdout(buf):
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
                err = str(e)

        if err:
            st.error(f"執行錯誤：{err}")
        output = buf.getvalue()
        if output:
            st.code(output, language=None)
        elif not err:
            st.warning("沒有輸出，請確認設定是否正確。")


# ════════════════════════════════════════════════════════════
#  Tab 2：持股管理
# ════════════════════════════════════════════════════════════
with tab_holdings:
    st.header("💼 持股管理")
    st.caption("直接在表格內編輯，點「套用變更」後當次分析立即生效。下載 secrets.toml 可永久儲存。")

    # 持股 → DataFrame
    rows = []
    for ticker, h in st.session_state.holdings.items():
        rows.append({
            "代號":   ticker,
            "名稱":   h.get("name", ""),
            "買入均價": float(h["buy_price"]),
            "持股數":  int(h["shares"]),
            "攤平候選": bool(h.get("avg_down", False)),
            "建倉中":  bool(h.get("building", False)),
        })

    df_h = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["代號","名稱","買入均價","持股數","攤平候選","建倉中"])

    edited_h = st.data_editor(
        df_h,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "代號":    st.column_config.TextColumn("代號", help="例：2313.TW"),
            "名稱":    st.column_config.TextColumn("名稱"),
            "買入均價": st.column_config.NumberColumn("買入均價", format="%.2f", min_value=0.0),
            "持股數":  st.column_config.NumberColumn("持股數", format="%d", min_value=0),
            "攤平候選": st.column_config.CheckboxColumn("攤平候選", help="深套，等超賣反彈訊號"),
            "建倉中":  st.column_config.CheckboxColumn("建倉中",  help="已建部位，持續順勢加碼"),
        },
    )

    col_a, col_b, col_c = st.columns([2, 2, 3])

    with col_a:
        if st.button("✅ 套用變更", type="primary", use_container_width=True):
            new_h = {}
            for _, row in edited_h.iterrows():
                t = str(row.get("代號", "")).strip()
                if not t:
                    continue
                new_h[t] = {
                    "name":      str(row.get("名稱", t)),
                    "buy_price": float(row.get("買入均價", 0)),
                    "shares":    int(row.get("持股數", 0)),
                    "avg_down":  bool(row.get("攤平候選", False)),
                    "building":  bool(row.get("建倉中",  False)),
                }
            st.session_state.holdings = new_h
            st.success(f"已套用！共 {len(new_h)} 筆持股，分析時將使用最新資料。")
            st.rerun()

    with col_b:
        toml_content = _gen_secrets_toml()
        st.download_button(
            "📥 下載 secrets.toml",
            data=toml_content,
            file_name="secrets.toml",
            mime="text/plain",
            use_container_width=True,
            help="下載後貼到 Streamlit Cloud Secrets，永久儲存持股設定",
        )

    with col_c:
        st.info("💡 **永久儲存方式**：下載 secrets.toml → 開啟 Streamlit Cloud → 你的 App → Settings → Secrets → 全部取代貼上 → Save")

    # 即時損益預覽
    st.divider()
    st.subheader("即時損益預覽")

    if st.button("🔄 更新現價"):
        _inject_holdings()
        import yfinance as yf
        import warnings
        warnings.filterwarnings("ignore")

        preview_rows = []
        for ticker, h in st.session_state.holdings.items():
            try:
                price = yf.Ticker(ticker).fast_info.get("last_price") or \
                        yf.Ticker(ticker).history(period="2d").iloc[-1]["Close"]
            except Exception:
                price = h["buy_price"]
            pnl = (price - h["buy_price"]) / h["buy_price"] * 100
            preview_rows.append({
                "代號":   ticker,
                "名稱":   h.get("name",""),
                "買入均價": h["buy_price"],
                "現價":   round(price, 2),
                "損益%":  round(pnl, 2),
                "持股數":  h["shares"],
                "市值":   round(price * h["shares"], 0),
            })

        df_preview = pd.DataFrame(preview_rows)
        st.dataframe(
            df_preview.style.applymap(
                lambda v: "color:red" if isinstance(v, float) and v < 0
                          else ("color:green" if isinstance(v, float) and v > 0 else ""),
                subset=["損益%"]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("點擊「更新現價」查看即時損益（使用 yfinance，延遲約 15 分鐘）")


# ════════════════════════════════════════════════════════════
#  Tab 3：觀察名單
# ════════════════════════════════════════════════════════════
with tab_watchlist:
    st.header("👁 觀察名單")
    st.caption("管理想追蹤但尚未持有的股票。")

    tw_list = st.session_state.watchlist.get("tw", [])

    # 顯示目前清單
    st.subheader("台股觀察名單")
    tw_str = st.text_area(
        "每行一個代號（只填數字，系統自動加 .TW）",
        value="\n".join(tw_list),
        height=200,
        help="例：\n3491\n6285\n2330",
    )

    if st.button("✅ 儲存觀察名單", type="primary"):
        new_tw = [c.strip() for c in tw_str.splitlines() if c.strip()]
        st.session_state.watchlist["tw"] = new_tw
        st.success(f"已儲存 {len(new_tw)} 支台股觀察名單。")
        st.rerun()

    st.info("💡 永久儲存同樣需要下載 secrets.toml（在「持股管理」頁面下載）")
