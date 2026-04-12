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

# ════════════════════════════════════════════════════════════
#  密碼鎖（整個 app 的入口）
# ════════════════════════════════════════════════════════════
def _check_pin():
    """回傳 True 表示已解鎖，False 表示顯示密碼頁面並停止後續渲染"""
    if st.session_state.get("authenticated"):
        return True

    # 讀取 PIN（優先 Secrets，備援 hardcode）
    try:
        correct_pin = st.secrets.get("APP_PIN", "0202")
    except Exception:
        correct_pin = "0202"

    # 置中卡片樣式
    st.markdown("""
    <style>
    .pin-box {
        max-width: 340px;
        margin: 12vh auto 0 auto;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        background: #1e1e2e;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        text-align: center;
    }
    .pin-title { font-size: 2rem; margin-bottom: 0.3rem; }
    .pin-sub   { color: #aaa; font-size: 0.95rem; margin-bottom: 1.5rem; }
    </style>
    <div class="pin-box">
      <div class="pin-title">📈 台股分析系統</div>
      <div class="pin-sub">請輸入 4 位數密碼</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        pin_input = st.text_input(
            "密碼",
            type="password",
            max_chars=4,
            placeholder="• • • •",
            label_visibility="collapsed",
            key="pin_input",
        )
        unlock_btn = st.button("解鎖", type="primary", use_container_width=True)

        if unlock_btn or (pin_input and len(pin_input) == 4):
            if pin_input == correct_pin:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("密碼錯誤，請重試")

    return False   # 尚未解鎖，後面的程式碼不執行

if not _check_pin():
    st.stop()

# ── Session State 初始化（第一次載入從 stocks.py / watchlist.py 讀） ─
def _init():
    if "holdings" not in st.session_state:
        from stocks import HOLDINGS
        from watchlist import WATCHLIST
        st.session_state.holdings  = {k: dict(v) for k, v in HOLDINGS.items()}
        st.session_state.watchlist = {k: list(v) for k, v in WATCHLIST.items()}

_init()


# ── 工具：把 session_state 注入 daily_analysis 模組 ─────────
def _inject_holdings():
    """讓 daily_analysis 使用 session_state 中最新的持股 + 觀察名單"""
    import importlib, daily_analysis
    importlib.reload(daily_analysis)          # 強制重載，避免 Streamlit module cache 問題
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

tab_analysis, tab_holdings, tab_watchlist, tab_insider, tab_beta, tab_guide = st.tabs(["📊 分析", "💼 持股管理", "👁 觀察名單", "🕵 內部人申報", "🧪 Beta", "📖 指標說明"])


# ════════════════════════════════════════════════════════════
#  Tab 1：分析
# ════════════════════════════════════════════════════════════
with tab_analysis:
    with st.sidebar:
        st.header("功能選單")
        mode = st.radio(
            "選擇功能",
            ["📊 盤後分析", "🔎 盤中掃描", "👁 觀察名單", "⚡ 快速查詢"],
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
- 🔎 盤中掃描：持倉即時量能＋內外盤
- 👁 觀察名單：所有觀察標的進場條件
- ⚡ 快速查詢：單股深度分析
        """)
        st.divider()
        if st.button("🔒 登出", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    if not run_btn:
        st.info("👈 左側選擇功能後點擊「執行分析」")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("### 📊 盤後分析")
            st.write("收盤後執行，持倉警示、攤平／加碼訊號。")
        with c2:
            st.markdown("### 🔎 盤中掃描")
            st.write("持倉即時量能與內外盤，當日操作建議。")
        with c3:
            st.markdown("### 👁 觀察名單")
            st.write("所有觀察標的即時進場條件，快速篩選機會。")
        with c4:
            st.markdown("### ⚡ 快速查詢")
            st.write("輸入股票代號，即時技術指標與進出場訊號。")
    else:
        _inject_holdings()   # 同時 reload + 注入最新持股 / 觀察名單
        with st.spinner("分析中，請稍候..."):
            buf = io.StringIO()
            err = None
            try:
                import daily_analysis as _da
                with contextlib.redirect_stdout(buf):
                    if "盤後分析" in mode:
                        _da.run()
                    elif "盤中掃描" in mode:
                        _da.intraday_scan()
                    elif "觀察名單" in mode:
                        _da.watchlist_scan()
                    elif "快速查詢" in mode:
                        if stock_code:
                            _da.quick_lookup(stock_code)
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
    st.header("👁 觀察名單管理")
    st.caption("管理想追蹤但尚未持有的股票。變更後當次分析立即生效。")

    tw_list = st.session_state.watchlist.get("tw", [])

    # 用 data_editor 顯示可編輯清單
    wl_df = pd.DataFrame({"代號": tw_list})
    edited_wl = st.data_editor(
        wl_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "代號": st.column_config.TextColumn("股票代號（只填數字）", help="例：2330、6285"),
        },
        hide_index=True,
        key="wl_editor",
    )

    col_save, col_info = st.columns([1, 2])
    with col_save:
        if st.button("✅ 套用變更", type="primary", use_container_width=True, key="wl_save_btn"):
            new_tw = [str(r).strip() for r in edited_wl["代號"].dropna() if str(r).strip()]
            st.session_state.watchlist["tw"] = new_tw
            st.success(f"已更新 {len(new_tw)} 支觀察名單，下次執行分析立即生效。")
            st.rerun()
    with col_info:
        st.info("💡 永久儲存：直接修改 GitHub 上的 `watchlist.py` 即可。")


# ════════════════════════════════════════════════════════════
#  Tab 4：內部人申報
# ════════════════════════════════════════════════════════════
with tab_insider:
    st.header("🕵 大股東增持掃描")
    st.caption(
        "資料來源：MOPS t93sb06_1「持股10%以上大股東最近異動」｜"
        "篩出**增持**公司 → 自動跑技術分析 → 給出進場評分"
    )

    col_i1, col_i2, col_i3 = st.columns([2, 2, 1])

    with col_i1:
        insider_days = st.select_slider(
            "回溯天數",
            options=[30, 60, 90],
            value=60,
            help="往前幾天的月報資料（月報有 1~2 個月時間差）",
        )
    with col_i2:
        insider_min_lots = st.number_input(
            "最小增持張數（張）",
            min_value=0,
            max_value=100000,
            value=500,
            step=100,
            help="預設 500 張（約 30 筆，分析約 1 分鐘）；設 0 = 全部（約 150 筆，需 4 分鐘）",
        )
    with col_i3:
        st.write("")
        st.write("")
        run_insider = st.button("🔍 掃描分析", type="primary", use_container_width=True)

    st.divider()

    if not run_insider:
        st.info(
            "點擊「掃描分析」後系統會：\n\n"
            "**① 抓資料**　MOPS 大股東持股異動，篩出本月比上月**增加**的公司\n\n"
            "**② 技術分析**　每支股票跑 MA / RSI / 量比 / MACD / ATR\n\n"
            "**③ 進場評分**　套用本系統進場條件（路徑A回調 / 路徑B突破），4分滿分\n\n"
            "**④ 一氣呵成輸出**　現價、估值、停損線、進場建議\n\n"
            "---\n"
            "**注意**：月報有 1~2 個月時間差，增持訊號是**中線參考**，不是當天訊號。\n"
            "API 全部免費（yfinance + TWSE OpenAPI），不消耗 FinMind 配額。"
        )
    else:
        with st.spinner("掃描中，請稍候（每支股票約 1~2 秒）..."):
            import io, contextlib
            from insider_scan import run_insider_scan
            buf = io.StringIO()
            err = None
            try:
                with contextlib.redirect_stdout(buf):
                    run_insider_scan(
                        days_back=insider_days,
                        min_lots=int(insider_min_lots),
                    )
            except Exception as e:
                err = str(e)

        if err:
            st.error(f"執行錯誤：{err}")
        else:
            output = buf.getvalue()
            if output:
                st.code(output, language=None)
            else:
                st.warning("沒有輸出，請確認網路是否可連到 MOPS。")


# ════════════════════════════════════════════════════════════
#  Tab 5：Beta — 大股東增持掃描（含流動性過濾）
# ════════════════════════════════════════════════════════════
with tab_beta:
    st.header("🧪 Beta：大股東增持掃描（含流動性過濾）")
    st.caption(
        "資料來源：MOPS t93sb06_1（持股10%以上大股東最近異動情形）＋ yfinance＋TWSE OpenAPI  |  "
        "月報資料有 1~2 個月時間差，僅供參考。"
    )
    st.info(
        "**Beta 新功能（相較「內部人申報」分頁）**\n\n"
        "- 顯示近20日日均交易量（張）\n"
        "- 自動過濾日均量不足門檻的標的（流動性不足略過）"
    )

    col_b1, col_b2, col_b3, col_b4 = st.columns([2, 2, 2, 1])

    with col_b1:
        beta_days = st.select_slider(
            "回溯天數",
            options=[30, 60, 90],
            value=60,
            key="beta_days",
            help="往前追溯幾天的大股東異動資料",
        )
    with col_b2:
        beta_min_lots = st.number_input(
            "最小增持張數（張）",
            min_value=0,
            max_value=100000,
            value=500,
            step=100,
            key="beta_min_lots",
            help="增持張數低於此值的紀錄不列入分析（0 = 不過濾）",
        )
    with col_b3:
        beta_min_vol = st.number_input(
            "最低日均量（張）",
            min_value=0,
            max_value=100000,
            value=500,
            step=100,
            key="beta_min_vol",
            help="近20日日均成交張數低於此值的標的將略過（流動性過濾）",
        )
    with col_b4:
        st.write("")
        st.write("")
        run_beta = st.button("🔍 掃描", type="primary", use_container_width=True, key="run_beta")

    st.divider()

    if not run_beta:
        st.info(
            "點擊「掃描」後系統會：\n\n"
            "**① 抓資料**　MOPS 大股東持股異動，篩出本月比上月**增加**的公司\n\n"
            "**② 流動性過濾**　日均量 < 門檻 → 略過，避免抓到私募或冷門股\n\n"
            "**③ 技術分析**　每支股票跑 MA / RSI / 量比 / MACD / ATR\n\n"
            "**④ 進場評分**　套用本系統進場條件，4分滿分\n\n"
            "---\n"
            "⚠️ 此為 Beta 測試版，穩定後將取代「內部人申報」分頁。"
        )
    else:
        with st.spinner("掃描中，請稍候（每支股票約 1~2 秒）..."):
            try:
                import insider_scan_beta as _isb
            except ImportError as _ie:
                st.error(f"無法載入 insider_scan_beta 模組：{_ie}")
                st.stop()

            buf = io.StringIO()
            _err = None
            try:
                with contextlib.redirect_stdout(buf):
                    _isb.run_insider_scan_beta(
                        days_back=int(beta_days),
                        min_lots=int(beta_min_lots),
                        min_avg_vol=int(beta_min_vol),
                    )
            except Exception as _e:
                _err = str(_e)

        if _err:
            st.error(f"執行錯誤：{_err}")
        else:
            _output = buf.getvalue()
            if _output:
                st.code(_output, language=None)
            else:
                st.warning("沒有輸出，請確認網路是否可連到 MOPS。")


# ════════════════════════════════════════════════════════════
#  Tab 6：指標說明
# ════════════════════════════════════════════════════════════
with tab_guide:
    st.header("📖 指標說明")
    st.caption("快速查閱本系統使用的技術指標含義與判斷標準")

    # ── 趨勢指標 ──────────────────────────────────────────
    st.subheader("📈 趨勢指標")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### MA5 / MA10　五日／十日均線")
        st.markdown("""
| 狀況 | 意義 |
|------|------|
| MA5 > MA10 | 多頭排列，短線趨勢向上 ✅ |
| MA5 < MA10 | 空頭排列，短線趨勢向下 ❌ |
| 現價 > MA5 | 短線強勢 |
| 現價 < MA5 | 短線偏弱，留意支撐 |

> **白話**：MA5 是最近 5 天的平均收盤價，MA10 是 10 天的。
> MA5 在 MA10 上方，代表近期漲勢比中期強，趨勢向上。
        """)

    with col2:
        st.markdown("#### 乖離率　現價偏離 MA5 的程度")
        st.markdown("""
| 數值 | 意義 |
|------|------|
| +5% 以上 | 短線偏熱，不追高 |
| +8% 且 RSI > 70 | 減碼訊號 🟢 |
| +12% 且 RSI > 78 | 強力停利訊號 🟢 |
| -5% 以下 | 偏離均線，可能超賣 |

> **白話**：漲太快、離均線太遠，通常會拉回修正。
> 乖離率越大 = 短線越危險，不是買點而是賣點。
        """)

    st.divider()

    # ── 動能指標 ──────────────────────────────────────────
    st.subheader("⚡ 動能指標")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### RSI　相對強弱指標（14日）")
        st.markdown("""
| 數值 | 意義 |
|------|------|
| > 70 | 超買區，漲勢過熱，留意回檔 🔥 |
| 50 ~ 70 | 多頭動能，正常上漲 ✅ |
| 45 ~ 50 | 中性偏多，可加碼甜蜜點 |
| 30 ~ 45 | 動能偏弱，謹慎 |
| < 30 | 超賣區，可能反彈，攤平參考 🧊 |

> **白話**：RSI 衡量「近期漲幅有多強」。
> 超過 70 代表大家都在搶買、過熱；低於 30 代表大家都在殺出、過冷，可能是反彈機會。
        """)

    with col4:
        st.markdown("#### MACD　指數平滑異同移動平均")
        st.markdown("""
| 狀況 | 意義 |
|------|------|
| MACD 柱 > 0 且增加中 | 多頭動能增強 ✅ |
| MACD 柱 > 0 但縮小 | 多頭動能減弱，注意 |
| MACD 柱 翻負 | 趨勢轉弱，考慮出場 🟠 |
| MACD 柱 持續負值 | 空頭趨勢中，不宜買入 |

> **白話**：MACD 柱（Histogram）是最重要的觀察點。
> 柱子由正轉負，代表買方力量輸給賣方，常是出場訊號。
        """)

    st.divider()

    # ── 量能指標 ──────────────────────────────────────────
    st.subheader("📊 量能指標")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### 量比　今日量 ÷ 5日均量")
        st.markdown("""
| 數值 | 意義 |
|------|------|
| > 1.5 | 爆量，市場高度關注 💡 |
| 1.0 ~ 1.5 | 量能正常偏多 ✅ |
| 0.8 ~ 1.0 | 量能普通 |
| < 0.8 | 量縮，市場觀望，不宜追買 |

> **白話**：量比 = 今天跟最近 5 天比，成交量有沒有放大。
> 漲勢要健康必須有量配合，量縮上漲可能是假突破。
        """)

    with col6:
        st.markdown("#### 內外盤　主動買賣比例（盤中）")
        st.markdown("""
| 數值 | 意義 |
|------|------|
| 外盤 > 60% | 買方主導，主動買進多 ↑ |
| 外盤 40~60% | 買賣平衡 |
| 外盤 < 40% | 賣方主導，主動賣出多 ↓ |

> **白話**：
> - **外盤**：有人主動用市價買入（願意追高）
> - **內盤**：有人主動用市價賣出（願意殺低）
>
> 外盤持續 > 內盤，說明買方積極，是多頭訊號。
> 只是輔助參考，不能單獨作為買賣依據。
        """)

    st.divider()

    # ── 風控指標 ──────────────────────────────────────────
    st.subheader("🛡 風控指標")
    col7, col8 = st.columns(2)

    with col7:
        st.markdown("#### ATR　平均真實波幅（14日）")
        st.markdown("""
| 用途 | 計算方式 |
|------|---------|
| 動態停損線 | 買入價 − 2 × ATR |

> **白話**：ATR 衡量這支股票「每天正常會波動多少」。
> 停損設在買入價減兩倍 ATR，讓正常波動不會觸發停損，
> 只有真的跌壞了才會觸發。
>
> 例：買入 100，ATR = 3，停損線 = 100 − 6 = **94**
        """)

    with col8:
        st.markdown("#### 本系統的進出場訊號總結")
        st.markdown("""
**進場條件（觀察名單，3/3 全達成）**
- MA5 > MA10（趨勢向上）
- RSI > 50（動能向上）
- 量比 > 1.2（有量配合）

**加碼條件（建倉中，4/4 全達成）**
- MA5 > MA10
- RSI 45~65（甜蜜點）
- 量比 > 0.8
- 現價不超過 MA5 的 5%

**出場訊號**
- 🔴 跌破 ATR 停損線 → 立即處理
- 🟢 乖離 >8% + RSI >70 → 減碼 30%
- 🟢 乖離 >12% + RSI >78 → 強力出清
        """)
