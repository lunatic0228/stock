"""K 線綜合分數的離線驗證。

這是**訊號品質研究，不是交易模擬**。只回答一個問題：
    綜合 K 線分數與未來報酬是否單調相關？

合格標準在跑出任何結果之前就宣告並印出 —— 結果才可否證，
而不是事後合理化。未達標就不要照原樣上線。

用法：
    python backtest_kline.py                  # 主驗證（5 張表 + PASS/FAIL）
    python backtest_kline.py --ablate         # 加上分項 ablation
    python backtest_kline.py --dump-stages 3037.TW

════════════════════════════════════════════════════════════════════
2026-08-05 執行結果：🔴 FAIL（全部標的與驗證組皆未通過）
════════════════════════════════════════════════════════════════════
樣本 5,915 根（7 檔 × 845 根，2023-01-12 ~ 2026-07-17）
因果性檢查：7 檔全通過，最大相對差 0.0e+00（含分數層一致性）

  3 桶（全部標的）    偏空 n=387 WR5 55%　偏多 n=496 WR5 50%　差 -5.0pp
  3 桶（驗證組）      偏空 n=229 WR5 54%　偏多 n=296 WR5 49%　差 -4.8pp
  基準              WR5 53%　中位 +0.3%

四項標準：① n 足夠 ✓　② 勝率差 ≥15pp ✗（-5.0）　③ 中位差 ≥1.5pp ✗（-0.5）
          ④ 單調 ✗（逆序 3 次）

分項 ablation（單獨使用的 3 桶 5 日勝率差）：
  candle -2.5pp／pattern +0.3pp／volprice -1.4pp／stage +0.2pp／inst +0.1pp
  → 沒有任何分項具備 edge，全部落在雜訊範圍內

階段合理性：主升段 vs 下跌段 20 日中位數只差 +0.8pp；
          頭部形成的 20 日中位數 +4.5% 反而是五個階段中最高
          → 階段標籤在此樣本上不帶方向資訊（視覺弧線正確但不具預測力）

體制切分：大盤 > MA60 時勝率差 -6.9pp；大盤 < MA60 時 +1.7pp
期間切分：前半 -8.8pp／後半 -0.0pp

【判讀】樣本期間這 7 檔處於強多頭（2313 從 59→302、3037 從 141→1082），
此體制下均值回歸主導 —— 買弱勢的期望值優於買強勢，因此偏多分數反而較差。

【處置】刻意不調權重去追過關數字：沒有任何分項有 edge，調權重只是對雜訊
過度配適，也會浪費計畫中限定的 3 組配置額度。改為在 kline_analysis.py 設
SCORE_VALIDATED = False，報告標頭明確標示分數未經驗證。K 線解讀／量價／
結構／階段仍作為結構化整理保留，但不宣稱有預測力。

【若要再嘗試】值得試的方向不是調權重，而是換問題設定：
  1. 擴大樣本到含空頭期間的標的與更長歷史（本樣本幾乎全是多頭）
  2. 改用「相對大盤的超額報酬」為標的，剝離 beta
  3. 驗證分數對「風險」（MAE／波動）而非「方向」的預測力
"""

import sys
import numpy as np
import pandas as pd

import daily_analysis as da
import kline_data as kd
import kline_analysis as ka

# ════════════════════════════════════════════════════════════
#  設定
# ════════════════════════════════════════════════════════════
TICKERS = {
    "2313.TW": "華通",   "2327.TW": "國巨",     "2344.TW": "華邦電",
    "2449.TW": "京元電", "3037.TW": "欣興",     "3711.TW": "日月光投控",
    "6805.TW": "富世達",
}
# 調參／驗證標的分離：用同一批資料調參又驗證必然樂觀
TUNE_SET   = {"2313.TW", "2344.TW", "6805.TW"}
VERIFY_SET = {"2327.TW", "2449.TW", "3037.TW", "3711.TW"}

HISTORY_DAYS = 1100
WARMUP_BARS  = 260        # MA60/BB百分位/滾動250 z-score 全部暖機完才開始評估
HORIZONS     = (3, 5, 10)
MAX_FWD      = max(HORIZONS)

# ── 事前宣告的合格標準（四項全過才算 PASS）──
PASS_MIN_N       = 30     # 偏多桶與偏空桶各自 n ≥ 30
PASS_WR_GAP_PP   = 15.0   # 偏多桶 5日勝率 − 偏空桶 5日勝率 ≥ 15pp
PASS_MED_GAP_PCT = 1.5    # 偏多桶 5日中位數 − 偏空桶 5日中位數 ≥ 1.5pp
PASS_MONOTONIC   = 1      # 5 桶 5日中位數允許的相鄰逆序次數上限

# 7 桶明細視角：分界與 UI 上的評語完全一致，讀表可 1:1 對應螢幕輸出
BUCKETS7 = [(-99, -6, '強力偏空'), (-6, -3.5, '偏空'), (-3.5, -1.5, '中性偏空'),
            (-1.5, 1.5, '中性'), (1.5, 3.5, '中性偏多'), (3.5, 6, '偏多'),
            (6, 99, '強力偏多')]
# 3 桶檢定視角：PASS/FAIL 以此為基礎，因為極端桶的 n 天生就小
BUCKETS3 = [(-99, -3.5, '偏空'), (-3.5, 3.5, '中性'), (3.5, 99, '偏多')]
BUCKETS5 = [(-99, -3.5, '偏空'), (-3.5, -1.5, '中性偏空'), (-1.5, 1.5, '中性'),
            (1.5, 3.5, '中性偏多'), (3.5, 99, '偏多')]

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


# ════════════════════════════════════════════════════════════
#  因果性斷言 —— 整支腳本最有價值的元件
# ════════════════════════════════════════════════════════════
def verify_causality(raw, full, ticker, n_checks=12, seed=7):
    """回測正確性的核心保險。

    指標是在完整 df 上一次算完再切片的（合法，因為用到的運算子都是因果的：
    rolling / ewm / diff / shift / cumsum）。但「合法」不能靠人工檢查斷定 ——
    這裡隨機抽 n_checks 個 t，比較：
      (a) 完整 frame 上算出的第 t 列
      (b) 只用 raw.iloc[:t+1] 重算後的最後一列
    任一欄位差異 > 1e-6 就代表有未來資訊洩漏，直接 raise 中止整個回測。

    沒有這個斷言，一個洩漏的特徵會產出漂亮但毫無意義的結果。
    """
    rng = np.random.default_rng(seed)
    lo, hi = WARMUP_BARS, len(full) - MAX_FWD - 2
    if hi <= lo:
        raise AssertionError(f"{ticker} 資料太短，無法做因果性檢查")
    picks = rng.choice(range(lo, hi), size=min(n_checks, hi - lo), replace=False)

    num_cols = [c for c in full.columns
                if pd.api.types.is_numeric_dtype(full[c])]
    worst = ('', 0.0)
    for t in sorted(picks):
        part = kd.add_kline_indicators(
            da.calculate_indicators(raw.iloc[:t + 1][OHLCV].copy()))
        for c in num_cols:
            a, b = full.iloc[t][c], part.iloc[-1][c]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b):
                raise AssertionError(f"{ticker} t={t} {c}: NaN 不一致 {a} vs {b}")
            scale = max(abs(float(a)), 1.0)
            diff = abs(float(a) - float(b)) / scale
            if diff > worst[1]:
                worst = (c, diff)
            if diff > 1e-6:
                raise AssertionError(
                    f"{ticker} t={t}（{full.index[t].date()}）欄位 {c} 洩漏未來資訊："
                    f"整段算 {a} vs 截斷重算 {b}（相對差 {diff:.2e}）")

    # 分數層也要驗：analyze_bar 只看得到 df.iloc[:t+1]，但它讀的欄位若不因果就會漏
    t = int(sorted(picks)[len(picks) // 2])
    part = kd.add_kline_indicators(
        da.calculate_indicators(raw.iloc[:t + 1][OHLCV].copy()))
    s_full = ka.analyze_bar(full.iloc[:t + 1], code=ticker, want_analog=False,
                            want_timing=False)
    s_part = ka.analyze_bar(part, code=ticker, want_analog=False, want_timing=False)
    if abs(s_full['score']['score'] - s_part['score']['score']) > 1e-6:
        raise AssertionError(
            f"{ticker} t={t} 綜合分數不一致："
            f"{s_full['score']['score']} vs {s_part['score']['score']}")

    print(f"  ✓ {ticker} 因果性檢查通過（{len(picks)} 個隨機時點，"
          f"最大相對差 {worst[1]:.1e} @ {worst[0] or '-'}；分數層亦一致）")


# ════════════════════════════════════════════════════════════
#  走查
# ════════════════════════════════════════════════════════════
def evaluate_ticker(ticker, name, raw, full, inst_series):
    verify_causality(raw, full, ticker)

    rows, prev_stage = [], None
    lo, hi = WARMUP_BARS, len(full) - MAX_FWD - 1
    for t in range(lo, hi):
        sub  = full.iloc[:t + 1]              # 只用到 t 為止，含 t
        date = full.index[t]
        inst = kd.inst_snapshot(inst_series, as_of=date) if inst_series else None
        # want_timing=False：回測只用 score 與 stage，跳過 timing_verdict
        # （它會逐根呼叫 entry_signals/exit_signals/bottom_score 等，純浪費）
        res  = ka.analyze_bar(sub, code=ticker, inst=inst, want_analog=False,
                              prev_stage_code=prev_stage, want_timing=False)
        prev_stage = res['stage']['code']

        entry = float(full['Open'].iloc[t + 1])   # t+1 開盤：最早能實際成交的價格
        prev_close = float(full['Close'].iloc[t])
        row = {
            'ticker': ticker, 'name': name, 'date': date,
            'score': res['score']['score'], 'verdict': res['score']['verdict'],
            'stage': res['stage']['code'], 'entry': entry,
            # 漲停開盤實際無法成交，在台股是真實且重大的假 edge 來源
            'untradeable': entry >= prev_close * 1.095,
        }
        for h in HORIZONS:
            row[f'fwd{h}'] = float(full['Close'].iloc[t + h]) / entry - 1
            row[f'mae{h}'] = float(full['Low'].iloc[t + 1:t + h + 1].min()) / entry - 1
        row['fwd20'] = (float(full['Close'].iloc[t + 20]) / entry - 1
                        if t + 20 < len(full) else np.nan)
        for c in res['score']['components']:
            row['c_' + c['name']] = c['sub']
            row['w_' + c['name']] = c['weight']
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
#  統計
# ════════════════════════════════════════════════════════════
def bucket_of(score, buckets):
    for lo, hi, label in buckets:
        if lo <= score < hi:
            return label
    return buckets[-1][2]


def agg(sub):
    if sub.empty:
        return None
    out = {'n': len(sub)}
    for h in HORIZONS:
        v = sub[f'fwd{h}'].dropna()
        out[f'wr{h}'] = float((v > 0).mean()) if len(v) else np.nan
        out[f'med{h}'] = float(v.median()) if len(v) else np.nan
    v5 = sub['fwd5'].dropna()
    out['mean5'] = float(v5.mean()) if len(v5) else np.nan
    out['p25'] = float(v5.quantile(0.25)) if len(v5) else np.nan
    out['p75'] = float(v5.quantile(0.75)) if len(v5) else np.nan
    out['mae5'] = float(sub['mae5'].dropna().median()) if len(sub) else np.nan
    return out


def print_bucket_table(df, buckets, title, score_col='score'):
    print(f"\n  ── {title} ──")
    print(f"  {'分數桶':<12}{'n':>6}{'WR3':>7}{'WR5':>7}{'WR10':>7}"
          f"{'中位5':>8}{'平均5':>8}{'p25':>8}{'p75':>8}{'MAE5中位':>10}")
    labels = [b[2] for b in buckets]
    stats = {}
    for lab in labels:
        s = agg(df[df[score_col].apply(lambda x: bucket_of(x, buckets)) == lab])
        stats[lab] = s
        if not s:
            print(f"  {lab:<12}{'0':>6}   （無樣本）")
            continue
        print(f"  {lab:<12}{s['n']:>6}{s['wr3']*100:>6.0f}%{s['wr5']*100:>6.0f}%"
              f"{s['wr10']*100:>6.0f}%{s['med5']*100:>+7.1f}%{s['mean5']*100:>+7.1f}%"
              f"{s['p25']*100:>+7.1f}%{s['p75']*100:>+7.1f}%{s['mae5']*100:>+9.1f}%")
    b = agg(df)
    print(f"  {'── 基準(全樣本)':<12}{b['n']:>6}{b['wr3']*100:>6.0f}%{b['wr5']*100:>6.0f}%"
          f"{b['wr10']*100:>6.0f}%{b['med5']*100:>+7.1f}%{b['mean5']*100:>+7.1f}%"
          f"{b['p25']*100:>+7.1f}%{b['p75']*100:>+7.1f}%{b['mae5']*100:>+9.1f}%")
    return stats, b


def judge(df, tag=""):
    """對照事前宣告的四項標準，逐項印出實際數字與 ✓/✗。"""
    s3, base = print_bucket_table(df, BUCKETS3, f"3 桶檢定視角{tag}")
    bull, bear = s3.get('偏多'), s3.get('偏空')
    print(f"\n  ── PASS / FAIL 判定{tag}（標準已於開頭宣告）──")

    checks = []
    if not bull or not bear:
        print("  ✗ 偏多或偏空桶無樣本，無法判定")
        return False

    ok_n = bull['n'] >= PASS_MIN_N and bear['n'] >= PASS_MIN_N
    checks.append((ok_n, f"兩桶 n ≥ {PASS_MIN_N}",
                   f"偏多 n={bull['n']}　偏空 n={bear['n']}"))

    wr_gap = (bull['wr5'] - bear['wr5']) * 100
    checks.append((wr_gap >= PASS_WR_GAP_PP, f"5日勝率差 ≥ {PASS_WR_GAP_PP}pp",
                   f"{bull['wr5']*100:.0f}% − {bear['wr5']*100:.0f}% = {wr_gap:+.1f}pp"))

    med_gap = (bull['med5'] - bear['med5']) * 100
    checks.append((med_gap >= PASS_MED_GAP_PCT, f"5日中位數差 ≥ {PASS_MED_GAP_PCT}pp",
                   f"{bull['med5']*100:+.1f}% − {bear['med5']*100:+.1f}% = {med_gap:+.1f}pp"))

    s5, _ = {}, None
    meds = []
    for lo, hi, lab in BUCKETS5:
        st = agg(df[df['score'].apply(lambda x: bucket_of(x, BUCKETS5)) == lab])
        meds.append(st['med5'] if st else np.nan)
    viol = sum(1 for i in range(len(meds) - 1)
               if not (np.isnan(meds[i]) or np.isnan(meds[i + 1])) and meds[i + 1] < meds[i])
    checks.append((viol <= PASS_MONOTONIC, f"5桶中位數逆序 ≤ {PASS_MONOTONIC} 次",
                   f"逆序 {viol} 次　序列 "
                   + " → ".join('NA' if np.isnan(m) else f"{m*100:+.1f}%" for m in meds)))

    for ok, label, detail in checks:
        print(f"  {'✓' if ok else '✗'} {label:<26}{detail}")
    passed = all(ok for ok, _, _ in checks)
    print(f"\n  {'🟢 PASS' if passed else '🔴 FAIL'}"
          f" —— {'綜合分數與前瞻報酬單調相關，可上線' if passed else '未通過，不可將分數當作已驗證的訊號'}")
    return passed


def print_per_ticker(df):
    print(f"\n  ── 逐檔 × 3 桶（證明 edge 不是單一檔撐起來的）──")
    print(f"  {'標的':<14}{'組':<6}{'偏空n':>7}{'偏空WR5':>9}"
          f"{'偏多n':>7}{'偏多WR5':>9}{'勝率差':>9}{'中位差':>9}")
    for tk, nm in TICKERS.items():
        sub = df[df['ticker'] == tk]
        if sub.empty:
            continue
        b = agg(sub[sub['score'] < -3.5])
        u = agg(sub[sub['score'] >= 3.5])
        grp = '調參' if tk in TUNE_SET else '驗證'
        if not b or not u:
            print(f"  {tk} {nm:<6}{grp:<6}"
                  f"{(b['n'] if b else 0):>7}{'-':>9}{(u['n'] if u else 0):>7}{'-':>9}"
                  f"{'樣本不足':>9}")
            continue
        print(f"  {tk} {nm:<6}{grp:<6}{b['n']:>7}{b['wr5']*100:>8.0f}%"
              f"{u['n']:>7}{u['wr5']*100:>8.0f}%"
              f"{(u['wr5']-b['wr5'])*100:>+8.1f}pp{(u['med5']-b['med5'])*100:>+8.1f}pp")


def print_stage_sanity(df):
    """階段合理性檢查，獨立於綜合分數。
    主升段應大幅優於下跌段；若否則分類器有問題，與分數表現無關。"""
    print(f"\n  ── 階段合理性檢查（獨立於分數）──")
    print(f"  {'階段':<12}{'n':>7}{'占比':>7}{'WR5':>7}{'中位5':>8}"
          f"{'中位10':>8}{'中位20':>8}")
    for code, label in ka.STAGE_LABELS.items():
        sub = df[df['stage'] == code]
        if sub.empty:
            print(f"  {label:<12}{0:>7}")
            continue
        v20 = sub['fwd20'].dropna()
        s = agg(sub)
        print(f"  {label:<12}{s['n']:>7}{s['n']/len(df)*100:>6.0f}%"
              f"{s['wr5']*100:>6.0f}%{s['med5']*100:>+7.1f}%{s['med10']*100:>+7.1f}%"
              f"{(v20.median()*100 if len(v20) else float('nan')):>+7.1f}%")
    up = df[df['stage'] == 'MAIN_UP']['fwd20'].dropna().median()
    dn = df[df['stage'] == 'DOWN']['fwd20'].dropna().median()
    if not (np.isnan(up) or np.isnan(dn)):
        gap = (up - dn) * 100
        print(f"  → 主升段 vs 下跌段 20 日中位數差 {gap:+.1f}pp "
              f"{'✓ 分類器方向正確' if gap > 2 else '⚠ 分類器未能區分方向，需檢查'}")


def print_regime_split(df, twii):
    print(f"\n  ── 體制切分（只在多頭體制存在的 edge 是 beta 代理，不是 K 線 edge）──")
    if twii is None:
        print("  ⚠ 無法取得 ^TWII，跳過體制切分")
    else:
        bull_dates = set(twii.index[twii['Close'] > twii['MA60']].date)
        df = df.copy()
        df['bull'] = df['date'].apply(lambda d: d.date() in bull_dates)
        for flag, lab in ((True, '大盤 > MA60（多頭）'), (False, '大盤 < MA60（空頭）')):
            sub = df[df['bull'] == flag]
            if sub.empty:
                continue
            b, u = agg(sub[sub['score'] < -3.5]), agg(sub[sub['score'] >= 3.5])
            if b and u:
                print(f"  {lab:<20}n={len(sub):>5}　偏空WR5 {b['wr5']*100:.0f}%"
                      f"（n={b['n']}）　偏多WR5 {u['wr5']*100:.0f}%（n={u['n']}）"
                      f"　差 {(u['wr5']-b['wr5'])*100:+.1f}pp")
            else:
                print(f"  {lab:<20}n={len(sub):>5}　極端桶樣本不足")

    print(f"\n  ── 期間前後半切分 ──")
    mid = df['date'].quantile(0.5)
    for flag, lab in ((True, f'前半（≤{str(mid)[:10]}）'), (False, '後半')):
        sub = df[(df['date'] <= mid) == flag]
        b, u = agg(sub[sub['score'] < -3.5]), agg(sub[sub['score'] >= 3.5])
        if b and u:
            print(f"  {lab:<20}n={len(sub):>5}　偏空WR5 {b['wr5']*100:.0f}%"
                  f"　偏多WR5 {u['wr5']*100:.0f}%　差 {(u['wr5']-b['wr5'])*100:+.1f}pp")
        else:
            print(f"  {lab:<20}n={len(sub):>5}　極端桶樣本不足")


def print_ablation(df):
    """逐項把權重歸零，用已存的 c_* 欄重算分數（免重跑）。
    這直接決定最終權重該怎麼設，比憑感覺調好。"""
    print(f"\n  ── 分項 ablation（3 桶 5 日勝率差）──")
    comps = ['candle', 'pattern', 'volprice', 'stage', 'inst']
    weights = {c: df[f'w_{c}'].iloc[0] for c in comps if f'w_{c}' in df.columns}

    def spread(scores):
        d = df.assign(_s=scores)
        b = agg(d[d['_s'] < -3.5])
        u = agg(d[d['_s'] >= 3.5])
        if not b or not u:
            return None, (b['n'] if b else 0), (u['n'] if u else 0)
        return (u['wr5'] - b['wr5']) * 100, b['n'], u['n']

    full = sum(df[f'c_{c}'] * weights[c] for c in weights)
    base_gap, bn, un = spread(full)
    print(f"  {'配置':<22}{'偏空n':>7}{'偏多n':>7}{'勝率差':>10}{'Δ vs 全開':>11}")
    print(f"  {'全部分項':<22}{bn:>7}{un:>7}"
          f"{(f'{base_gap:+.1f}pp' if base_gap is not None else 'NA'):>10}{'—':>11}")
    for drop in comps:
        if drop not in weights:
            continue
        s = sum(df[f'c_{c}'] * weights[c] for c in weights if c != drop)
        gap, bn2, un2 = spread(s)
        delta = (f"{gap - base_gap:+.1f}pp"
                 if (gap is not None and base_gap is not None) else 'NA')
        print(f"  {'去掉 ' + drop:<22}{bn2:>7}{un2:>7}"
              f"{(f'{gap:+.1f}pp' if gap is not None else 'NA'):>10}{delta:>11}")
    print(f"  ℹ 去掉某項後勝率差「變大」→ 該項在拖累分數，權重應下調或移除")
    print(f"  ℹ 只用單項也能有價差 → 該項是主要 edge 來源")
    print(f"\n  {'單獨使用':<22}{'偏空n':>7}{'偏多n':>7}{'勝率差':>10}")
    for only in comps:
        if only not in weights:
            continue
        s = df[f'c_{only}'] * 10.0        # 放大到 ±10 才落進同樣的分桶
        gap, bn2, un2 = spread(s)
        print(f"  {'只用 ' + only:<22}{bn2:>7}{un2:>7}"
              f"{(f'{gap:+.1f}pp' if gap is not None else 'NA'):>10}")


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════
def main(do_ablate=False):
    print("=" * 78)
    print("   K 線綜合分數 — 離線驗證")
    print("=" * 78)
    print(f"  標的 {len(TICKERS)} 檔｜歷史 {HISTORY_DAYS} 交易日｜暖機 {WARMUP_BARS} 根"
          f"｜前瞻 {HORIZONS}")
    print(f"  調參組 {sorted(TUNE_SET)}")
    print(f"  驗證組 {sorted(VERIFY_SET)}")
    print(f"  進場價 = t+1 開盤（讀到報告時今天已收盤，最早能成交的價格）")
    print(f"  權重 candle {ka.W_CANDLE} / pattern {ka.W_PATTERN} / "
          f"volprice {ka.W_VOLPRICE} / stage {ka.W_STAGE} / inst {ka.W_INST}")
    print()
    print("  ── 事前宣告的合格標準（四項全過才算 PASS）──")
    print(f"  ① 偏多桶與偏空桶各自 n ≥ {PASS_MIN_N}")
    print(f"  ② 偏多桶 5日勝率 − 偏空桶 5日勝率 ≥ {PASS_WR_GAP_PP}pp")
    print(f"  ③ 偏多桶 5日中位數 − 偏空桶 5日中位數 ≥ {PASS_MED_GAP_PCT}pp")
    print(f"  ④ 5 桶 5日中位數相鄰逆序 ≤ {PASS_MONOTONIC} 次")
    print("=" * 78)

    print("\n▌ 資料下載與因果性檢查")
    parts = []
    for ticker, name in TICKERS.items():
        full = kd.fetch_long(ticker, days=HISTORY_DAYS, trim=False)
        if full is None or len(full) < WARMUP_BARS + MAX_FWD + 40:
            print(f"  ⚠ {ticker} 資料不足，略過")
            continue
        raw = full[OHLCV].copy()
        code = ticker.replace('.TW', '')
        inst_series = kd.fetch_inst_series(code, days=int(HISTORY_DAYS * 1.5))
        try:
            part = evaluate_ticker(ticker, name, raw, full, inst_series)
        except AssertionError as e:
            print(f"\n  🔴 因果性檢查失敗，中止回測：\n     {e}")
            return
        parts.append(part)
        print(f"    {ticker} {name}　評估 {len(part)} 根　"
              f"{part['date'].iloc[0].date()} ~ {part['date'].iloc[-1].date()}"
              f"　法人資料 {len(inst_series)} 日")

    if not parts:
        print("  ⚠ 無可用資料")
        return
    df = pd.concat(parts, ignore_index=True)

    print(f"\n▌ 樣本：{len(df):,} 根（{len(parts)} 檔）")
    n_unt = int(df['untradeable'].sum())
    print(f"  其中 t+1 開盤即漲停無法成交 {n_unt} 根（{n_unt/len(df)*100:.1f}%）"
          f" —— 下方主表已排除，另列含入版本對照")

    tradeable = df[~df['untradeable']]

    print("\n" + "=" * 78)
    print("   表 1：7 桶 × 前瞻報酬（7 檔合併，已排除漲停無法成交）")
    print("=" * 78)
    print_bucket_table(tradeable, BUCKETS7, "綜合K線分數 vs 前瞻報酬")

    print("\n" + "=" * 78)
    print("   表 2：3 桶檢定 + PASS/FAIL")
    print("=" * 78)
    passed_all = judge(tradeable, "（全部標的）")
    print("\n  【只看驗證組 —— 這才是沒被調參污染的結果】")
    passed_verify = judge(tradeable[tradeable['ticker'].isin(VERIFY_SET)], "（驗證組）")

    print("\n  【含漲停無法成交的版本（對照用，會高估）】")
    _ = print_bucket_table(df, BUCKETS3, "3 桶（含 untradeable）")

    print("\n" + "=" * 78)
    print("   表 3：逐檔")
    print("=" * 78)
    print_per_ticker(tradeable)

    print("\n" + "=" * 78)
    print("   表 4：階段合理性 + 體制切分")
    print("=" * 78)
    print_stage_sanity(tradeable)
    twii = kd.fetch_long('^TWII', days=HISTORY_DAYS, trim=False)
    print_regime_split(tradeable, twii)

    if do_ablate:
        print("\n" + "=" * 78)
        print("   表 5：分項 ablation")
        print("=" * 78)
        print_ablation(tradeable)

    print("\n" + "=" * 78)
    print(f"   總結：全部標的 {'PASS' if passed_all else 'FAIL'}"
          f"　驗證組 {'PASS' if passed_verify else 'FAIL'}")
    if not passed_verify:
        print("   🔴 驗證組未通過 → 不可把綜合分數當作已驗證的訊號。")
        print("      報告標頭應掛「⚠ 綜合分數未通過回測驗證，僅作結構化整理用」，")
        print("      並依 --ablate 的結果找出沒有預測力的分項再調整（最多 3 組配置）。")
    print("=" * 78)


if __name__ == "__main__":
    args = sys.argv[1:]
    if '--dump-stages' in args:
        code = next((a for a in args if not a.startswith('-')), '3037')
        ka._dump_stages(code, verbose='--verbose' in args)
    else:
        main(do_ablate='--ablate' in args)
