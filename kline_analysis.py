"""K 線趨勢診斷 — 分析層

輸出「今日 K 線在說什麼 → 量價是否配合 → 處於趨勢哪個階段 → 法人態度
→ 綜合分數 → 歷史上類似情況後來怎麼走 → 現在是不是進場時機」，
並與現有 A/C 策略並列對照。

設計約定
--------
1. 所有分析函式都作用在**傳入 df 的最後一根**（與 daily_analysis 的慣例相同，
   重用其偵測器零成本）。回測傳 df.iloc[:t+1]，即時報告傳完整 df。
2. 全部門檻與權重集中在本檔頂端，方便 backtest_kline.py 調參。
3. 單向依賴 daily_analysis，不修改它。

免責：K 線型態的學術實證效力薄弱（Marshall/Young/Rose 2006、Horton 2009）。
本模組若有 edge，來自趨勢階段 × 量能體制 × 法人籌碼的組合，而非型態名稱本身。
"""

import numpy as np
import pandas as pd

import daily_analysis as da
import kline_data as kd

# ════════════════════════════════════════════════════════════
#  門檻常數
# ════════════════════════════════════════════════════════════
# ── 單根 K 線形狀（實體／影線占振幅 %）──
BODY_LONG       = 70.0    # 長紅／長黑
BODY_MID        = 40.0    # 中紅／中黑
BODY_SMALL      = 15.0    # 以下視為十字類（實體幾乎不存在）
SHADOW_DOMINANT = 55.0    # 槌子／流星／T字：單邊影線主導
SHADOW_BALANCED = 25.0    # 紡錘：上下影線都不短
MARUBOZU_MAX    = 10.0    # 光頭光腳：上下影合計上限
DOJI_RANGE_ATR  = 1.5     # 長腳十字：振幅需達 ATR 的倍數

# ── 位階（Range_pos，60 日區間位置 0~1）──
POS_HIGH = 0.70
POS_LOW  = 0.30

# ── 量能（Vol_ratio）──
VOL_EXPAND = 1.2          # 量增
VOL_DRY    = 0.8          # 量縮
VOL_BURST  = 2.0          # 爆量
VOL_EXHAUST = 0.6         # 量能枯竭（與 da.detect_volume_shrinkage 同門檻）

# ── 價格方向（當日漲跌 %）──
PRICE_FLAT = 0.5          # 絕對值小於此視為價平

# ── 多根型態 ──
ENGULF_MIN_BODY  = 40.0   # 吞噬：今日實體下限
ENGULF_PREV_BODY = 15.0   # 吞噬：昨日實體下限（吞掉一根十字沒有意義）
SOLDIER_MIN_BODY = 30.0   # 紅三兵／黑三兵：每根實體下限
BREAK_LOOKBACK   = 20     # 破底翻／穿頭破腳的參考區間
GAP_LOOKBACK     = 60     # 未回補缺口的追溯範圍
ISLAND_MAX_BARS  = 15     # 島狀反轉：兩個跳空之間的最大間隔

# ── 薄量防護（成交金額，新台幣）──
MIN_TURNOVER_TWD = 20_000_000

# ── 台股漲跌停 ±10%：觸及時 K 線被截斷，型態判讀失真 ──
LIMIT_PCT = 9.5           # 視為觸及漲跌停的漲跌幅門檻


# ════════════════════════════════════════════════════════════
#  1. 單根 K 線解剖
# ════════════════════════════════════════════════════════════
def candle_anatomy(df):
    """解剖最後一根 K 線的幾何。所有比例對振幅正規化，跨價位可比。"""
    r = df.iloc[-1]
    o, h, l, c = float(r['Open']), float(r['High']), float(r['Low']), float(r['Close'])
    prev_close = float(df.iloc[-2]['Close']) if len(df) >= 2 else o
    atr = float(r['ATR']) if pd.notna(r.get('ATR')) else np.nan

    rng  = h - l
    body = abs(c - o)
    day_chg = (c / prev_close - 1) * 100 if prev_close else 0.0

    # 觸及漲跌停：K 線被 ±10% 截斷，實體／影線比例失去原意
    at_limit = abs(day_chg) >= LIMIT_PCT

    # 開高低收全同。漲跌停鎖死是其中一種，但也可能只是完全沒有波動的一字線
    if rng <= 0:
        locked = at_limit
        return {
            'cls': '一價到底' if locked else '一字線',
            'cls_note': '一價到底（漲跌停鎖死）' if locked else '一字線（全日無波動）',
            'family': 'locked', 'limit_locked': locked, 'at_limit': at_limit,
            'body_pct': 0.0, 'up_pct': 0.0, 'low_pct': 0.0,
            'rng_pct': 0.0, 'rng_atr': 0.0, 'close_pos': 0.5,
            'color': 'red' if day_chg > 0 else ('black' if day_chg < 0 else 'flat'),
            'day_chg': day_chg, 'open': o, 'high': h, 'low': l, 'close': c,
            'bias': 0.0,
            'key_levels': {'confirm': h, 'invalidate': l},
        }

    body_pct = body / rng * 100
    up_pct   = (h - max(o, c)) / rng * 100
    low_pct  = (min(o, c) - l) / rng * 100
    rng_atr  = rng / atr if atr and atr > 0 else np.nan
    color    = 'red' if c > o else ('black' if c < o else 'flat')
    close_pos = (c - l) / rng

    # ── 分類 ──
    if body_pct >= BODY_LONG:
        cls = '長紅K' if color == 'red' else '長黑K'
        family = 'longred' if color == 'red' else 'longblack'
    elif body_pct >= BODY_MID:
        cls = '中紅K' if color == 'red' else '中黑K'
        family = 'longred' if color == 'red' else 'longblack'
    elif body_pct < BODY_SMALL:
        if low_pct >= SHADOW_DOMINANT:
            cls, family = 'T字線', 'hammer'
        elif up_pct >= SHADOW_DOMINANT:
            cls, family = '倒T字線（墓碑）', 'star'
        elif rng_atr == rng_atr and rng_atr >= DOJI_RANGE_ATR:
            cls, family = '長腳十字', 'doji'
        else:
            cls, family = '十字線', 'doji'
    else:
        if low_pct >= SHADOW_DOMINANT:
            cls, family = '槌子線', 'hammer'
        elif up_pct >= SHADOW_DOMINANT:
            cls, family = '流星線', 'star'
        elif up_pct >= SHADOW_BALANCED and low_pct >= SHADOW_BALANCED:
            cls, family = '紡錘線', 'doji'
        else:
            cls = '小紅K' if color == 'red' else '小黑K'
            family = 'small'

    # 附註
    parts = []
    if up_pct + low_pct < MARUBOZU_MAX and body_pct >= BODY_MID:
        parts.append('光頭光腳')
    elif low_pct >= 35 and family in ('longred', 'longblack', 'small'):
        parts.append('帶下影')
    elif up_pct >= 35 and family in ('longred', 'longblack', 'small'):
        parts.append('帶上影')
    if at_limit:
        parts.append('漲停' if day_chg > 0 else '跌停')
    cls_note = f"{cls}（{'／'.join(parts)}）" if parts else cls

    # ── 形狀本身的多空傾向（尚未套用位階／量能情境）──
    base = {
        'longred': 0.9 if body_pct >= BODY_LONG else 0.6,
        'longblack': -0.9 if body_pct >= BODY_LONG else -0.6,
        'hammer': 0.5 if cls == '槌子線' else 0.4,
        'star': -0.5 if cls == '流星線' else -0.4,
        'doji': 0.0,
        'small': 0.3 if color == 'red' else (-0.3 if color == 'black' else 0.0),
    }[family]
    # 收盤在日內的位置也帶資訊：收在最高附近 = 買方掌控到最後
    bias = float(np.clip(base + (close_pos - 0.5) * 0.3, -1.0, 1.0))

    return {
        'cls': cls, 'cls_note': cls_note, 'family': family,
        'limit_locked': False, 'at_limit': at_limit,
        'body_pct': body_pct, 'up_pct': up_pct, 'low_pct': low_pct,
        'rng_pct': rng / c * 100 if c else 0.0, 'rng_atr': rng_atr,
        'close_pos': close_pos, 'color': color, 'day_chg': day_chg,
        'open': o, 'high': h, 'low': l, 'close': c, 'bias': bias,
        'key_levels': {'confirm': h, 'invalidate': l},
    }


# ── 情境解讀矩陣：同一種形狀在不同位階／量能下意義不同 ──────────
_READING = {
    'hammer': {
        ('low', 'expand'): '長下影＋放量＝實質承接，有資金願意在低位接走賣壓，落底機率較高',
        ('low', 'dry'):    '下殺被承接，但量縮代表買盤並不積極，止跌尚未確認',
        ('low', 'flat'):   '低位長下影，賣壓有人接，但力道普通，需要隔日紅K驗證',
        ('high', 'expand'): '高檔衝高被打下來又拉回，多空激戰、換手劇烈，追高風險已升高',
        ('high', 'dry'):   '高檔無量震盪，多方後繼無力，觀望',
        ('high', 'flat'):  '高檔上下影皆長，方向不明，宜減碼觀察',
        ('mid', 'expand'): '中段回檔遇買盤承接，若能守住此低點，回檔可能結束',
        ('mid', 'dry'):    '中段量縮試探下方，尚未表態',
        ('mid', 'flat'):   '中段震盪整理，等方向',
    },
    'star': {
        ('high', 'expand'): '高檔放量長上影＝衝高遇沉重賣壓，主力可能藉拉高出貨，優先減碼',
        ('high', 'dry'):   '高檔量縮衝不上去，上方套牢賣壓明顯，多方轉弱',
        ('high', 'flat'):  '高檔留長上影，上檔壓力確認，漲勢受阻',
        ('low', 'expand'): '低位反彈遇賣壓打回，量增卻收上影＝解套賣壓沉重，反彈受限',
        ('low', 'dry'):    '低位彈升無力，賣壓仍在，尚未落底',
        ('low', 'flat'):   '低位衝高回落，反彈力道不足',
        ('mid', 'expand'): '中段衝關失敗，留意是否形成短波高點',
        ('mid', 'dry'):    '中段無量衝高回落，觀望',
        ('mid', 'flat'):   '中段遇壓回落',
    },
    'longred': {
        ('low', 'expand'): '低位放量紅K＝轉折候選，但單一根不構成反轉，需結構確認（不破今日低、量能延續）',
        ('low', 'dry'):    '低位量縮紅K＝技術性彈升而非實質轉強，多為下跌途中的反抽',
        ('low', 'flat'):   '低位紅K，量能普通，止跌訊號偏弱',
        ('high', 'expand'): '高檔放量長紅＝軋空或末升段噴出，此時追高的期望值最差',
        ('high', 'dry'):   '高檔量縮上漲＝價量背離，動能已在衰退，隨時可能反轉',
        ('high', 'flat'):  '高檔續漲但量能未擴張，留意背離',
        ('mid', 'expand'): '中段放量攻擊，量價配合良好，趨勢延續機率較高',
        ('mid', 'dry'):    '中段量縮上漲，漲勢基礎不穩',
        ('mid', 'flat'):   '中段溫和上漲',
    },
    'longblack': {
        ('low', 'expand'): '低位放量長黑＝停損盤湧出或法人出貨，此時不宜接刀',
        ('low', 'dry'):    '低位量縮長黑＝無人承接的滑落，跌勢尚未結束但也未見恐慌，仍在尋底',
        ('low', 'flat'):   '低位續跌，賣壓延續',
        ('high', 'expand'): '高檔放量長黑＝出貨訊號明確，持股應優先減碼',
        ('high', 'dry'):   '高檔長黑但量縮，先視為回檔，跌破關鍵支撐再認定轉空',
        ('high', 'flat'):  '高檔轉弱，漲勢可能結束',
        ('mid', 'expand'): '中段放量下殺，支撐失守風險升高',
        ('mid', 'dry'):    '中段量縮下跌，屬正常回檔',
        ('mid', 'flat'):   '中段回檔',
    },
    'doji': {
        ('low', 'expand'): '低位放量十字＝多空在此激烈換手，常出現在轉折前後，方向待突破',
        ('low', 'dry'):    '低位量縮十字＝賣壓耗盡但買盤未進場，打底過程中的常見長相',
        ('low', 'flat'):   '低位變盤徵兆，方向未定',
        ('high', 'expand'): '高檔放量十字＝多空易手，漲勢動能中斷，警戒',
        ('high', 'dry'):   '高檔量縮十字，多方無力再攻，偏空看待',
        ('high', 'flat'):  '高檔變盤徵兆，留意轉折',
        ('mid', 'expand'): '中段放量十字，方向待突破',
        ('mid', 'dry'):    '中段量縮盤整，能量收縮中',
        ('mid', 'flat'):   '中段整理，等突破',
    },
    'small': {
        ('low', 'expand'): '低位小實體帶量，多空拉鋸，尚未表態',
        ('low', 'dry'):    '低位量縮小實體，觀望氣氛濃厚',
        ('low', 'flat'):   '低位小幅波動，仍在整理',
        ('high', 'expand'): '高檔小實體帶量＝上攻乏力，量大卻走不動要留意',
        ('high', 'dry'):   '高檔量縮小實體，動能衰退',
        ('high', 'flat'):  '高檔停滯',
        ('mid', 'expand'): '中段小實體帶量，方向待明',
        ('mid', 'dry'):    '中段量縮整理',
        ('mid', 'flat'):   '中段小幅整理',
    },
}


def _pos_bucket(range_pos):
    if range_pos != range_pos:          # NaN
        return 'mid'
    if range_pos >= POS_HIGH:
        return 'high'
    if range_pos < POS_LOW:
        return 'low'
    return 'mid'


def _vol_bucket(vol_ratio):
    if vol_ratio != vol_ratio:
        return 'flat'
    if vol_ratio >= VOL_EXPAND:
        return 'expand'
    if vol_ratio < VOL_DRY:
        return 'dry'
    return 'flat'


def candle_reading(df, a, vp=None, stage=None, avg_price=None):
    """今日這根 K 線在說什麼。回傳 (facts, meaning)：
       facts   — 客觀觀察（開高走低、影線占比、量、位置）
       meaning — 情境解讀（依位階 × 量能 × 階段）＋ 確認／失效價位
    """
    r = df.iloc[-1]
    close, high, low, open_ = a['close'], a['high'], a['low'], a['open']
    range_pos = float(r['Range_pos']) if pd.notna(r.get('Range_pos')) else np.nan
    vol_ratio = float(r['Vol_ratio']) if pd.notna(r.get('Vol_ratio')) else np.nan
    ma10 = float(r['MA10']) if pd.notna(r.get('MA10')) else np.nan
    ma20 = float(r['MA20']) if pd.notna(r.get('MA20')) else np.nan

    pos_b, vol_b = _pos_bucket(range_pos), _vol_bucket(vol_ratio)
    facts, meaning = [], []

    if a['family'] == 'locked':
        facts.append(f"開高低收同為 {close:.2f}（{a['day_chg']:+.2f}%）")
        if a['limit_locked']:
            meaning.append(f"{'漲' if a['day_chg'] > 0 else '跌'}停鎖死："
                           f"K 線沒有實體／影線資訊，型態判定失效")
            meaning.append("成交量也不可比（大量掛單未撮合），本日量價訊號略過")
            meaning.append(f"⚠ 明日若開盤即打開 {close:.2f} 且無法收回 → 排隊的委託開始成交，"
                           f"{'漲勢' if a['day_chg'] > 0 else '跌勢'}可能結束")
        else:
            meaning.append("全日零波動，多為極低量或暫停交易，本日不具型態意義")
        return facts, meaning

    # ── 客觀事實 ──
    if close > open_ and low < open_ - (high - low) * 0.15:
        facts.append(f"開低走高：開 {open_:.2f} 一度殺到 {low:.2f}，"
                     f"收 {close:.2f}，收盤位於日內 {a['close_pos']*100:.0f}%")
    elif close < open_ and high > open_ + (high - low) * 0.15:
        facts.append(f"開高走低：開 {open_:.2f} 一度衝到 {high:.2f}，"
                     f"收 {close:.2f}，收盤位於日內 {a['close_pos']*100:.0f}%")
    else:
        facts.append(f"開 {open_:.2f} → 收 {close:.2f}（{a['day_chg']:+.2f}%），"
                     f"收盤位於日內 {a['close_pos']*100:.0f}%")

    if a['low_pct'] >= SHADOW_BALANCED and a['close_pos'] > 0.5:
        facts.append(f"下影 {a['low_pct']:.0f}% 且收盤高於日中點 → {low:.2f} 附近有實質承接")
    if a['up_pct'] >= SHADOW_BALANCED and a['close_pos'] < 0.5:
        facts.append(f"上影 {a['up_pct']:.0f}% 且收盤低於日中點 → {high:.2f} 附近賣壓沉重")

    if vol_ratio == vol_ratio:
        vlab = {'expand': '放量', 'dry': '量縮', 'flat': '量平'}[vol_b]
        clab = {'red': '收紅', 'black': '收黑', 'flat': '平盤'}[a['color']]
        tail = {
            ('expand', 'red'):   '有人主動進場，不是無量虛漲',
            ('expand', 'black'): '有人急著出貨，賣壓是真實的',
            ('dry', 'red'):      '上漲缺乏成交量支撐，追價意願低',
            ('dry', 'black'):    '下跌沒什麼量，多為浮額殺出而非法人拋售',
        }.get((vol_b, a['color']), '量能中性')
        facts.append(f"量比 {vol_ratio:.2f}（{vlab}）＋{clab} → {tail}")

    if avg_price:
        rel = '獲利' if close > avg_price else '虧損'
        note = '籌碼相對安定' if close > avg_price else '短線套牢籌碼偏多'
        facts.append(f"收盤 {close:.2f} {'>' if close > avg_price else '<'} "
                     f"均價 {avg_price:.2f} → 當日進場者多為{rel}，{note}")

    if range_pos == range_pos and ma20 == ma20:
        posname = {'high': '高位區', 'mid': '中段', 'low': '低位區'}[pos_b]
        facts.append(f"位置：MA20 {'上方' if close > ma20 else '下方'} "
                     f"{(close/ma20-1)*100:+.1f}%、60日區間 {range_pos*100:.0f}%（{posname}）")

    # ── 情境解讀 ──
    core = _READING.get(a['family'], {}).get((pos_b, vol_b))
    if core:
        meaning.append(core)

    # 與 MA10 的相對位置決定「反抗」還是「反轉」
    if ma10 == ma10 and a['color'] == 'red' and close < ma10:
        meaning.append(f"但收盤 {close:.2f} 仍在 MA10({ma10:.2f}) 之下，"
                       f"屬「反抗」不是「反轉」；下跌段中的紅K過半只是技術性反彈")
    elif ma10 == ma10 and a['color'] == 'black' and close > ma10:
        meaning.append(f"收盤 {close:.2f} 仍守住 MA10({ma10:.2f})，"
                       f"目前僅為回檔，跌破才需認定轉弱")

    if stage and stage.get('label'):
        meaning.append(f"目前階段判定為「{stage['label']}」，"
                       f"單根 K 線的訊息要放在這個框架下看")

    if a.get('at_limit'):
        meaning.append(f"⚠ 本日觸及{'漲' if a['day_chg'] > 0 else '跌'}停（{a['day_chg']:+.2f}%），"
                       f"K 線被 ±10% 截斷，實體／影線比例失去原本含意")

    # ── 確認／失效價位（每個結論都必須配上可驗證的價位）──
    meaning.append(f"⚠ 明日站上 {high:.2f}（今日高）且量不縮 → 換手成功，訊號成立")
    meaning.append(f"⚠ 跌破 {low:.2f}（今日低）→ 今天的量變成接錯的套牢籌碼")

    return facts, meaning


# ════════════════════════════════════════════════════════════
#  2. 多根型態（近 5 根）
# ════════════════════════════════════════════════════════════
def _body_pct(row):
    rng = float(row['High']) - float(row['Low'])
    if rng <= 0:
        return 0.0
    return abs(float(row['Close']) - float(row['Open'])) / rng * 100


def unfilled_gaps(df, lookback=GAP_LOOKBACK):
    """近 lookback 根內尚未回補的跳空缺口。

    向下缺口區間 = [今日High, 昨日Low]，被之後任一根的 High 觸及即視為回補。
    向上缺口區間 = [昨日High, 今日Low]，被之後任一根的 Low 觸及即視為回補。
    """
    n = len(df)
    if n < 3:
        return []
    start = max(1, n - lookback)
    highs, lows = df['High'].values, df['Low'].values
    out = []
    for j in range(start, n):
        # 向上跳空
        if lows[j] > highs[j - 1]:
            lo, hi = highs[j - 1], lows[j]
            if not (lows[j + 1:] <= lo).any():
                out.append({'type': 'up', 'lo': float(lo), 'hi': float(hi),
                            'date': df.index[j], 'bars_ago': n - 1 - j})
        # 向下跳空
        elif highs[j] < lows[j - 1]:
            lo, hi = highs[j], lows[j - 1]
            if not (highs[j + 1:] >= hi).any():
                out.append({'type': 'down', 'lo': float(lo), 'hi': float(hi),
                            'date': df.index[j], 'bars_ago': n - 1 - j})
    return out


def multibar_patterns(df, a=None):
    """近 5 根的組合型態。回傳 list[{'name','dir','strength','msg'}]。

    dir: +1 看多 / -1 看空 / 0 中性（方向待突破）
    能重用 daily_analysis 既有偵測器的絕不重寫。
    """
    if len(df) < 4:
        return []
    a = a or candle_anatomy(df)
    out = []

    cur, prev = df.iloc[-1], df.iloc[-2]
    o, h, l, c = a['open'], a['high'], a['low'], a['close']
    po, ph, pl, pc = (float(prev['Open']), float(prev['High']),
                      float(prev['Low']), float(prev['Close']))
    vol_ratio = float(cur['Vol_ratio']) if pd.notna(cur.get('Vol_ratio')) else 1.0
    prev_body = _body_pct(prev)

    if a['family'] == 'locked':
        out.append({'name': a['cls'], 'dir': 0, 'strength': 0.0,
                    'msg': ('漲跌停鎖死，多根型態判定失效，本日型態分歸零'
                            if a['limit_locked'] else '全日無波動，無型態可判')})
        return out

    vol_boost = 0.3 if vol_ratio >= VOL_EXPAND else 0.0
    vol_note  = '，帶量' if vol_ratio >= VOL_EXPAND else ''

    # ── 吞噬 ──
    if a['body_pct'] >= ENGULF_MIN_BODY and prev_body >= ENGULF_PREV_BODY:
        if a['color'] == 'red' and pc < po and o <= pc and c >= po:
            out.append({'name': '看多吞噬', 'dir': 1, 'strength': 1.3 + vol_boost,
                        'msg': f"今日實體({o:.2f}~{c:.2f}) 完全包住昨日({po:.2f}~{pc:.2f}){vol_note}"})
        elif a['color'] == 'black' and pc > po and o >= pc and c <= po:
            out.append({'name': '看空吞噬', 'dir': -1, 'strength': 1.3 + vol_boost,
                        'msg': f"今日實體({o:.2f}~{c:.2f}) 完全包住昨日({po:.2f}~{pc:.2f}){vol_note}"})

    # ── 孕育線／內含（能量收縮，方向待突破）──
    if h <= ph and l >= pl:
        out.append({'name': '孕育線（內含）', 'dir': 0, 'strength': 0.0,
                    'msg': f"今日高低({l:.2f}~{h:.2f}) 完全落在昨日區間內，"
                           f"能量收縮；突破 {ph:.2f} 或跌破 {pl:.2f} 才表態"})

    # ── 紅三兵／黑三兵 ──
    if len(df) >= 3:
        b3 = df.iloc[-3:]
        bodies = [_body_pct(b3.iloc[k]) for k in range(3)]
        closes = [float(b3.iloc[k]['Close']) for k in range(3)]
        reds   = [float(b3.iloc[k]['Close']) > float(b3.iloc[k]['Open']) for k in range(3)]
        if all(reds) and closes[0] < closes[1] < closes[2] and min(bodies) >= SOLDIER_MIN_BODY:
            out.append({'name': '紅三兵', 'dir': 1, 'strength': 1.5,
                        'msg': f"連 3 根紅K收盤逐級走高（{closes[0]:.2f}→{closes[2]:.2f}）"})
        elif (not any(reds)) and closes[0] > closes[1] > closes[2] and min(bodies) >= SOLDIER_MIN_BODY:
            out.append({'name': '黑三兵', 'dir': -1, 'strength': 1.5,
                        'msg': f"連 3 根黑K收盤逐級走低（{closes[0]:.2f}→{closes[2]:.2f}）"})

    # ── 今日跳空 ──
    if l > ph:
        out.append({'name': '向上跳空', 'dir': 1, 'strength': 1.0,
                    'msg': f"今日低 {l:.2f} > 昨日高 {ph:.2f}，缺口 {ph:.2f}~{l:.2f}"})
    elif h < pl:
        out.append({'name': '向下跳空', 'dir': -1, 'strength': 1.0,
                    'msg': f"今日高 {h:.2f} < 昨日低 {pl:.2f}，缺口 {h:.2f}~{pl:.2f}"})

    # ── 破底翻 / 穿頭破腳 ──
    if len(df) > BREAK_LOOKBACK:
        prev_low  = float(df['Low'].iloc[-(BREAK_LOOKBACK + 1):-1].min())
        prev_high = float(df['High'].iloc[-(BREAK_LOOKBACK + 1):-1].max())
        if l < prev_low and c > pc:
            out.append({'name': '破底翻', 'dir': 1, 'strength': 1.0,
                        'msg': f"今日低 {l:.2f} 破近{BREAK_LOOKBACK}日低 {prev_low:.2f}，"
                               f"但收 {c:.2f} 高於昨收 {pc:.2f}"})
        if h > prev_high and c < pc:
            out.append({'name': '穿頭破腳', 'dir': -1, 'strength': 1.0,
                        'msg': f"今日高 {h:.2f} 破近{BREAK_LOOKBACK}日高 {prev_high:.2f}，"
                               f"但收 {c:.2f} 低於昨收 {pc:.2f}"})

    # ── 貫穿線／烏雲罩頂 ──
    prev_mid = (po + pc) / 2
    if prev_body >= BODY_MID:
        if pc < po and a['color'] == 'red' and o < pc and c > prev_mid:
            out.append({'name': '貫穿線', 'dir': 1, 'strength': 1.0,
                        'msg': f"昨日長黑，今日開低 {o:.2f} 卻收 {c:.2f}，"
                               f"穿越昨日實體中點 {prev_mid:.2f}"})
        elif pc > po and a['color'] == 'black' and o > pc and c < prev_mid:
            out.append({'name': '烏雲罩頂', 'dir': -1, 'strength': 1.0,
                        'msg': f"昨日長紅，今日開高 {o:.2f} 卻收 {c:.2f}，"
                               f"跌破昨日實體中點 {prev_mid:.2f}"})

    # ── 島狀反轉 ──
    # 關鍵不只是「兩側各有一個反向跳空」，而是中間那串 K 棒被兩個缺口
    # 隔離在同一個價格區域（形成孤島）。少了這個條件會把一般的跳空
    # 一漲一跌都誤判成島狀反轉（實測觸發率會高到 8%，真實應 <1%）。
    lows, highs = df['Low'].values, df['High'].values
    n = len(df)
    if l > ph and n >= 4:        # 今日向上跳空 → 往回找向下跳空
        for k in range(n - 2, max(1, n - 1 - ISLAND_MAX_BARS), -1):
            if highs[k] >= lows[k - 1]:
                continue
            island_high = float(highs[k:n - 1].max())
            if island_high < lows[k - 1] and island_high < l:
                out.append({'name': '島狀反轉（底）', 'dir': 1, 'strength': 1.5,
                            'msg': f"{df.index[k].date()} 向下跳空後，{n - 1 - k} 根 K 棒"
                                   f"全數困在 {island_high:.2f} 之下，今日向上跳空脫離孤島"})
                break
    elif h < pl and n >= 4:      # 今日向下跳空 → 往回找向上跳空
        for k in range(n - 2, max(1, n - 1 - ISLAND_MAX_BARS), -1):
            if lows[k] <= highs[k - 1]:
                continue
            island_low = float(lows[k:n - 1].min())
            if island_low > highs[k - 1] and island_low > h:
                out.append({'name': '島狀反轉（頂）', 'dir': -1, 'strength': 1.5,
                            'msg': f"{df.index[k].date()} 向上跳空後，{n - 1 - k} 根 K 棒"
                                   f"全數浮在 {island_low:.2f} 之上，今日向下跳空墜落孤島"})
                break

    # ── 長腳十字（變盤徵兆）──
    if a['cls'] == '長腳十字':
        out.append({'name': '長腳十字', 'dir': 0, 'strength': 0.0,
                    'msg': f"實體 {a['body_pct']:.0f}% 但振幅達 {a['rng_atr']:.1f}×ATR，"
                           f"多空劇烈拉鋸 → 變盤徵兆"})

    # ── 重用 daily_analysis 既有偵測器 ──
    ok, msg = da.detect_long_lower_shadow(df)
    if ok:
        out.append({'name': '長下影線', 'dir': 1, 'strength': 1.0,
                    'msg': msg.strip().lstrip('💡').strip()})
    elif a['low_pct'] >= 45 and a['close_pos'] > 0.5:
        # 介於 45~60% 之間：承接跡象存在但不夠強，減半計分
        out.append({'name': '長下影線（未達門檻）', 'dir': 1, 'strength': 0.5,
                    'msg': f"下影 {a['low_pct']:.0f}%（未達 60% 門檻，減半計分）"})

    ok, msg = da.detect_sharp_drop_bounce(df)
    if ok:
        out.append({'name': '急跌反彈型', 'dir': 1, 'strength': 1.0,
                    'msg': msg.strip().lstrip('💡').strip()})

    return out


# ════════════════════════════════════════════════════════════
#  3. 量價配合
# ════════════════════════════════════════════════════════════
def _price_dir(day_chg):
    if day_chg > PRICE_FLAT:
        return 'up'
    if day_chg < -PRICE_FLAT:
        return 'down'
    return 'flat'


def volume_price(df, a=None):
    """量價配合分析。四象限 + 位階消解歧義 + 爆量/枯竭 + 量能趨勢 + 背離。"""
    a = a or candle_anatomy(df)
    r = df.iloc[-1]

    vol_ratio = float(r['Vol_ratio']) if pd.notna(r.get('Vol_ratio')) else np.nan
    range_pos = float(r['Range_pos']) if pd.notna(r.get('Range_pos')) else np.nan
    turnover  = float(r['Turnover']) if pd.notna(r.get('Turnover')) else np.nan
    vol_ma5   = float(r['Vol_MA5']) if pd.notna(r.get('Vol_MA5')) else np.nan
    vol_ma20  = float(r['Vol_MA20']) if pd.notna(r.get('Vol_MA20')) else np.nan
    vol_slope = float(r['Vol_slope20']) if pd.notna(r.get('Vol_slope20')) else np.nan

    pdir = _price_dir(a['day_chg'])
    vbkt = _vol_bucket(vol_ratio)
    pos_b = _pos_bucket(range_pos)
    msgs = []

    # 台股成交金額偏低時，量比一筆大單就能失真
    thin = turnover == turnover and turnover < MIN_TURNOVER_TWD

    # ── 四象限 ──
    if pdir == 'flat' or vbkt == 'flat':
        quadrant = {'up': '溫和上漲', 'down': '溫和下跌', 'flat': '價平量平'}[pdir]
        bias = 0.1 if pdir == 'up' else (-0.1 if pdir == 'down' else 0.0)
    elif pdir == 'up' and vbkt == 'expand':
        quadrant = '價漲量增'
        if pos_b == 'high' and range_pos == range_pos and range_pos >= 0.85:
            bias = 0.3
            msgs.append('位階修正：已在 60 日區間頂部 → 視為末升段噴出，非健康攻擊，追高期望值差')
        elif pos_b == 'low':
            bias = 0.5
            msgs.append('位階修正：處低位區 → 判為「跌深有量反彈」，非「主升攻擊」')
        else:
            bias = 0.8
            msgs.append('中段放量上攻，量價配合健康，動能有量支撐')
    elif pdir == 'up' and vbkt == 'dry':
        quadrant = '價漲量縮'
        bias = -0.5
        msgs.append('⚠ 價漲量縮＝背離，上漲缺乏成交量支撐，追高危險')
    elif pdir == 'down' and vbkt == 'expand':
        quadrant = '價跌量增'
        if pos_b == 'high':
            bias = -0.9
            msgs.append('位階修正：高檔放量下跌 → 出貨特徵明確')
        elif pos_b == 'low':
            bias = -0.2
            msgs.append('位階修正：低檔放量下跌 → 恐慌殺尾，有機會是換手落底（需隔日紅K確認）')
        else:
            bias = -0.6
            msgs.append('中段放量下跌，支撐失守風險升高')
    else:   # down + dry
        quadrant = '價跌量縮'
        if pos_b == 'low':
            bias = 0.3
            msgs.append('位階修正：低檔價跌量縮 → 惜售、賣壓衰竭傾向（偏正面）')
        elif pos_b == 'high':
            bias = -0.1
            msgs.append('位階修正：高檔無量下跌 → 買盤退場，並非好事')
        else:
            bias = 0.0
            msgs.append('中段量縮下跌，屬正常回檔')

    # ── 爆量／枯竭 ──
    burst = vol_ratio == vol_ratio and vol_ratio >= VOL_BURST
    dry   = vol_ratio == vol_ratio and vol_ratio <= VOL_EXHAUST
    if burst:
        if pos_b == 'high':
            bias -= 0.3
            msgs.append(f'爆量（量比 {vol_ratio:.2f} ≥ {VOL_BURST}）在高檔 → 出貨警訊')
        elif pos_b == 'low':
            bias += 0.2
            msgs.append(f'爆量（量比 {vol_ratio:.2f} ≥ {VOL_BURST}）在低檔 → 換手落底候選')
        else:
            msgs.append(f'爆量（量比 {vol_ratio:.2f} ≥ {VOL_BURST}）於中段，方向由收盤決定')
    elif dry:
        msgs.append(f'量能枯竭（量比 {vol_ratio:.2f} ≤ {VOL_EXHAUST}），成交意願極低')
    elif vol_ratio == vol_ratio:
        msgs.append(f'爆量檢查：{vol_ratio:.2f} 未達 {VOL_BURST:.2f} 門檻（非爆量）')

    # ── 量能趨勢 ──
    if vol_ma5 == vol_ma5 and vol_ma20 == vol_ma20 and vol_ma20 > 0:
        ratio5_20 = vol_ma5 / vol_ma20 - 1
        if ratio5_20 > 0.15:
            vol_trend = 'up'
        elif ratio5_20 < -0.15:
            vol_trend = 'down'
        else:
            vol_trend = 'flat'
        tlab = {'up': '增溫中', 'flat': '持平', 'down': '退潮中'}[vol_trend]
        msgs.append(f"量能趨勢：Vol_MA5 {vol_ma5/1000:,.0f}張 vs "
                    f"Vol_MA20 {vol_ma20/1000:,.0f}張（{ratio5_20*100:+.0f}%）→ {tlab}"
                    + (f"　20日量能斜率 {vol_slope:+.2f}%/根" if vol_slope == vol_slope else ""))
        # 上漲卻量能退潮 = 動能衰退
        if pdir == 'up' and vol_trend == 'down':
            bias -= 0.2
            msgs.append('上漲但量能整體退潮 → 動能衰退，漲勢基礎轉弱')
    else:
        vol_trend = 'flat'

    # ── 量價背離（OBV）──
    obv_bull, obv_msg = da.detect_obv_divergence(df)
    obv_bear, obv_bear_msg = _obv_top_divergence(df)
    if obv_bull:
        bias += 0.3
        msgs.append('OBV：' + obv_msg.strip().lstrip('💡').strip() + ' → 🟢 底背離')
    if obv_bear:
        bias -= 0.3
        msgs.append('OBV：' + obv_bear_msg + ' → 🔴 頂背離')

    if thin:
        msgs.insert(0, f'⚠ 成交金額僅 {turnover/1e8:.2f} 億（< {MIN_TURNOVER_TWD/1e8:.1f} 億），'
                       f'量比與型態訊號不可靠 → 量價分項歸零')
        bias = 0.0

    return {
        'quadrant': quadrant, 'price_dir': pdir, 'vol_bkt': vbkt, 'pos_bkt': pos_b,
        'vol_ratio': vol_ratio, 'burst': burst, 'dry': dry, 'thin': thin,
        'vol_trend': vol_trend, 'obv_bull_div': obv_bull, 'obv_bear_div': obv_bear,
        'bias': float(np.clip(bias, -1.0, 1.0)), 'msgs': msgs,
    }


def _rsi_top_divergence(df, lookback=20):
    """RSI 頂背離：價格創近 lookback 日新高，但 RSI 未同步創高。

    daily_analysis.detect_rsi_divergence 只做底背離，這裡補對稱的頂背離
    （頭部形成的判定需要它）。
    """
    if len(df) < lookback + 2 or 'RSI' not in df.columns:
        return False, None
    win = df.iloc[-lookback:]
    if win['RSI'].isna().any():
        return False, None
    close_now, rsi_now = float(win.iloc[-1]['Close']), float(win.iloc[-1]['RSI'])
    prior = win.iloc[:-1]
    if close_now < float(prior['Close'].max()):
        return False, None
    if rsi_now >= float(prior['RSI'].max()):
        return False, None
    return True, (f"近{lookback}日價格創新高但 RSI {rsi_now:.1f} "
                  f"未超前高 {float(prior['RSI'].max()):.1f}（動能減弱）")


def _obv_top_divergence(df, lookback=20):
    """頂背離：價格創近 lookback 日新高，但 OBV 未創新高。

    daily_analysis 只有底背離（detect_obv_divergence），這裡補對稱的頂背離。
    """
    if len(df) < lookback + 2 or 'OBV' not in df.columns:
        return False, None
    win = df.iloc[-lookback:]
    close_now = float(win.iloc[-1]['Close'])
    obv_now   = float(win.iloc[-1]['OBV'])
    prior = win.iloc[:-1]
    if close_now < float(prior['Close'].max()):
        return False, None
    if obv_now >= float(prior['OBV'].max()):
        return False, None
    return True, f'近{lookback}日價格創新高，OBV 未同步創高（量能未跟上）'


# ════════════════════════════════════════════════════════════
#  4. 擺盪結構與趨勢階段
# ════════════════════════════════════════════════════════════
PIVOT_K         = 4      # 樞紐確認：前後各 k 根。最後 k 根無法確認，刻意排除
PIVOT_ATR_MULT  = 1.5    # 相鄰樞紐價差小於此倍數的 ATR 即視為雜訊而合併
PIVOT_LOOKBACK  = 250    # 樞紐搜尋範圍（限制範圍避免逐根走查時變成 O(n²)）
STRUCT_PIVOTS   = 3      # 結構判定取最近幾個同型樞紐（見 swing_structure 說明）
RANGE_TOL       = 0.05   # 高低點首尾變化小於此比例視為持平

STAGE_LABELS = {
    'BASE':     '底部整理',
    'EARLY_UP': '初升段',
    'MAIN_UP':  '主升段',
    'TOP':      '頭部形成',
    'DOWN':     '下跌段',
}
STAGE_ICONS = {
    'BASE': '🟰', 'EARLY_UP': '🌱', 'MAIN_UP': '📈', 'TOP': '⚠', 'DOWN': '📉',
}
# 方向分：只回答「未來 3~10 日方向」。
# MAIN_UP 高於 EARLY_UP 是刻意的——已確立的上升趨勢延續機率最高；
# 進場品質（追高風險、RR）由 timing_verdict 單獨處理，不摺進方向分。
STAGE_SUB = {'MAIN_UP': 1.0, 'EARLY_UP': 0.8, 'BASE': 0.0, 'TOP': -0.7, 'DOWN': -1.0}
# 方向家族：主升／初升同屬偏多，兩者互跳對方向判斷無害；
# 真正有害的是 偏多↔偏空 之間的亂跳，衡量穩定度時要分開看
STAGE_FAMILY = {'MAIN_UP': 'up', 'EARLY_UP': 'up', 'BASE': 'flat',
                'TOP': 'down', 'DOWN': 'down'}
# 遲滯分兩級：同家族內換標籤（主升↔初升）幾乎免費，換方向家族要求較大優勢。
# 這樣才能只壓抑有害的方向亂跳，而不拖慢無害的標籤細分。
STAGE_HYST_SAME = 0.02
STAGE_HYST_FAM  = 0.15


def find_pivots(df, k=PIVOT_K, lookback=PIVOT_LOOKBACK, atr_filter=True):
    """擺盪樞紐點（swing high / swing low）。

    第 i 根的 High 為前後各 k 根中最高 → 擺盪高點（Low 反之）。
    只回傳「已確認」的樞紐：最後 k 根右側資料不足，無法確認，刻意排除 ——
    這是本模組防 look-ahead 的關鍵之一，也意味著最近的波段點在即時情境下
    永遠拿不到（任何「已落底」的說法至少晚 k 根）。
    """
    n = len(df)
    if n < 2 * k + 2:
        return []
    start = max(0, n - lookback)
    highs = df['High'].values[start:]
    lows  = df['Low'].values[start:]
    m = len(highs)

    raw = []
    for i in range(k, m - k):
        win_h = highs[i - k:i + k + 1]
        win_l = lows[i - k:i + k + 1]
        if highs[i] >= win_h.max():
            raw.append([i, 'H', float(highs[i])])
        if lows[i] <= win_l.min():
            raw.append([i, 'L', float(lows[i])])
    if not raw:
        return []

    def alternate(items):
        """強制 H/L 交替：連續同型只保留較極端者。"""
        out = []
        for it in items:
            if out and out[-1][1] == it[1]:
                if (it[1] == 'H' and it[2] > out[-1][2]) or \
                   (it[1] == 'L' and it[2] < out[-1][2]):
                    out[-1] = it
            else:
                out.append(it)
        return out

    raw.sort(key=lambda x: (x[0], 0 if x[1] == 'H' else 1))
    piv = alternate(raw)

    # 雜訊過濾：橫盤時 k 窗規則會產生一堆微小的假樞紐
    if atr_filter:
        atr = df['ATR'].values[start:]
        for _ in range(10):
            changed = False
            for j in range(len(piv) - 1):
                a_i, b_i = piv[j][0], piv[j + 1][0]
                thresh = atr[b_i] if atr[b_i] == atr[b_i] else 0.0
                if thresh > 0 and abs(piv[j + 1][2] - piv[j][2]) < PIVOT_ATR_MULT * thresh:
                    del piv[j + 1]
                    piv = alternate(piv)
                    changed = True
                    break
            if not changed:
                break

    return [{'pos': start + i, 'date': df.index[start + i], 'price': px, 'type': t}
            for i, t, px in piv]


def _seq_dir(prices, tol=RANGE_TOL):
    """一串樞紐價格的方向：'up' / 'down' / 'flat'。

    刻意用「首尾比較」而非「最後兩點比較」。只比最後兩個同型樞紐極度脆弱：
    大跌勢中的兩個微小反彈就能讓結構讀成 HH（實測 2313 就是這樣被誤判），
    而首尾跨越 3 個樞紐約 20~40 根 K 棒，對單一異常樞紐不敏感。
    """
    if len(prices) < 2 or prices[0] <= 0:
        return 'flat'
    chg = prices[-1] / prices[0] - 1
    if abs(chg) < tol:
        return 'flat'
    return 'up' if chg > 0 else 'down'


def swing_structure(df, pivots=None):
    """波段結構：HH-HL（上升）／LH-LL（下降）／range（區間）／unclear。

    另回報「目前腿」：自最後一個已確認樞紐到最後一根的方向／幅度／根數。
    這一段尚未被確認為樞紐，但只用到 ≤ 最後一根的資料，因此安全。
    """
    pivots = pivots if pivots is not None else find_pivots(df)
    close = float(df.iloc[-1]['Close'])
    res = {'pattern': 'unclear', 'pivots': pivots[-6:], 'last_H': None, 'last_L': None,
           'h_dir': None, 'l_dir': None, 'broken': None,
           'leg_dir': None, 'leg_pct': None, 'leg_bars': None, 'confirm_lag': PIVOT_K,
           'note': '樞紐不足，結構未定'}

    Hs = [p for p in pivots if p['type'] == 'H']
    Ls = [p for p in pivots if p['type'] == 'L']
    res['last_H'] = Hs[-1]['price'] if Hs else None
    res['last_L'] = Ls[-1]['price'] if Ls else None

    if pivots:
        last = pivots[-1]
        res['leg_dir']  = 'up' if close > last['price'] else 'down'
        res['leg_pct']  = (close / last['price'] - 1) * 100
        res['leg_bars'] = len(df) - 1 - last['pos']

    # ── 結構破壞優先於歷史樞紐序列 ──
    # 樞紐天生落後 k 根，未確認的那一腿可能很大（實測 2327 的目前腿達 -55%，
    # 整段崩跌都還沒形成樞紐），此時舊樞紐排列出來的標籤是過時且誤導的。
    # 收盤跌破最後確認低點 = 下降結構延伸中，這是比歷史排列更即時的事實。
    if res['last_L'] is not None and close < res['last_L']:
        res['pattern'] = 'LH-LL'
        res['broken']  = 'down'
        res['note'] = (f"收盤 {close:.1f} 已跌破最後確認低點 {res['last_L']:.1f}"
                       f"（{(close/res['last_L']-1)*100:+.1f}%）→ 下降結構延伸中，"
                       f"新低樞紐尚待確認")
        return res
    if res['last_H'] is not None and close > res['last_H']:
        res['pattern'] = 'HH-HL'
        res['broken']  = 'up'
        res['note'] = (f"收盤 {close:.1f} 已突破最後確認高點 {res['last_H']:.1f}"
                       f"（{(close/res['last_H']-1)*100:+.1f}%）→ 上升結構延伸中，"
                       f"新高樞紐尚待確認")
        return res

    if len(Hs) < 2 or len(Ls) < 2:
        return res

    hp = [p['price'] for p in Hs[-STRUCT_PIVOTS:]]
    lp = [p['price'] for p in Ls[-STRUCT_PIVOTS:]]
    h_dir, l_dir = _seq_dir(hp), _seq_dir(lp)
    res['h_dir'], res['l_dir'] = h_dir, l_dir
    span = f"高點 {hp[0]:.1f}→{hp[-1]:.1f}　低點 {lp[0]:.1f}→{lp[-1]:.1f}"

    if h_dir == 'up' and l_dir != 'down':
        res['pattern'], res['note'] = 'HH-HL', f'高點走高、低點不破 → 上升結構（{span}）'
    elif h_dir == 'down' and l_dir != 'up':
        res['pattern'], res['note'] = 'LH-LL', f'高點走低、低點不撐 → 下降結構（{span}）'
    elif l_dir == 'up' and h_dir != 'down':
        res['pattern'], res['note'] = 'HH-HL', f'低點抬高、高點未破低 → 上升結構（{span}）'
    elif l_dir == 'down' and h_dir != 'up':
        res['pattern'], res['note'] = 'LH-LL', f'低點下移、高點未走高 → 下降結構（{span}）'
    elif h_dir == 'flat' and l_dir == 'flat':
        res['pattern'], res['note'] = 'range', f'高低點皆持平（±{RANGE_TOL*100:.0f}% 內）→ 區間整理（{span}）'
    else:
        # 高點走高但低點走低（擴張），或高點走低但低點走高（收斂三角）
        kind = '擴張' if h_dir == 'up' else '收斂'
        res['pattern'], res['note'] = 'unclear', f'高低點方向相反（{kind}）→ 結構轉換中（{span}）'
    return res


def stage_features(df, struct=None):
    """趨勢階段判定所需的特徵。全部只讀 df（最後一根為判定對象）。"""
    r = df.iloc[-1]
    g = lambda c: float(r[c]) if c in df.columns and pd.notna(r[c]) else np.nan

    struct = struct if struct is not None else swing_structure(df)
    close = g('Close')
    ma5, ma10, ma20, ma60 = g('MA5'), g('MA10'), g('MA20'), g('MA60')

    obv_bear, _ = _obv_top_divergence(df)
    rsi_bear, _ = _rsi_top_divergence(df)
    macd_dir, _ = da.detect_macd_convergence(df)
    bb_squeeze, _ = da.detect_bb_squeeze(df)

    vol_ma5, vol_ma20 = g('Vol_MA5'), g('Vol_MA20')
    if vol_ma5 == vol_ma5 and vol_ma20 == vol_ma20 and vol_ma20 > 0:
        vr = vol_ma5 / vol_ma20 - 1
        vol_trend = 'up' if vr > 0.15 else ('down' if vr < -0.15 else 'flat')
    else:
        vol_trend = 'flat'

    return {
        'close': close, 'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
        'ma_align_full': all(x == x for x in (ma5, ma10, ma20, ma60)) and ma5 > ma10 > ma20 > ma60,
        'ma_align_bear': all(x == x for x in (ma5, ma10, ma20, ma60)) and ma5 < ma10 < ma20 < ma60,
        'ma20_slope': g('MA20_slope'), 'ma60_slope': g('MA60_slope'),
        'dev_ma20': g('dev_ma20'), 'dev_ma60': g('dev_ma60'),
        'pos60': g('Range_pos'), 'pos252': g('Range_pos252'),
        'dd_252h': g('dd_252h'), 'up_252l': g('up_252l'),
        'adx': g('ADX'), 'di_plus': g('DI_plus'), 'di_minus': g('DI_minus'),
        'roc20': g('ROC_20'),
        'bbw_pct': g('BBW_pct'), 'bb_squeeze': bb_squeeze,
        'bars_since_high': g('bars_since_60d_high'),
        'bars_since_low': g('bars_since_60d_low'),
        'struct': struct['pattern'], 'vol_trend': vol_trend,
        'obv_bear_div': obv_bear, 'rsi_bear_div': rsi_bear,
        'macd_bearish': macd_dir == 'bearish',
    }


def _stage_checks(f):
    """每個階段的證據項 (標籤, 是否命中, 權重)。權重來自各項的判別力，非等權。"""
    nn = lambda v: v == v            # not NaN
    adx, dip, dim = f['adx'], f['di_plus'], f['di_minus']
    s20, s60 = f['ma20_slope'], f['ma60_slope']
    pos = f['pos60']

    return {
        'MAIN_UP': [
            # 站上 MA10 是主升段的必要條件。少了這一項，MA 多頭排列在頭部
            # 可以續存數週，會讓主升段一路贏過頭部形成（實測 2313 就是如此）
            ('站上 MA10', nn(f['ma10']) and f['close'] > f['ma10'], 2.0),
            ('MA5>MA10>MA20>MA60 多頭排列', f['ma_align_full'], 2.0),
            ('ADX≥25 且 +DI>-DI', nn(adx) and adx >= 25 and dip > dim, 2.0),
            ('MA20 斜率 >+2%/10根', nn(s20) and s20 > 2.0, 1.5),
            ('結構 HH-HL', f['struct'] == 'HH-HL', 1.5),
            ('60日位階 >70%', nn(pos) and pos > 0.70, 1.0),
            ('乖離 MA60 >+8%', nn(f['dev_ma60']) and f['dev_ma60'] > 8.0, 1.0),
        ],
        'EARLY_UP': [
            ('站上 MA20', nn(f['ma20']) and f['close'] > f['ma20'], 1.5),
            ('站上 MA60 且 MA60 斜率 >-1%', nn(f['ma60']) and f['close'] > f['ma60']
             and nn(s60) and s60 > -1.0, 1.5),
            ('MA20 斜率 0~+2%（剛翻正）', nn(s20) and 0.0 < s20 <= 2.0, 1.5),
            ('距 60 日低點 <25 根（剛打底）', nn(f['bars_since_low']) and f['bars_since_low'] < 25, 1.5),
            ('結構 HH-HL 或 range', f['struct'] in ('HH-HL', 'range'), 1.0),
            ('60日位階 35~80%（未漲多）', nn(pos) and 0.35 < pos < 0.80, 1.0),
            ('量能增溫', f['vol_trend'] == 'up', 1.0),
            ('ADX 15~25（趨勢剛啟動）', nn(adx) and 15 <= adx <= 25, 0.5),
        ],
        'TOP': [
            # 頭部的定義性特徵：長天期均線還在漲，短天期已轉折
            ('MA60 斜率 >+1% 但 MA20 斜率 <+1%（長多短轉）', nn(s60) and s60 > 1.0
             and nn(s20) and s20 < 1.0, 2.0),
            ('剛創高後跌破 MA10（距60日高<25根）', nn(f['bars_since_high'])
             and f['bars_since_high'] < 25 and nn(f['ma10']) and f['close'] < f['ma10'], 2.0),
            # MA5 下穿 MA10 是最經典的第一個頭部訊號，不必等結構轉空
            ('MA5 下穿 MA10（短均死叉）', nn(f['ma5']) and nn(f['ma10'])
             and f['ma5'] < f['ma10'], 1.5),
            ('自 60 日高回落 >8% 但仍在年線高位（>50%）', nn(f['pos60']) and nn(f['pos252'])
             and f['pos60'] < 0.92 and f['pos252'] > 0.50
             and nn(f['dd_252h']) and f['dd_252h'] < -8.0, 1.5),
            ('OBV 頂背離', f['obv_bear_div'], 1.5),
            ('RSI 頂背離', f['rsi_bear_div'], 1.0),
            ('MACD 多頭收斂（動能衰退）', f['macd_bearish'], 1.0),
        ],
        'DOWN': [
            ('MA5<MA10<MA20<MA60 空頭排列', f['ma_align_bear'], 2.0),
            ('MA20 斜率 <-1.5%/10根', nn(s20) and s20 < -1.5, 2.0),
            ('結構 LH-LL', f['struct'] == 'LH-LL', 1.5),
            ('ADX≥20 且 -DI>+DI', nn(adx) and adx >= 20 and dim > dip, 1.5),
            ('距年高 <-15%', nn(f['dd_252h']) and f['dd_252h'] < -15.0, 1.0),
            ('跌破 MA60', nn(f['ma60']) and f['close'] < f['ma60'], 1.0),
        ],
        'BASE': [
            ('MA20 斜率走平（|斜率|<0.8%）', nn(s20) and abs(s20) < 0.8, 2.0),
            # 沒有這一項，瀑布式下跌會被誤判成打底
            # （實測 2313 在 -15% 的急殺段被標成底部整理）
            ('近20日跌幅 <10%（未在急跌中）', nn(f['roc20']) and f['roc20'] > -10.0, 2.0),
            ('年線位階 <50%（真的在低檔）', nn(f['pos252']) and f['pos252'] < 0.50, 1.5),
            ('60日位階 <40%', nn(pos) and pos < 0.40, 1.5),
            ('布林帶寬處 120 日低檔（<30 百分位）', nn(f['bbw_pct']) and f['bbw_pct'] < 0.30, 1.5),
            ('ADX <20（無趨勢）', nn(adx) and adx < 20, 1.5),
            ('結構 range', f['struct'] == 'range', 1.5),
            ('量能退潮', f['vol_trend'] == 'down', 0.5),
        ],
    }


def classify_stage(df, f=None, struct=None, prev_code=None, hysteresis=None):
    """五階段分類器。加權投票取 argmax，不用 if/elif 串接。

    串接的順序偏誤會讓邊界個案亂跳；投票法還能回報信心度與次高階段 ——
    接近轉折時 DOWN 與 BASE 分數收斂本身就是資訊（「下跌段末端／可能落底中」），
    不是失敗。

    prev_code 給定時套用遲滯：新階段必須比前一階段多出一定分數才換，否則沿用
    前一根的判定。跨方向家族的門檻較高（STAGE_HYST_FAM），同家族內幾乎免費
    （STAGE_HYST_SAME）。只讀前一根的標籤，因此仍然完全因果。
    hysteresis 明確給值時（例如 0.0）覆蓋上述兩級門檻。
    """
    f = f if f is not None else stage_features(df, struct)
    checks = _stage_checks(f)

    scores, evidence = {}, {}
    for code, items in checks.items():
        total = sum(w for _, _, w in items)
        hit   = sum(w for _, ok, w in items if ok)
        scores[code] = hit / total if total else 0.0
        evidence[code] = items

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best, best_score = ranked[0]
    runner, runner_score = ranked[1]

    held = False
    if prev_code and prev_code in scores and prev_code != best:
        if hysteresis is not None:
            need = hysteresis
        else:
            need = (STAGE_HYST_FAM if STAGE_FAMILY[best] != STAGE_FAMILY[prev_code]
                    else STAGE_HYST_SAME)
        if best_score - scores[prev_code] < need:
            best, best_score = prev_code, scores[prev_code]
            held = True
            ranked = sorted(((c, s) for c, s in scores.items() if c != best),
                            key=lambda x: -x[1])
            runner, runner_score = ranked[0]

    confidence = best_score - runner_score

    return {
        'code': best, 'label': STAGE_LABELS[best], 'icon': STAGE_ICONS[best],
        'score': best_score, 'confidence': confidence, 'held': held,
        'runner_up': runner, 'runner_label': STAGE_LABELS[runner],
        'runner_score': runner_score, 'family': STAGE_FAMILY[best],
        'scores': scores, 'evidence': evidence[best], 'features': f,
        # 信心低於 0.4 時分數要打折：階段判定本身不確定，不該貢獻滿分方向
        'sub': STAGE_SUB[best] * max(confidence, 0.4),
        'contested': confidence < 0.10,
    }


def classify_stage_stable(df, f=None, struct=None):
    """帶遲滯的階段判定（即時單根呼叫用）。

    先算前一根的無遲滯判定當作 prev_code，再算本根。成本 2 倍，
    換來不會因為單日雜訊就跳階段。
    """
    prev = None
    if len(df) >= kd.WARMUP_BARS + 1:
        try:
            prev = classify_stage(df.iloc[:-1], hysteresis=0.0)['code']
        except Exception:
            prev = None
    return classify_stage(df, f=f, struct=struct, prev_code=prev)


# ════════════════════════════════════════════════════════════
#  5. 綜合 K 線分數
# ════════════════════════════════════════════════════════════
# 每個分項輸出 [-1,+1]，ΣW = 10 → 分數直接落在 -10~+10。
# 初始權重僅為起點，最終權重由 backtest_kline.py 的 ablation 表決定。
W_CANDLE, W_PATTERN, W_VOLPRICE, W_STAGE, W_INST = 2.0, 2.0, 2.5, 2.5, 1.0
W_SUM = W_CANDLE + W_PATTERN + W_VOLPRICE + W_STAGE + W_INST

# ── 回測驗證狀態（backtest_kline.py，2026-08-05）──────────────
# 結果：FAIL。7 檔 × 845 根（2023-01 ~ 2026-07，5,915 個樣本）走查後，
#   偏多桶 5日勝率 50% vs 偏空桶 55%（差 -5.0pp，方向與預期相反）
#   驗證組（未參與調參的 4 檔）同樣 -4.8pp
#   ablation：每個分項單獨使用的勝率差都在 ±2.5pp 內，全部落在雜訊範圍
#   階段合理性：主升段 vs 下跌段 20 日中位數只差 +0.8pp（未能區分方向）
#
# 原因判讀：樣本期間這 7 檔處於強多頭（2313 從 59→302、3037 從 141→1082），
# 此體制下均值回歸主導 —— 買弱勢的期望值優於買強勢，因此偏多分數反而較差。
#
# 刻意不調權重去追過關數字：沒有任何分項有 edge，調權重只是對雜訊過度配適。
# 因此本分數定位為「結構化整理工具」，不是已驗證的方向訊號，
# 報告標頭會明確標示。若未來重跑通過，把 SCORE_VALIDATED 改為 True。
SCORE_VALIDATED = False
SCORE_VALIDATION_NOTE = (
    "綜合分數未通過回測驗證（backtest_kline.py：偏多桶 5 日勝率 50% "
    "vs 偏空桶 55%，n=5,915）。本分數僅作結構化整理，不可當方向訊號使用。"
)

SCORE_BANDS = [
    (6.0,  '強力偏多', '⏫'), (3.5,  '偏多',   '↗'), (1.5,  '中性偏多', '↗'),
    (-1.5, '中性',    '→'),  (-3.5, '中性偏空', '↘'), (-6.0, '偏空',   '↘'),
]
SCORE_FLOOR = ('強力偏空', '⏬')


def score_verdict(score):
    for lo, label, icon in SCORE_BANDS:
        if score >= lo:
            return label, icon
    return SCORE_FLOOR


def _candle_sub(a, pos_bkt, vol_bkt):
    """單根 K 線的方向分。同樣的形狀在不同位階／量能下要打折甚至轉向。"""
    if a['family'] == 'locked':
        return 0.0, '漲跌停鎖死／無波動，不計分'
    b = a['bias']
    notes = [a['cls_note']]
    if pos_bkt == 'high' and b > 0:
        if vol_bkt == 'dry':
            b *= -0.5
            notes.append('高檔量縮上攻 → 正分轉負（背離）')
        else:
            b *= 0.6
            notes.append('高檔 → 正分打折（追高風險）')
    elif pos_bkt == 'low' and b < 0:
        b *= 0.7
        notes.append('低檔 → 負分打折（跌勢已反映）')
    if a.get('at_limit'):
        b *= 0.7
        notes.append('觸及漲跌停 → K 線被截斷，可信度打折')
    return float(np.clip(b, -1.0, 1.0)), '；'.join(notes)


def _pattern_sub(pats):
    if not pats:
        return 0.0, '無型態'
    raw = sum(p['dir'] * p['strength'] for p in pats)
    hits = [p['name'] for p in pats if p['dir'] != 0]
    return float(np.clip(raw / 3.0, -1.0, 1.0)), ('＋'.join(hits[:3]) if hits else '僅中性型態')


def _inst_sub(inst):
    """法人方向分。inst 由呼叫端解析好（回測傳 as-of 快照，絕不查『現在』）。"""
    if not inst or not inst.get('dir'):
        return 0.0, '無法人資料'
    d, days = inst['dir'], inst.get('days', 0)
    if d == 'buy':
        return (1.0, f'外資連買 {days} 日') if days >= 3 else (0.5, f'外資連買 {days} 日')
    if d == 'sell':
        return (-1.0, f'外資連賣 {days} 日') if days >= 3 else (-0.5, f'外資連賣 {days} 日')
    return 0.0, '法人中性'


def kline_score(df, a=None, pats=None, vp=None, stage=None, inst=None):
    """綜合 K 線分數（-10 ~ +10）＋ 分項明細。

    歷史類比刻意不計入分數：其樣本數逐根變動，會讓分數非定態、回測無法分桶。
    它以獨立證據並列顯示。
    """
    a     = a if a is not None else candle_anatomy(df)
    pats  = pats if pats is not None else multibar_patterns(df, a)
    vp    = vp if vp is not None else volume_price(df, a)
    stage = stage if stage is not None else classify_stage_stable(df)

    c_sub, c_note = _candle_sub(a, vp['pos_bkt'], vp['vol_bkt'])
    p_sub, p_note = _pattern_sub(pats)
    i_sub, i_note = _inst_sub(inst)
    v_sub = vp['bias']
    s_sub = stage['sub']

    comps = [
        {'name': 'candle',   'label': '今日K線', 'weight': W_CANDLE,   'sub': c_sub, 'detail': c_note},
        {'name': 'pattern',  'label': '近5日型態', 'weight': W_PATTERN,  'sub': p_sub, 'detail': p_note},
        {'name': 'volprice', 'label': '量價配合', 'weight': W_VOLPRICE, 'sub': v_sub, 'detail': vp['quadrant']},
        {'name': 'stage',    'label': '趨勢階段', 'weight': W_STAGE,    'sub': s_sub,
         'detail': f"{stage['label']}（信心 {stage['confidence']:.2f}）"},
        {'name': 'inst',     'label': '法人動向', 'weight': W_INST,     'sub': i_sub, 'detail': i_note},
    ]
    for c in comps:
        c['contrib'] = c['weight'] * c['sub']

    score = float(np.clip(sum(c['contrib'] for c in comps), -10.0, 10.0))
    label, icon = score_verdict(score)
    return {'score': score, 'verdict': label, 'icon': icon, 'components': comps}


# ════════════════════════════════════════════════════════════
#  6. 進出場時機判斷 + 現有 A/C 策略對照
# ════════════════════════════════════════════════════════════
RR_MIN          = 1.5     # 風險報酬比門檻。低於此即使分數偏多也降級為「觀察」
ENTRY_SCORE_MIN = 3.5     # 進場所需的最低綜合分數（對應「偏多」）


def _existing_ac(df, buy_price=None, inst=None):
    """現有 A/C 策略的判定。純顯示，不影響本模組任何結論。

    呼叫方式與 mechanical_scan / watchlist_v2_scan 一致，確保對照忠實。
    """
    r = df.iloc[-1]
    close = float(r['Close'])
    g = lambda c: float(r[c]) if c in df.columns and pd.notna(r[c]) else np.nan

    entry_score, entry_msgs = da.entry_signals(df)
    ov_score, ov_detail, ov_level = da.detect_oversold(df)
    ex_count, ex_signals = da.detect_selling_exhaustion(df)
    bs_score, bs_details = da.bottom_score(df)
    probe = da.calc_probe_stoploss(df)

    adx, roc20, ma10 = g('ADX'), g('ROC_20'), g('MA10')
    a_gate = {
        'adx_ok':   adx == adx and adx >= 25,
        'roc_ok':   roc20 == roc20 and roc20 >= 3.0,
        'score_ok': entry_score >= 4,
        'adx': adx, 'roc20': roc20, 'score': entry_score, 'msgs': entry_msgs,
    }
    a_gate['pass'] = a_gate['adx_ok'] and a_gate['roc_ok'] and a_gate['score_ok']

    # 條件③必須是「最近一個交易日外資淨買超」，不是連續買超天數 ——
    # daily_analysis 的 _inst_buy_latest 就是這個定義（L3754），
    # 用連續天數會讓對照區與既有 V2 掃描給出不同答案，失去對照的意義
    if inst and inst.get('latest_date'):
        inst_buy = bool(inst['latest_buy'])
        c3_label = (f"法人買超（外資 {inst['latest_date'][5:]} "
                    f"{'買+' if inst_buy else '賣-'}{abs(inst['latest_foreign']):,}張）")
    else:
        inst_buy, c3_label = False, "法人買超（無資料）"
    c_conds = [
        (f"超跌評分 {ov_score}/7 ≥ 3", ov_score >= 3),
        (f"賣壓衰竭 {ex_count} 項 ≥ 1", ex_count >= 1),
        (c3_label, inst_buy),
    ]
    c_gate = {
        'premise':  ma10 == ma10 and close < ma10,
        'ma10': ma10, 'conds': c_conds,
        'met': sum(1 for _, ok in c_conds if ok),
        'oversold': (ov_score, ov_detail, ov_level),
        'exhaust': (ex_count, ex_signals),
        'probe': probe,
    }
    c_gate['pass'] = c_gate['premise'] and c_gate['met'] >= 2

    exit_msgs = da.exit_signals(df, buy_price) if buy_price else []
    return {'A': a_gate, 'C': c_gate, 'bottom': (bs_score, bs_details),
            'exit_msgs': exit_msgs}


def timing_verdict(df, sc, stage, vp, a, struct, gaps=None, inst=None,
                   buy_price=None, analog=None):
    """是否進場時機。回傳具體價位，不講空話。

    RR < RR_MIN 時即使分數偏多也降級為「觀察」——這是攔住「下跌段反彈被
    標成買點」最有效的一道閘，也是本模組與策略 C 最常出現分歧的原因。
    """
    r = df.iloc[-1]
    close = float(r['Close'])
    g = lambda c: float(r[c]) if c in df.columns and pd.notna(r[c]) else np.nan
    ma5, ma10, ma20 = g('MA5'), g('MA10'), g('MA20')
    atr, low20, high22 = g('ATR'), g('Low_20'), g('High22')
    bb_low, bb_up = g('BB_lower'), g('BB_upper')
    gaps = gaps if gaps is not None else unfilled_gaps(df)
    code = stage['code']

    existing = _existing_ac(df, buy_price, inst)
    plan = {'zone': None, 'stop': None, 'stop_note': '', 'targets': [], 'rr': None}

    def pct(px):
        return (px / close - 1) * 100

    if code == 'TOP':
        plan['stop_note'] = '頭部形成階段不建構進場計畫'
    elif code in ('BASE', 'DOWN'):
        # 綁定「現價下方最接近的支撐」給窄區間。
        # 用 BB下緣~MA10 這種寬帶會產生 19% 寬的進場區（實測 2313），
        # 停損被迫拉到區間下緣之外，RR 算出來毫無意義也不可執行。
        cands = [(a['low'], f"今日低 {a['low']:.2f}"),
                 (struct.get('last_L'), f"最近確認低點 {struct.get('last_L') or 0:.2f}"),
                 (ma10, f"MA10 {ma10:.2f}"), (ma20, f"MA20 {ma20:.2f}"),
                 (bb_low, f"布林下緣 {bb_low:.2f}"), (low20, f"近20日低 {low20:.2f}")]
        below = [(p, n) for p, n in cands if p and p == p and p < close]
        if below:
            base, base_note = max(below, key=lambda x: x[0])
        else:
            base, base_note = close * 0.97, f"現價×0.97 {close*0.97:.2f}"
        plan['zone'] = (base, base * 1.02)
        plan['zone_note'] = f"以{base_note}為進場基準"
        plan['stop'] = base * 0.97
        plan['stop_note'] = f"{base_note}×0.97"
        if low20 == low20 and low20 * 0.97 < plan['stop']:
            plan['stop_alt'] = (low20 * 0.97, f"近20日低 {low20:.2f}×0.97")
        tg = []
        if ma10 == ma10:
            tg.append((ma10, 'MA10'))
        down_gaps = [x for x in gaps if x['type'] == 'down' and x['lo'] > close]
        if down_gaps:
            ng = min(down_gaps, key=lambda x: x['lo'])
            tg.append((ng['lo'], f"未回補缺口下緣（{ng['date'].date()}）"))
        if ma20 == ma20:
            tg.append((ma20, 'MA20／布林中軌'))
        if struct.get('last_H') and struct['last_H'] > close:
            tg.append((struct['last_H'], '最近確認樞紐高'))
        plan['targets'] = sorted({round(p, 2): (p, n) for p, n in tg if p > close}.values())
    else:   # EARLY_UP / MAIN_UP：回調買，不追價
        lo = ma10 if ma10 == ma10 else close * 0.97
        hi = ma5 * 1.02 if ma5 == ma5 else close
        plan['zone'] = (min(lo, hi), max(lo, hi))
        cand = []
        if ma10 == ma10:
            cand.append((ma10 * 0.97, f"MA10 {ma10:.1f}×0.97"))
        if high22 == high22 and atr == atr:
            cand.append((high22 - 2 * atr, f"吊燈 近22日高{high22:.1f}−2×ATR"))
        if not cand:
            cand = [(close * 0.95, '現價×0.95')]
        plan['stop'], plan['stop_note'] = max(cand, key=lambda x: x[0])   # 取較緊者
        tg = []
        if struct.get('last_H') and struct['last_H'] > close:
            tg.append((struct['last_H'], '前波樞紐高'))
        if struct.get('last_H') and struct.get('last_L'):
            mm = struct['last_H'] + (struct['last_H'] - struct['last_L'])
            if mm > close:
                tg.append((mm, '量測目標（樞紐高+波幅）'))
        if bb_up == bb_up and bb_up > close:
            tg.append((bb_up, '布林上軌'))
        plan['targets'] = sorted({round(p, 2): (p, n) for p, n in tg if p > close}.values())

    # ── 進場參考價與 RR ──
    # 必須區分「現在就買」與「等回調再買」：兩者的停損距離完全不同。
    # 現價已高於建議進場區時，用現價算 RR 才回答得了「是否現在進場」，
    # 同時另外報一個「回調到區間後的 RR」讓使用者知道等待有什麼好處。
    if plan['zone'] and plan['stop'] and plan['targets']:
        zlo, zhi = plan['zone']
        plan['in_zone'] = zlo <= close <= zhi
        plan['entry_ref'] = close if plan['in_zone'] else (zhi if close > zhi else zlo)

        # 停損必須低於進場區下緣，否則「進場即已在停損之下」邏輯不成立
        if plan['stop'] >= zlo:
            plan['stop'] = zlo * 0.97
            plan['stop_note'] = f"進場區下緣 {zlo:.2f}×0.97"

        t1 = plan['targets'][0][0]
        sp_now = pct(plan['stop'])
        plan['rr'] = abs(pct(t1) / sp_now) if sp_now else None
        if not plan['in_zone'] and plan['entry_ref']:
            e = plan['entry_ref']
            sp_z = (plan['stop'] / e - 1) * 100
            plan['rr_zone'] = abs(((t1 / e - 1) * 100) / sp_z) if sp_z else None
    elif plan['stop'] and plan['targets']:
        sp = pct(plan['stop'])
        plan['rr'] = abs(pct(plan['targets'][0][0]) / sp) if sp else None

    # ── 結論 ──
    score, rr = sc['score'], plan['rr']
    reasons = []
    if buy_price:
        answer = '⚪ 持有中'
        reasons.append(f"成本 {buy_price:.2f}，目前損益 {(close/buy_price-1)*100:+.1f}%")
        reasons.append(f"K 線綜合 {score:+.1f}（{sc['verdict']}）；出場訊號見下方對照區")
    elif code == 'TOP':
        answer = '🔴 不宜進場'
        reasons.append('階段判定為頭部形成：長天期均線仍在漲但短均已轉折，'
                       '此位置進場等於在派發區接手')
    elif rr is not None and rr < RR_MIN:
        answer = '🟡 尚未到進場時機，列入觀察'
        reasons.append(f"以現價計風險報酬比 {rr:.2f} 未達 {RR_MIN}"
                       f"（停損 {pct(plan['stop']):+.1f}%／"
                       f"第一目標 {pct(plan['targets'][0][0]):+.1f}%）→ 現在進場賠賺不對稱")
        if plan.get('rr_zone') and plan['rr_zone'] >= RR_MIN:
            reasons.append(f"但若回調到 {plan['entry_ref']:.2f} 再進，RR 可改善到 "
                           f"{plan['rr_zone']:.2f} → 值得掛單等，而不是現在追")
    elif score >= ENTRY_SCORE_MIN:
        if code == 'DOWN':
            answer = '🟡 尚未到進場時機，列入觀察'
            reasons.append(f"綜合 {score:+.1f} 偏多但階段仍為下跌段 → 屬反彈性質，"
                           f"非趨勢翻多；要做只能小量試單，不可當趨勢單")
        elif rr is None:
            answer = '🟡 尚未到進場時機，列入觀察'
            reasons.append(f"綜合 {score:+.1f}（{sc['verdict']}）偏多，但上方無明確目標價"
                           f"（已在區間頂部或缺乏壓力參考）→ 無法評估風險報酬，不建議追")
        else:
            answer = '🟢 可分批進場'
            reasons.append(f"綜合 {score:+.1f}（{sc['verdict']}）＋階段 {stage['label']}"
                           f"＋RR {rr:.2f} ≥ {RR_MIN}")
    elif score <= -3.5:
        answer = '🔴 不宜進場'
        reasons.append(f"綜合 {score:+.1f}（{sc['verdict']}），多空條件皆不利")
    else:
        answer = '🟡 尚未到進場時機，列入觀察'
        reasons.append(f"綜合 {score:+.1f}（{sc['verdict']}）未達進場門檻 {ENTRY_SCORE_MIN:+.1f}")

    if answer.startswith('🟢') and not SCORE_VALIDATED:
        reasons.append("⚠ 此結論由規則推導，而綜合分數未通過回測驗證 —— "
                       "請以停損價位與 RR 是否可接受為決策依據，不要因為分數偏多就加大部位")
    if stage['contested']:
        reasons.append(f"階段判定不明確（{stage['label']} {stage['score']:.2f} vs "
                       f"{stage['runner_label']} {stage['runner_score']:.2f}）→ 宜降低部位")
    if analog and analog.get('trust') == 'ok':
        h5 = analog['horizons'].get(5)
        b5 = analog['baseline'].get(5)
        if h5 and b5:
            edge = (h5['win_rate'] - b5['win_rate']) * 100
            reasons.append(f"歷史類比 5 日勝率 {h5['win_rate']*100:.0f}%"
                           f"（基準 {b5['win_rate']*100:.0f}%，{edge:+.0f}pp，n={analog['n']}）")

    # ── 等什麼（每條都必須帶數字）──
    triggers = []
    if ma10 == ma10 and close < ma10:
        triggers.append(f"站上 MA10 {ma10:.2f} 且量比 ≥ {VOL_EXPAND}")
    if ma20 == ma20 and close < ma20:
        triggers.append(f"站上 MA20 {ma20:.2f}（中期壓力）")
    triggers.append(f"回測 {a['low']:.2f} 不破且量比 ≤ {VOL_DRY}"
                    f"（打第二支腳，停損可縮到 {a['low']*0.98:.2f} → RR 改善）")
    mh = g('MACD_hist')
    if mh == mh and mh < 0:
        triggers.append(f"MACD 柱由 {mh:.3f} 收斂轉正")
    if struct.get('last_L') and struct.get('last_H'):
        triggers.append(f"結構出現第一個 HL（不破 {struct['last_L']:.2f} "
                        f"且突破 {struct['last_H']:.2f}）")
    if inst and inst.get('dir') != 'buy':
        triggers.append(f"外資轉為連買 ≥ 3 日（目前 "
                        f"{'連賣 ' + str(inst.get('days', 0)) + ' 日' if inst.get('dir') == 'sell' else '中性'}）")

    abandon = []
    if struct.get('last_L'):
        abandon.append(f"跌破 {struct['last_L']:.2f}（最近確認低點）→ 下降結構延續，下跌段未完")
    abandon.append(f"跌破今日低 {a['low']:.2f} → 今天的量變成套牢籌碼")
    if a['color'] == 'red':
        abandon.append("明日開高走低收黑且量大於今日 → 今日紅K為假突破")

    return {'answer': answer, 'reasons': reasons, 'plan': plan,
            'triggers': triggers, 'abandon': abandon, 'existing': existing}


# ════════════════════════════════════════════════════════════
#  7. 編排
# ════════════════════════════════════════════════════════════
def analyze_bar(df, code=None, inst=None, pool_frames=None, want_analog=False,
                buy_price=None, stable_stage=True, prev_stage_code=None,
                want_timing=True):
    """純計算，不 print。回測與報告共用的唯一路徑 ——
    這保證螢幕上看到的分數與回測驗證的分數是同一個東西。

    inst 由呼叫端解析好（dict: dir/days/note）。回測必須傳 as-of 快照，
    絕不讓本函式去查「現在」的法人資料。

    want_timing=False 時跳過 timing_verdict（它會逐根呼叫 5 個 daily_analysis
    的偵測函式）。回測只需要 score 與 stage，關掉可大幅縮短執行時間。
    """
    a      = candle_anatomy(df)
    pats   = multibar_patterns(df, a)
    vp     = volume_price(df, a)
    piv    = find_pivots(df)
    struct = swing_structure(df, piv)
    feats  = stage_features(df, struct)
    if prev_stage_code is not None:
        # 回測逐根走查時直接串接前一根的標籤，省掉 classify_stage_stable 的重算
        stage = classify_stage(df, f=feats, struct=struct, prev_code=prev_stage_code)
    else:
        stage = (classify_stage_stable(df, f=feats, struct=struct) if stable_stage
                 else classify_stage(df, f=feats, struct=struct))
    sc     = kline_score(df, a, pats, vp, stage, inst)
    gaps   = unfilled_gaps(df)

    analog = None
    if want_analog:
        analog = analog_forecast(df, code=code, pool_frames=pool_frames, stage=stage)

    timing = (timing_verdict(df, sc, stage, vp, a, struct, gaps, inst, buy_price, analog)
              if want_timing else None)
    return {'anatomy': a, 'patterns': pats, 'vp': vp, 'pivots': piv, 'struct': struct,
            'stage': stage, 'score': sc, 'gaps': gaps, 'analog': analog,
            'timing': timing, 'inst': inst}


# ════════════════════════════════════════════════════════════
#  10. 歷史類比統計
# ════════════════════════════════════════════════════════════
# 兩段式比對：先硬篩體制，再軟距離比幾何。
# 13 維精確分桶會回傳 n=0；純 kNN 又會把下跌段的 K 配到主升段（幾何長得像），
# 正是要避免的錯誤。所以體制用硬條件，幾何用軟距離。
FP_FEATURES = {           # 特徵 → 距離權重
    'body_signed': 1.5, 'low_pct': 1.2, 'up_pct': 1.0, 'rng_atr': 0.8,
    'ret1': 1.2, 'ret5': 1.0, 'rsi': 1.2, 'dev_ma20': 1.2,
    'pos60': 1.5, 'vol_log': 1.5, 'macd_atr': 0.8, 'bb_pos': 0.8, 'adx': 0.8,
}
FP_Z_WINDOW       = 250   # 滾動 z-score 視窗（因果；絕不可用全樣本 mean/std）
FP_DIST_MAX       = 1.0
FP_DIST_RELAX     = 1.4
FP_TOP_N          = 200
MAX_FWD           = 10    # 最長前瞻期；決定候選日的合格界線
MIN_SAMPLES_OK    = 30    # 達此數才視為可參考
MIN_SAMPLES_SHOW  = 15    # 低於此數只印 n，抑制所有百分位

REGIME_LABELS = {'UP': '多頭', 'BASE': '盤整', 'TOP': '轉弱', 'DOWN': '空頭'}
_fp_cache = {}


def regime_group(df):
    """向量化的體制分組（UP/BASE/TOP/DOWN），供指紋硬篩使用。

    刻意不用 classify_stage：那需要逐根重算樞紐，750 根 × 7 檔會慢到不可用。
    這裡用均線位置與斜率做快篩，全向量化且同樣因果。
    分組精神與五階段一致，但粒度較粗 —— 報告中會標明是「體制快篩分組」，
    不會與畫面上顯示的五階段標籤混稱。
    """
    close = df['Close']
    ma20, ma60 = df['MA20'], df['MA60']
    s20, s60 = df['MA20_slope'], df['MA60_slope']
    pos252 = df['Range_pos252']

    grp = pd.Series('BASE', index=df.index)
    grp[(close > ma20) & (s20 > 0.5)] = 'UP'
    grp[(close < ma20) & (s20 < -0.5)] = 'DOWN'
    # 頭部優先於空頭：長天期還在漲、短天期已轉折、且仍在年線高位
    grp[(s60 > 1.0) & (s20 < 1.0) & (pos252 > 0.5) & (close < df['MA10'])] = 'TOP'
    grp[ma60.isna()] = 'BASE'
    return grp


def build_fp_frame(df, ticker):
    """每根一列的指紋矩陣 + 硬篩欄 + 前瞻報酬。全部因果。"""
    key = (ticker, len(df), str(df.index[-1]))
    if key in _fp_cache:
        return _fp_cache[key]

    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    rng = (h - l).replace(0, np.nan)
    atr = df['ATR'].replace(0, np.nan)

    raw = pd.DataFrame(index=df.index)
    raw['body_signed'] = (c - o) / rng * 100
    raw['up_pct']      = (h - np.maximum(o, c)) / rng * 100
    raw['low_pct']     = (np.minimum(o, c) - l) / rng * 100
    raw['rng_atr']     = (h - l) / atr
    raw['ret1']        = c.pct_change() * 100
    raw['ret5']        = c.pct_change(5) * 100
    raw['rsi']         = df['RSI']
    raw['dev_ma20']    = df['dev_ma20']
    raw['pos60']       = df['Range_pos'] * 100
    raw['vol_log']     = np.log(df['Vol_ratio'].replace(0, np.nan))
    raw['macd_atr']    = df['MACD_hist'] / atr
    raw['bb_pos']      = ((c - df['BB_lower'])
                          / (df['BB_upper'] - df['BB_lower']).replace(0, np.nan))
    raw['adx']         = df['ADX']
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 滾動 z-score（因果）+ clip，讓不同股票的特徵可比
    fp = pd.DataFrame(index=df.index)
    for col in FP_FEATURES:
        m = raw[col].rolling(FP_Z_WINDOW, min_periods=120).mean()
        s = raw[col].rolling(FP_Z_WINDOW, min_periods=120).std()
        fp[col] = ((raw[col] - m) / s.replace(0, np.nan)).clip(-3, 3)

    # 硬篩欄
    fp['regime'] = regime_group(df)
    chg = c.pct_change() * 100
    fp['price_dir'] = np.where(chg > PRICE_FLAT, 'up',
                               np.where(chg < -PRICE_FLAT, 'down', 'flat'))
    vr = df['Vol_ratio']
    fp['vol_bkt'] = np.where(vr >= 1.5, 'exp', np.where(vr < VOL_DRY, 'dry', 'norm'))

    # 前瞻報酬：entry = 次日開盤（讀到報告時今天已收盤，最早能成交的價格）
    entry = o.shift(-1)
    fp['entry'] = entry
    for n in (3, 5, 10):
        fp[f'fwd{n}'] = c.shift(-n) / entry - 1
        fp[f'mae{n}'] = (l.shift(-1).rolling(n).min().shift(-(n - 1)) / entry - 1)
    # 候選日的前瞻視窗結束日期。用它判斷候選是否完全落在查詢日之前，
    # 同儕池跨股票時以日期比對才正確（位置索引不通用）。
    fp['fwd_end'] = pd.Series(df.index, index=df.index).shift(-MAX_FWD)
    fp['ticker'] = ticker
    fp['bar_date'] = df.index

    _fp_cache[key] = fp
    return fp


def _horizon_stats(sub, horizons=(3, 5, 10)):
    out = {}
    for n in horizons:
        v = sub[f'fwd{n}'].dropna()
        if len(v) == 0:
            continue
        mae = sub[f'mae{n}'].dropna()
        out[n] = {
            'n': len(v), 'win_rate': float((v > 0).mean()),
            'median': float(v.median()), 'mean': float(v.mean()),
            'p25': float(v.quantile(0.25)), 'p75': float(v.quantile(0.75)),
            'mae_median': float(mae.median()) if len(mae) else float('nan'),
        }
    return out


def analog_forecast(df, code=None, pool_frames=None, stage=None, horizons=(3, 5, 10)):
    """歷史上長得像今天的日子，後來怎麼走。

    防 look-ahead 四道防線：
      (i)  候選日的前瞻視窗必須完整且結束於查詢日之前（以日期比對，跨股票通用）
      (ii) 特徵正規化只用滾動 250 根的因果 mean/std
      (iii) 樞紐帶 k 根確認延遲（體制快篩不用樞紐，故此處不受影響）
      (iv) backtest_kline.verify_causality 逐點斷言

    基準率是強制項：「5 日勝率 57%」在基準 51% 時 edge 只有 6pp，
    少了這行整個統計就是自我欺騙。
    """
    ticker = code or 'SELF'
    self_fp = build_fp_frame(df, ticker)
    target = self_fp.iloc[-1]
    tdate  = df.index[-1]

    cols = list(FP_FEATURES.keys())
    w = np.array([FP_FEATURES[c] for c in cols], dtype=float)
    tgt = target[cols].values.astype(float)
    if np.isnan(tgt).any():
        return {'trust': 'insufficient', 'n': 0, 'min_show': MIN_SAMPLES_SHOW,
                'fp_note': '暖機不足（滾動 z-score 需 ≥120 根），無法建立指紋'}

    fp_note = (f"體制={REGIME_LABELS.get(target['regime'], target['regime'])}（快篩分組）"
               f"｜價{'漲' if target['price_dir']=='up' else ('跌' if target['price_dir']=='down' else '平')}"
               f"｜{'放量' if target['vol_bkt']=='exp' else ('量縮' if target['vol_bkt']=='dry' else '量平')}"
               f"　(實體{target['body_signed']:+.1f}σ 下影{target['low_pct']:+.1f}σ "
               f"RSI{target['rsi']:+.1f}σ 乖離{target['dev_ma20']:+.1f}σ)")

    def eligible(frame):
        """候選日的前瞻視窗必須完整，且結束於查詢日之前。"""
        m = frame['fwd_end'].notna() & (frame['fwd_end'] < tdate)
        m &= frame[cols].notna().all(axis=1)
        m &= frame['fwd10'].notna()
        return frame[m]

    # ── 放寬階梯：固定順序、永遠印出目前層級、絕不默默放寬 ──
    # regime 硬篩永不放寬（放寬它就等於把空頭的日子拿去預測多頭，失去意義）
    LADDER = [
        (0, '僅本股', FP_DIST_MAX, False, True),
        (1, f'距離放寬到 {FP_DIST_RELAX}', FP_DIST_RELAX, False, True),
        (2, '納入同儕池', FP_DIST_RELAX, True, True),
        (3, '同儕池＋放掉量能分組', FP_DIST_RELAX, True, False),
    ]

    chosen = None
    for level, note, dmax, use_peers, use_vol in LADDER:
        frames = [self_fp]
        if use_peers and pool_frames:
            for pt, pdf in pool_frames.items():
                if pt == ticker or pdf is None or len(pdf) < 150:
                    continue
                try:
                    frames.append(build_fp_frame(pdf, pt))
                except Exception:
                    continue
        pool = pd.concat([eligible(f) for f in frames]) if frames else None
        if pool is None or pool.empty:
            continue

        m = (pool['regime'] == target['regime']) & (pool['price_dir'] == target['price_dir'])
        if use_vol:
            m &= (pool['vol_bkt'] == target['vol_bkt'])
        cand = pool[m]
        if cand.empty:
            continue

        d = np.sqrt(((cand[cols].values.astype(float) - tgt) ** 2 * w).sum(axis=1) / w.sum())
        hit = cand.assign(dist=d)
        hit = hit[hit['dist'] <= dmax].nsmallest(FP_TOP_N, 'dist')
        chosen = (level, note, dmax, hit, pool, len(frames))
        if len(hit) >= MIN_SAMPLES_OK:
            break

    if chosen is None:
        return {'trust': 'insufficient', 'n': 0, 'min_show': MIN_SAMPLES_SHOW,
                'fp_note': fp_note + '　→ 同體制樣本為 0'}

    level, note, dmax, hit, pool, n_frames = chosen
    n = len(hit)
    trust = ('ok' if n >= MIN_SAMPLES_OK else
             'weak' if n >= MIN_SAMPLES_SHOW else 'insufficient')
    if trust == 'insufficient':
        return {'trust': 'insufficient', 'n': n, 'min_show': MIN_SAMPLES_SHOW,
                'fp_note': fp_note,
                'pool_note': f"放寬層級 {level}：{note}（{n_frames} 檔）"}

    stats    = _horizon_stats(hit, horizons)
    baseline = _horizon_stats(pool, horizons)     # 無條件基準（同池、同期間）

    edge_notes = []
    for nh in horizons:
        if nh in stats and nh in baseline:
            wr = (stats[nh]['win_rate'] - baseline[nh]['win_rate']) * 100
            md = (stats[nh]['median'] - baseline[nh]['median']) * 100
            edge_notes.append(
                f"{nh}日：勝率 {wr:+.0f}pp、中位數 {md:+.1f}pp vs 基準"
                + ("（有優勢）" if wr > 3 and md > 0 else
                   "（無明顯優勢）" if abs(wr) <= 3 else "（劣於基準）"))

    p5 = stats.get(5, {}).get('win_rate', 0.5)
    ci = 1.96 * float(np.sqrt(max(p5 * (1 - p5), 1e-9) / n))

    top = [{'ticker': str(r['ticker']), 'date': str(pd.Timestamp(r['bar_date']).date()),
            'fwd5': float(r['fwd5']), 'dist': float(r['dist'])}
           for _, r in hit.nsmallest(5, 'dist').iterrows()]

    return {
        'trust': trust, 'n': n, 'min_show': MIN_SAMPLES_SHOW,
        'dist_max': dmax, 'mean_dist': float(hit['dist'].mean()),
        'relax_level': level,
        'pool_note': f"放寬層級 {level}：{note}（樣本池 {n_frames} 檔、"
                     f"合格候選 {len(pool):,} 根）",
        'fp_note': fp_note, 'horizons': stats, 'baseline': baseline,
        'horizon_list': [h for h in horizons if h in stats],
        'edge_notes': edge_notes, 'ci95': ci, 'top_matches': top,
    }


# ════════════════════════════════════════════════════════════
#  8. 報告輸出
# ════════════════════════════════════════════════════════════
def _sec(title):
    print(f"\n  ── {title} ──")


def print_kline_report(res, df, ticker, name="", fq=None, status_note="",
                       data_note="", buy_price=None, shares=0):
    a, vp, stage, sc = res['anatomy'], res['vp'], res['stage'], res['score']
    struct, timing = res['struct'], res['timing']
    r = df.iloc[-1]
    g = lambda c: float(r[c]) if c in df.columns and pd.notna(r[c]) else float('nan')
    close = a['close']
    now = da.now_tw()

    print()
    print("=" * 60)
    print(f"   🕯 K線趨勢診斷   {ticker}  {name}")
    print(f"   {now.strftime('%Y-%m-%d  %H:%M')}  {status_note}")
    print(f"   歷史 {len(df)} 根｜yfinance(還原權息) + Fugle 收盤 + FinMind 法人")
    print("=" * 60)
    if not SCORE_VALIDATED:
        print(f"  ⚠ {SCORE_VALIDATION_NOTE}")
        print(f"    以下的 K 線解讀、量價、結構、階段仍是有用的結構化整理，")
        print(f"    但「綜合分數」不具實證預測力，請以價位與條件判斷為主。")
        print("=" * 60)
    if data_note:
        print(f"\n  {data_note}")

    # ── 快照 ──
    print()
    print(f"  現價 {close:.2f}  ({a['day_chg']:+.2f}%)   開 {a['open']:.2f}  "
          f"高 {a['high']:.2f}  低 {a['low']:.2f}"
          + (f"  均價 {fq['avg']:.2f}" if fq and fq.get('avg') else ""))
    print(f"  MA5 {g('MA5'):.2f}   MA10 {g('MA10'):.2f}   "
          f"MA20 {g('MA20'):.2f}   MA60 {g('MA60'):.2f}")
    print(f"  RSI {g('RSI'):.1f}   MACD柱 {g('MACD_hist'):+.3f}   "
          f"量比 {g('Vol_ratio'):.2f}   ATR {g('ATR'):.2f}（{g('ATR_pct'):.1f}%）")
    print(f"  ADX {g('ADX'):.1f}（+DI {g('DI_plus'):.1f} / -DI {g('DI_minus'):.1f}）  "
          f"距60日高 {(close/g('High60')-1)*100:+.1f}%  距60日低 {(close/g('Low60')-1)*100:+.1f}%")
    if buy_price:
        print(f"  持倉 {shares:,} 股　成本 {buy_price:.2f}　"
              f"損益 {(close/buy_price-1)*100:+.1f}%")

    # ── 今日K線解讀 ──
    _sec("今日K線解讀")
    ra = f"{a['rng_atr']:.2f}" if a['rng_atr'] == a['rng_atr'] else "-"
    print(f"  型態：{a['cls_note']}  實體 {a['body_pct']:.0f}%  "
          f"上影 {a['up_pct']:.0f}%  下影 {a['low_pct']:.0f}%  振幅 {ra}×ATR")
    facts, meaning = candle_reading(df, a, vp, stage,
                                   avg_price=fq.get('avg') if fq else None)
    for i, f in enumerate(facts):
        print(f"  {'└' if i == len(facts)-1 else '├'} {f}")
    print(f"\n  💬 這根K線在說什麼")
    for m in meaning:
        print(f"     {m}")

    # ── 近5日型態 ──
    _sec("近5日型態")
    pats = res['patterns']
    if pats:
        for p in pats:
            mark = '✓' if p['dir'] != 0 else 'ℹ'
            sign = f"[{p['dir']*p['strength']:+.1f}]" if p['dir'] != 0 else "[ 0.0]"
            print(f"  {mark} {p['name']:<14}{sign}  {p['msg']}")
    else:
        print("  （近 5 根無明顯組合型態）")
    for gp in res['gaps']:
        if gp['bars_ago'] <= 40:
            side = '上方壓力' if gp['lo'] > close else '下方支撐'
            print(f"  ⚠ 未回補{'向上' if gp['type']=='up' else '向下'}缺口 "
                  f"{gp['lo']:.2f}~{gp['hi']:.2f}（{gp['date'].date()}）→ {side}")
    last5 = df['Close'].pct_change().iloc[-5:] * 100
    print(f"  近5日：" + " / ".join(f"{v:+.1f}%" for v in last5))
    pc = next(c for c in sc['components'] if c['name'] == 'pattern')
    print(f"  型態分項 {pc['sub']:+.2f}")

    # ── 量價配合 ──
    _sec("量價配合")
    print(f"  今日：價{'漲' if a['day_chg']>0 else ('跌' if a['day_chg']<0 else '平')} "
          f"{a['day_chg']:+.2f}% ／ 量比 {vp['vol_ratio']:.2f} → 【{vp['quadrant']}】")
    for i, m in enumerate(vp['msgs']):
        print(f"  {'└' if i == len(vp['msgs'])-1 else '├'} {m}")
    print(f"  量價分項 {vp['bias']:+.2f}")

    # ── 趨勢結構與階段 ──
    _sec("趨勢結構與階段")
    print(f"  階段判定：{stage['icon']} {stage['label']}"
          f"（信心 {stage['confidence']:.2f}）　"
          f"次高：{stage['runner_label']}（{stage['runner_score']:.2f}）"
          + ("　⚠ 遲滯沿用前一根" if stage.get('held') else ""))
    for label, ok, w in stage['evidence']:
        print(f"  {'✓' if ok else '✗'} {label}　[×{w}]")
    print(f"  ├ 均線：MA5 {g('MA5'):.2f} / MA10 {g('MA10'):.2f} / "
          f"MA20 {g('MA20'):.2f} / MA60 {g('MA60'):.2f}")
    print(f"  ├ 斜率：MA20 {g('MA20_slope'):+.2f}%/10根　MA60 {g('MA60_slope'):+.2f}%/20根")
    print(f"  ├ 波段結構：{struct['pattern']}　{struct['note']}")
    if struct['pivots']:
        chain = " → ".join(f"{p['type']}{p['price']:.1f}({p['date'].strftime('%m-%d')})"
                           for p in struct['pivots'])
        print(f"  │   {chain}")
        print(f"  │   （樞紐需右側 {struct['confirm_lag']} 根確認，故最新樞紐落後）")
    if struct['leg_dir']:
        print(f"  ├ 目前腿：自 {struct['pivots'][-1]['price']:.2f} "
              f"{'反彈' if struct['leg_dir']=='up' else '下跌'} {struct['leg_pct']:+.1f}%，"
              f"已 {struct['leg_bars']} 根")
    print(f"  └ 位置：距年高 {g('dd_252h'):+.1f}%　距年低 {g('up_252l'):+.1f}%　"
          f"60日區間 {g('Range_pos')*100:.0f}%")
    if stage['contested']:
        print(f"  ℹ 兩階段分數接近 → 應解讀為「{stage['label']}／{stage['runner_label']} "
              f"難分」，而非已確立其中之一")
    print(f"  階段分項 {stage['sub']:+.2f}")

    # ── 法人動向 ──
    _sec("法人動向")
    inst = res['inst']
    if inst and inst.get('rows'):
        for d, v in inst['rows']:
            print(f"  {d}  外資 {v['foreign']:>+8,}張  投信 {v['trust']:>+7,}張  "
                  f"自營 {v['dealer']:>+7,}張  合計 {v['total']:>+8,}張")
        print(f"  方向：{inst.get('note', '')}"
              f"　最近一日（{inst['latest_date']}）外資 "
              f"{'買超' if inst['latest_buy'] else '賣超'} "
              f"{abs(inst['latest_foreign']):,} 張")
    else:
        print("  （無法人資料：非台股、未設 FINMIND_TOKEN，或 API 無回應）")
    print("  ℹ 法人資料為 T+1，非今日")

    # ── 綜合分數 ──
    _sec("綜合K線分數")
    print(f"  分數  {sc['score']:+.1f} / ±10      判定：{sc['icon']} {sc['verdict']}"
          + ("" if SCORE_VALIDATED else "　⚠ 未通過回測驗證"))
    print(f"  ┌{'─'*56}┐")
    print(f"  │ {'項目':<10}{'權重':>6}{'分項':>8}{'貢獻':>8}   {'說明':<16}│")
    for c in sc['components']:
        detail = c['detail'][:22]
        print(f"  │ {c['label']:<10}{c['weight']:>6.1f}{c['sub']:>+8.2f}"
              f"{c['contrib']:>+8.2f}   {detail:<16}│")
    print(f"  ├{'─'*56}┤")
    print(f"  │ {'合計':<10}{'':>6}{'':>8}{sc['score']:>+8.2f}{'':<19}│")
    print(f"  └{'─'*56}┘")

    # ── 歷史類比 ──
    _sec("歷史類比統計")
    _print_analog(res['analog'])

    # ── 進出場時機 ──
    _sec("進出場時機建議")
    t, plan = timing, timing['plan']
    print(f"  結論：{t['answer']}")
    for rs in t['reasons']:
        print(f"  理由：{rs}" if rs is t['reasons'][0] else f"        {rs}")
    if plan['zone'] and plan['stop']:
        sp = (plan['stop'] / close - 1) * 100
        zone_note = ""
        if plan.get('in_zone') is False:
            zone_note = (f"　⚠ 現價 {close:.2f} 已"
                         f"{'高於' if close > plan['zone'][1] else '低於'}此區，"
                         f"需等{'回調' if close > plan['zone'][1] else '止跌'}")
        print(f"\n   進場價區  {plan['zone'][0]:.2f} ~ {plan['zone'][1]:.2f}{zone_note}")
        if plan.get('zone_note'):
            print(f"             （{plan['zone_note']}）")
        print(f"   停損      {plan['stop']:.2f}（{plan['stop_note']}，"
              f"以現價計風險 {sp:+.1f}%）")
        if plan.get('stop_alt'):
            alt_p, alt_n = plan['stop_alt']
            print(f"             ＊另一參考：{alt_n} = {alt_p:.2f}"
                  f"（{(alt_p/close-1)*100:+.1f}%，較寬）")
        for i, (px, nm) in enumerate(plan['targets'][:3], 1):
            mark = "   ← 實際會先遇到" if i == 1 else ""
            print(f"   目標{'①②③'[i-1]}    {px:>8.2f}  {nm:<18}"
                  f"({(px/close-1)*100:+.1f}%){mark}")
        if plan['rr'] is not None:
            ok = plan['rr'] >= RR_MIN
            print(f"   風險報酬  以現價 {close:.2f} 進：停損 {sp:+.1f}%／目標① "
                  f"{(plan['targets'][0][0]/close-1)*100:+.1f}% → RR {plan['rr']:.2f}"
                  f"  {'✓' if ok else f'❌（門檻 {RR_MIN}）'}")
            if plan.get('rr_zone') is not None:
                e = plan['entry_ref']
                print(f"             若等到 {e:.2f} 進：停損 "
                      f"{(plan['stop']/e-1)*100:+.1f}%／目標① "
                      f"{(plan['targets'][0][0]/e-1)*100:+.1f}% → RR {plan['rr_zone']:.2f}"
                      f"  {'✓' if plan['rr_zone'] >= RR_MIN else '❌'}")
            if not ok:
                print(f"   👉 所以結論是「等」而不是「買」：現在進場賠賺不對稱")
    if t['triggers']:
        print(f"\n  等什麼（任一成立即重評）：")
        for i, tr in enumerate(t['triggers'], 1):
            print(f"   {'①②③④⑤⑥'[min(i-1,5)]} {tr}")
    if t['abandon']:
        print(f"  直接放棄的情況：")
        for ab in t['abandon']:
            print(f"   ✗ {ab}")

    # ── 現有 A/C 對照 ──
    _sec("現有 A / C 策略對照（僅顯示，不影響上方任何結論）")
    _print_existing(t['existing'], close, sc, stage)

    print()
    print("=" * 60)
    print("  ℹ 本報告為規則推導與歷史統計，非投資建議。K線型態的學術實證")
    print("    效力薄弱，任何情境都請先設好停損再進場。")
    print("=" * 60)


def _print_analog(an):
    if an is None:
        print("  （歷史類比尚未啟用）")
        return
    if an.get('trust') == 'insufficient':
        print(f"  指紋：{an['fp_note']}")
        print(f"  命中 n = {an['n']}（門檻 {an['min_show']}）→ 樣本不足，本項不列入判斷")
        return
    print(f"  指紋：{an['fp_note']}")
    print(f"  範圍：{an['pool_note']}")
    warn = "  ⚠ 樣本偏少，僅供參考" if an['trust'] == 'weak' else "  ✓ 可參考"
    print(f"  命中 n = {an['n']}（距離 ≤ {an['dist_max']:.2f}，"
          f"平均相似距離 {an['mean_dist']:.2f}）{warn}")
    print(f"       {'期間':>6}{'勝率':>7}{'中位數':>9}{'平均':>9}"
          f"{'p25':>8}{'p75':>8}{'期間最大逆行':>13}")
    for h in an['horizon_list']:
        s = an['horizons'][h]
        print(f"       {h:>4} 日{s['win_rate']*100:>6.0f}%{s['median']*100:>+8.1f}%"
              f"{s['mean']*100:>+8.1f}%{s['p25']*100:>+7.1f}%{s['p75']*100:>+7.1f}%"
              f"{s['mae_median']*100:>+12.1f}%")
    for h in an['horizon_list']:
        b = an['baseline'][h]
        print(f"     基準{h:>2}日{b['win_rate']*100:>6.0f}%{b['median']*100:>+8.1f}%"
              f"{b['mean']*100:>+8.1f}%{b['p25']*100:>+7.1f}%{b['p75']*100:>+7.1f}%"
              f"{b['mae_median']*100:>+12.1f}%")
    for line in an['edge_notes']:
        print(f"  → {line}")
    print(f"  ℹ 勝率 95% 信賴區間 ±{an['ci95']*100:.0f}pp（n={an['n']}）")
    if an.get('top_matches'):
        print(f"  最相似 {len(an['top_matches'])} 天：" + "  ".join(
            f"{m['ticker']} {m['date']}({m['fwd5']*100:+.1f}%)" for m in an['top_matches']))
    print("  ⚠ 歷史類比 ≠ 預測；樣本期間的市場環境與現在可能不同。")


def _print_existing(ex, close, sc, stage):
    A, C = ex['A'], ex['C']
    bs_score, bs_details = ex['bottom']
    print(f"  【策略A 趨勢跟蹤】ADX {A['adx']:.1f} {'≥' if A['adx_ok'] else '<'} 25 "
          f"{'✓' if A['adx_ok'] else '✗'}　"
          f"近20日 {A['roc20']:+.1f}% {'≥' if A['roc_ok'] else '<'} 3% "
          f"{'✓' if A['roc_ok'] else '✗'}　→ "
          f"{'前提成立' if A['adx_ok'] and A['roc_ok'] else '前提不成立'}")
    print(f"     entry_signals score {A['score']}/4 "
          f"{'✓ 達標' if A['score_ok'] else '✗ 未達'}"
          f"　→ 策略A {'🟢 觸發' if A['pass'] else '⏳ 不符'}")
    for m in A['msgs'][:5]:
        print(f"     {m.strip()}")

    print(f"  【策略C 低谷反彈】前提 {close:.2f} "
          f"{'<' if C['premise'] else '≥'} MA10 {C['ma10']:.2f} "
          f"{'✓' if C['premise'] else '✗'}")
    for i, (label, ok) in enumerate(C['conds'], 1):
        print(f"     {'①②③'[i-1]} {label} {'✓' if ok else '✗'}")
    p = C['probe']
    print(f"     → {C['met']}/3 達標 "
          f"{'✓ 觸發' if C['pass'] else '✗ 未觸發'}"
          f"　C停損 {p['stop_price']:.1f}　C目標 MA10 {p['target_price']:.1f}"
          f"　RR 1:{p['rr_ratio']:.2f}")

    # bottom_score 的 details 是 (標記, 名稱, 權重, 訊息或None) 的 tuple
    print(f"  【底部評分】{bs_score}/10.0")
    hit = [f"{m}{nm}[×{w}]" for m, nm, w, _ in bs_details if m == '✓']
    miss = [f"{m}{nm}[×{w}]" for m, nm, w, _ in bs_details if m != '✓']
    for group in (hit, miss):
        for k in range(0, len(group), 3):
            print("     " + "　".join(group[k:k + 3]))
    for _, _, _, msg in bs_details:
        if msg:
            print(f"     {msg.strip()}")

    # ── 衝突揭露：本模組最有資訊量的輸出 ──
    ka_bull = sc['score'] >= ENTRY_SCORE_MIN
    ka_wait = '🟢' not in ex.get('_answer', '') if False else None
    conflicts = []
    if C['pass'] and sc['score'] < ENTRY_SCORE_MIN:
        conflicts.append(
            f"策略C 觸發（{C['met']}/3）但 K 線模組綜合僅 {sc['score']:+.1f}"
            f"（{sc['verdict']}）")
    if A['pass'] and sc['score'] < 0:
        conflicts.append(f"策略A 觸發但 K 線模組綜合 {sc['score']:+.1f} 偏空")
    if bs_score >= 2.0 and stage['code'] == 'DOWN':
        conflicts.append(f"底部評分 {bs_score} 已過 2.0 門檻，但階段判定為下跌段")
    if (not C['pass']) and (not A['pass']) and ka_bull:
        conflicts.append(f"A/C 皆未觸發，但 K 線模組綜合 {sc['score']:+.1f} 偏多")

    if conflicts:
        print(f"  ⚖ 衝突揭露：")
        for c in conflicts:
            print(f"     • {c}")
        print(f"     差異來源：策略C 只問「是否超跌＋衰竭」，不看趨勢階段與 RR；")
        print(f"     K 線模組會因 RR < {RR_MIN} 或階段為下跌段／頭部而降級。")
        print(f"     兩者都對，衡量標準不同 —— 決策權在你。")
    else:
        print(f"  ⚖ 三方判定方向一致（K線 {sc['score']:+.1f}／"
              f"策略A {'觸發' if A['pass'] else '不符'}／"
              f"策略C {'觸發' if C['pass'] else '不符'}／底部 {bs_score}）")
    print(f"  ⚠ 本模組為對照用，未改動任何既有訊號。")


# ════════════════════════════════════════════════════════════
#  9. 入口
# ════════════════════════════════════════════════════════════
def _resolve_inst(ticker):
    """解析法人動向（僅即時報告使用；回測絕不呼叫，改傳 as-of 快照）。"""
    if not ticker.endswith(('.TW', '.TWO')):
        return None
    code = ticker.replace('.TWO', '').replace('.TW', '')
    snap = kd.inst_snapshot(kd.fetch_inst_series(code))
    if not snap:
        return None
    d, days = snap['dir'], snap['days']
    snap['note'] = ('外資連買 {} 日，籌碼偏多'.format(days) if d == 'buy' else
                    '外資連賣 {} 日，出場訊號加強'.format(days) if d == 'sell' else
                    '外資中性')
    return snap


def kline_report(raw_code, days=750, use_peer_pool=True, show_analog=True,
                 live=True, buy_price=None, shares=0, name=""):
    """單檔 K 線趨勢診斷報告。"""
    ticker, label, df = kd.resolve_ticker(raw_code, days=days)
    if df is None:
        print(f"\n  ⚠  無法取得 {raw_code} 資料，請確認代號是否正確\n")
        return None
    if len(df) < kd.WARMUP_BARS:
        print(f"\n  ⚠  {ticker} 歷史資料僅 {len(df)} 根，不足 {kd.WARMUP_BARS} 根，無法分析\n")
        return None

    status, status_note = da.market_status()
    data_note, fq = "", None
    if live:
        df, data_note, fq = kd.attach_live_price(df, ticker, status)

    inst = _resolve_inst(ticker) if live else None
    pool = _peer_pool(ticker, days) if (use_peer_pool and show_analog) else None
    res  = analyze_bar(df, code=ticker, inst=inst, pool_frames=pool,
                       want_analog=show_analog, buy_price=buy_price)
    print_kline_report(res, df, ticker, name or label, fq, status_note,
                       data_note, buy_price, shares)
    return res


def _peer_pool(exclude_ticker, days):
    """同儕池：延遲抓取，只在歷史類比需要放寬樣本時才用到。"""
    peers = [t for t in PEER_TICKERS if t != exclude_ticker]
    return kd.fetch_long_batch(peers, days=days)


PEER_TICKERS = ['2313.TW', '2327.TW', '2344.TW', '2449.TW',
                '3037.TW', '3711.TW', '6805.TW']


def kline_all_holdings(days=750, use_peer_pool=True, show_analog=True):
    """逐檔診斷所有持倉。"""
    holdings = getattr(da, 'HOLDINGS', {}) or {}
    if not holdings:
        print("\n  ⚠  未設定持倉（HOLDINGS 為空），請改用單檔查詢\n")
        return
    items = list(holdings.items())
    if len(items) > 10:
        print(f"\n  ⚠  標的數 {len(items)} > 10，FinMind 法人 API 可能限流，建議分批\n")
    pool = kd.fetch_long_batch(PEER_TICKERS, days=days) if (use_peer_pool and show_analog) else None
    for ticker, info in items:
        try:
            kline_report(ticker, days=days, use_peer_pool=False, show_analog=show_analog,
                         buy_price=info.get('buy_price'), shares=info.get('shares', 0),
                         name=info.get('name', ''))
        except Exception as e:
            print(f"\n  ⚠  {ticker} 分析失敗：{e}\n")
    _ = pool


# ════════════════════════════════════════════════════════════
#  自我檢查（步驟 2~5 的驗證）
# ════════════════════════════════════════════════════════════
def _check_anatomy(code="3037", n=20):
    ticker, label, df = kd.resolve_ticker(code)
    if df is None:
        print(f"  ⚠ 無法取得 {code}")
        return
    print(f"\n  ── {ticker} 近 {n} 根 K 線解剖 ──")
    print(f"  {'日期':<12}{'開':>8}{'高':>8}{'低':>8}{'收':>8}"
          f"{'漲跌%':>8}{'實體':>6}{'上影':>6}{'下影':>6}{'ATR倍':>7}  分類")
    for k in range(len(df) - n, len(df)):
        sub = df.iloc[:k + 1]
        a = candle_anatomy(sub)
        ra = f"{a['rng_atr']:.2f}" if a['rng_atr'] == a['rng_atr'] else "  -"
        print(f"  {str(df.index[k].date()):<12}{a['open']:>8.2f}{a['high']:>8.2f}"
              f"{a['low']:>8.2f}{a['close']:>8.2f}{a['day_chg']:>+8.2f}"
              f"{a['body_pct']:>5.0f}%{a['up_pct']:>5.0f}%{a['low_pct']:>5.0f}%"
              f"{ra:>7}  {a['cls_note']}")


def _check_patterns(code="3037"):
    ticker, label, df = kd.resolve_ticker(code)
    if df is None:
        return
    counts, samples = {}, {}
    total = 0
    for k in range(kd.WARMUP_BARS, len(df)):
        sub = df.iloc[:k + 1]
        total += 1
        for p in multibar_patterns(sub):
            counts[p['name']] = counts.get(p['name'], 0) + 1
            samples.setdefault(p['name'], []).append(str(df.index[k].date()))
    # 這幾項本質就該極稀有（漲跌停、島狀反轉），不套用「過於嚴格」的警告
    rare_ok = {'一字線', '一價到底', '島狀反轉（底）', '島狀反轉（頂）', '長腳十字'}
    print(f"\n  ── {ticker} 多根型態觸發率（{total} 根）──")
    print(f"  {'型態':<20}{'次數':>6}{'觸發率':>9}   最近 3 次")
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        rate = cnt / total * 100
        if rate > 25:
            flag = " ⚠ 過於寬鬆"
        elif rate < 0.5 and name not in rare_ok:
            flag = " ⚠ 過於嚴格"
        else:
            flag = ""
        print(f"  {name:<20}{cnt:>6}{rate:>8.1f}%   "
              f"{', '.join(samples[name][-3:])}{flag}")

    gaps = unfilled_gaps(df)
    print(f"\n  未回補缺口（近 {GAP_LOOKBACK} 根）：{len(gaps)} 個")
    for g in gaps[-5:]:
        print(f"    {'向上' if g['type']=='up' else '向下'}缺口 "
              f"{g['lo']:.2f}~{g['hi']:.2f}　{g['date'].date()}（{g['bars_ago']} 根前）")


def _check_volume_price(code="3037"):
    ticker, label, df = kd.resolve_ticker(code)
    if df is None:
        return
    counts, bias_sum = {}, {}
    total = 0
    for k in range(kd.WARMUP_BARS, len(df)):
        sub = df.iloc[:k + 1]
        vp = volume_price(sub)
        counts[vp['quadrant']] = counts.get(vp['quadrant'], 0) + 1
        bias_sum[vp['quadrant']] = bias_sum.get(vp['quadrant'], 0.0) + vp['bias']
        total += 1
    print(f"\n  ── {ticker} 量價象限分布（{total} 根）──")
    print(f"  {'象限':<12}{'次數':>6}{'占比':>8}{'平均分項':>10}")
    for q, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {q:<12}{cnt:>6}{cnt/total*100:>7.1f}%{bias_sum[q]/cnt:>+10.2f}")


def _check_pivots(code="3037", show=14):
    ticker, label, df = kd.resolve_ticker(code)
    if df is None:
        return
    piv = find_pivots(df)
    st  = swing_structure(df, piv)
    print(f"\n  ── {ticker} 擺盪樞紐（最近 {show} 個，共 {len(piv)} 個）──")
    prev = None
    for p in piv[-show:]:
        delta = f"{(p['price']/prev['price']-1)*100:+7.1f}%" if prev else "       -"
        bars  = f"{p['pos']-prev['pos']:>4} 根" if prev else "     -"
        print(f"    {p['type']}  {p['price']:>9.2f}  {p['date'].date()}  {delta} {bars}")
        prev = p
    print(f"\n  結構：{st['pattern']}　{st['note']}")
    print(f"  最後高點 {st['last_H']}　最後低點 {st['last_L']}")
    if st['leg_dir']:
        print(f"  目前腿：自最後樞紐 {piv[-1]['price']:.2f} "
              f"{'反彈' if st['leg_dir']=='up' else '下跌'} {st['leg_pct']:+.1f}%，"
              f"已 {st['leg_bars']} 根（樞紐需右側 {st['confirm_lag']} 根確認）")


def _dump_stages(code="3037", n=260, verbose=False):
    """逐日走查階段判定。這是階段分類器最重要的驗證：
    弧線必須讀成 主升段 → 頭部形成 → 下跌段，且不可逐日亂跳。"""
    ticker, label, df = kd.resolve_ticker(code, days=1000)
    if df is None:
        print(f"  ⚠ 無法取得 {code}")
        return
    n = min(n, len(df) - kd.WARMUP_BARS)
    rows, prev = [], None
    for k in range(len(df) - n, len(df)):
        sub = df.iloc[:k + 1]
        s = classify_stage(sub, prev_code=prev)     # 逐根串接遲滯（因果）
        prev = s['code']
        rows.append((df.index[k], float(sub.iloc[-1]['Close']), s))

    print(f"\n  ── {ticker} 階段序列（最近 {n} 根，逐根以 df.iloc[:t+1] 走查）──")
    if verbose:
        print(f"  {'日期':<12}{'收盤':>9}  {'階段':<8}{'信心':>6}  次高")
        for d, c, s in rows:
            mark = ' ⚠並列' if s['contested'] else ''
            print(f"  {str(d.date()):<12}{c:>9.2f}  {s['label']:<8}"
                  f"{s['confidence']:>6.2f}  {s['runner_label']}{mark}")

    # 區段化：只印階段變化的轉折點，一眼看出弧線
    print(f"\n  階段區段（合併連續相同階段）：")
    segs, cur = [], None
    for d, c, s in rows:
        if cur is None or cur['code'] != s['code']:
            cur = {'code': s['code'], 'label': s['label'], 'icon': s['icon'],
                   'start': d, 'end': d, 'bars': 1,
                   'p0': c, 'p1': c, 'conf': [s['confidence']]}
            segs.append(cur)
        else:
            cur['end'], cur['p1'], cur['bars'] = d, c, cur['bars'] + 1
            cur['conf'].append(s['confidence'])
    for g in segs:
        chg = (g['p1'] / g['p0'] - 1) * 100
        print(f"    {g['icon']} {g['label']:<6}{str(g['start'].date())}~{str(g['end'].date())}"
              f"  {g['bars']:>4} 根  {g['p0']:>8.2f}→{g['p1']:>8.2f} ({chg:+6.1f}%)"
              f"  平均信心 {np.mean(g['conf']):.2f}")

    # 亂跳檢查。分兩層看：
    #   標籤層 —— 主升↔初升互跳對方向判斷無害，只是標籤細分
    #   方向層 —— 偏多↔偏空 之間亂跳才真正有害
    flips = sum(1 for g in segs if g['bars'] <= 2)
    fam_segs, cur_f = [], None
    for _, _, s in rows:
        if cur_f is None or cur_f['fam'] != s['family']:
            cur_f = {'fam': s['family'], 'bars': 1}
            fam_segs.append(cur_f)
        else:
            cur_f['bars'] += 1
    fam_flips = sum(1 for g in fam_segs if g['bars'] <= 2)

    dist = {}
    for _, _, s in rows:
        dist[s['label']] = dist.get(s['label'], 0) + 1
    print(f"\n  標籤區段 {len(segs)} 個，其中 ≤2 根 {flips} 個（{flips/len(segs)*100:.0f}%）"
          f"　平均 {len(rows)/len(segs):.1f} 根")
    # 用「平均方向區段長度」而非短命佔比：區段總數少時佔比會失真
    # （3037 只有 6 個方向區段，2 個短命就顯示 33%，實際上平均 43 根才換一次）
    fam_avg = len(rows) / max(len(fam_segs), 1)
    ok = fam_avg >= 8.0
    print(f"  方向區段 {len(fam_segs)} 個，平均 {fam_avg:.1f} 根換一次方向"
          f"（≤2 根的 {fam_flips} 個）"
          f"{'  ✓ 方向穩定' if ok else '  ⚠ 方向亂跳，需提高遲滯或 PIVOT_K'}")
    print(f"  階段分布：" + "　".join(f"{k} {v}({v/len(rows)*100:.0f}%)"
                                    for k, v in sorted(dist.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    code = next((a for a in args if not a.startswith('-')), "3037")
    if '--patterns' in args:
        _check_patterns(code)
    elif '--volprice' in args:
        _check_volume_price(code)
    elif '--anatomy' in args:
        _check_anatomy(code)
    elif '--pivots' in args:
        _check_pivots(code)
    elif '--dump-stages' in args:
        _dump_stages(code, verbose='--verbose' in args)
    elif '--selfcheck' in args:
        _check_anatomy(code)
        _check_patterns(code)
        _check_volume_price(code)
        _check_pivots(code)
        _dump_stages(code)
    else:
        kline_report(code, show_analog='--no-analog' not in args,
                     use_peer_pool='--no-peers' not in args)
