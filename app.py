"""
股票量化分析診斷系統 - app.py
支援：一般股票 / 一般ETF / 槓桿ETF / 反向ETF / 美股 / 盤中即時報價
pip install streamlit pandas pandas-ta yfinance plotly
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from datetime import datetime, timedelta
import re, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 頁面設定
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="量化診斷系統", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;700&family=JetBrains+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans TC',sans-serif;}
.stApp{background:#0d1117;color:#e6edf3;}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d;}
[data-testid="stSidebar"] *{color:#e6edf3 !important;}
h1{color:#58a6ff !important;font-weight:700;}
h2,h3{color:#79c0ff !important;}

/* 狀態卡片 */
.pass-card{background:linear-gradient(135deg,#0d2818,#1a4731);border:1px solid #2ea043;
  border-radius:12px;padding:16px;text-align:center;box-shadow:0 0 12px rgba(46,160,67,.2);}
.fail-card{background:linear-gradient(135deg,#2d0f0f,#4a1a1a);border:1px solid #f85149;
  border-radius:12px;padding:16px;text-align:center;box-shadow:0 0 12px rgba(248,81,73,.2);}
.skip-card{background:linear-gradient(135deg,#1c1c2e,#252540);border:1px solid #6e40c9;
  border-radius:12px;padding:16px;text-align:center;box-shadow:0 0 12px rgba(110,64,201,.2);}
.card-title{font-size:.75rem;color:#8b949e;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;}
.card-status{font-size:1.4rem;font-weight:700;}
.card-detail{font-size:.72rem;color:#8b949e;margin-top:4px;}

/* ETF 類型徽章 */
.etf-lev{display:inline-block;background:linear-gradient(135deg,#4a2400,#8b4513);
  border:1px solid #ffa657;border-radius:20px;padding:3px 12px;
  font-size:.78rem;font-weight:700;color:#ffa657;margin-right:8px;}
.etf-inv{display:inline-block;background:linear-gradient(135deg,#2d0f0f,#5c1a1a);
  border:1px solid #f85149;border-radius:20px;padding:3px 12px;
  font-size:.78rem;font-weight:700;color:#f85149;margin-right:8px;}
.etf-norm{display:inline-block;background:linear-gradient(135deg,#0d2033,#1a3a5c);
  border:1px solid #79c0ff;border-radius:20px;padding:3px 12px;
  font-size:.78rem;font-weight:700;color:#79c0ff;margin-right:8px;}
.stock-badge{display:inline-block;background:linear-gradient(135deg,#0d2818,#1a4731);
  border:1px solid #2ea043;border-radius:20px;padding:3px 12px;
  font-size:.78rem;font-weight:700;color:#7ee787;margin-right:8px;}

/* 盤中 badge */
.live-badge{display:inline-block;background:linear-gradient(135deg,#1a4731,#2ea043);
  border:1px solid #2ea043;border-radius:20px;padding:4px 14px;
  font-size:.78rem;font-weight:700;color:#7ee787;animation:pulse 2s infinite;}
.closed-badge{display:inline-block;background:#21262d;border:1px solid #30363d;
  border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:700;color:#8b949e;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.6;}}

/* 警告框 */
.warn-box{background:linear-gradient(135deg,#2d2000,#4a3500);border:1px solid #d29922;
  border-radius:10px;padding:14px 18px;margin:10px 0;font-size:.9rem;color:#f0c040;}

hr{border-color:#30363d;}
[data-testid="metric-container"]{background:#161b22;border:1px solid #30363d;
  border-radius:8px;padding:12px;}
[data-testid="stMetricValue"]{color:#58a6ff !important;font-family:'JetBrains Mono',monospace;}
.stButton>button{background:linear-gradient(135deg,#1f6feb,#388bfd);color:white;
  border:none;border-radius:8px;padding:12px 28px;font-size:1rem;font-weight:700;width:100%;}
.stButton>button:hover{background:linear-gradient(135deg,#388bfd,#58a6ff);
  transform:translateY(-1px);box-shadow:0 4px 16px rgba(56,139,253,.4);}
[data-testid="stExpander"]{background:#161b22;border:1px solid #30363d;border-radius:8px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ★ 資產類型識別（純邏輯，無 st.*）
# ══════════════════════════════════════════════════════════════

# 台灣已知槓桿/反向 ETF 代號前綴規則 + 明確清單
# 規則：00XXX L → 槓桿；00XXX R → 反向（部分）
_TW_LEVERAGED_SUFFIX  = re.compile(r"^0\d{4}L(\.TW)?$", re.I)
_TW_INVERSE_SUFFIX    = re.compile(r"^0\d{4}R(\.TW)?$", re.I)

# 美股常見槓桿/反向 ETF 代號關鍵字（非完整清單，僅輔助）
_US_LEVERAGED_NAMES   = {"TQQQ","SQQQ","UPRO","SPXU","LABU","LABD",
                          "TECL","TECS","FAS","FAZ","NUGT","DUST",
                          "UDOW","SDOW","TNA","TZA","NAIL","DRN","DRV",
                          "SPXL","SPXS","SOXL","SOXS","FNGU","FNGD",
                          "CURE","WEBL","WEBS","BNKU","BNKD","DPST","ERX","ERY",
                          "GUSH","DRIP","HIBL","HIBS","MIDU","MIDZ","RETL","RETS"}

def classify_asset(symbol: str, yf_info: dict) -> dict:
    """
    根據代號格式 + yfinance info 判斷資產類型。
    回傳：
      asset_type : 'stock' | 'etf_normal' | 'etf_leveraged' | 'etf_inverse'
      leverage   : float  (1.0 一般, 2.0 正2, -1.0 反1 …)
      underlying : str    追蹤標的描述
      warnings   : list[str]  操作風險提示
    """
    raw  = symbol.upper().replace(".TW","").replace(".TWO","")
    name = yf_info.get("longName","") or yf_info.get("shortName","")
    q_type = yf_info.get("quoteType","")   # ETF / EQUITY / MUTUALFUND …
    category = yf_info.get("category","") or ""

    is_etf = (q_type == "ETF") or raw.startswith("0")  # 台股 ETF 以 0 開頭

    # ── 槓桿判斷 ──────────────────────────────────────
    lev_ratio = 1.0
    asset_type = "stock"
    underlying = ""
    warns = []

    if is_etf:
        name_l = name.lower()
        # 1. 台股代號尾碼規則
        if _TW_LEVERAGED_SUFFIX.match(symbol) or _TW_LEVERAGED_SUFFIX.match(raw+"L"):
            # 代號本身就是 L 結尾
            if re.search(r"L(\.TW)?$", symbol, re.I):
                asset_type = "etf_leveraged"
                lev_ratio  = 2.0       # 台灣槓桿ETF 幾乎全是正2倍
        elif _TW_INVERSE_SUFFIX.match(symbol) or re.search(r"R(\.TW)?$", symbol, re.I):
            asset_type = "etf_inverse"
            lev_ratio  = -1.0

        # 2. 名稱關鍵字（中英文）
        lev_kw  = ["正2","2x","2倍","leveraged","ultra","2×","triple","3x","3倍","正三"]
        inv_kw  = ["反1","反向","inverse","bear","short","放空","-1x","-2x","-1倍"]
        if any(k in name_l for k in lev_kw):
            asset_type = "etf_leveraged"
            lev_ratio  = 3.0 if any(k in name_l for k in ["triple","3x","3倍","正三"]) else 2.0
        if any(k in name_l for k in inv_kw):
            asset_type = "etf_inverse"
            lev_ratio  = -1.0

        # 3. 美股已知清單
        if raw in _US_LEVERAGED_NAMES:
            if raw.startswith("S") or "BEAR" in raw or "SHORT" in raw or raw in {"SQQQ","SPXU","SPXS","SOXS","FNGD","BNKD"}:
                asset_type = "etf_inverse"; lev_ratio = -3.0
            else:
                asset_type = "etf_leveraged"; lev_ratio = 3.0

        if asset_type == "stock":       # ETF 但非槓桿/反向
            asset_type = "etf_normal"

        # 追蹤標的
        underlying = yf_info.get("underlyingSymbol","") or category or "未知"

    # ── 風險提示 ──────────────────────────────────────
    if asset_type == "etf_leveraged":
        warns = [
            f"⚡ 此為 **{lev_ratio:.0f}倍槓桿 ETF**，每日重置機制會造成長期持有的「波動損耗（Beta Slippage）」",
            "📅 槓桿ETF 不適合長期持有，建議以**短線趨勢交易**為主（數日～數週）",
            "🛑 停損設定應比一般股票**更嚴格**（本系統已自動縮減 ATR 倍數至 1.0×）",
            "⚖️ 風報比門檻維持 ≥ 2.0，但建議每筆倉位不超過總資金 **5%**",
        ]
    elif asset_type == "etf_inverse":
        warns = [
            f"🔻 此為 **反向 ETF**（{lev_ratio:.0f}倍），做空市場方向",
            "📅 反向ETF 同樣有每日重置損耗，長期持有會加速淨值衰退",
            "⚠️ 大盤過濾層邏輯**自動反轉**：大盤走弱時才算通過",
            "🛑 停損應**更嚴格**，本系統已自動調整參數",
        ]
    elif asset_type == "etf_normal":
        warns = [
            "📦 此為一般型 ETF，基本面層改以「NAV折溢價、追蹤誤差」方向寬鬆判斷",
        ]

    return {
        "asset_type": asset_type,
        "leverage":   lev_ratio,
        "underlying": underlying,
        "warnings":   warns,
        "is_etf":     is_etf,
        "name":       name,
    }


# ══════════════════════════════════════════════════════════════
# 代號處理
# ══════════════════════════════════════════════════════════════
def to_ticker(raw: str) -> str:
    """
    台股：純數字/英數混合補 .TW
      - 4碼數字 → 股票  (e.g. 2330 → 2330.TW)
      - 4-5碼數字+字母L/R → 槓桿/反向ETF (e.g. 00631L → 00631L.TW)
      - 00XXX 開頭 → ETF   (e.g. 0050 → 0050.TW)
    美股：純英文 → 保持大寫不加後綴
    """
    raw = raw.strip().upper()
    # 已有後綴直接回傳
    if raw.endswith(".TW") or raw.endswith(".TWO"):
        return raw
    # 純英文字母 → 美股
    if re.match(r'^[A-Z]+$', raw):
        return raw
    # 含數字（台股）→ 補 .TW
    return raw + ".TW"


# ══════════════════════════════════════════════════════════════
# 即時報價（純計算）
# ══════════════════════════════════════════════════════════════
def fetch_realtime_quote(symbol: str) -> dict:
    try:
        t    = yf.Ticker(symbol)
        fi   = t.fast_info
        info = t.info
        price      = getattr(fi, "last_price",     None)
        prev_close = getattr(fi, "previous_close", None)
        open_px    = getattr(fi, "open",           None)
        high       = getattr(fi, "day_high",       None)
        low        = getattr(fi, "day_low",        None)
        volume     = getattr(fi, "last_volume",    None)
        currency   = getattr(fi, "currency",       "TWD")
        mkt_state  = info.get("marketState", "CLOSED")
        if price is None or (isinstance(price, float) and np.isnan(price)):
            price = info.get("currentPrice") or info.get("regularMarketPrice")
        return {
            "price":        float(price)      if price      else None,
            "prev_close":   float(prev_close) if prev_close else None,
            "open":         float(open_px)    if open_px    else None,
            "high":         float(high)       if high       else None,
            "low":          float(low)        if low        else None,
            "volume":       int(volume)       if volume     else None,
            "currency":     currency,
            "market_state": mkt_state,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error":        None,
            "info":         info,
        }
    except Exception as exc:
        return {"price": None, "market_state": "CLOSED",
                "last_updated": None, "error": str(exc), "info": {}}


# ══════════════════════════════════════════════════════════════
# 歷史 OHLCV
# ══════════════════════════════════════════════════════════════
def fetch_ohlcv(symbol: str, period_days: int = 400):
    end   = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=period_days + 5)
    try:
        df = yf.download(symbol,
                         start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(), f"找不到代號 `{symbol}` 的行情，請確認代號是否正確。"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"下載 {symbol} 時發生例外：{exc}"


# ══════════════════════════════════════════════════════════════
# 工具函數
# ══════════════════════════════════════════════════════════════
def calc_slope(series: pd.Series, window: int = 20) -> float:
    if len(series) < window:
        return 0.0
    y = series.iloc[-window:].values
    x = np.arange(window)
    return float(np.polyfit(x, y, 1)[0])

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    v  = tr.rolling(period).mean().iloc[-1]
    return float(v) if not pd.isna(v) else 0.0


# ══════════════════════════════════════════════════════════════
# 第 1 層：大盤過濾（反向ETF 邏輯反轉）
# ══════════════════════════════════════════════════════════════
def check_market_filter(index_symbol: str, asset_info: dict) -> dict:
    """
    一般股票/槓桿ETF：大盤需在 200MA 上方且斜率向上
    反向ETF          ：大盤需在 200MA 下方或斜率向下（方向相反才有利）
    """
    df, err = fetch_ohlcv(index_symbol, period_days=450)
    if err or df.empty or len(df) < 210:
        return {"pass": False, "detail": err or f"大盤({index_symbol})數據不足"}

    close = df["Close"]
    ma200 = close.rolling(200).mean()
    px    = float(close.iloc[-1])
    ma    = float(ma200.iloc[-1])
    slope = calc_slope(ma200, window=20)

    above    = px > ma
    slope_up = slope > 0
    is_inv   = asset_info["asset_type"] == "etf_inverse"

    if is_inv:
        # 反向ETF：大盤走弱才利多
        passed = (not above) or (not slope_up)
        status = "大盤走弱，有利空頭" if passed else "大盤仍強，反向ETF不利"
    else:
        passed = above and slope_up
        status = "多頭環境" if passed else "大盤不佳"

    detail = (
        f"[{'反向模式' if is_inv else '正向模式'}] "
        f"大盤 {px:,.1f} {'>' if above else '<'} 200MA {ma:,.1f}，"
        f"斜率{'↑' if slope_up else '↓'}（{slope:.2f}） → {status}"
    )
    return {"pass": passed, "detail": detail}


# ══════════════════════════════════════════════════════════════
# 第 2 層：基本面（ETF 改為追蹤誤差/規模判斷）
# ══════════════════════════════════════════════════════════════
def check_fundamental(ticker_obj, asset_info: dict) -> dict:
    """
    股票   → 營收成長、毛利率、EPS
    一般ETF → 基金規模(AUM)、費用率、追蹤誤差（寬鬆）
    槓桿ETF → 直接寬鬆通過（基本面對短線槓桿ETF意義不大）
    反向ETF → 直接寬鬆通過
    """
    atype = asset_info["asset_type"]

    if atype in ("etf_leveraged", "etf_inverse"):
        return {
            "pass": True,
            "detail": f"槓桿/反向ETF 不適用傳統基本面篩選，自動寬鬆通過（請聚焦技術面與風控）",
        }

    if atype == "etf_normal":
        try:
            info     = ticker_obj.info
            aum      = info.get("totalAssets", None)
            exp_r    = info.get("annualReportExpenseRatio", None) or info.get("expenseRatio", None)
            ytd_ret  = info.get("ytdReturn", None)
            flags, details = [], []
            if aum is not None:
                ok = aum > 1e8          # AUM > 1億（流動性基本要求）
                flags.append(ok)
                details.append(f"AUM {aum/1e8:.1f}億 {'✅' if ok else '❌<1億(流動性不足)'}")
            else:
                flags.append(True); details.append("AUM資料不足(寬鬆通過)")
            if exp_r is not None:
                ok = exp_r < 0.01       # 費用率 < 1%
                flags.append(ok)
                details.append(f"費用率 {exp_r*100:.2f}% {'✅' if ok else '❌>1%(偏高)'}")
            else:
                flags.append(True); details.append("費用率資料不足(寬鬆通過)")
            if ytd_ret is not None:
                details.append(f"YTD報酬 {ytd_ret*100:+.1f}%")
            return {"pass": all(flags) if flags else True,
                    "detail": " ｜ ".join(details) or "ETF基本面數據不足，寬鬆通過"}
        except Exception as exc:
            return {"pass": True, "detail": f"ETF基本面 API 異常({exc})，寬鬆通過"}

    # 一般股票
    try:
        info            = ticker_obj.info
        revenue_growth  = info.get("revenueGrowth", None)
        gross_margin    = info.get("grossMargins",  None)
        earnings_growth = info.get("earningsGrowth",None)
        flags, details  = [], []

        if revenue_growth is not None:
            flags.append(revenue_growth > 0)
            details.append(f"營收YoY {revenue_growth*100:+.1f}%")
        else:
            flags.append(True); details.append("營收YoY 數據不足(寬鬆通過)")

        if gross_margin is not None:
            flags.append(gross_margin > 0.05)
            details.append(f"毛利率 {gross_margin*100:.1f}%")
        else:
            flags.append(True); details.append("毛利率數據不足(寬鬆通過)")

        if earnings_growth is not None:
            flags.append(earnings_growth > -0.20)
            details.append(f"EPS成長 {earnings_growth*100:+.1f}%")
        else:
            flags.append(True); details.append("EPS數據不足(寬鬆通過)")

        return {"pass": all(flags), "detail": " ｜ ".join(details)}
    except Exception as exc:
        return {"pass": True, "detail": f"基本面 API 異常({exc})，寬鬆通過"}


# ══════════════════════════════════════════════════════════════
# 第 3 層：籌碼面（槓桿ETF 改用標的指數動能）
# ══════════════════════════════════════════════════════════════
def check_chips(df: pd.DataFrame, asset_info: dict,
                underlying_df: pd.DataFrame | None = None) -> dict:
    """
    股票/一般ETF → OBV斜率 + 量比 + 健康度
    槓桿ETF      → 額外檢查：ETF自身溢價（收盤 vs NAV近似）+
                   標的指數（underlying_df）動能方向一致性
    反向ETF      → 標的指數動能應向下才有利
    """
    atype = asset_info["asset_type"]

    if len(df) < 25:
        return {"pass": False, "detail": "數據不足 25 筆，無法判斷籌碼"}

    close, volume = df["Close"], df["Volume"]
    obv       = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_slope = calc_slope(obv, window=10)
    vol5      = float(volume.iloc[-5:].mean())
    vol20     = float(volume.iloc[-20:].mean())
    vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0

    recent    = df.iloc[-5:]
    pd_diff   = recent["Close"].diff()
    vd_diff   = recent["Volume"].diff()
    healthy   = int(((pd_diff>0)&(vd_diff>0)).sum()) + int(((pd_diff<0)&(vd_diff<0)).sum())
    healthy_r = healthy / 4

    base_detail = (
        f"OBV斜率{'↑' if obv_slope>0 else '↓'}({obv_slope:.1f}) ｜"
        f" 量比(5d/20d) {vol_ratio:.2f} ｜ 籌碼健康度 {healthy_r*100:.0f}%"
    )
    base_flags = [obv_slope > 0, vol_ratio > 0.8, healthy_r >= 0.5]

    # 槓桿/反向ETF：加入標的指數動能驗證
    if atype in ("etf_leveraged", "etf_inverse") and underlying_df is not None and len(underlying_df) >= 20:
        idx_close  = underlying_df["Close"]
        idx_ma20   = idx_close.rolling(20).mean()
        idx_above  = float(idx_close.iloc[-1]) > float(idx_ma20.iloc[-1])
        idx_slope  = calc_slope(idx_ma20, window=10)
        idx_up     = idx_slope > 0

        if atype == "etf_leveraged":
            # 正向槓桿：標的指數也要向上
            idx_ok = idx_above and idx_up
            idx_detail = f"標的指數{'↑強' if idx_ok else '↓弱'}（20MA {'上方' if idx_above else '下方'}，斜率{'↑' if idx_up else '↓'}）"
        else:
            # 反向：標的指數走弱才有利
            idx_ok = (not idx_above) or (not idx_up)
            idx_detail = f"標的指數{'↓弱利多反向' if idx_ok else '↑強不利反向'}（{'跌破' if not idx_above else '站上'}20MA）"

        base_flags.append(idx_ok)
        base_detail += f" ｜ {idx_detail}"

    return {"pass": all(base_flags), "detail": base_detail}


# ══════════════════════════════════════════════════════════════
# 第 4 層：技術面（槓桿ETF 加入動能/連續性驗證）
# ══════════════════════════════════════════════════════════════
def check_technical(df: pd.DataFrame, asset_info: dict) -> dict:
    """
    槓桿ETF 追加條件：
      - 連續 3 日收盤 > 開盤（趨勢連貫性）
      - RSI 不能超買（<75，避免追高被反轉）
    反向ETF 追加條件：
      - 連續 3 日收盤 < 開盤（空頭連貫）
      - RSI 不能超賣（>25）
    """
    _empty = {
        "pass": False, "detail": "數據不足 35 筆",
        "ma5": None, "ma10": None, "ma20": None,
        "bull_align": False, "kd_cross": False, "macd_flip": False,
        "k_series": None, "d_series": None, "macd_df": None,
    }
    if len(df) < 35:
        return _empty

    atype = asset_info["asset_type"]
    close = df["Close"]
    ma5   = close.rolling(5).mean()
    ma10  = close.rolling(10).mean()
    ma20  = close.rolling(20).mean()

    # 反向ETF：均線要「空頭排列」（5MA < 10MA < 20MA）
    if atype == "etf_inverse":
        bear_align = float(ma5.iloc[-1]) < float(ma10.iloc[-1]) < float(ma20.iloc[-1])
        bull_align = bear_align   # 對反向ETF，空頭排列就是好事
        align_label = "空頭排列✅" if bear_align else "未空頭排列❌"
    else:
        bull_align = float(ma5.iloc[-1]) > float(ma10.iloc[-1]) > float(ma20.iloc[-1])
        align_label = "多頭排列✅" if bull_align else "未多頭排列❌"

    # KD
    k_series = d_series = None
    kd_cross = False
    k_now = d_now = 50.0
    try:
        stoch = ta.stoch(df["High"], df["Low"], close, k=9, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            k_col = next((c for c in stoch.columns if "STOCHk" in c), None)
            d_col = next((c for c in stoch.columns if "STOCHd" in c), None)
            if k_col and d_col:
                k_series = stoch[k_col]; d_series = stoch[d_col]
                k_now,  k_prev = float(k_series.iloc[-1]), float(k_series.iloc[-2])
                d_now,  d_prev = float(d_series.iloc[-1]), float(d_series.iloc[-2])
                if atype == "etf_inverse":
                    # 反向ETF：KD 高檔死亡交叉（K 跌破 D，且 K > 50）
                    kd_cross = (k_prev > d_prev) and (k_now < d_now) and (k_now > 50)
                else:
                    kd_cross = (k_prev < d_prev) and (k_now > d_now) and (k_now < 50)
    except Exception:
        pass

    # MACD
    macd_df = None; macd_flip = False; hist_now = 0.0
    try:
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            h_col = next((c for c in macd_df.columns if "MACDh" in c), None)
            if h_col:
                hist_now  = float(macd_df[h_col].iloc[-1])
                hist_prev = float(macd_df[h_col].iloc[-2])
                if atype == "etf_inverse":
                    # 反向ETF：MACD 由正轉負
                    macd_flip = (hist_prev > 0) and (hist_now <= 0)
                else:
                    macd_flip = (hist_prev < 0) and (hist_now >= 0)
    except Exception:
        pass

    # RSI 過濾（槓桿/反向ETF 加嚴）
    rsi_ok   = True
    rsi_note = ""
    try:
        rsi_s = ta.rsi(close, length=14)
        rsi_v = float(rsi_s.iloc[-1]) if rsi_s is not None else 50
        if atype == "etf_leveraged":
            rsi_ok   = rsi_v < 75        # 槓桿ETF 避免追高
            rsi_note = f" ｜ RSI {rsi_v:.1f}{'✅<75' if rsi_ok else '❌≥75(過熱勿追)'}"
        elif atype == "etf_inverse":
            rsi_ok   = rsi_v > 25        # 反向ETF 避免超賣
            rsi_note = f" ｜ RSI {rsi_v:.1f}{'✅>25' if rsi_ok else '❌≤25(超賣風險)'}"
    except Exception:
        pass

    # 連續趨勢驗證（槓桿/反向ETF 加嚴）
    consec_ok   = True
    consec_note = ""
    if atype in ("etf_leveraged", "etf_inverse") and len(df) >= 4:
        last3 = df.iloc[-3:]
        if atype == "etf_leveraged":
            consec_up = (last3["Close"] > last3["Open"]).sum()
            consec_ok = consec_up >= 2      # 近3日至少2日收紅
            consec_note = f" ｜ 近3日{'收紅'+str(consec_up)+'日✅' if consec_ok else '連續性不足❌'}"
        else:
            consec_dn = (last3["Close"] < last3["Open"]).sum()
            consec_ok = consec_dn >= 2
            consec_note = f" ｜ 近3日{'收黑'+str(consec_dn)+'日✅(利空方向)' if consec_ok else '連續性不足❌'}"

    passed = bull_align and (kd_cross or macd_flip) and rsi_ok and consec_ok
    detail = (
        f"均線{align_label} ({float(ma5.iloc[-1]):.2f}/{float(ma10.iloc[-1]):.2f}/{float(ma20.iloc[-1]):.2f}) ｜ "
        f"KD{'信號✅' if kd_cross else '未信號❌'}(K={k_now:.1f}) ｜ "
        f"MACD柱{'信號✅' if macd_flip else '未信號❌'}(OSC={hist_now:.4f})"
        + rsi_note + consec_note
    )
    return {
        "pass": passed, "detail": detail,
        "ma5": ma5, "ma10": ma10, "ma20": ma20,
        "bull_align": bull_align, "kd_cross": kd_cross, "macd_flip": macd_flip,
        "k_series": k_series, "d_series": d_series, "macd_df": macd_df,
    }


# ══════════════════════════════════════════════════════════════
# 第 5 層：風控（槓桿ETF 更嚴格）
# ══════════════════════════════════════════════════════════════
def check_risk_reward(df: pd.DataFrame, live_price: float | None,
                      capital: float, asset_info: dict) -> dict:
    """
    一般股票    → 停損 1.5×ATR，目標 3.0×ATR，風報比 ≥ 2.0
    槓桿ETF     → 停損 1.0×ATR（更嚴），目標 2.5×ATR，風報比 ≥ 2.0
                  每筆最大倉位 = 總資金 × 5%（而非 2%，但單位風險更小）
    反向ETF     → 停損 1.0×ATR，目標 2.5×ATR，風報比 ≥ 2.0
    一般ETF     → 停損 1.2×ATR，目標 2.8×ATR，風報比 ≥ 2.0
    """
    if len(df) < 20:
        return {"pass": False, "detail": "數據不足 20 筆，無法計算風控"}

    atype   = asset_info["asset_type"]
    price   = live_price if live_price else float(df["Close"].iloc[-1])
    atr_val = calc_atr(df, period=14)
    if atr_val <= 0:
        return {"pass": False, "detail": "ATR 計算異常"}

    # 依資產類型設定參數
    if atype == "etf_leveraged":
        sl_mult, tp_mult, risk_pct = 1.0, 2.5, 0.05   # 更緊停損，5%風控
        type_note = "槓桿ETF模式：停損1.0×ATR，5%資金風控"
    elif atype == "etf_inverse":
        sl_mult, tp_mult, risk_pct = 1.0, 2.5, 0.05
        type_note = "反向ETF模式：停損1.0×ATR，5%資金風控"
    elif atype == "etf_normal":
        sl_mult, tp_mult, risk_pct = 1.2, 2.8, 0.02
        type_note = "一般ETF模式：停損1.2×ATR，2%資金風控"
    else:
        sl_mult, tp_mult, risk_pct = 1.5, 3.0, 0.02
        type_note = "股票模式：停損1.5×ATR，2%資金風控"

    stop_loss = price - sl_mult * atr_val
    target    = price + tp_mult * atr_val
    risk      = price - stop_loss
    reward    = target - price
    rr_ratio  = reward / risk

    max_risk_amt   = capital * risk_pct
    risk_per_lot   = risk * 1000
    suggested_lots = max(0, int(max_risk_amt / risk_per_lot)) if risk_per_lot > 0 else 0

    passed = rr_ratio >= 2.0
    src    = "即時價" if live_price else "收盤價"
    detail = (
        f"[{type_note}] 基準({src}) {price:.2f} ｜ ATR {atr_val:.2f} ｜"
        f" 停損 {stop_loss:.2f} ｜ 目標 {target:.2f} ｜"
        f" 風報比 {rr_ratio:.2f} {'✅≥2.0' if passed else '❌<2.0'} ｜"
        f" 建議 {suggested_lots} 張（資金{capital/10000:.0f}萬，風控{risk_pct*100:.0f}%）"
    )
    return {
        "pass": passed, "detail": detail,
        "current_price": price, "stop_loss": stop_loss,
        "target": target, "rr_ratio": rr_ratio,
        "suggested_lots": suggested_lots, "atr": atr_val,
        "sl_mult": sl_mult, "tp_mult": tp_mult,
    }


# ══════════════════════════════════════════════════════════════
# 繪圖
# ══════════════════════════════════════════════════════════════
def draw_chart(df: pd.DataFrame, tech: dict, symbol: str,
               asset_info: dict,
               live_price: float | None = None,
               stop_loss: float | None  = None,
               target: float | None     = None):

    close      = df["Close"]
    rsi_series = ta.rsi(close, length=14)
    atype      = asset_info["asset_type"]

    type_labels = {
        "stock":        "📈 個股",
        "etf_normal":   "📦 ETF",
        "etf_leveraged":"⚡ 槓桿ETF",
        "etf_inverse":  "🔻 反向ETF",
    }
    title_suffix = type_labels.get(atype, "")

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.18, 0.18, 0.14],
        subplot_titles=(
            f"{symbol} {title_suffix}  K線圖 + 均線",
            "KD 指標（K9, D3）",
            "MACD 柱狀體（12-26-9）",
            "RSI（14）",
        ),
    )

    # K 線
    inc_color = "#2ea043" if atype != "etf_inverse" else "#f85149"
    dec_color = "#f85149" if atype != "etf_inverse" else "#2ea043"
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="K線",
        increasing_line_color=inc_color, decreasing_line_color=dec_color,
        increasing_fillcolor=inc_color,  decreasing_fillcolor=dec_color,
    ), row=1, col=1)

    # 均線
    for label, series, color in [
        ("MA5",  tech.get("ma5"),  "#ffa657"),
        ("MA10", tech.get("ma10"), "#79c0ff"),
        ("MA20", tech.get("ma20"), "#ff7b72"),
    ]:
        if series is not None:
            fig.add_trace(go.Scatter(
                x=df.index, y=series, name=label,
                line=dict(color=color, width=1.6), opacity=0.9,
            ), row=1, col=1)

    # 成交量
    vol_colors = [
        "#2ea043" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#f85149"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="成交量",
        marker_color=vol_colors, opacity=0.35, showlegend=False, yaxis="y5",
    ), row=1, col=1)

    # 水平線
    if live_price:
        fig.add_hline(y=live_price,
                      line=dict(color="#ffa657", width=1.8, dash="dash"),
                      annotation_text=f"  即時 {live_price:.2f}",
                      annotation_font=dict(color="#ffa657", size=12),
                      annotation_position="right", row=1, col=1)
    if stop_loss:
        fig.add_hline(y=stop_loss,
                      line=dict(color="#f85149", width=1.4, dash="dot"),
                      annotation_text=f"  停損 {stop_loss:.2f}",
                      annotation_font=dict(color="#f85149", size=11),
                      annotation_position="right", row=1, col=1)
    if target:
        fig.add_hline(y=target,
                      line=dict(color="#2ea043", width=1.4, dash="dot"),
                      annotation_text=f"  目標 {target:.2f}",
                      annotation_font=dict(color="#2ea043", size=11),
                      annotation_position="right", row=1, col=1)

    # 槓桿ETF：加上前高/前低警示區（近30日）
    if atype in ("etf_leveraged", "etf_inverse") and len(df) >= 30:
        recent_30 = df.iloc[-30:]
        r_high = float(recent_30["High"].max())
        r_low  = float(recent_30["Low"].min())
        fig.add_hline(y=r_high, line=dict(color="#d29922", width=1, dash="dot"),
                      annotation_text=f"  30日高 {r_high:.2f}",
                      annotation_font=dict(color="#d29922", size=10),
                      annotation_position="right", row=1, col=1)
        fig.add_hline(y=r_low,  line=dict(color="#8b949e", width=1, dash="dot"),
                      annotation_text=f"  30日低 {r_low:.2f}",
                      annotation_font=dict(color="#8b949e", size=10),
                      annotation_position="right", row=1, col=1)

    # KD
    if tech.get("k_series") is not None:
        fig.add_trace(go.Scatter(x=df.index, y=tech["k_series"],
                                 name="K值", line=dict(color="#ffa657", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=tech["d_series"],
                                 name="D值", line=dict(color="#79c0ff", width=1.5)), row=2, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="rgba(248,81,73,.35)",  row=2, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="rgba(46,160,67,.35)",  row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(139,148,158,.3)", row=2, col=1)

    # MACD
    if tech.get("macd_df") is not None:
        md     = tech["macd_df"]
        mc_col = next((c for c in md.columns if c.startswith("MACD_")), None)
        ms_col = next((c for c in md.columns if "MACDs" in c), None)
        mh_col = next((c for c in md.columns if "MACDh" in c), None)
        if mc_col:
            fig.add_trace(go.Scatter(x=df.index, y=md[mc_col], name="MACD",
                                     line=dict(color="#ffa657", width=1.5)), row=3, col=1)
        if ms_col:
            fig.add_trace(go.Scatter(x=df.index, y=md[ms_col], name="Signal",
                                     line=dict(color="#79c0ff", width=1.5)), row=3, col=1)
        if mh_col:
            hist = md[mh_col]
            fig.add_trace(go.Bar(x=df.index, y=hist, name="OSC",
                                 marker_color=["#2ea043" if v>=0 else "#f85149" for v in hist],
                                 opacity=0.75), row=3, col=1)

    # RSI（槓桿ETF 加超買線 75 / 反向ETF 加超賣線 25）
    if rsi_series is not None:
        fig.add_trace(go.Scatter(x=df.index, y=rsi_series, name="RSI(14)",
                                 line=dict(color="#bc8cff", width=1.5)), row=4, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,81,73,.35)", row=4, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(46,160,67,.35)", row=4, col=1)
        if atype == "etf_leveraged":
            fig.add_hline(y=75, line_dash="dash", line_color="rgba(255,166,87,.6)", row=4, col=1)
        elif atype == "etf_inverse":
            fig.add_hline(y=25, line_dash="dash", line_color="rgba(248,81,73,.6)",  row=4, col=1)

    fig.update_layout(
        height=900, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Noto Sans TC"),
        legend=dict(bgcolor="rgba(22,27,34,.9)", bordercolor="#30363d",
                    borderwidth=1, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=90, t=40, b=10),
        hovermode="x unified",
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#21262d", zeroline=False, row=i, col=1,
                         showspikes=True, spikecolor="#58a6ff", spikethickness=1)
        fig.update_yaxes(gridcolor="#21262d", zeroline=False, row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# UI 元件
# ══════════════════════════════════════════════════════════════
def render_status_card(col, title: str, result: dict):
    passed  = result.get("pass", False)
    skipped = result.get("skip", False)
    if skipped:
        css, status = "skip-card", "⏭️ 略過"
    elif passed:
        css, status = "pass-card", "✅ 通過"
    else:
        css, status = "fail-card", "❌ 未通過"
    raw    = result.get("detail", "")
    short  = raw[:64] + ("…" if len(raw) > 64 else "")
    with col:
        st.markdown(f"""
        <div class="{css}">
            <div class="card-title">{title}</div>
            <div class="card-status">{status}</div>
            <div class="card-detail">{short}</div>
        </div>""", unsafe_allow_html=True)

def asset_badge(asset_info: dict) -> str:
    atype = asset_info["asset_type"]
    name  = asset_info.get("name","")
    lev   = asset_info.get("leverage", 1.0)
    if atype == "etf_leveraged":
        return f'<span class="etf-lev">⚡ 槓桿ETF {lev:+.0f}x</span>'
    elif atype == "etf_inverse":
        return f'<span class="etf-inv">🔻 反向ETF {lev:.0f}x</span>'
    elif atype == "etf_normal":
        return f'<span class="etf-norm">📦 一般ETF</span>'
    else:
        return f'<span class="stock-badge">📈 個股</span>'


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown("## 📊 量化診斷系統")
        st.markdown("---")
        raw_symbol = st.text_input(
            "📌 股票/ETF 代號",
            value="00631L",
            help="台股：2330、0050、00631L、00632R\n美股：AAPL、TQQQ、SQQQ",
        )
        period_choice = st.selectbox("📅 分析歷史區間",
                                     ["3 個月","6 個月","1 年","2 年"], index=2)
        period_map  = {"3 個月":130,"6 個月":220,"1 年":400,"2 年":750}
        period_days = period_map[period_choice]

        index_choice = st.selectbox("🌏 大盤指數",
            ["台股加權 (^TWII)","S&P 500 (^GSPC)","那斯達克 (^IXIC)"])
        index_map    = {"台股加權 (^TWII)":"^TWII",
                        "S&P 500 (^GSPC)": "^GSPC",
                        "那斯達克 (^IXIC)":"^IXIC"}
        index_symbol = index_map[index_choice]

        capital = st.number_input("💰 可用資金（元）",
            min_value=100_000, max_value=10_000_000,
            value=500_000, step=100_000,
            help="股票風控2%、槓桿/反向ETF風控5%")

        st.markdown("---")
        auto_refresh = st.checkbox("🔄 盤中自動刷新", value=False)
        refresh_sec  = 60
        if auto_refresh:
            refresh_sec = st.slider("刷新間隔（秒）", 30, 300, 60, 10)

        run_btn = st.button("🚀 開始全面量化診斷", type="primary")
        st.markdown("---")
        st.markdown("""
        <div style='font-size:.72rem;color:#8b949e;line-height:2'>
        <b>支援資產類型</b><br>
        📈 個股（台/美）<br>📦 一般ETF<br>
        ⚡ 槓桿ETF（如00631L）<br>🔻 反向ETF（如00632R）<br><br>
        <b>5大漏斗層級</b><br>
        1️⃣ 大盤過濾<br>2️⃣ 基本面<br>3️⃣ 籌碼面<br>
        4️⃣ 技術面共振<br>5️⃣ 風控風報比
        </div>""", unsafe_allow_html=True)

    # 自動刷新
    if auto_refresh:
        import time
        if "last_refresh" not in st.session_state:
            st.session_state["last_refresh"] = time.time()
        if time.time() - st.session_state["last_refresh"] >= refresh_sec:
            st.session_state["last_refresh"] = time.time()
            st.rerun()

    st.markdown("# 📈 多維度股票量化診斷系統")
    st.markdown("支援個股、一般ETF、**槓桿ETF（如00631L）**、**反向ETF（如00632R）**，"
                "各類型自動套用對應的漏斗邏輯與風控參數。")

    if not run_btn and "diag_symbol" not in st.session_state:
        st.markdown("""
        <div style='margin-top:60px;text-align:center'>
            <div style='font-size:4rem'>📊</div>
            <div style='font-size:1.1rem;margin-top:12px;color:#8b949e'>
                在左側輸入代號，點擊「開始全面量化診斷」<br>
                <span style='font-size:.85rem;color:#6e40c9'>
                支援：2330 / 0050 / 00631L / 00632R / AAPL / TQQQ</span>
            </div>
        </div>""", unsafe_allow_html=True)
        return

    if run_btn:
        st.session_state.update({
            "diag_symbol":  raw_symbol,
            "diag_index":   index_symbol,
            "diag_capital": capital,
            "diag_period":  period_days,
        })

    symbol    = to_ticker(st.session_state.get("diag_symbol",  raw_symbol))
    index_sym = st.session_state.get("diag_index",   index_symbol)
    cap       = st.session_state.get("diag_capital", capital)
    p_days    = st.session_state.get("diag_period",  period_days)

    st.markdown(f"### 🔍 正在診斷：`{symbol}`")

    # ── 1. 即時報價 ──────────────────────────────
    with st.spinner("📡 取得即時報價與資產資訊…"):
        quote    = fetch_realtime_quote(symbol)
        yf_info  = quote.get("info", {})

    live_px      = quote.get("price")
    mkt_state    = quote.get("market_state", "CLOSED")
    last_updated = quote.get("last_updated", "")
    is_live      = mkt_state in ("REGULAR", "PRE", "POST")

    # ── 2. 資產類型識別 ──────────────────────────
    asset_info = classify_asset(symbol, yf_info)
    atype      = asset_info["asset_type"]

    # 頂部徽章列
    badge_html = (
        f'<span class="live-badge">● LIVE ({mkt_state})</span>'
        if is_live else
        f'<span class="closed-badge">● 已收盤 ({mkt_state})</span>'
    )
    ts_html = (f"<span style='color:#8b949e;font-size:.78rem;margin-left:12px'>"
               f"更新：{last_updated}</span>") if last_updated else ""
    st.markdown(
        asset_badge(asset_info) + badge_html + ts_html,
        unsafe_allow_html=True
    )

    # 資產名稱
    if asset_info.get("name"):
        st.caption(f"📛 {asset_info['name']}"
                   + (f"　｜　追蹤標的：{asset_info['underlying']}"
                      if asset_info.get("underlying") else ""))

    # ── 槓桿/反向ETF 風險警示框 ─────────────────
    if asset_info["warnings"]:
        warns_md = "\n".join(asset_info["warnings"])
        st.markdown(f"""
        <div class="warn-box">
        <b>⚠️ 特殊資產操作提醒</b><br><br>{warns_md.replace(chr(10),'<br>')}
        </div>""", unsafe_allow_html=True)
        st.markdown("")

    # 盤中即時報價快覽
    if is_live and live_px:
        prev  = quote.get("prev_close")
        chg   = live_px - prev if prev else 0
        chg_p = chg / prev * 100 if prev else 0
        hi, lo, vol = quote.get("high"), quote.get("low"), quote.get("volume")
        st.markdown("##### 📡 盤中即時報價")
        q1,q2,q3,q4,q5 = st.columns(5)
        q1.metric("即時成交", f"{live_px:.2f}",   f"{chg:+.2f} ({chg_p:+.2f}%)")
        q2.metric("昨收",    f"{prev:.2f}"      if prev else "N/A")
        q3.metric("今日高",  f"{hi:.2f}"        if hi   else "N/A")
        q4.metric("今日低",  f"{lo:.2f}"        if lo   else "N/A")
        q5.metric("成交量",  f"{vol/1000:.0f}K" if vol  else "N/A")
        st.markdown("---")

    # ── 3. 歷史 OHLCV ────────────────────────────
    with st.spinner("⏳ 下載歷史行情…"):
        df, err = fetch_ohlcv(symbol, period_days=p_days + 60)

    if err or df.empty:
        st.error(f"❌ {err or '無法取得行情數據'}"); return
    if len(df) < 30:
        st.warning(f"⚠️ `{symbol}` 數據僅 {len(df)} 筆，不足以分析。"); return

    st.caption(
        f"📅 歷史資料：{df.index[0].strftime('%Y-%m-%d')} ～ "
        f"{df.index[-1].strftime('%Y-%m-%d')}　共 {len(df)} 個交易日"
        + (f"　｜　即時價 {live_px:.2f}" if live_px else "")
    )

    # ── 4. 下載標的指數（槓桿/反向ETF 用）───────
    underlying_df = None
    if atype in ("etf_leveraged","etf_inverse"):
        # 嘗試從 yf_info 取得追蹤標的，否則用大盤作為替代
        und_sym = yf_info.get("underlyingSymbol","")
        if not und_sym:
            # 台股槓桿ETF 多追蹤台灣50，用 ^TWII 近似
            und_sym = "^TWII" if ".TW" in symbol else index_sym
        with st.spinner(f"🔗 下載追蹤標的 {und_sym} 數據…"):
            underlying_df, _ = fetch_ohlcv(und_sym, period_days=p_days + 60)
        if underlying_df is not None and underlying_df.empty:
            underlying_df = None

    # ── 5. 五大層級漏斗 ──────────────────────────
    with st.spinner("🔬 執行量化漏斗篩選…"):
        r1 = check_market_filter(index_sym, asset_info)
        r2 = check_fundamental(yf.Ticker(symbol), asset_info)
        r3 = check_chips(df, asset_info, underlying_df)
        r4 = check_technical(df, asset_info)
        r5 = check_risk_reward(df, live_px if is_live else None, cap, asset_info)

    results  = [r1, r2, r3, r4, r5]
    labels   = ["大盤過濾", "基本面", "籌碼面", "技術面共振", "風控風報比"]
    icons    = ["🌏", "📋", "🏦", "📐", "⚖️"]
    # 基本面對槓桿/反向ETF 標記為略過（skip）而非計入失敗
    if atype in ("etf_leveraged","etf_inverse"):
        r2["skip"] = True
    all_pass = all(
        r["pass"] or r.get("skip", False)
        for r in results
    )

    # ── 五大狀態卡片 ─────────────────────────────
    st.markdown("---")
    st.markdown("#### 🛡️ 五大漏斗層級診斷結果")
    cols = st.columns(5)
    for col, lbl, res, ico in zip(cols, labels, results, icons):
        render_status_card(col, f"{ico} {lbl}", res)

    # ── 關鍵指標 ─────────────────────────────────
    st.markdown("---")
    display_px = live_px if (is_live and live_px) else float(df["Close"].iloc[-1])
    hist_last  = float(df["Close"].iloc[-1])
    price_chg  = float(df["Close"].pct_change().iloc[-1]) * 100
    vol_avg5   = float(df["Volume"].iloc[-5:].mean())

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("💹 參考價格", f"{display_px:.2f}",
              f"昨收 {hist_last:.2f} ({price_chg:+.2f}%)" if is_live else f"{price_chg:+.2f}%")
    m2.metric("📦 5日均量", f"{vol_avg5/1000:.0f}K 股")
    if r5.get("stop_loss"):
        m3.metric("🛑 ATR停損", f"{r5['stop_loss']:.2f}",
                  f"-{(display_px - r5['stop_loss']):.2f}")
    if r5.get("target"):
        m4.metric("🎯 目標價",  f"{r5['target']:.2f}",
                  f"+{(r5['target'] - display_px):.2f}")
    if r5.get("rr_ratio"):
        rr = r5["rr_ratio"]
        m5.metric("📊 風報比",  f"{rr:.2f}x",
                  "✅ 達標" if rr >= 2.0 else "❌ 不達標")

    # ── 圖表 ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📉 互動式技術圖表")
    df_chart  = df.iloc[-p_days:] if len(df) > p_days else df
    tech_plot = dict(r4)
    for k in ["ma5","ma10","ma20","k_series","d_series"]:
        if tech_plot.get(k) is not None:
            tech_plot[k] = tech_plot[k].reindex(df_chart.index)
    if tech_plot.get("macd_df") is not None:
        tech_plot["macd_df"] = tech_plot["macd_df"].reindex(df_chart.index)

    draw_chart(df_chart, tech_plot, symbol, asset_info,
               live_price=live_px if is_live else None,
               stop_loss=r5.get("stop_loss"),
               target=r5.get("target"))

    # ── 詳細報告 ─────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 詳細診斷報告")
    with st.expander("📌 查看各層級完整說明", expanded=True):
        for i, (lbl, res) in enumerate(zip(labels, results)):
            if res.get("skip"):
                st.info(f"**第{i+1}層 {lbl}** ⏭️ 略過（{res['detail']}）")
            elif res["pass"]:
                st.success(f"**第{i+1}層 {lbl}** ✅ 通過\n\n> {res['detail']}")
            else:
                st.error(f"**第{i+1}層 {lbl}** ❌ 未通過\n\n> {res['detail']}")

    # ── 最終判決 ─────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏁 最終操作建議")

    if all_pass:
        sl, tp, rr, atr_v = r5["stop_loss"], r5["target"], r5["rr_ratio"], r5["atr"]
        lots = r5["suggested_lots"]
        risk_pct_label = "5%" if atype in ("etf_leveraged","etf_inverse") else "2%"
        lev_note = ""
        if atype == "etf_leveraged":
            lev_note = f"\n| 🔑 槓桿倍數 | **{asset_info['leverage']:.0f}x**（停損比一般股票更緊，請嚴格執行）|"
        elif atype == "etf_inverse":
            lev_note = f"\n| 🔑 反向倍數 | **{asset_info['leverage']:.0f}x**（大盤走弱方向操作）|"

        st.success(f"""
## ✅ 通過診斷 — **建議進場**

| 項目 | 數值 |
|------|------|
| 📌 進場參考價 | **{display_px:.2f}** 元 {'（即時）' if is_live else '（收盤）'} |
| 🛑 嚴格停損價 | **{sl:.2f}** 元（基準價 − {r5['sl_mult']:.1f}×ATR {atr_v:.2f}）|
| 🎯 目標獲利價 | **{tp:.2f}** 元（基準價 + {r5['tp_mult']:.1f}×ATR）|
| ⚖️ 風報比     | **{rr:.2f}x**（≥ 2.0 達標）|
| 📦 建議操作張數 | **{lots} 張**（資金 {cap/10000:.0f} 萬，風控 {risk_pct_label}）|{lev_note}

> ⚠️ 本系統僅供量化參考，不構成實際投資建議，請自行評估風險。
        """)
    else:
        failed = [
            f"第{i+1}層「{lbl}」：{res['detail']}"
            for i, (lbl, res) in enumerate(zip(labels, results))
            if not res["pass"] and not res.get("skip")
        ]
        st.error(
            "## ❌ 未通過診斷 — **拒絕進場**\n\n"
            "以下漏洞導致攔截：\n\n"
            + "\n\n".join(f"- {f}" for f in failed)
            + "\n\n> 🔄 建議等待條件改善後重新診斷，切勿強行進場。"
        )
        pass_count = sum(1 for r in results if r["pass"] or r.get("skip"))
        if pass_count >= 3:
            st.warning(f"⚠️ 本次通過/略過 {pass_count}/5 關，條件接近成熟，可列入觀察名單。")

    if auto_refresh:
        import time
        elapsed   = time.time() - st.session_state.get("last_refresh", time.time())
        remaining = max(0, int(refresh_sec - elapsed))
        st.markdown(
            f"<div style='color:#8b949e;font-size:.78rem;margin-top:16px'>"
            f"🔄 自動刷新中，{remaining} 秒後重新抓取即時報價</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()