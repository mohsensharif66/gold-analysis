import numpy as np
import pandas as pd
from app.db.database import utc_now_iso

def calculate_indicators(close: pd.Series, high: pd.Series | None = None,
                         low: pd.Series | None = None) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    out = pd.DataFrame(index=close.index)
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    for span in (20, 50, 200):
        out[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_middle"], out["bb_upper"], out["bb_lower"] = mid, mid + 2*std, mid - 2*std
    if high is not None and low is not None:
        high, low = pd.to_numeric(high), pd.to_numeric(low)
        hh9, ll9 = high.rolling(9).max(), low.rolling(9).min()
        hh26, ll26 = high.rolling(26).max(), low.rolling(26).min()
        hh52, ll52 = high.rolling(52).max(), low.rolling(52).min()
        out["tenkan_sen"] = (hh9 + ll9) / 2
        out["kijun_sen"] = (hh26 + ll26) / 2
        out["senkou_span_a"] = ((out["tenkan_sen"] + out["kijun_sen"]) / 2).shift(26)
        out["senkou_span_b"] = ((hh52 + ll52) / 2).shift(26)
        out["chikou_span"] = close.shift(-26)
    else:
        for c in ("tenkan_sen","kijun_sen","senkou_span_a","senkou_span_b","chikou_span"):
            out[c] = np.nan
    return out

def persist_indicators(db, symbol: str, timeframe: str) -> int:
    with db.connect() as conn:
        rows = conn.execute("""
            SELECT open_time_utc,open,high,low,close FROM candles
            WHERE symbol_code=? AND timeframe=? ORDER BY open_time_utc
        """, (symbol, timeframe)).fetchall()
    if not rows:
        return 0
    df = pd.DataFrame([dict(r) for r in rows])
    ind = calculate_indicators(df["close"], df["high"], df["low"])
    cols = ["rsi_14","macd","macd_signal","macd_hist","ema_20","ema_50","ema_200",
            "bb_middle","bb_upper","bb_lower","tenkan_sen","kijun_sen",
            "senkou_span_a","senkou_span_b","chikou_span"]
    records = []
    for i, row in df.iterrows():
        vals = [None if pd.isna(ind.iloc[i][c]) else float(ind.iloc[i][c]) for c in cols]
        records.append((symbol,timeframe,row.open_time_utc,float(row.close),*vals,utc_now_iso()))
    with db.connect() as conn:
        conn.executemany("""
            INSERT INTO indicators(symbol_code,timeframe,open_time_utc,close,rsi_14,macd,
            macd_signal,macd_hist,ema_20,ema_50,ema_200,bb_middle,bb_upper,bb_lower,
            tenkan_sen,kijun_sen,senkou_span_a,senkou_span_b,chikou_span,computed_at_utc)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol_code,timeframe,open_time_utc) DO UPDATE SET
              close=excluded.close,rsi_14=excluded.rsi_14,macd=excluded.macd,
              macd_signal=excluded.macd_signal,macd_hist=excluded.macd_hist,
              ema_20=excluded.ema_20,ema_50=excluded.ema_50,ema_200=excluded.ema_200,
              bb_middle=excluded.bb_middle,bb_upper=excluded.bb_upper,bb_lower=excluded.bb_lower,
              tenkan_sen=excluded.tenkan_sen,kijun_sen=excluded.kijun_sen,
              senkou_span_a=excluded.senkou_span_a,senkou_span_b=excluded.senkou_span_b,
              chikou_span=excluded.chikou_span,computed_at_utc=excluded.computed_at_utc
        """, records)
    return len(records)
