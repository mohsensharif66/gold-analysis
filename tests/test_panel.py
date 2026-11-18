import json
import numpy as np
import pandas as pd

from app.data_layer.panel import aligned_panel, load_ohlc, has_range, complete_bars
from tests.helpers import make_db, put_yf, put_xaus_hist


def test_ohlc_comes_from_raw_payload_and_is_scaled_to_toman(tmp_path):
    db = make_db(tmp_path)
    put_yf(db, "GOLD18_IRR", pd.Timestamp("2024-03-02"), 100, 110, 95, 105)
    df = load_ohlc(db, "GOLD18_IRR", include_provisional=False)
    r = df.iloc[0]
    assert (r.open, r.high, r.low, r.close) == (100, 110, 95, 105)      # rial payload x0.1 == stored toman
    assert not r.provisional


def test_missing_open_stays_nan_never_invented(tmp_path):
    db = make_db(tmp_path)
    put_xaus_hist(db, pd.Timestamp("2024-03-04"), 2050, 2000, 2030)
    r = load_ohlc(db, "XAU_USD", False).iloc[0]
    assert np.isnan(r.open) and r.high == 2050 and r.low == 2000


def test_inconsistent_high_low_is_repaired_and_flagged(tmp_path):
    db = make_db(tmp_path)
    put_yf(db, "GOLD18_IRR", pd.Timestamp("2024-03-02"), 100, 101, 99, 120)   # close above 'high'
    r = load_ohlc(db, "GOLD18_IRR", False).iloc[0]
    assert r.repaired and r.high == 120


def test_degenerate_close_only_history_is_not_treated_as_ohlc(tmp_path):
    db = make_db(tmp_path)
    for i in range(30):
        put_yf(db, "GOLD18_IRR", pd.Timestamp("2024-03-02") + pd.Timedelta(days=i), 100 + i, 100 + i, 100 + i, 100 + i)
    assert not has_range(load_ohlc(db, "GOLD18_IRR", False))


def test_xau_is_lagged_one_day_no_lookahead(tmp_path):
    db = make_db(tmp_path)
    days = pd.date_range("2024-03-04", periods=8)
    for i, d in enumerate(days):
        put_yf(db, "GOLD18_IRR", d, 100 + i, 101 + i, 99 + i, 100 + i)
        put_yf(db, "USD_IRR", d, 50, 51, 49, 50)
        put_xaus_hist(db, d, 2001 + i * 10, 1999 + i * 10, 2000 + i * 10)
    panel, info = aligned_panel(db, xau_lag_days=1)
    d = days[4]
    assert panel.loc[d, "xau"] == 2000 + 3 * 10          # value of D-1, not D
    assert panel.loc[days[0], "xau"] != panel.loc[days[0], "xau"]   # nothing before first obs -> NaN


def test_stale_carry_is_blocked(tmp_path):
    db = make_db(tmp_path)
    put_xaus_hist(db, pd.Timestamp("2024-03-01"), 2001, 1999, 2000)
    for i in range(12):
        d = pd.Timestamp("2024-03-10") + pd.Timedelta(days=i)
        put_yf(db, "GOLD18_IRR", d, 100, 101, 99, 100); put_yf(db, "USD_IRR", d, 50, 51, 49, 50)
    panel, _ = aligned_panel(db, max_stale_days=5)
    assert panel["xau"].isna().all()                      # >5 days old -> NaN rather than silently carried
