import pandas as pd
from app.engines.indicators import calculate_indicators

def test_indicator_columns_exist():
    close = pd.Series(range(1, 250), dtype=float)
    high, low = close + 1, close - 1
    result = calculate_indicators(close, high, low)
    assert {"rsi_14","macd","ema_20","ema_200","bb_upper","tenkan_sen"} <= set(result.columns)
    assert result["ema_20"].notna().any()
