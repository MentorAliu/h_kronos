import pandas as pd
import pytest

from hourly_prediction.validation import CandleValidationError, validate_candles


def make_candles(start: str, periods: int, freq: str) -> pd.DataFrame:
    timestamps = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i for i in range(periods)],
            "high": [101.0 + i for i in range(periods)],
            "low": [99.0 + i for i in range(periods)],
            "close": [100.5 + i for i in range(periods)],
            "volume": [10.0 + i for i in range(periods)],
        }
    )


def test_valid_hourly_candles_pass_and_compute_amount() -> None:
    df = make_candles("2026-06-01 00:00:00", 3, "1h")

    clean = validate_candles(
        df,
        timeframe="1h",
        now=pd.Timestamp("2026-06-01 04:00:00", tz="UTC"),
    )

    assert list(clean.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert clean["amount"].tolist() == pytest.approx(
        (clean["close"] * clean["volume"]).tolist()
    )


def test_valid_15m_candles_pass() -> None:
    df = make_candles("2026-06-01 00:00:00", 4, "15min")

    clean = validate_candles(
        df,
        timeframe="15m",
        now=pd.Timestamp("2026-06-01 01:15:00", tz="UTC"),
    )

    assert len(clean) == 4


def test_missing_required_column_fails() -> None:
    df = make_candles("2026-06-01 00:00:00", 3, "1h").drop(columns=["volume"])

    with pytest.raises(CandleValidationError, match="Missing required columns"):
        validate_candles(
            df,
            timeframe="1h",
            now=pd.Timestamp("2026-06-01 04:00:00", tz="UTC"),
        )


def test_duplicate_timestamps_fail() -> None:
    df = make_candles("2026-06-01 00:00:00", 3, "1h")
    df.loc[2, "timestamp"] = df.loc[1, "timestamp"]

    with pytest.raises(CandleValidationError, match="duplicate"):
        validate_candles(
            df,
            timeframe="1h",
            now=pd.Timestamp("2026-06-01 04:00:00", tz="UTC"),
        )


def test_missing_interval_gap_fails() -> None:
    df = make_candles("2026-06-01 00:00:00", 4, "1h")
    df = df.drop(index=2).reset_index(drop=True)

    with pytest.raises(CandleValidationError, match="gap"):
        validate_candles(
            df,
            timeframe="1h",
            now=pd.Timestamp("2026-06-01 05:00:00", tz="UTC"),
        )


def test_unsorted_timestamps_fail() -> None:
    df = make_candles("2026-06-01 00:00:00", 3, "1h")
    df = df.iloc[[1, 0, 2]].reset_index(drop=True)

    with pytest.raises(CandleValidationError, match="sorted"):
        validate_candles(
            df,
            timeframe="1h",
            now=pd.Timestamp("2026-06-01 04:00:00", tz="UTC"),
        )


def test_unfinished_latest_candle_is_dropped() -> None:
    df = make_candles("2026-06-01 00:00:00", 3, "1h")

    clean = validate_candles(
        df,
        timeframe="1h",
        now=pd.Timestamp("2026-06-01 02:30:00", tz="UTC"),
    )

    assert clean["timestamp"].tolist() == [
        pd.Timestamp("2026-06-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-06-01 01:00:00", tz="UTC"),
    ]


def test_invalid_numeric_value_fails() -> None:
    df = make_candles("2026-06-01 00:00:00", 3, "1h")
    df["close"] = df["close"].astype(object)
    df.loc[1, "close"] = "not-a-number"

    with pytest.raises(CandleValidationError, match="numeric"):
        validate_candles(
            df,
            timeframe="1h",
            now=pd.Timestamp("2026-06-01 04:00:00", tz="UTC"),
        )
