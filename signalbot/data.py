"""Market data download + CSV cache. Crypto: Binance public data API. Forex: Yahoo Finance."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "15m": 900_000, "1d": 86_400_000}


def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith(("USDT", "USDC", "BUSD"))


def fetch_binance(symbol: str, interval: str = "1h", years: float | None = None,
                  start_ms: int | None = None) -> pd.DataFrame:
    years = years if years is not None else (7.0 if interval == "1d" else 3.0)
    start = start_ms if start_ms is not None else int((time.time() - years * 365 * 86400) * 1000)
    rows: list[list] = []
    while True:
        r = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": interval, "startTime": start, "limit": 1000},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.set_index("time").astype(float)


def fetch_yahoo(symbol: str, interval: str = "1h", range_: str | None = None) -> pd.DataFrame:
    if interval not in ("1h", "1d"):  # Yahoo has no 4h bars: build them from 1h
        return resample(fetch_yahoo(symbol, "1h", range_), interval)
    # Yahoo limits 1h data to the last ~730 days; daily data goes back 10+ years.
    range_ = range_ or ("10y" if interval == "1d" else "730d")
    r = requests.get(
        YAHOO_URL.format(ticker=f"{symbol}=X"),
        params={"interval": interval, "range": range_},
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {k: q[k] for k in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(res["timestamp"], unit="s", utc=True),
    )
    df.index.name = "time"
    return df.dropna(subset=["open", "high", "low", "close"]).fillna({"volume": 0.0})


def resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[interval]
    out = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["close"])


def drop_unclosed(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Remove the last candle if it has not closed yet."""
    if df.empty:
        return df
    bar = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    if df.index[-1] + bar > pd.Timestamp.now(tz="UTC"):
        return df.iloc[:-1]
    return df


def cache_path(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}_{interval}.csv"


def fetch(symbol: str, interval: str = "1h") -> pd.DataFrame:
    symbol = symbol.upper()
    df = fetch_binance(symbol, interval) if is_crypto(symbol) else fetch_yahoo(symbol, interval)
    df = df[~df.index.duplicated()].sort_index()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_path(symbol, interval))
    return df


def update(symbol: str, interval: str = "1h") -> pd.DataFrame:
    """Fast refresh: download only the bars after the cached ones (full download if no/old cache)."""
    symbol = symbol.upper()
    path = cache_path(symbol, interval)
    if not path.exists():
        return fetch(symbol, interval)
    old = pd.read_csv(path, index_col="time", parse_dates=True)
    bar = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    age = pd.Timestamp.now(tz="UTC") - old.index[-1]
    try:
        if is_crypto(symbol):
            new = fetch_binance(symbol, interval, start_ms=int(old.index[-1].timestamp() * 1000))
        elif age <= pd.Timedelta(days=5):
            daily = interval == "1d"
            new = fetch_yahoo(symbol, interval, "1mo" if daily else "10d")
            new = new if daily else new.iloc[2:]  # resampled first buckets may be partial
        else:
            return fetch(symbol, interval)
    except requests.RequestException:
        return old  # offline: keep what we have
    if new.empty or new.index[0] > old.index[-1] + bar:  # nothing new / gap -> be safe
        return old if new.empty else fetch(symbol, interval)
    df = pd.concat([old[old.index < new.index[0]], new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    return df


def load(symbol: str, interval: str = "1h", refresh: bool = False, full: bool = False) -> pd.DataFrame:
    """refresh=True: incremental update; full=True: re-download everything."""
    path = cache_path(symbol, interval)
    if full:
        df = fetch(symbol, interval)
    elif refresh or not path.exists():
        df = update(symbol, interval)
    else:
        df = pd.read_csv(path, index_col="time", parse_dates=True)
    return drop_unclosed(df, interval)
