"""Crypto perpetual funding rate from Binance's public data archive (data.binance.vision).

Features are aligned to the bar CLOSE time and only use funding settled strictly before it.
Finished months are cached on disk; the current (unpublished) month is skipped, so the most
recent weeks have NaN funding features until the month's file is published.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from .data import DATA_DIR, HEADERS, INTERVAL_MS

ARCHIVE = ("https://data.binance.vision/data/futures/um/monthly/fundingRate/"
           "{sym}/{sym}-fundingRate-{ym}.zip")


def _month_file(symbol: str, ym: str):
    return DATA_DIR / "funding" / symbol / f"{ym}.csv"


def _download_month(symbol: str, ym: str) -> pd.DataFrame | None:
    r = requests.get(ARCHIVE.format(sym=symbol, ym=ym), headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip().lower() for c in df.columns]
    return pd.DataFrame({"time": pd.to_datetime(df["calc_time"], unit="ms", utc=True),
                         "rate": df["last_funding_rate"].astype(float)})


def load_funding(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    frames = []
    for p in pd.period_range(start.tz_localize(None).to_period("M"), now.tz_localize(None).to_period("M"), freq="M"):
        ym, path = str(p), _month_file(symbol, str(p))
        if path.exists():
            cached = pd.read_csv(path)
            cached["time"] = pd.to_datetime(cached["time"], utc=True, format="ISO8601")  # mixed ms/no-ms
            frames.append(cached)
            continue
        df = _download_month(symbol, ym) if p < now.tz_localize(None).to_period("M") else None
        if df is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["time", "rate"])
    out = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    out["time"] = pd.to_datetime(out["time"], utc=True, format="ISO8601")
    return out


def funding_features(symbol: str, bar_index: pd.DatetimeIndex, interval: str) -> pd.DataFrame:
    fr = load_funding(symbol, bar_index[0] - pd.Timedelta(days=100))
    cols = ["fund_last", "fund_mean_1d", "fund_mean_7d", "fund_z"]
    if fr.empty:
        return pd.DataFrame(index=bar_index, columns=cols, dtype=float)
    s = fr["rate"]
    f = pd.DataFrame({
        "time": fr["time"].astype("datetime64[ns, UTC]"),
        "fund_last": s,
        "fund_mean_1d": s.rolling(3).mean(),       # 8h funding -> 3 settlements = 1 day
        "fund_mean_7d": s.rolling(21).mean(),
    })
    f["fund_z"] = (f["fund_mean_1d"] - s.rolling(90).mean()) / s.rolling(90).std()
    close = pd.DataFrame({"t": (bar_index + pd.Timedelta(milliseconds=INTERVAL_MS[interval]))
                          .astype("datetime64[ns, UTC]")})
    m = pd.merge_asof(close, f, left_on="t", right_on="time", allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=2))   # stale funding -> NaN instead of old values
    m.index = bar_index
    return m[cols]
