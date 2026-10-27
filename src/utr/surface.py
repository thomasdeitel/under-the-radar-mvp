"""Hourly surface and sea observations (NOAA): airport weather, plus pier sea-surface temperature.

These are the "current conditions" the predictor can use between the twice-daily soundings.
Archive files (ISD hourly airport reports, NDBC pier history) feed training; live feeds
(NWS API, NDBC realtime) feed the hourly run. Both are reduced to the same hourly table.
"""
from __future__ import annotations

import gzip
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .fetch import UA, _download, _head, _sha256

COLUMNS = ["t_c", "td_c", "slp_hpa", "wspd_ms", "wdir_deg", "sst_c", "pier_air_c", "pier_wspd_ms", "pier_pres_hpa"]


# ---------------------------------------------------------------- parsing

def parse_isd(path) -> pd.DataFrame:
    """Hourly table from an ISD fixed-width .gz file (mandatory section only)."""
    rows = []
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            if len(line) < 105:
                continue
            try:
                ts = pd.Timestamp(year=int(line[15:19]), month=int(line[19:21]), day=int(line[21:23]),
                                  hour=int(line[23:25]), minute=int(line[25:27]))
                wdir, wspd = int(line[60:63]), int(line[65:69])
                temp, dew, slp = int(line[87:92]), int(line[93:98]), int(line[99:104])
            except ValueError:
                continue
            rows.append((ts, np.nan if wdir == 999 or wspd == 9999 else wdir,
                         np.nan if wspd == 9999 else wspd / 10.0,
                         np.nan if temp == 9999 else temp / 10.0,
                         np.nan if dew == 9999 else dew / 10.0,
                         np.nan if slp == 99999 else slp / 10.0))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=["time", "wdir_deg", "wspd_ms", "t_c", "td_c", "slp_hpa"]).set_index("time")
    return _to_hourly_nearest(df)


def _to_hourly_nearest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each clock hour, the report closest to the top of the hour."""
    df = df.sort_index()
    hour = df.index.round("h")
    off = np.abs((df.index - hour).total_seconds())
    df = df.assign(_hour=hour, _off=off).sort_values("_off").drop_duplicates("_hour")
    out = df.set_index("_hour").drop(columns="_off").sort_index()
    out.index.name = "time"
    return out


def parse_ndbc(path) -> pd.DataFrame:
    """Hourly means from an NDBC standard-meteorological file (.gz or plain text)."""
    p = str(path)
    text = gzip.open(p, "rt", encoding="latin-1").read() if p.endswith(".gz") else Path(p).read_text()
    return parse_ndbc_text(text)


def parse_ndbc_text(text: str) -> pd.DataFrame:
    """Hourly means of the pier's 6-minute records, from the file's text."""
    all_lines = [l for l in text.splitlines() if l.strip()]
    hi = next(i for i, l in enumerate(all_lines) if l.lstrip("#").split()[0] in ("YY", "YYYY"))
    head = [{"YYYY": "YY", "WD": "WDIR", "BAR": "PRES"}.get(h, h) for h in all_lines[hi].lstrip("#").split()]
    body = [l for l in all_lines[hi + 1:] if not l.startswith("#")]     # skips the units row
    df = pd.read_csv(io.StringIO("\n".join(body)), sep=r"\s+", names=head, na_values=["MM"], dtype=float)
    if "mm" not in df:
        df["mm"] = 0.0
    yr = df["YY"].astype(int).where(df["YY"] > 100, df["YY"] + 1900)
    df.index = pd.to_datetime(dict(year=yr, month=df["MM"].astype(int), day=df["DD"].astype(int),
                                   hour=df["hh"].astype(int), minute=df["mm"].astype(int)))
    for col, bad in (("WSPD", 99.0), ("PRES", 9999.0), ("ATMP", 999.0), ("WTMP", 999.0)):
        if col in df:
            df.loc[df[col] >= bad, col] = np.nan
    cols = {"WTMP": "sst_c", "ATMP": "pier_air_c", "WSPD": "pier_wspd_ms", "PRES": "pier_pres_hpa"}
    hourly = df[list(cols)].rename(columns=cols).resample("h").mean()
    return hourly


def merge_hourly(isd: pd.DataFrame, pier: pd.DataFrame) -> pd.DataFrame:
    df = isd.join(pier, how="outer").sort_index()
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS]


# ---------------------------------------------------------------- archive download

def _fetch_file(url: str, dest: Path, manifest: dict, force: bool = False) -> dict:
    """Conditional download of one file; returns a status dict (never raises)."""
    key = dest.name
    res = {"file": key, "downloaded": False, "problems": []}
    try:
        head = _head(url)
    except Exception as exc:
        res["problems"].append(f"{key}: not available ({exc})")
        return res
    prev = manifest.get(key, {})
    if dest.exists() and not force and prev.get("etag") == head["etag"] and prev.get("last_modified") == head["last_modified"]:
        res["note"] = "up to date"
        return res
    try:
        _download(url, dest, head["length"])
        gzip.open(dest).read(1024)          # integrity: header + first block decompress
    except Exception as exc:
        dest.unlink(missing_ok=True)
        res["problems"].append(f"{key}: download failed ({exc})")
        return res
    manifest[key] = {"etag": head["etag"], "last_modified": head["last_modified"], "bytes": dest.stat().st_size,
                     "sha256": _sha256(dest), "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    res["downloaded"] = True
    return res


def fetch_surface_archive(raw_dir=config.RAW_DIR, years=None, isd_base=config.ISD_BASE,
                          ndbc_base=config.NDBC_HIST, force: bool = False) -> list[dict]:
    """Download yearly ISD and NDBC files into raw_dir/surface/. Missing years are reported, not fatal."""
    d = Path(raw_dir) / "surface"
    d.mkdir(parents=True, exist_ok=True)
    mpath = d / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    this_year = pd.Timestamp.now("UTC").year
    years = list(years) if years else list(range(config.SURFACE_START_YEAR, this_year + 1))
    results = []
    for y in years:
        for url, name in ((f"{isd_base}/{y}/{config.ISD_STATION}-{y}.gz", f"isd_{y}.gz"),
                          (f"{ndbc_base}/{config.NDBC_STATION}h{y}.txt.gz", f"ndbc_{y}.txt.gz")):
            r = _fetch_file(url, d / name, manifest, force)
            results.append(r)
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    return results


def load_surface_archive(raw_dir=config.RAW_DIR, processed_dir=config.PROCESSED_DIR, rebuild: bool = False) -> pd.DataFrame:
    d = Path(raw_dir) / "surface"
    cache = Path(processed_dir) / "surface_hourly.csv.gz"
    files = sorted(d.glob("*.gz"))
    if cache.exists() and not rebuild and files and cache.stat().st_mtime >= max(f.stat().st_mtime for f in files):
        return pd.read_csv(cache, index_col="time", parse_dates=["time"])
    isd = [parse_isd(f) for f in sorted(d.glob("isd_*.gz"))]
    pier = [parse_ndbc(f) for f in sorted(d.glob("ndbc_*.txt.gz"))]
    isd_all = pd.concat([x for x in isd if len(x)]).sort_index() if isd else pd.DataFrame(columns=COLUMNS)
    pier_all = pd.concat(pier).sort_index() if pier else pd.DataFrame()
    isd_all = isd_all[~isd_all.index.duplicated()]
    pier_all = pier_all[~pier_all.index.duplicated()]
    df = merge_hourly(isd_all, pier_all)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


# ---------------------------------------------------------------- live feeds

def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={**UA, "Accept": "application/geo+json, text/plain, */*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_nws(payload: dict) -> pd.DataFrame:
    """Hourly-ish table from api.weather.gov observation JSON (units are SI)."""
    rows = []
    for feat in payload.get("features", []):
        p = feat["properties"]
        v = lambda k: (p.get(k) or {}).get("value")
        ts = pd.Timestamp(p["timestamp"]).tz_convert("UTC").tz_localize(None)
        w = v("windSpeed")
        slp = v("seaLevelPressure")
        rows.append((ts, v("temperature"), v("dewpoint"), None if slp is None else slp / 100.0,
                     None if w is None else w / 3.6, v("windDirection")))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=["time", "t_c", "td_c", "slp_hpa", "wspd_ms", "wdir_deg"]).set_index("time")
    df = df.astype(float)
    return _to_hourly_nearest(df)


def fetch_live_surface(limit: int = 200) -> pd.DataFrame:
    """Latest hourly table from the NWS API (airport) and NDBC realtime2 (pier). Raises on total failure."""
    errors = []
    try:
        air = parse_nws(json.loads(_get(f"{config.NWS_OBS}?limit={limit}")))
    except Exception as exc:
        air, _ = pd.DataFrame(columns=COLUMNS), errors.append(f"airport feed: {exc}")
    try:
        text = _get(f"{config.NDBC_REALTIME}/{config.NDBC_STATION.upper()}.txt").decode("latin-1")
        pier = parse_ndbc_text(text)
    except Exception as exc:
        pier, _ = pd.DataFrame(), errors.append(f"pier feed: {exc}")
    if air.empty and pier.empty:
        raise RuntimeError("; ".join(errors) or "no live data")
    df = merge_hourly(air, pier)
    df.attrs["errors"] = errors
    return df
