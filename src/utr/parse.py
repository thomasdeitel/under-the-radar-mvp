"""Parser for NOAA NCEI IGRA v2 'data-por' files (fixed-width radiosonde soundings)."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

MISSING = (-9999, -8888)
MIN_PRESSURE_PA = 60000  # parse only the lowest ~4 km; ducting work needs < 3 km


@dataclass
class Sounding:
    station: str
    time: pd.Timestamp          # nominal synoptic time (00Z or 12Z), UTC, naive
    lat: float
    lon: float
    n_levels_reported: int
    p_hpa: np.ndarray           # ordered surface -> upward
    t_c: np.ndarray
    rh_pct: np.ndarray
    dpdp_c: np.ndarray


def nominal_time(year: int, month: int, day: int, hour: int, reltime: int) -> pd.Timestamp | None:
    """Snap a sounding to its 00Z/12Z synoptic slot (releases start ~1 h early). None if neither."""
    h = hour
    if h == 99:
        rh = reltime // 100 if 0 <= reltime < 9999 else 99
        h = rh
    if h == 99 or h < 0 or h > 23:
        return None
    base = pd.Timestamp(year=year, month=month, day=day)
    slot = int(round(h / 12.0)) * 12
    if abs(h - slot) > 2:
        return None
    return base + pd.Timedelta(hours=slot)


def _num(field: str, scale: float = 1.0) -> float:
    field = field.strip()
    if not field:
        return np.nan
    v = int(field)
    return np.nan if v in MISSING else v / scale


def iter_soundings(path, start_year: int = 2000) -> Iterator[Sounding]:
    """Yield soundings from an IGRA .txt.zip (or plain .txt) file, oldest first."""
    path = str(path)
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        inner = zf.namelist()[0]
        stream = io.TextIOWrapper(zf.open(inner), encoding="latin-1")
    else:
        stream = open(path, encoding="latin-1")
    try:
        yield from _iter_stream(stream, start_year)
    finally:
        stream.close()


def _iter_stream(stream, start_year: int) -> Iterator[Sounding]:
    line = stream.readline()
    while line:
        if not line.startswith("#"):
            line = stream.readline()
            continue
        station = line[1:12]
        year, month, day = int(line[13:17]), int(line[18:20]), int(line[21:23])
        hour, reltime, numlev = int(line[24:26]), int(line[27:31]), int(line[32:36])
        lat, lon = int(line[55:62]) / 10000.0, int(line[63:71]) / 10000.0
        t = nominal_time(year, month, day, hour, reltime) if year >= start_year else None
        keep = t is not None
        p, tc, rh, dp = [], [], [], []
        for _ in range(numlev):
            rec = stream.readline()
            if not keep:
                continue
            press = rec[9:15].strip()
            if not press:
                continue
            pv = int(press)
            if pv in MISSING or pv < MIN_PRESSURE_PA:
                continue
            p.append(pv / 100.0)
            tc.append(_num(rec[22:27], 10.0))
            rh.append(_num(rec[28:33], 10.0))
            dp.append(_num(rec[34:39], 10.0))
        line = stream.readline()
        if keep:
            yield Sounding(station, t, lat, lon, numlev, np.array(p), np.array(tc),
                           np.array(rh), np.array(dp))
