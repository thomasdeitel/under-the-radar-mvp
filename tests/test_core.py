import http.server
import io
import threading
import zipfile
from functools import partial

import numpy as np
import pandas as pd
import pytest

from utr import config
from utr.dataset import make_pairs
from utr.features import summarize
from utr.fetch import _download, fetch_station
from utr.parse import Sounding, iter_soundings, nominal_time
from utr.quality import check_archive, check_series
from utr.refractivity import (duct_metrics, modified_refractivity, refractivity,
                              sat_vapor_pressure_hpa, vapor_pressure_hpa)


# ---------- refractivity ----------
def test_refractivity_matches_hand_calculation():
    e = vapor_pressure_hpa(15.0, rh_pct=60.0)
    assert e == pytest.approx(0.6 * sat_vapor_pressure_hpa(15.0))
    n = refractivity(1013.25, 15.0, e)
    assert 317.0 < float(n) < 321.0   # 77.6*1013.25/288.15 + 3.73e5*10.2/288.15^2 ~ 318.7


def test_vapor_pressure_falls_back_to_dewpoint_depression():
    e = vapor_pressure_hpa(np.array([20.0]), rh_pct=np.array([np.nan]), dpdp_c=np.array([5.0]))
    assert e[0] == pytest.approx(sat_vapor_pressure_hpa(15.0))


def test_duct_detected_where_m_decreases_with_height():
    z = np.array([0, 100, 200, 300, 400.0])
    m = np.array([350, 360, 340, 345, 350.0])
    d = duct_metrics(z, m)
    assert d["duct_present"] == 1
    assert d["trap_strength_m"] == pytest.approx(20.0)
    assert d["trap_thickness_m"] == pytest.approx(100.0)
    assert d["lowest_trap_base_m"] == pytest.approx(100.0)
    assert d["surface_duct"] == 0


def test_no_duct_in_monotonic_profile_and_surface_duct_flag():
    z = np.linspace(0, 3000, 31)
    assert duct_metrics(z, modified_refractivity(300 - 0.03 * z, z))["duct_present"] == 0
    m = np.array([360, 340, 350, 360.0])
    d = duct_metrics(np.array([0, 20, 120, 220.0]), m)
    assert d["surface_duct"] == 1


# ---------- sounding summaries ----------
def _profile(inversion: bool):
    z = np.arange(0, 3300, 150.0)
    p = 1013.25 * (1 - 2.25577e-5 * z) ** 5.25588
    t = 20 - 6.5 * z / 1000
    rh = np.full_like(z, 60.0)
    if inversion:
        t = t + 6 * np.clip((z - 800) / 200, 0, 1)
        rh = np.where(z < 900, 90.0, 20.0)
    return Sounding("TEST", pd.Timestamp("2024-06-01 12:00"), 32.8, -117.1, len(z), p, t, rh,
                    np.full_like(z, np.nan))


def test_standard_atmosphere_has_no_duct_and_passes_qc():
    row = summarize(_profile(False))
    assert row["valid"] and row["duct_present"] == 0


def test_dry_warm_layer_over_moist_layer_is_a_duct():
    row = summarize(_profile(True))
    assert row["valid"] and row["duct_present"] == 1
    assert 600 <= row["lowest_trap_base_m"] <= 1100


def test_qc_flags_short_sounding():
    s = _profile(False)
    short = Sounding(s.station, s.time, s.lat, s.lon, 4, s.p_hpa[:4], s.t_c[:4], s.rh_pct[:4], s.dpdp_c[:4])
    row = summarize(short)
    assert not row["valid"] and "levels" in row["qc_issues"]


# ---------- parser ----------
def _igra_text(year=2024, hour=11, levels=None):
    levels = levels or [(1, 1, 101300, 133, 180, 700), (2, 0, 92500, 800, 140, 600), (2, 0, 85000, 1500, 110, 400),
                        (3, 0, -9999, -9999, 100, 300), (1, 0, 50000, 5800, -150, -9999)]
    head = f"#USM00072293 {year:4d} 06 01 {hour:02d} 1115 {len(levels):4d} ncdc-nws ncdc-nws   328333 -1171166"
    rows = [head]
    for l1, l2, pr, gph, tmp, rh in levels:
        rows.append(f"{l1}{l2} {-9999:5d} {pr:6d} {gph:5d} {tmp:5d} {rh:5d} {-9999:5d} {-9999:5d} {-9999:5d}"
                    .replace(f" {pr:6d} {gph:5d} {tmp:5d} ", f" {pr:6d}A{gph:5d}B{tmp:5d}A"))
    return "\n".join(rows) + "\n"


def test_parser_reads_fixed_width_fields_and_skips_missing_pressure_and_high_levels(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text(_igra_text())
    s = list(iter_soundings(f, start_year=2000))
    assert len(s) == 1
    snd = s[0]
    assert snd.time == pd.Timestamp("2024-06-01 12:00")
    assert snd.station == "USM00072293"
    assert snd.p_hpa.tolist() == [1013.0, 925.0, 850.0]     # no-pressure level and 500 hPa dropped
    assert snd.t_c.tolist() == [18.0, 14.0, 11.0]
    assert snd.rh_pct[0] == 70.0


def test_parser_skips_old_years_and_off_hour_soundings(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text(_igra_text(year=1995) + _igra_text(hour=6))
    assert list(iter_soundings(f, start_year=2000)) == []


def test_nominal_time_snaps_to_synoptic_slots():
    assert nominal_time(2024, 6, 1, 11, 1115) == pd.Timestamp("2024-06-01 12:00")
    assert nominal_time(2024, 6, 1, 23, 2330) == pd.Timestamp("2024-06-02 00:00")
    assert nominal_time(2024, 6, 1, 99, 1130) == pd.Timestamp("2024-06-01 12:00")
    assert nominal_time(2024, 6, 1, 6, 600) is None


# ---------- quality / warnings ----------
def test_check_archive_catches_truncated_and_missing_files(tmp_path):
    assert check_archive(tmp_path / "nope.zip")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK\x03\x04" + b"0" * 2000)
    assert any("not a valid zip" in p or "corrupt" in p for p in check_archive(bad))


def _series(times, valid=True):
    df = pd.DataFrame({"valid": valid, "qc_issues": ""}, index=pd.DatetimeIndex(times))
    return df


def test_series_check_warns_when_stale_missing_or_incomplete():
    now = pd.Timestamp("2026-10-26 18:00")
    times = pd.date_range("2026-10-12 00:00", "2026-10-26 12:00", freq="12h")
    assert check_series(_series(times), now=now)["status"] == "OK"
    stale = check_series(_series(times[:-10]), now=now)
    assert stale["status"] == "WARN" and any("STALE" in w for w in stale["warnings"])
    holes = check_series(_series(times.delete([5, 6])), now=now)
    assert any("MISSING" in w for w in holes["warnings"])
    df = _series(times)
    df.loc[times[-3:], "valid"] = False
    df.loc[times[-3:], "qc_issues"] = "only 5 levels below 3 km"
    assert any("INCOMPLETE" in w for w in check_series(df, now=now)["warnings"])


# ---------- pairing ----------
def test_pairs_use_the_earlier_sounding_as_input():
    idx = pd.date_range("2024-01-01 00:00", periods=6, freq="12h")
    from utr.features import FEATURE_COLUMNS, TARGETS
    df = pd.DataFrame(0.0, index=idx, columns=sorted(set(FEATURE_COLUMNS + TARGETS)))
    df["valid"] = True
    df["trap_strength_m"] = np.arange(6) * 10.0
    p = make_pairs(df, 12)
    assert len(p) == 5
    assert p.loc[idx[3], "trap_strength_m"] == 30.0 and p.loc[idx[3], "src_trap_strength_m"] == 20.0
    assert len(make_pairs(df, 24)) == 4


# ---------- fetch (local server stands in for NOAA) ----------
@pytest.fixture
def server(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _make_zip(path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("TEST-data.txt", "x" * 5000)
    path.write_bytes(buf.getvalue())


def test_fetch_downloads_once_verifies_and_publishes_to_shared_folder(server, tmp_path):
    root, url = server
    _make_zip(root / "TEST-data.txt.zip")
    raw, shared = tmp_path / "raw", tmp_path / "shared"
    r1 = fetch_station("TEST", raw, shared, base_url=url, retries=1)
    assert r1["downloaded"] and not r1["problems"]
    assert (shared / "TEST-data.txt.zip").exists() and not list(shared.glob("*.part"))
    r2 = fetch_station("TEST", raw, shared, base_url=url, retries=1)
    assert not r2["downloaded"] and "up to date" in r2["note"]


def test_fetch_reports_corrupt_download_and_rejects_short_transfer(server, tmp_path):
    root, url = server
    (root / "TEST-data.txt.zip").write_bytes(b"PK\x03\x04" + b"1" * 3000)
    r = fetch_station("TEST", tmp_path / "raw", tmp_path / "shared", base_url=url, retries=1)
    assert r["problems"] and not (tmp_path / "shared" / "TEST-data.txt.zip").exists()
    with pytest.raises(IOError):
        _download(f"{url}/TEST-data.txt.zip", tmp_path / "d.zip", expected_len=10**9)
    assert not list(tmp_path.glob("*.part"))


def test_fetch_failure_is_reported_not_raised(tmp_path):
    r = fetch_station("TEST", tmp_path / "raw", None, base_url="http://127.0.0.1:9", retries=1)
    assert r["problems"] and "failed" in r["problems"][0]
