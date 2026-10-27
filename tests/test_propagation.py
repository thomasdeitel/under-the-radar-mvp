import gzip

import numpy as np
import pandas as pd
import pytest

from utr.nowcast import next_slot, runtime_row, surface_feature_table
from utr.nowcast_eval import Light
from utr.propagation import (Scenario, free_space_axis_amplitude, propagation_factor_db,
                             solve_split_step, standard_atmosphere_m, starter)
from utr.surface import parse_isd, parse_ndbc_text
from utr.teng import TengConfig, solve_teng


def _small():
    return Scenario(rmax_m=8000, dr_m=100.0, zmax_m=300.0)


def test_split_step_matches_exact_two_ray_solution_in_uniform_air():
    sc = _small()
    r, u = solve_split_step(sc, np.full_like(sc.z, 330.0))
    a = 0.5 * sc.sigma_p**2
    c = 1 + 2j * a * r[-1] / sc.k
    h = 40.0
    exact = abs(c**-0.5 * (np.exp(-a * (h - sc.src_height_m) ** 2 / c) - np.exp(-a * (h + sc.src_height_m) ** 2 / c)))
    assert abs(u[-1, int(h / sc.dz_m) - 1]) == pytest.approx(exact, rel=1e-4)


def test_free_space_reference_is_the_analytic_gaussian_beam():
    sc = _small()
    assert free_space_axis_amplitude(sc, np.array([0.0]))[0] == pytest.approx(1.0)
    assert free_space_axis_amplitude(sc, np.array([50000.0]))[0] < 0.5


def test_starter_vanishes_at_the_conducting_ground():
    sc = _small()
    assert abs(starter(sc)[0]) < 1e-3


def test_trapping_layer_carries_signal_farther_than_standard_air():
    sc = Scenario(rmax_m=40000, dr_m=100.0, zmax_m=400.0)
    z = sc.z
    std = standard_atmosphere_m(z)
    duct = std + np.where(z > 80, -40.0 * np.clip((z - 80) / 20, 0, 1), 0.0)   # M falls by 40 across 80-100 m
    _, u_std = solve_split_step(sc, std)
    _, u_duct = solve_split_step(sc, duct)
    band = (z > 20) & (z < 70)
    assert np.abs(u_duct[-1][band]).mean() > 3 * np.abs(u_std[-1][band]).mean()


def test_teng_network_solver_agrees_with_reference_solver():
    sc = _small()
    m = standard_atmosphere_m(sc.z)
    r, u_ref = solve_split_step(sc, m)
    out = solve_teng(sc, m, TengConfig(kz_cut=1.0, dr_m=100.0))
    pf_ref = propagation_factor_db(sc, r, u_ref)
    pf = propagation_factor_db(sc, out["ranges"], out["fields"])
    mask = (pf_ref > -40) & (r[:, None] >= 5000) & (sc.z[None, :] > 5) & (sc.z[None, :] < 0.7 * sc.zmax_m)
    assert np.abs(pf - pf_ref)[mask].mean() < 1.0      # dB
    assert out["n_params"] > 0


# ---------------------------------------------------------------- hourly observations
def _isd_line(ts, wdir, wspd, temp, dew, slp):
    line = list(" " * 110)

    def put(a, b, text):
        line[a:b] = list(text)

    put(15, 27, ts.strftime("%Y%m%d%H%M"))
    put(60, 63, f"{wdir:03d}")
    put(65, 69, f"{wspd:04d}")
    put(87, 92, f"{temp:+05d}")
    put(93, 98, f"{dew:+05d}")
    put(99, 104, f"{slp:05d}")
    return "".join(line)


def test_isd_parser_reads_values_and_maps_missing(tmp_path):
    lines = [_isd_line(pd.Timestamp("2024-01-01 00:53"), 270, 52, 161, 117, 10177),
             _isd_line(pd.Timestamp("2024-01-01 01:53"), 999, 9999, 9999, 9999, 99999)]
    p = tmp_path / "isd.gz"
    with gzip.open(p, "wt") as f:
        f.write("\n".join(lines) + "\n")
    df = parse_isd(p)
    assert df.loc["2024-01-01 01:00", "t_c"] == pytest.approx(16.1)
    assert df.loc["2024-01-01 01:00", "wspd_ms"] == pytest.approx(5.2)
    assert df.loc["2024-01-01 01:00", "slp_hpa"] == pytest.approx(1017.7)
    assert df.iloc[1][["t_c", "td_c", "slp_hpa", "wspd_ms"]].isna().all()


def test_ndbc_parser_averages_six_minute_records_to_hours():
    text = ("#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n"
            "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  mi    ft\n"
            "2024 01 01 00 00 339  1.0  1.9 99.00 99.00 99.00 999 1016.8  15.0  16.0 999.0 99.0 99.00\n"
            "2024 01 01 00 06 324  2.0  1.5 99.00 99.00 99.00 999 9999.0  16.0  17.0 999.0 99.0 99.00\n")
    h = parse_ndbc_text(text)
    assert h.loc["2024-01-01 00:00", "sst_c"] == pytest.approx(16.5)
    assert h.loc["2024-01-01 00:00", "pier_pres_hpa"] == pytest.approx(1016.8)   # 9999 ignored


def _hourly(n=30):
    idx = pd.date_range("2024-06-01", periods=n, freq="h")
    return pd.DataFrame({"t_c": 20.0, "td_c": 12.0, "slp_hpa": 1015.0, "wspd_ms": 3.0, "wdir_deg": 270.0,
                         "sst_c": 18.0, "pier_air_c": 19.0, "pier_wspd_ms": 3.0, "pier_pres_hpa": 1015.0}, index=idx)


def test_surface_features_include_sea_air_gap_and_trends():
    h = _hourly()
    h.loc[h.index[-1], "t_c"] = 23.0
    last = surface_feature_table(h).iloc[-1]
    assert last["d_t_3h"] == pytest.approx(3.0)
    assert last["sea_air_vp_gap_hpa"] > 0          # sea-saturated air is moister than the air above it
    assert last["wind_u"] > 0                      # west wind blows toward the east


def test_next_slot_is_the_next_00z_or_12z():
    assert next_slot(pd.Timestamp("2025-03-10 09:00")) == pd.Timestamp("2025-03-10 12:00")
    assert next_slot(pd.Timestamp("2025-03-10 12:00")) == pd.Timestamp("2025-03-10 12:00")
    assert next_slot(pd.Timestamp("2025-03-10 13:00")) == pd.Timestamp("2025-03-11 00:00")


def test_runtime_row_uses_latest_sounding_at_or_before_now():
    from utr.features import FEATURE_COLUMNS
    idx = pd.to_datetime(["2025-03-09 12:00", "2025-03-10 00:00"])
    summ = pd.DataFrame(1.0, index=idx, columns=FEATURE_COLUMNS)
    summ["valid"] = True
    summ.loc[idx[1], "trap_strength_m"] = 9.0
    h = _hourly(60)
    h.index = pd.date_range("2025-03-09", periods=60, freq="h")
    row, info = runtime_row(summ, surface_feature_table(h), pd.Timestamp("2025-03-10 09:00"))
    assert info["last_sounding"] == idx[1] and info["hours_since_sounding"] == 9.0
    assert info["target_sounding"] == pd.Timestamp("2025-03-10 12:00")
    assert row["src_trap_strength_m"].iloc[0] == 9.0 and row["prev2_trap_strength_m"].iloc[0] == 1.0


# ---------------------------------------------------------------- lights
def _preds(p, s_std, t_std):
    n = len(p)
    return {"duct_present": {"nn": np.asarray(p)},
            "trap_strength_m": {"nn_std": np.asarray(s_std), "nn": np.zeros(n)},
            "trap_thickness_m": {"nn_std": np.asarray(t_std), "nn": np.zeros(n)}}


def test_lights_follow_the_signal_for_each_output():
    rng = np.random.default_rng(0)
    lamp = Light(_preds(rng.uniform(0, 1, 500), rng.uniform(0, 20, 500), rng.uniform(0, 100, 500)))
    c = lamp.colors(_preds([0.99, 0.5, 0.5], [1.0, 1.0, 30.0], [1.0, 1.0, 1.0]))
    assert c["duct_present"][0] == "green" and c["duct_present"][1] == "red"
    assert c["trap_strength_m"][0] == "green" and c["trap_strength_m"][2] == "red"
    assert c["overall"][0] == "green" and c["overall"][2] == "red"
