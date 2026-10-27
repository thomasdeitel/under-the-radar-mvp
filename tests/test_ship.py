import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utr import ship


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("ship")
    schema = ship.synthetic_demo(d / "demo.csv", days=21)
    sm = ship.train(d / "demo.csv", schema, d / "model")
    return d, schema, sm


def test_it_learns_a_known_relationship_from_three_weeks_of_made_up_data(trained):
    _, _, sm = trained
    r = sm.report["tests"]["evaporation_duct_height_m"]
    assert r["skill_vs_baseline"] > 0.3           # beats simply repeating the last value
    assert r["models_used"]


def test_model_directory_holds_everything_needed_and_report_is_readable(trained):
    d, _, _ = trained
    for name in ("model.pkl", "schema.json", "report.md", "report.json", "data_store.csv"):
        assert (d / "model" / name).exists()
    assert "What the model learned" in (d / "model" / "report.md").read_text()


def test_predictions_have_a_light_for_every_row_and_a_daily_summary(trained):
    d, _, _ = trained
    pred, _ = ship.predict(d / "demo.csv", d / "model", last_only=False)
    assert set(pred["light"]) <= {"green", "yellow", "red"}
    assert len(pred) > 400 and pred["prediction"].notna().all()
    daily = ship.daily_summary(pred)
    assert set(daily["light"]) <= {"green", "yellow", "red"} and len(daily) >= 20


def test_learn_appends_new_data_and_keeps_history(tmp_path):
    schema = ship.synthetic_demo(tmp_path / "a.csv", days=21, seed=0)
    ship.train(tmp_path / "a.csv", schema, tmp_path / "m")
    more = ship.synthetic_demo(tmp_path / "b.csv", days=42, seed=0)       # same stream, longer
    sm = ship.train(tmp_path / "b.csv", more, tmp_path / "m")
    assert len(sm.history) == 2 and sm.history[1]["rows"] > sm.history[0]["rows"]
    # the store de-duplicates by time, so overlapping files are not double counted
    store = pd.read_csv(tmp_path / "m" / "data_store.csv")
    assert store["time"].is_unique


def test_data_check_reports_problems(tmp_path):
    schema = ship.Schema(targets=["y"], valid_ranges={"t": [-10, 40]})
    idx = pd.date_range("2026-01-01", periods=72, freq="h")
    df = pd.DataFrame({"time": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "t": 20.0, "y": np.nan})
    df.loc[3, "t"] = 99.0                    # impossible
    df.loc[df.index[10:40], "t"] = np.nan   # a long hole
    df.to_csv(tmp_path / "x.csv", index=False)
    _, rep = ship.to_grid(ship.read_table(tmp_path / "x.csv", schema), schema)
    assert rep["status"] == "FAIL"                              # target has no observations
    assert any("outside" in w for w in rep["warnings"])
    assert any("days of data" in w for w in rep["warnings"])
    with pytest.raises(ValueError):
        ship.train(tmp_path / "x.csv", schema, tmp_path / "m")


def test_missing_time_column_is_a_clear_error(tmp_path):
    pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="time"):
        ship.read_table(tmp_path / "x.csv", ship.Schema())


def test_surrogate_mode_learns_inputs_to_same_time_target(tmp_path):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-01", periods=600, freq="h")
    x1, x2 = rng.normal(size=600), rng.normal(size=600)
    y = 3 * x1 - 2 * x2 + rng.normal(0, 0.2, 600)
    df = pd.DataFrame({"time": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "x1": x1, "x2": x2, "sim_output": y})
    df.to_csv(tmp_path / "s.csv", index=False)
    sm = ship.train(tmp_path / "s.csv", ship.Schema(targets=["sim_output"], horizon_hours=0.0), tmp_path / "m")
    assert sm.report["tests"]["sim_output"]["skill_vs_baseline"] > 0.8


def test_binary_target_gets_probabilities(tmp_path):
    rng = np.random.default_rng(2)
    idx = pd.date_range("2026-01-01", periods=700, freq="h")
    x = rng.normal(size=700)
    y = (x + rng.normal(0, 0.5, 700) > 0).astype(float)
    pd.DataFrame({"time": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "x": x, "detected": y}).to_csv(tmp_path / "b.csv", index=False)
    ship.train(tmp_path / "b.csv", ship.Schema(targets=["detected"], horizon_hours=0.0), tmp_path / "m")
    pred, _ = ship.predict(tmp_path / "b.csv", tmp_path / "m", last_only=False)
    assert pred["prediction"].between(0, 1).all()


def test_template_files_are_valid(tmp_path):
    paths = ship.write_template(tmp_path)
    schema = ship.Schema.load(paths[1])
    df = ship.read_table(paths[0], schema)
    assert "air_temp_c" in df.columns and len(df) == 2


def test_ship_module_makes_no_network_calls():
    src = Path(ship.__file__).read_text()
    for banned in ("urllib", "requests", "socket", "http.client", "urlopen"):
        assert banned not in src
