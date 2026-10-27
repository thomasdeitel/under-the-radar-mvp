# Operator guide: learn from your own data, offline

For a team that holds data we cannot see. You give the program a CSV of your observations; it
checks them, learns from them, and predicts each hour or day with a green/yellow/red light.
Nothing is sent anywhere: the `ship` commands use only numpy, pandas and scikit-learn and make no
network calls (a test enforces this). It runs on an ordinary CPU in seconds to minutes. No GPU.

## 1. Install (one time)

```
pip install .            # or: pip install -r requirements.txt  and  export PYTHONPATH=src
```
On a machine without internet, build wheels elsewhere with `pip download -r requirements.txt` and install from that folder.

## 2. Prepare your data

```
python -m utr ship template --out my_inputs
```
This writes `input_template.csv` and `schema.json`. Put your observations in the same shape:

- One row per observation time, UTC, in a column named `time`.
- Any number of other numeric columns: air temperature, dew point or humidity, pressure, wind,
  sea temperature, anything you record. Add, remove or rename columns freely. Empty cells are fine.
- The things you want predicted are **targets**, listed in `schema.json`. They can be sparse (for
  example one value a day). A target can be a measurement, a yes/no (0 or 1), or the output of your
  own slow simulation, so the model learns to reproduce it from the inputs.

Edit `schema.json`:

| Field | Meaning |
|---|---|
| `targets` | column names to predict |
| `horizon_hours` | how far ahead. `3` = from now, predict 3 h from now. `0` = learn the mapping from the inputs at time t to the target at time t (use this when the target is a simulation output) |
| `step_hours` | the grid the data are averaged onto (default 1 h) |
| `angle_columns` | direction columns in degrees, such as wind direction |
| `valid_ranges` | physical limits per column; values outside become missing |
| `lags_hours` | how far back the model looks for trends |

Check the file before training:

```
python -m utr ship check my_inputs/my_data.csv --schema my_inputs/schema.json
```
It reports missing values, impossible values, long gaps and too little data.

## 3. Learn

```
python -m utr ship train my_data.csv --schema my_inputs/schema.json --model-dir ship_model
```
Writes `ship_model/report.md` (read this first), the model, and a copy of the data it has seen.
The newest 20% of the data is held back and used only to score the model, so the report is an
honest test. Three weeks of hourly data is enough to start; expect rough results until there is more.

## 4. Predict

```
python -m utr ship predict recent.csv --model-dir ship_model --out predictions.csv --all --daily
```
`--all` gives a prediction from every row (an hourly table); without it you get one prediction from
the newest row. `--daily` also writes a per-day summary with one light per day.

Each prediction row has `issue_time`, `valid_time`, `target`, `prediction`, `light` and a `note`.

| Light | Meaning |
|---|---|
| green | the models agree and the inputs look like the ones it learned from |
| yellow | use with care: the models disagree, the inputs are unusual or not current, or the model has not yet beaten the simple baseline |
| red | do not rely on it; rerun your full simulation |

## 5. Keep learning

When new data arrives, run:

```
python -m utr ship learn new_data.csv --model-dir ship_model
```
New rows are added to the stored data (overlapping times are not double counted) and the model is
retrained. `report.md` keeps a history, so you can see whether it is improving as data accumulates.

## What the report tells you, and what it cannot

- **Skill vs baseline** compares the model to repeating the last known value or the average. If it
  is not positive, the model has not learned anything useful yet and says so in the lights.
- **Do the lights mean anything on your data** shows error in each light band. Error should rise
  from green to red. With a few weeks of data this is noisy.
- The model learns correlations in your data. It does not know physics you did not give it, and it
  is only as good as the range of conditions in the data it saw. Inputs far outside that range
  turn the light yellow or red.
- Try it first on `examples/ship_demo/`, which is **synthetic** data with a made-up relationship.
  It shows that the workflow runs, not that the model works on real atmospheric data.
