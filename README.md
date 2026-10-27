# Under the Radar MVP

Navy 215 "Under the Radar" (i4NS). Keeps radar-relevant atmospheric information current between
the twice-daily soundings, shows how much to trust it with a green/yellow/red light, and can be
handed to an operator who learns it on their own data offline.

Three parts, all in `python -m utr ...`:

| Part | What it does | Commands |
|---|---|---|
| **A. Data loading** | Scheduled download of NOAA radiosonde, airport-weather and pier/sea-temperature data, with integrity checks, warnings for stale / missing / incomplete data, and publishing to a shared folder | `fetch`, `fetch-surface`, `check` |
| **B. Hourly estimate + coverage map** | Every hour, estimates the duct state expected at the next sounding from the latest soundings, earlier ducts and current weather and sea temperature. Builds a radar coverage map from the latest sounding with a TENG-style neural-network solver and says how stale it probably is | `nowcast-eval`, `nowcast`, `coverage-eval`, `coverage` |
| **C. Learn from your own data** | For operators who cannot share data: put your observations in a CSV, the program checks them, learns from them, and predicts each hour or day with a light. Fully offline | `ship template / check / train / learn / predict` |

Part C is documented for operators in [docs/OPERATOR_GUIDE.md](docs/OPERATOR_GUIDE.md).

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m utr fetch && python -m utr fetch-surface     # NOAA data (a few minutes the first time)
python -m utr check                                    # warnings about stale / missing data
python -m utr nowcast-eval                             # train + test the hourly estimate (~5 min)
python -m utr coverage-eval                            # learn how fast coverage maps go stale (~6 min)
python -m utr nowcast                                  # live: estimate + lights from the newest NOAA observations
python -m utr coverage                                 # live: coverage map + staleness light -> results/latest_coverage.png
python -m utr ship demo --out demo && python -m utr ship train demo/synthetic_demo.csv --schema demo/schema.json --model-dir demo/model
pytest -q                                              # 37 tests, no network needed
```
Run `nowcast` and `coverage` hourly from cron or a task scheduler. Exit code is non-zero when there are warnings.

## How the pieces map to the deck MVP

| Deck MVP | Here |
|---|---|
| Turn a new atmospheric reading into a radar coverage map | `coverage`: sounding to refractivity profile to range-height propagation map; about 6 s per 60 km radial on 2 CPU cores |
| Hybrid of neural networks with numerical time-stepping (TENG) | `src/utr/teng.py`: a neural network holds the field and is advanced in range by natural-gradient (TENG) steps. See limits below |
| Green / yellow / red; red means rerun the full simulation | Lights on the hourly estimate (per output), on the coverage map (staleness), and on every operator prediction |
| Runs on ordinary computers, training done ashore | CPU only; `ship` retrains in seconds to minutes |
| Update every 3 hours (to be confirmed) | The hourly estimate refreshes whenever new surface or sea observations arrive; soundings arrive twice a day |

## Results (San Diego, test period 2022 onward, never used for fitting)

**Hourly estimate of the duct state at the next sounding** (full model; `results/nowcast.md`):

| Output | Model | Seasonal average | Keep last sounding |
|---|---|---|---|
| Duct present (Brier) | 0.133 | 0.167 | 0.234 |
| Duct strength (RMSE, M-units) | 15.3 | 16.6 | 17.8 |
| Duct thickness (MAE, m) | 67.9 | 69.4 | 82.7 |

- Adding hourly weather and sea temperature helps duct strength and thickness a little (strength error down about 5%, confidence interval excludes zero) and does not change the duct-present score. Adding the sounding before last (earlier ducts) helps nothing measurable.
- It clearly beats keeping the last sounding, and beats the seasonal average modestly on presence and strength. On thickness it is about even.
- Lights per output: error rises from green to red (duct strength mean error 7.8 / 13.4 / 14.6 M-units; thickness 51 / 78 / 93 m; presence Brier 0.06 / 0.22 / 0.25). About one estimate in three has at least one red light.

**Coverage maps** (`results/coverage.md`, `results/coverage_example.png`):
- A map built from the sounding 12 h earlier differs from the map built from the new sounding by 3.4 dB on average (90th percentile 7.1 dB). That is the staleness problem, measured with our reference solver.
- A model predicts how far the map has moved from the current conditions (rank correlation 0.56). Green hours average 2.2 dB of change, yellow 4.0, red 6.2.
- The TENG-style network solver matches the reference split-step solver to about 0.4 dB beyond 5 km (checked on 3 soundings; 0.07 dB with the finer setting).

## Limitations (please read)

- **No Navy data was used.** The reference "full simulation" is our own parabolic-equation solver, not the Navy's. The radar settings (1 GHz, 3 degree beam, 25 m antenna height) and the "covered" threshold (-30 dB) are placeholders. Speed against the Navy's simulator is untested.
- **Coverage maps are one radial in a horizontally uniform atmosphere**, valid beyond about 5 km, from IGRA soundings whose vertical resolution is too coarse to see thin evaporation ducts. Low-angle propagation only.
- **The TENG-style solver is slower than the plain numerical solver in this 1D-in-height setting** (about 6 s versus 0.04 s per radial). Its case rests on problems where grids are expensive, which this prototype does not test. Evolving the network's frequencies as well as its amplitudes is stable only with damping and gave no accuracy gain at 13 times the cost, so it is off by default. This is a faithful application of the TENG idea to the parabolic equation, not Bertram's own method or code; it should be reviewed with him.
- The duct measure comes from IGRA profiles (5 M-unit threshold), so about 75% of San Diego soundings count as ducts. Duct thickness and strength are noisy targets.
- Hourly labels do not exist: ducts are only observed at 00Z and 12Z, so "now-cast" is scored as the estimate made 0, 3, 6 or 9 h before a sounding.
- A model trained on San Diego did not transfer to Abu Dhabi (see `results/summary.md`); retrain per region. That is what Part C is for.
- NOAA's airport-weather archive lags by months; live runs use the National Weather Service feed instead, which is not covered by the held-out test.

## Layout

```
src/utr/   fetch, surface, quality (A) | refractivity, propagation, teng, coverage, nowcast* (B) | ship (C)
tests/     37 tests; a local web server stands in for NOAA
results/   generated metrics: summary.md (earlier prediction study), nowcast.md, coverage.md, figures
examples/  ship_demo (SYNTHETIC data and the report it produces), ship_template
docs/      OPERATOR_GUIDE.md, MVP.md
```
Data sources: NOAA NCEI IGRA v2 radiosondes; NOAA NCEI Integrated Surface Database (San Diego airport); NOAA NDBC (Scripps Pier); National Weather Service API. All public.
