# Under the Radar — MVP

Automated NOAA atmospheric data collection + a small neural-network correction model for predicting
**radar ducting** (trapping layers in modified refractivity) from radiosonde soundings.

Navy 215 "Under the Radar" (i4NS). MVP goal: *reduce manual work and data-loading errors, and show an
improvement over the current prediction, without changing the existing simulation methods.*

## What it does

| Piece | What | Command |
|---|---|---|
| **1. Automated data loading** | Scheduled download of NOAA NCEI IGRA v2 radiosonde archives. Conditional download (ETag/Last-Modified), size + zip integrity verification, sha256 manifest, atomic publish into a shared folder (no half-written files). | `python -m utr fetch` |
| **2. Automatic warnings** | Flags STALE (newest sounding too old), MISSING (gaps in the 00Z/12Z series) and INCOMPLETE (soundings that fail QC) data. Exit code 0 = OK, 1 = WARN, 2 = FAIL, so schedulers/alerts can act on it. | `python -m utr check` |
| **3. Prediction model + honest evaluation** | From the latest sounding, predicts duct presence / strength / thickness at the next sounding (12 h, 24 h ahead). A 5-member MLP ensemble is compared on a held-out time period against persistence (stand-in for "keep using the old prediction"), seasonal climatology, and gradient boosting. | `python -m utr evaluate` |

Refractivity physics: N = 77.6 P/T + 3.73e5 e/T²; modified refractivity M = N + 0.157 z; a trapping layer is where dM/dz < 0
(strength threshold 5 M-units, computed below 3 km AGL).

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m utr fetch                 # San Diego (72293) + Abu Dhabi stress station; add --shared-dir /path/to/shared
python -m utr check                 # prints warnings, writes results/data_quality.json
python -m utr evaluate              # ~1 min; writes results/metrics.json, summary.md, skill.png
pytest -q                           # 16 tests, no network needed
```

Scheduled use (example, every 3 h): `0 */3 * * * cd /path/to/repo && PYTHONPATH=src python -m utr fetch --shared-dir /shared/models && PYTHONPATH=src python -m utr check || mail -s "data warning" you@example.com`

## Results (test period 2022 onward, never used for training)

Skill = 1 − error(model)/error(reference); positive is better. Full tables with confidence intervals: `results/summary.md`; figure: `results/skill.png`.

San Diego, 12 h lead:

| Target | NN skill vs persistence | NN skill vs climatology |
|---|---|---|
| Duct present | +0.42 | +0.20 |
| Duct strength | +0.10 | +0.07 |
| Duct thickness | +0.20 | −0.21 |

Takeaways:
- The NN clearly beats "keep using the previous state" on every target.
- Versus a seasonal average it helps for duct presence (and slightly for strength) but is **worse for thickness**.
- A San Diego–trained model does **not transfer** to Abu Dhabi; it must be retrained per region (it is cheap).
- Ensemble spread is a usable confidence signal (higher spread → larger error; green/yellow/red terciles in the report).
- Inference ≈ 9 ms per sounding on CPU.

## Limitations (please read)

- **Speed vs the Navy simulator is not tested.** Only this model's latency is measured.
- Ground truth here is *derived from IGRA soundings*, not from the Navy's high-fidelity simulation or measured radar propagation. "Persistence" is a stand-in for the current system, not the system itself.
- IGRA's vertical resolution is coarse; thin evaporation ducts are not observable.
- Targets are the next sounding at the same station (00Z/12Z), not gridded forecasts. Real forecast inputs (GFS/ERA5/Copernicus) are the natural next step.
- Data freshness depends on NOAA's IGRA update cadence (the check will warn when it lags).

See `docs/MVP.md` for how this maps to the project MVP and next steps.

## Layout

```
src/utr/  config, fetch, quality, parse, refractivity, features, dataset, model, experiments, cli
tests/    pytest suite (local HTTP server stands in for NOAA)
results/  generated metrics, summary, figure, data-quality report
docs/     MVP mapping and next steps
```
Data source: NOAA NCEI Integrated Global Radiosonde Archive v2 (public).
