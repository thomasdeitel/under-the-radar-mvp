# MVP mapping and next steps

## Stated MVP (team deck): Automated Model Data Collection and Loading
- **What it does:** automates downloading atmospheric data and moving it to a shared folder.
- **How:** scheduled program downloads new data, checks for missing/incomplete files, loads to the shared drive.
- **Problem addressed:** manual downloads (sometimes once a day) and no automatic warnings for incomplete data.
- **Goal:** less manual work and fewer loading errors without changing existing simulation methods.

| Deck statement | In this repo |
|---|---|
| Scheduled download | `utr fetch` (cron/launchd/Task Scheduler) |
| Check for missing/incomplete files | `utr fetch` size/zip verification; `utr check` STALE/MISSING/INCOMPLETE warnings |
| Load into shared drive | atomic publish via `--shared-dir` |
| Don't change simulation methods | model only reads data and predicts; nothing touches the simulator |
| Prediction improvement (team direction: test current predictions, a neural network adjusts them) | `utr evaluate`, NN vs persistence/climatology/GBM |

## What this prototype uses as a stand-in
NOAA IGRA radiosondes stand in for the shipboard model data, and "persistence" stands in for the current prediction. Both should be replaced with real inputs when available.

## Next steps
1. Get a sample of the actual model output/ground truth the current system produces, so the NN corrects the real predictions instead of a proxy.
2. Time the existing simulator vs this model to test the "faster" claim.
3. Add forecast-grid inputs (NOAA GFS, ERA5 or Copernicus) and hourly cadence if bandwidth allows.
4. Handle thickness better (target transform, more vertical detail from model levels).
5. Extend toward radar propagation (the team's plan to build on the TENG method).
6. Confirm shipboard constraints (standalone laptops, compute limits, legacy integration) before choosing the deployment form.
