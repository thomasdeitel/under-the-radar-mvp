# MVP mapping and next steps

## Stated MVP (team deck, week 5)
Fast radar coverage update with a confidence indicator:
- Turns a new atmospheric reading into a radar coverage map in 20-40 minutes.
- Hybrid approach combining neural networks with numerical time-stepping (TENG) to update wave propagation solutions.
- Confidence indicator green, yellow or red; red means rerun the full simulation.
- Software on the ship, ordinary computers, training done ashore.
- Update rate target every 3 hours (to be confirmed).

## What exists
| Deck item | Status |
|---|---|
| Atmospheric reading to coverage map | Working on NOAA soundings: `utr coverage`, about 6 s per radial. Our own reference solver stands in for the Navy simulation |
| TENG-style hybrid | Working prototype in `teng.py`; matches the reference solver to about 0.4 dB. Not Bertram's code. Slower than a plain solver in this setting |
| Green / yellow / red | Working and tested on held-out data for the hourly estimate and for coverage staleness; see `results/` |
| Ordinary computers, training ashore | CPU only. Operator workflow `utr ship ...` learns from the operator's own data offline |
| 3-hour updates | Hourly estimate runs on whatever observations are newest; true 3-hour sounding data does not exist in public archives |

## Questions only the Navy or Bertram can answer (do not guess)
1. Is the TENG "time" in Bertram's method range (as used here) or real time as conditions change? What is his custom optimizer and can we use it?
2. What does the high-fidelity simulation take in and put out (which columns), so the operator guide can name them and the model can be trained on its outputs (`horizon_hours: 0` surrogate mode)?
3. What accuracy counts as good enough (dB, range, detection probability) so the light thresholds can be set to a requirement instead of percentiles?
4. Which radar frequencies, antenna heights and thresholds matter, so the placeholders can be replaced?
5. What shipboard observations exist at what cadence (sea temperature, humidity, soundings, refractivity sensors)?

## Next steps
1. Review the TENG-style solver and the interpretation of "time" with Bertram.
2. Send the operator guide and the synthetic demo to the Navy contact so they can run it when their data is ready (about 3 weeks).
3. Replace placeholders (frequency, beam, threshold) with the real radar parameters.
4. Time the Navy's simulator against this pipeline to test the speed claim.
5. Add forecast-grid inputs (GFS, ERA5, Copernicus) for locations without soundings.
