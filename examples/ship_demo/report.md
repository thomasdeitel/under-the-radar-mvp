# What the model learned from your data

- Trained 2026-10-27T00:33:36Z on 504 time steps from 2026-01-01 00:00:00 to 2026-01-21 23:00:00 (21 days).
- Data check: **OK**
- Mode: forecast 3.0 h ahead

## Held-out test (the newest data, never used for fitting, model choice or light cut-offs)

| Target | Examples (train/val/test) | Model | Simple baseline | Skill vs baseline | Learned models used |
|---|---|---|---|---|---|
| evaporation_duct_height_m | 101/33/33 | RMSE 1.67 | RMSE 3.46 (last value) | +0.77 | ridge, gbm |

Skill is 1 minus (model error / baseline error); positive means the model is better than the baseline. With only a few weeks of data the test block is small, so treat these as a first read, not a verdict.

## Do the lights mean anything on your data?

| Target | Green error (n) | Yellow error (n) | Red error (n) |
|---|---|---|---|
| evaporation_duct_height_m | 1.07 (9) | 1.91 (7) | 1.31 (17) |

Error should rise from green to red. If it does not, the lights are not yet trustworthy; more data will fix that.

## Training history

- 2026-10-27T00:33:36Z: 504 steps (2026-01-01 00:00:00 to 2026-01-21 23:00:00); test skill evaporation_duct_height_m +0.77
