# eICU Dataset TODO

**Owner:** Emma
**Deadline:** TBD

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Review SOFA code: https://github.com/nus-mornin-lab/oxygenation_kc | ⬜ |
| 2 | Compute SOFA for `sepsis_adult_eicu_v1.csv` | ⬜ |
| 3 | Compute SOFA for `sepsis_adult_eicu_v2.csv` | ⬜ |
| 4 | Update `code/config.py` → set `has_precomputed_sofa: True` | ⬜ |
| 5 | Run `python code/01_explore_data.py` (set `ACTIVE_DATASET='eicu_v1'`) | ⬜ |
| 6 | Run `python code/02_drift_analysis.py` | ⬜ |
| 7 | Share results on WhatsApp | ⬜ |

## Placeholder Scripts

Placeholder scripts are ready in `code/eicu/`:
- `code/eicu/01_explore_data.py` - Copy from `code/mimic/` and adapt
- `code/eicu/02_drift_analysis.py` - Copy from `code/mimic/` and adapt

## Help

- Reference code: `code/mimic/` (working example)
- SOFA SQL: `reference/sql/`
- Questions: Ask Sebastian or Ziyue
