# Zhejiang ICU Dataset TODO

**Owner:** Ziyue
**Deadline:** December 10, 2025

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Prepare dataset CSV file | ⬜ |
| 2 | Save to `data/zhejiang/zhejiang_icu_data.csv` | ⬜ |
| 3 | Ensure SOFA scores computed (or compute them) | ⬜ |
| 4 | Document column names (outcome, year, gender, age, etc.) | ⬜ |
| 5 | Update `code/config.py` → `zhejiang` entry with column mappings | ⬜ |
| 6 | Run `python code/01_explore_data.py` (set `ACTIVE_DATASET='zhejiang'`) | ⬜ |
| 7 | Run `python code/02_drift_analysis.py` | ⬜ |
| 8 | Share results on WhatsApp | ⬜ |

## Placeholder Scripts

Placeholder scripts are ready in `code/zhejiang/`:
- `code/zhejiang/01_explore_data.py` - Copy from `code/mimic/` and adapt
- `code/zhejiang/02_drift_analysis.py` - Copy from `code/mimic/` and adapt

## Help

- Reference code: `code/mimic/` (working example)
- Config template: See `mimic` entry in `code/config.py`
- SOFA code: `reference/sql/` or https://github.com/nus-mornin-lab/oxygenation_kc
- Questions: Ask Sebastian
