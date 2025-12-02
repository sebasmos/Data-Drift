# Amsterdam ICU Dataset TODO

**Owner:** TBD
**Deadline:** TBD

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Confirm data availability | ⬜ |
| 2 | Obtain permissions/IRB | ⬜ |
| 3 | Prepare CSV: `data/amsterdam/amsterdam_icu_data.csv` | ⬜ |
| 4 | Compute SOFA scores (if needed) | ⬜ |
| 5 | Update `code/config.py` → `amsterdam_icu` entry | ⬜ |
| 6 | Run `python code/01_explore_data.py` | ⬜ |
| 7 | Run `python code/02_drift_analysis.py` | ⬜ |

## Placeholder Scripts

Placeholder scripts are ready in `code/amsterdam/`:
- `code/amsterdam/01_explore_data.py` - Copy from `code/mimic/` and adapt
- `code/amsterdam/02_drift_analysis.py` - Copy from `code/mimic/` and adapt

## Help

- Reference code: `code/mimic/` (working example)
- Template: See `data/chinese/TODO.md`
- Questions: Team meeting
