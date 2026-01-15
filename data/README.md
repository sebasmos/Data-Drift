# Data Directory

Place your CSV files in the corresponding folders below.

## Folder Structure

```
data/
├── mimic_combined/
│   └── mimic_combined.csv      ← Place file here
├── eicu_combined/
│   └── eicu_combined.csv       ← Place file here
├── saltz/
│   └── saltz_data.csv          ← Place file here
├── zhejiang/
│   └── zhejiang_data.csv       ← Place file here
└── mimic_iv_lc/
    ├── mouthcare_interval_frequency.csv
    └── turning_interval_frequency.csv
```

## Dataset Summary

| Folder | CSV Filename | N | Period | Source |
|--------|--------------|---|--------|--------|
| `mimic_combined/` | `mimic_combined.csv` | 112,468 | 2001-2022 | [PhysioNet MIMIC](https://physionet.org/content/mimiciv/) |
| `eicu_combined/` | `eicu_combined.csv` | 661,358 | 2014-2021 | [PhysioNet eICU](https://physionet.org/content/eicu-crd/) |
| `saltz/` | `saltz_data.csv` | 27,259 | 2013-2021 | [AmsterdamUMCdb](https://amsterdammedicaldatascience.nl/) |
| `zhejiang/` | `zhejiang_data.csv` | 7,932 | 2011-2022 | Private |
| `mimic_iv_lc/` | `mouthcare_*.csv`, `turning_*.csv` | 17,594 | 2008-2019 | [PhysioNet MIMIC](https://physionet.org/content/mimiciv/) |

## Required Columns

### mimic_combined.csv

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Patient age |
| `gender` | str | M/F |
| `ethnicity` | str | Race/ethnicity |
| `death_hosp` | int | Hospital mortality (0/1) |
| `anchor_year_group` | str | Time period (e.g., "2001 - 2008") |
| `sofa` | float | SOFA score |
| `oasis` | float | OASIS score |
| `sapsii` | float | SAPS-II score |
| `apsiii` | float | APS-III score |

### eicu_combined.csv

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Patient age |
| `gender` | str | Male/Female |
| `ethnicity` | str | Race/ethnicity |
| `hosp_mort` | int | Hospital mortality (0/1) |
| `hospitaldischargeyear` | int | Discharge year |
| `sofa` | float | SOFA score |
| `oasis` | float | OASIS score |
| `sapsii` | float | SAPS-II score |
| `apsiii` | float | APS-III score |
| `apachescore` | float | APACHE score |

### saltz_data.csv

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Patient age |
| `gender` | str | M/F |
| `death_hosp` | int | Hospital mortality (0/1) |
| `anchor_year_group` | int | Year (e.g., 2013) |
| `sofa` | float | SOFA score |
| `oasis` | float | OASIS score |
| `sapsii` | float | SAPS-II score |
| `apsiii` | float | APS-III score |

*Note: No race/ethnicity column (European dataset)*

### zhejiang_data.csv

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Patient age |
| `gender` | str | M/F |
| `death_hosp` | int | Hospital mortality (0/1) |
| `anchor_year_group` | str | Time period (e.g., "2011 - 2013") |
| `sofa` | float | SOFA score |
| `oasis` | float | OASIS score |
| `sapsii` | float | SAPS-II score |
| `apsiii` | float | APS-III score |

*Note: No race/ethnicity column (Asian dataset)*

### mimic_iv_lc/ (Care Phenotypes)

| Column | Type | Description |
|--------|------|-------------|
| `admission_age` | int | Patient age |
| `gender` | str | M/F |
| `race` | str | Race/ethnicity |
| `outcome` | int | Hospital mortality (0/1) |
| `admission_year` | int | Year |
| `sofa` | float | SOFA score |
| `mouthcare_interval_frequency` or `turning_interval_frequency` | str | Care quartile (Q1-Q4) |

## Creating Combined Datasets

If you have raw MIMIC-III/IV or eICU files, run:

```bash
# Merge MIMIC-III + MIMIC-IV → mimic_combined.csv
python code/merge_mimic.py

# Merge eICU files → eicu_combined.csv
python code/merge_eicu.py
```

## Data Access

| Dataset | Access |
|---------|--------|
| MIMIC | PhysioNet credentialing required → [physionet.org](https://physionet.org/) |
| eICU | PhysioNet credentialing required → [physionet.org](https://physionet.org/) |
| AmsterdamUMCdb | Data use agreement → [amsterdammedicaldatascience.nl](https://amsterdammedicaldatascience.nl/) |
| Zhejiang | Private (contact authors) |

## Verify Setup

```bash
python code/batch_analysis.py --fast
```

If successful, you'll see output for each dataset.
