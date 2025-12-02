# Subgroup-Specific Drift in Clinical Prediction Models

[![LICENSE](https://img.shields.io/badge/license-CC%20BY--NC--SA-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/Data-Drift)

> **TL;DR:** Does model drift affect all patient subgroups equally? We analyze SOFA score performance across demographic groups in ICU patients to test if some subgroups experience faster degradation than others.

---

## 🚀 Quick Start

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Setup
uv venv
source .venv/bin/activate  # macOS/Linux or .venv\Scripts\activate (Windows)
uv pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run Analysis

**Option 1: Use generic scripts (recommended)**
```bash
# 1. Configure dataset in code/config.py
# ACTIVE_DATASET = 'mimic'

# 2. Run generic scripts
python code/01_explore_data.py
python code/02_drift_analysis.py

# 3. View results
open output/mimic/mimic_drift_analysis.png
```

**Option 2: Use dataset-specific scripts**
```bash
# Run MIMIC-specific scripts
python code/mimic/01_explore_data.py
python code/mimic/02_drift_analysis.py
```

---

## 📊 Datasets

| Dataset | Status | Owner | Location | TODO |
|---------|--------|-------|----------|------|
| `mimic` | ✅ Ready | Sebastian | `data/mimic/` | - |
| `mimic_mouthcare` | ✅ Ready | Sebastian | `data/mimic/` | Run analysis |
| `eicu_v1` | ⚠️ Needs SOFA | Emma | `data/eicu/` | [TODO](data/eicu/TODO.md) |
| `eicu_v2` | ⚠️ Needs SOFA | Emma | `data/eicu/` | [TODO](data/eicu/TODO.md) |
| `chinese_icu` | 🔜 Dec 10 | Ziyue | `data/chinese/` | [TODO](data/chinese/TODO.md) |
| `amsterdam_icu` | 🔜 Pending | TBD | `data/amsterdam/` | [TODO](data/amsterdam/TODO.md) |

---

## 📂 Project Structure

```
Data-Drift/
├── code/                           # All analysis code (organized by dataset)
│   ├── config.py                   # Global configuration
│   ├── 01_explore_data.py          # Generic exploration script
│   ├── 02_drift_analysis.py        # Generic drift analysis script
│   ├── README.md                   # Code organization guide
│   │
│   ├── mimic/                      # ✅ MIMIC-specific code (COMPLETE)
│   │   ├── 01_explore_data.py
│   │   └── 02_drift_analysis.py
│   │
│   ├── eicu/                       # ⚠️ eICU placeholders (Emma)
│   │   ├── 01_explore_data.py      # TODO: Implement
│   │   └── 02_drift_analysis.py    # TODO: Implement
│   │
│   ├── chinese/                    # 🔜 Chinese ICU placeholders (Ziyue)
│   │   ├── 01_explore_data.py      # TODO: Implement
│   │   └── 02_drift_analysis.py    # TODO: Implement
│   │
│   └── amsterdam/                  # 🔜 Amsterdam placeholders (TBD)
│       ├── 01_explore_data.py      # TODO: Implement
│       └── 02_drift_analysis.py    # TODO: Implement
│
├── data/                           # All datasets
│   ├── mimic/                      ✅ Data files + README
│   ├── eicu/                       ⚠️ Data files + TODO.md
│   ├── chinese/                    🔜 TODO.md
│   └── amsterdam/                  🔜 TODO.md
│
├── output/                         # Results (auto-generated)
│   ├── mimic/
│   └── ...
│
└── reference/                      # Reference only (SQL, notebooks, old code)
    ├── sql/
    ├── notebooks/
    ├── legacy/
    └── archive/
```

### Code Organization

**Each dataset has its own folder in `code/`:**
- `code/mimic/` - Complete MIMIC analysis scripts ✅
- `code/eicu/` - Placeholder scripts for Emma ⚠️
- `code/chinese/` - Placeholder scripts for Ziyue 🔜
- `code/amsterdam/` - Placeholder scripts for TBD 🔜

**See [code/README.md](code/README.md) for details on code organization.**

---

## 🔬 Methodology

### SOFA Score

**SOFA (Sequential Organ Failure Assessment)** evaluates 6 organ systems:
- Respiratory, Cardiovascular, Renal, Coagulation, Liver, Neurological
- **Range:** 0-24 (higher = worse)

### Analysis

1. **01_explore_data.py** - Validates data, shows distributions
2. **02_drift_analysis.py** - Analyzes drift across:
   - Overall performance by time
   - By race, gender, age, care frequency
   - Generates multi-panel figure + CSV tables

---

## 📈 Outputs

Each dataset produces in `output/<dataset>/`:

- `<dataset>_drift_analysis.png` - Multi-panel visualization
- `<dataset>_yearly_performance.csv` - Overall metrics
- `<dataset>_race_performance.csv` - Race-stratified metrics
- `<dataset>_gender_performance.csv` - Gender-stratified metrics
- `<dataset>_care_performance.csv` - Care frequency metrics
- `<dataset>_age_performance.csv` - Age group metrics

---

## 🛠️ Configuration

Edit `code/config.py` to switch datasets or customize parameters:

```python
# Switch dataset
ACTIVE_DATASET = 'mimic'  # Options: 'mimic', 'mimic_mouthcare', 'eicu_v1', etc.

# Customize analysis
ANALYSIS_CONFIG = {
    'min_sample_size': 30,
    'age_bins': [0, 50, 65, 80, 200],
    'age_labels': ['<50', '50-65', '65-80', '80+'],
    'figure_dpi': 300,
}
```

---

## 📝 Citation

```bibtex
@software{data_drift_2025,
  title={Subgroup-Specific Drift in Clinical Prediction Models},
  author={Hamza and Xiaoli and Celi, Leo Anthony and Cajas Ord{\'o}{\~n}ez, Sebasti{\'a}n Andr{\'e}s},
  year={2025},
  url={https://github.com/HamzaNabulsi/Data-Drift}
}
```

See [CITATION.cff](CITATION.cff) for full metadata.

---

## 🔗 Resources

- **SOFA Code:** `reference/sql/` or https://github.com/nus-mornin-lab/oxygenation_kc
- **SOFA Reference:** https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score

---

## ⚖️ License

[![LICENSE](https://img.shields.io/badge/license-CC%20BY--NC--SA-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 🔄 Status

- ✅ MIMIC analyzed
- ⚠️ eICU needs SOFA computation (Emma)
- 🔜 Chinese ICU (Ziyue - Dec 10)
- 🔜 Amsterdam ICU (TBD)
- 📊 Additional metrics in development
