# Code Organization

All analysis code is organized by dataset. Each dataset has its own folder with two main scripts.

## Structure

```
code/
├── config.py                    # Global configuration (all datasets)
├── 01_explore_data.py           # Generic exploration script
├── 02_drift_analysis.py         # Generic drift analysis script
│
├── mimic/                       # ✅ MIMIC dataset (COMPLETE)
│   ├── 01_explore_data.py       # MIMIC-specific exploration
│   └── 02_drift_analysis.py     # MIMIC-specific drift analysis
│
├── eicu/                        # ⚠️ eICU dataset (TODO: Emma)
│   ├── 01_explore_data.py       # Placeholder - Emma to implement
│   └── 02_drift_analysis.py     # Placeholder - Emma to implement
│
├── chinese/                     # 🔜 Chinese ICU (TODO: Ziyue - Dec 10)
│   ├── 01_explore_data.py       # Placeholder - Ziyue to implement
│   └── 02_drift_analysis.py     # Placeholder - Ziyue to implement
│
└── amsterdam/                   # 🔜 Amsterdam ICU (TODO: TBD)
    ├── 01_explore_data.py       # Placeholder - TBD to implement
    └── 02_drift_analysis.py     # Placeholder - TBD to implement
```

## Scripts

### Root Level (Generic)

**`config.py`** - Configuration for all datasets
- Dataset paths and column mappings
- Analysis parameters (age bins, figure settings, etc.)
- Switch datasets by changing `ACTIVE_DATASET`

**`01_explore_data.py`** - Generic data exploration
- Works with any dataset configured in `config.py`
- Validates data structure
- Shows distributions

**`02_drift_analysis.py`** - Generic drift analysis
- Works with any dataset configured in `config.py`
- Analyzes subgroup-specific drift
- Generates visualizations

### Dataset-Specific Folders

Each dataset folder contains customized versions of the analysis scripts.

**Use dataset-specific scripts when:**
- Dataset requires special preprocessing
- Custom analysis needed
- Different visualization requirements

**Use generic scripts when:**
- Standard analysis is sufficient
- Dataset follows common structure

## Usage

### Option 1: Use Generic Scripts (Recommended for new datasets)

```bash
# Edit config.py to select dataset
# ACTIVE_DATASET = 'mimic'

# Run generic scripts
python code/01_explore_data.py
python code/02_drift_analysis.py
```

### Option 2: Use Dataset-Specific Scripts

```bash
# Run MIMIC-specific analysis
python code/mimic/01_explore_data.py
python code/mimic/02_drift_analysis.py

# Run eICU-specific analysis (once Emma implements)
python code/eicu/01_explore_data.py
python code/eicu/02_drift_analysis.py
```

## For New Team Members

### If you're adding a new dataset:

1. **Add data** to `data/<your_dataset>/`
2. **Configure** in `code/config.py`
3. **Start with generic scripts:**
   ```bash
   python code/01_explore_data.py
   python code/02_drift_analysis.py
   ```
4. **If needed, customize:**
   - Copy generic scripts to `code/<your_dataset>/`
   - Modify for dataset-specific needs
   - Update this README

### Current Assignments:

| Dataset | Owner | Status | Scripts Location |
|---------|-------|--------|------------------|
| MIMIC | Sebastian | ✅ Complete | `code/mimic/` |
| eICU | Emma | ⚠️ Pending SOFA | `code/eicu/` (placeholders) |
| Chinese ICU | Ziyue | 🔜 Dec 10 | `code/chinese/` (placeholders) |
| Amsterdam | TBD | 🔜 Pending | `code/amsterdam/` (placeholders) |

## Placeholders

Placeholder files (`code/<dataset>/*.py`) show:
- What needs to be implemented
- Who is responsible
- Reference to working example (`code/mimic/`)

**To implement a placeholder:**
1. Open the placeholder file
2. Copy code from `code/mimic/` equivalent
3. Adapt for your dataset
4. Remove the TODO message
5. Test with your data
