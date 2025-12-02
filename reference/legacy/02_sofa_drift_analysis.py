"""
SOFA Score Drift Analysis
Shows unequal performance drift across patient subgroups over time
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = r"C:\Users\sebastian.cajasordon\Desktop\AnpassenNN\mit-projects\model-drift\data"
OUTPUT_PATH = r"C:\Users\sebastian.cajasordon\Desktop\AnpassenNN\mit-projects\model-drift\output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(os.path.join(DATA_PATH, "turning_interval_frequency.csv"))
print(f"Loaded {len(df)} records")

# ============================================================
# PREPROCESSING
# ============================================================
# Convert outcome to binary (Deceased=1, Survivor=0)
df['outcome_binary'] = (df['outcome'] == 'Deceased').astype(int)

# Create age groups
df['age_group'] = pd.cut(df['admission_age'], 
                         bins=[0, 50, 65, 80, 200], 
                         labels=['<50', '50-65', '65-80', '80+'])

# Create care quartiles
df['care_quartile'] = pd.qcut(df['turning_interval_frequency'], 
                               q=4, 
                               labels=['Q1_High', 'Q2', 'Q3', 'Q4_Low'])

# Define year order for proper sorting
year_order = ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019']

print(f"\nAge group distribution:\n{df['age_group'].value_counts()}")
print(f"\nCare quartile distribution:\n{df['care_quartile'].value_counts()}")

# ============================================================
# METRICS FUNCTION
# ============================================================
def calculate_metrics(data, score_col='sofa', outcome_col='outcome_binary'):
    """Calculate SOFA prediction performance metrics."""
    if len(data) < 30:
        return None
    
    y_true = data[outcome_col].values
    y_score = data[score_col].values
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    
    # Skip if no variance in outcome
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    
    # Binary prediction using median threshold
    threshold = np.median(y_score)
    y_pred = (y_score >= threshold).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = None
    
    return {
        'AUC': auc,
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'N': len(y_true),
        'Mortality_Rate': y_true.mean(),
        'Mean_SOFA': y_score.mean()
    }

# ============================================================
# ANALYSIS 1: OVERALL PERFORMANCE BY YEAR
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 1: OVERALL SOFA PERFORMANCE BY YEAR")
print("="*60)

yearly_results = []
for year in year_order:
    subset = df[df['admission_year'] == year]
    metrics = calculate_metrics(subset)
    if metrics:
        metrics['Year'] = year
        yearly_results.append(metrics)
        print(f"{year}: AUC={metrics['AUC']:.3f}, N={metrics['N']}, Mortality={metrics['Mortality_Rate']:.1%}")

yearly_df = pd.DataFrame(yearly_results)

# ============================================================
# ANALYSIS 2: DRIFT BY RACE
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 2: SOFA PERFORMANCE BY RACE OVER TIME")
print("="*60)

race_results = []
for year in year_order:
    for race in ['White', 'Black', 'Asian', 'Hispanic', 'Other']:
        subset = df[(df['admission_year'] == year) & (df['race'] == race)]
        metrics = calculate_metrics(subset)
        if metrics:
            metrics['Year'] = year
            metrics['Race'] = race
            race_results.append(metrics)

race_df = pd.DataFrame(race_results)
if len(race_df) > 0:
    pivot = race_df.pivot(index='Year', columns='Race', values='AUC')
    print(pivot.round(3))

# ============================================================
# ANALYSIS 3: DRIFT BY GENDER
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 3: SOFA PERFORMANCE BY GENDER OVER TIME")
print("="*60)

gender_results = []
for year in year_order:
    for gender in df['gender'].unique():
        subset = df[(df['admission_year'] == year) & (df['gender'] == gender)]
        metrics = calculate_metrics(subset)
        if metrics:
            metrics['Year'] = year
            metrics['Gender'] = gender
            gender_results.append(metrics)

gender_df = pd.DataFrame(gender_results)
if len(gender_df) > 0:
    print(gender_df.pivot(index='Year', columns='Gender', values='AUC').round(3))

# ============================================================
# ANALYSIS 4: DRIFT BY CARE QUARTILE (Q1 vs Q4)
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 4: SOFA PERFORMANCE BY CARE QUARTILE")
print("="*60)

care_results = []
for year in year_order:
    for q in ['Q1_High', 'Q4_Low']:
        subset = df[(df['admission_year'] == year) & (df['care_quartile'] == q)]
        metrics = calculate_metrics(subset)
        if metrics:
            metrics['Year'] = year
            metrics['Care_Quartile'] = q
            care_results.append(metrics)

care_df = pd.DataFrame(care_results)
if len(care_df) > 0:
    print(care_df.pivot(index='Year', columns='Care_Quartile', values='AUC').round(3))

# ============================================================
# ANALYSIS 5: DRIFT BY AGE GROUP
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 5: SOFA PERFORMANCE BY AGE GROUP")
print("="*60)

age_results = []
for year in year_order:
    for age in ['<50', '50-65', '65-80', '80+']:
        subset = df[(df['admission_year'] == year) & (df['age_group'] == age)]
        metrics = calculate_metrics(subset)
        if metrics:
            metrics['Year'] = year
            metrics['Age_Group'] = age
            age_results.append(metrics)

age_df = pd.DataFrame(age_results)
if len(age_df) > 0:
    print(age_df.pivot(index='Year', columns='Age_Group', values='AUC').round(3))

# ============================================================
# CREATE VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("CREATING FIGURES...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('SOFA Score Performance Drift Analysis\nDoes Model Drift Affect All Subgroups Equally?', 
             fontsize=14, fontweight='bold')

# X-axis labels (shortened years)
x_labels = ['08-10', '11-13', '14-16', '17-19']

# Plot 1: Overall AUC over time
ax1 = axes[0, 0]
ax1.plot(range(len(yearly_df)), yearly_df['AUC'], 'o-', linewidth=2, markersize=10, color='#2c3e50')
ax1.fill_between(range(len(yearly_df)), yearly_df['AUC'], alpha=0.3, color='#2c3e50')
ax1.set_xticks(range(len(yearly_df)))
ax1.set_xticklabels(x_labels)
ax1.set_xlabel('Admission Year')
ax1.set_ylabel('AUC')
ax1.set_title('A. Overall SOFA Performance')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.55, 0.85])

# Plot 2: AUC by Race over time
ax2 = axes[0, 1]
colors_race = {'White': '#3498db', 'Black': '#e74c3c', 'Asian': '#2ecc71', 'Hispanic': '#9b59b6', 'Other': '#f39c12'}
for race in ['White', 'Black', 'Asian', 'Hispanic']:
    subset = race_df[race_df['Race'] == race]
    if len(subset) > 1:
        ax2.plot(range(len(subset)), subset['AUC'], 'o-', label=race, linewidth=2, 
                 markersize=8, color=colors_race.get(race, 'gray'))
ax2.set_xticks(range(4))
ax2.set_xticklabels(x_labels)
ax2.set_xlabel('Admission Year')
ax2.set_ylabel('AUC')
ax2.set_title('B. Performance by Race')
ax2.legend(loc='lower left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.55, 0.85])

# Plot 3: AUC by Gender over time
ax3 = axes[0, 2]
colors_gender = {'Male': '#3498db', 'Female': '#e74c3c'}
for gender in gender_df['Gender'].unique():
    subset = gender_df[gender_df['Gender'] == gender]
    ax3.plot(range(len(subset)), subset['AUC'], 'o-', label=gender, linewidth=2, 
             markersize=10, color=colors_gender.get(gender, 'gray'))
ax3.set_xticks(range(4))
ax3.set_xticklabels(x_labels)
ax3.set_xlabel('Admission Year')
ax3.set_ylabel('AUC')
ax3.set_title('C. Performance by Gender')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0.55, 0.85])

# Plot 4: Q1 vs Q4 Care Quartile over time
ax4 = axes[1, 0]
colors_care = {'Q1_High': '#27ae60', 'Q4_Low': '#e74c3c'}
labels_care = {'Q1_High': 'Q1 (High-freq care)', 'Q4_Low': 'Q4 (Low-freq care)'}
for q in ['Q1_High', 'Q4_Low']:
    subset = care_df[care_df['Care_Quartile'] == q]
    if len(subset) > 0:
        ax4.plot(range(len(subset)), subset['AUC'], 'o-', label=labels_care[q], linewidth=2, 
                 markersize=10, color=colors_care[q])
ax4.set_xticks(range(4))
ax4.set_xticklabels(x_labels)
ax4.set_xlabel('Admission Year')
ax4.set_ylabel('AUC')
ax4.set_title('D. Performance by Care Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0.55, 0.85])

# Plot 5: AUC by Age Group over time
ax5 = axes[1, 1]
colors_age = {'<50': '#3498db', '50-65': '#2ecc71', '65-80': '#f39c12', '80+': '#e74c3c'}
for age in ['<50', '50-65', '65-80', '80+']:
    subset = age_df[age_df['Age_Group'] == age]
    if len(subset) > 1:
        ax5.plot(range(len(subset)), subset['AUC'], 'o-', label=age, linewidth=2, 
                 markersize=8, color=colors_age.get(age, 'gray'))
ax5.set_xticks(range(4))
ax5.set_xticklabels(x_labels)
ax5.set_xlabel('Admission Year')
ax5.set_ylabel('AUC')
ax5.set_title('E. Performance by Age Group')
ax5.legend(loc='lower left', fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0.55, 0.85])

# Plot 6: Mortality Rate over time by subgroup
ax6 = axes[1, 2]
ax6.plot(range(len(yearly_df)), yearly_df['Mortality_Rate'], 'o-', linewidth=2, 
         markersize=10, color='#c0392b', label='Overall')
ax6.set_xticks(range(4))
ax6.set_xticklabels(x_labels)
ax6.set_xlabel('Admission Year')
ax6.set_ylabel('Mortality Rate')
ax6.set_title('F. Mortality Rate Over Time')
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0.2, 0.4])

plt.tight_layout()
output_file = os.path.join(OUTPUT_PATH, 'sofa_drift_analysis.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figure saved: {output_file}")

# ============================================================
# SAVE DATA TABLES
# ============================================================
yearly_df.to_csv(os.path.join(OUTPUT_PATH, 'yearly_performance.csv'), index=False)
race_df.to_csv(os.path.join(OUTPUT_PATH, 'race_performance.csv'), index=False)
gender_df.to_csv(os.path.join(OUTPUT_PATH, 'gender_performance.csv'), index=False)
care_df.to_csv(os.path.join(OUTPUT_PATH, 'care_quartile_performance.csv'), index=False)
age_df.to_csv(os.path.join(OUTPUT_PATH, 'age_performance.csv'), index=False)
print("✅ CSV tables saved to output folder")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*60)
print("SUMMARY: KEY FINDINGS")
print("="*60)

if len(yearly_df) >= 2:
    auc_change = yearly_df['AUC'].iloc[-1] - yearly_df['AUC'].iloc[0]
    print(f"Overall AUC change (first to last period): {auc_change:+.3f}")

print("\nDrift by subgroup (AUC change from 2008-2010 to 2017-2019):")

for name, results_df, col in [('Race', race_df, 'Race'), 
                               ('Gender', gender_df, 'Gender'),
                               ('Care Quartile', care_df, 'Care_Quartile'),
                               ('Age Group', age_df, 'Age_Group')]:
    print(f"\n{name}:")
    for group in results_df[col].unique():
        subset = results_df[results_df[col] == group]
        if len(subset) >= 2:
            first_auc = subset['AUC'].iloc[0]
            last_auc = subset['AUC'].iloc[-1]
            change = last_auc - first_auc
            print(f"  {group}: {first_auc:.3f} → {last_auc:.3f} ({change:+.3f})")

plt.show()
print("\n✅ Analysis complete!")