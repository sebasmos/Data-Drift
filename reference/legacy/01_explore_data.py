"""
Step 2: Explore the data structure
Run this first to understand what we're working with
"""
import pandas as pd
import os

DATA_PATH = r"C:\Users\sebastian.cajasordon\Desktop\AnpassenNN\mit-projects\model-drift\data"

# Load data
print("Loading data...")
turning = pd.read_csv(os.path.join(DATA_PATH, "turning_interval_frequency.csv"))
mouthcare = pd.read_csv(os.path.join(DATA_PATH, "mouthcare_interval_frequency.csv"))

# Convert outcome to binary (Deceased=1, Survivor=0)
turning['outcome_binary'] = (turning['outcome'] == 'Deceased').astype(int)
mouthcare['outcome_binary'] = (mouthcare['outcome'] == 'Deceased').astype(int)

print(f"\n{'='*60}")
print("TURNING DATASET")
print(f"{'='*60}")
print(f"Shape: {turning.shape[0]} rows, {turning.shape[1]} columns")
print(f"\nColumns: {turning.columns.tolist()}")
print(f"\nFirst 5 rows:\n{turning.head()}")

print(f"\n{'='*60}")
print("KEY VARIABLES")
print(f"{'='*60}")
print(f"Years available: {sorted(turning['admission_year'].unique())}")
print(f"Races: {turning['race'].unique().tolist()}")
print(f"Genders: {turning['gender'].unique().tolist()}")
print(f"Outcome values: {turning['outcome'].unique().tolist()}")
print(f"SOFA range: {turning['sofa'].min()} to {turning['sofa'].max()}")

print(f"\n{'='*60}")
print("OUTCOME DISTRIBUTION")
print(f"{'='*60}")
print(turning['outcome'].value_counts())
print(f"\nMortality rate: {turning['outcome_binary'].mean():.1%}")

print(f"\n{'='*60}")
print("PATIENTS BY YEAR")
print(f"{'='*60}")
print(turning.groupby('admission_year').agg({
    'stay_id': 'count',
    'outcome_binary': 'mean'
}).rename(columns={'stay_id': 'N_patients', 'outcome_binary': 'Mortality_rate'}))

print(f"\n{'='*60}")
print("PATIENTS BY RACE")
print(f"{'='*60}")
print(turning.groupby('race').agg({
    'stay_id': 'count',
    'outcome_binary': 'mean'
}).rename(columns={'stay_id': 'N_patients', 'outcome_binary': 'Mortality_rate'}))

print(f"\n{'='*60}")
print("NUMERIC SUMMARIES")
print(f"{'='*60}")
print(turning[['sofa', 'admission_age', 'turning_interval_frequency', 'days', 'cci']].describe())

print(f"\n{'='*60}")
print("MOUTHCARE DATASET")
print(f"{'='*60}")
print(f"Shape: {mouthcare.shape[0]} rows, {mouthcare.shape[1]} columns")
print(f"Mortality rate: {mouthcare['outcome_binary'].mean():.1%}")