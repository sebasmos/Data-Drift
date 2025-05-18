!pip install equiflow
import pandas as pd
from tableone import TableOne
import sklearn as sk
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error
from fairlearn.metrics import equalized_odds_ratio, demographic_parity_ratio
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
import shap
from tableone import tableone
from matplotlib import pyplot as plt
import pydata_google_auth
import seaborn as sns
import pandas_gbq
from equiflow import EquiFlow


warnings.filterwarnings("ignore")

credentials = pydata_google_auth.get_user_credentials(['https://www.googleapis.com/auth/cloud-platform'],)
project = 'capacheiv'
pandas_gbq.context.credentials = credentials

#importing data from eICU 1, 2015 and 2016
query = ""
for fn in ['icustays', 'apache_vars', 'apache_pt_results', 'sepsis']:
    with open(fn+'.sql', 'r') as file:
        query += file.read()
query += "select * from sepsis"
project = 'sccm-datathon-2024-participant'

gbq_data_15_16 = pd.read_gbq(query, dialect='standard', project_id=project)

EquiFlow._clean_missing = lambda self: None

eqfl = EquiFlow(
    data=gbq_data_15_16,
    initial_cohort_label='All Patients 2015-16',
    categorical=['gender', 'ethnicity'],  
    nonnormal=['temperature', 'respiratoryrate', 'heartrate', 'meanbp', 'wbc']
)

# Exclusion 1: Exclude patients with missing or unknown gender.
# (Adjust the condition if your data uses a different value for missing/unknown.)
eqfl.add_exclusion(
    mask=gbq_data_15_16.gender.notnull() & (gbq_data_15_16.gender != 'Unknown'),
    exclusion_reason='Missing/Unknown Gender',
    new_cohort_label='Known Gender'
)

# Exclusion 2: Exclude patients with missing or unknown ethnicity.
# (Again, adjust the condition based on your data.)
eqfl.add_exclusion(
    mask=gbq_data_15_16.ethnicity.notnull() & (gbq_data_15_16.ethnicity != 'Other/Unknown'),
    exclusion_reason='Missing/Unknown Ethnicity',
    new_cohort_label='Known Ethnicity'
)

# Generate the flow diagram
eqfl.plot_flows()