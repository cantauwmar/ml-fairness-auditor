import os
import sys

import pandas as pd

try:
    from folktables import ACSDataSource, ACSIncome
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: folktables.\n"
        "Install it first (recommended): pip install -r requirements.txt\n"
        "Then re-run: python data/download_acs_income.py"
    ) from exc

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'acs_income.csv.gz')

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=None, download=True)  # all 50 states + PR

features, label, group = ACSIncome.df_to_pandas(acs_data)
df = features.copy()
df['PINCP'] = label  # income target

RENAME = {
    'AGEP': 'age',
    'COW':  'workclass',
    'SCHL': 'education-num',
    'MAR':  'marital-status',
    'OCCP': 'occupation',
    'POBP': 'native-country',
    'RELP': 'relationship',
    'WKHP': 'hours-per-week',
    'SEX':  'sex',
    'RAC1P': 'race',
    'PINCP': 'income',
}
df = df.rename(columns=RENAME)
df['income'] = df['income'].map({True: '>50K', False: '<=50K'})

df.to_csv(OUTPUT_PATH, index=False, compression='gzip')
print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
