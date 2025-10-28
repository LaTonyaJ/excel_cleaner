import sys
import pathlib
import pandas as pd

# Ensure project root is on sys.path so `utils` can be imported when running this script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils.cleaner import clean_dataframe

if __name__ == '__main__':
    df = pd.read_csv('assets/cleaned_sample.csv')
    cfg = {
        'trim_whitespace': True,
        'drop_blank_rows': True,
        'drop_blank_cols': True,
        'normalize_columns': True,
        'infer_types': True,
        'drop_duplicates': True,
        'date_detect_thresh': 0.5,
    }
    cleaned, report = clean_dataframe(df, cfg)
    print('Dtypes after cleaning:')
    print(cleaned.dtypes)
    print('\nSample rows:')
    print(cleaned.head())
    print('\nReport:')
    print(report)
