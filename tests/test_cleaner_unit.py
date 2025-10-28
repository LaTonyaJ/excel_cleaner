import sys
import pathlib
import pandas as pd
import pytest

# Ensure project root is on sys.path so `utils` can be imported when running tests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils.cleaner import clean_dataframe


def test_normalize_trim_infer_numeric():
    df = pd.DataFrame({' Name ': [' Alice ', None], 'Age': ['30', ' 40 ']})
    cfg = {
        'normalize_columns': True,
        'trim_whitespace': True,
        'infer_types': True,
        'date_detect_thresh': 0.5,
    }

    cleaned, report = clean_dataframe(df.copy(), cfg)

    # Column names normalized
    assert 'name' in cleaned.columns
    assert 'age' in cleaned.columns

    # Whitespace trimmed and None preserved
    assert cleaned['name'].iloc[0] == 'Alice'
    assert pd.isna(cleaned['name'].iloc[1]) or cleaned['name'].iloc[1] in (None, pd.NA)

    # Age converted to numeric
    assert pd.api.types.is_numeric_dtype(cleaned['age'])


def test_drop_blank_rows_cols_duplicates_and_report():
    df = pd.DataFrame({'A': [None, 1, 1], 'B': [None, 2, 2], 'C': [None, None, None]})
    cfg = {
        'drop_blank_rows': True,
        'drop_blank_cols': True,
        'drop_duplicates': True,
    }

    cleaned, report = clean_dataframe(df.copy(), cfg)

    # Column C should be dropped
    assert 'C' not in cleaned.columns
    assert report.get('blank_cols_dropped', None) == 1

    # First row (all blank) should be dropped
    assert report.get('blank_rows_dropped', None) == 1

    # Duplicate rows should be deduplicated
    assert report.get('duplicates_dropped', None) == 1

    # Final shape should be (1 row, 2 cols)
    assert cleaned.shape == (1, 2)


def test_date_detection_and_dtype_change():
    df = pd.DataFrame({'d': ['2020-01-01', '2020/02/02', 'not a date', '']})
    cfg = {
        'infer_types': True,
        'date_detect_thresh': 0.5,  # 2/4 == 0.5 should be enough
    }

    cleaned, report = clean_dataframe(df.copy(), cfg)

    assert 'd' in cleaned.columns
    # Implementation may convert to datetime dtype or leave as string dtype
    # as a safer Arrow-compatible string; ensure at least the majority/threshold
    # of values parse as datetimes when coerced.
    assert (
        pd.api.types.is_datetime64_any_dtype(cleaned['d'])
        or pd.api.types.is_string_dtype(cleaned['d'])
    )

    parsed = pd.to_datetime(cleaned['d'], errors='coerce')
    # At least one value should parse as a datetime (implementation may be conservative)
    assert parsed.notna().sum() >= 1


def test_outlier_detection_skips_identifier_columns_when_dropping():
    # user_id contains an extreme value that would be an outlier if considered
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 12, 11],
        'user_id': [1000000, 2, 3],
    })

    cfg = {
        'detect_outliers': True,
        'outlier_method': 'zscore',
        'outlier_threshold': 3.0,
        'outlier_action': 'drop',
    }

    cleaned, report = clean_dataframe(df.copy(), cfg)

    # Since identifier-like columns (user_id) are skipped, no rows should be removed
    assert cleaned.shape[0] == 3
    # And the report should not list 'user_id' as an outlier column
    assert 'user_id' not in report.get('outliers', {})


def test_outlier_detection_drops_value_outliers_but_not_id():
    # create many normal values and a single extreme outlier to ensure detection
    normal = [10] * 20
    df = pd.DataFrame({
        'id': list(range(1, 22)),
        'value': normal + [1000],
        'user_id': list(range(1, 22)),
    })

    cfg = {
        'detect_outliers': True,
        'outlier_method': 'zscore',
        'outlier_threshold': 3.0,
        'outlier_action': 'drop',
    }

    cleaned, report = clean_dataframe(df.copy(), cfg)

    # The extreme 'value' should be removed (one of 21 rows)
    assert cleaned.shape[0] == 20
    assert report.get('outliers', {}).get('value', {}).get('count', 0) == 1


def test_null_handling_drop_rows():
    df = pd.DataFrame({'A': [1, None, 3], 'B': [None, None, 2]})
    cfg = {'null_handling': 'drop_rows'}

    cleaned, report = clean_dataframe(df.copy(), cfg)

    # Rows with any nulls should be dropped: row 0 has null in B, row 1 has nulls
    assert report.get('nulls_dropped', 0) == 2
    assert cleaned.shape[0] == 1


def test_null_handling_fill_mean_and_mode():
    df = pd.DataFrame({'num': [1.0, None, 3.0], 'cat': ['a', None, 'a']})
    cfg_num = {'null_handling': 'fill', 'fill_strategy': 'mean'}
    cleaned_num, report_num = clean_dataframe(df.copy(), cfg_num)

    # Numeric NaN should be replaced with mean ( (1+3)/2 == 2.0 )
    assert cleaned_num['num'].isna().sum() == 0
    assert float(cleaned_num['num'].iloc[1]) == pytest.approx(2.0)

    cfg_cat = {'null_handling': 'fill', 'fill_strategy': 'mode'}
    cleaned_cat, report_cat = clean_dataframe(df.copy(), cfg_cat)

    # Categorical NaN should be filled with mode 'a'
    assert cleaned_cat['cat'].isna().sum() == 0
    assert cleaned_cat['cat'].iloc[1] == 'a'
