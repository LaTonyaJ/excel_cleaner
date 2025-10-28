import re
from typing import Tuple, Dict, Any

import pandas as pd


def _normalize_column_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "", name)
    return name.lower()


def clean_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans a DataFrame according to config and returns (cleaned_df, report).

    Config keys:
      - trim_whitespace: bool
      - drop_duplicates: bool
      - drop_blank_rows: bool
      - drop_blank_cols: bool
      - normalize_columns: bool
      - infer_types: bool
      - date_detect_thresh: float (0-1)

    Report contains counts and column/dtype changes.
    """
    report = {}
    original_shape = df.shape
    report['original_shape'] = original_shape

    # Normalize column names
    col_renames = {}
    if config.get('normalize_columns'):
        new_cols = [_normalize_column_name(c) for c in df.columns]
        for old, new in zip(df.columns, new_cols):
            if old != new:
                col_renames[old] = new
        df.columns = new_cols
    report['col_renames'] = col_renames

    # Trim whitespace in object columns
    if config.get('trim_whitespace'):
        obj_cols = df.select_dtypes(include=['object']).columns
        for c in obj_cols:
            # guard: skip if all null
            try:
                df[c] = df[c].astype(str).where(df[c].notna(), None)
                df[c] = df[c].str.strip()
            except Exception:
                # leave column unchanged on failure
                pass

    # NULL handling: drop or fill nulls per configuration
    # Config keys (optional): null_handling: 'none'|'drop_rows'|'fill'
    # If 'fill' then fill_strategy: 'mean'|'median'|'mode'|'constant' and fill_constant used when strategy == 'constant'
    null_handling = config.get('null_handling', 'none')
    report['nulls_dropped'] = 0
    report['nulls_filled'] = {}
    if null_handling == 'drop_rows':
        before = len(df)
        df = df.dropna(how='any')
        report['nulls_dropped'] = before - len(df)
    elif null_handling == 'fill':
        strategy = config.get('fill_strategy', 'mode')
        const = config.get('fill_constant', None)
        filled_counts = {}
        for c in df.columns:
            col = df[c]
            na_count = col.isna().sum()
            if na_count == 0:
                continue
            try:
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(col):
                    val = col.mean()
                    df[c] = col.fillna(val)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(col):
                    val = col.median()
                    df[c] = col.fillna(val)
                elif strategy == 'mode':
                    mode_vals = col.mode(dropna=True)
                    if not mode_vals.empty:
                        val = mode_vals.iloc[0]
                        df[c] = col.fillna(val)
                    else:
                        df[c] = col.fillna('') if col.dtype == object else col.fillna(0)
                elif strategy == 'constant':
                    df[c] = col.fillna(const)
                else:
                    # Fallback: fill with empty string for objects, 0 for numerics
                    df[c] = col.fillna('') if col.dtype == object else col.fillna(0)

                filled_counts[c] = na_count
            except Exception:
                # leave as-is on failure
                continue

        report['nulls_filled'] = filled_counts

    # Drop completely blank rows
    if config.get('drop_blank_rows'):
        before = len(df)
        df = df.dropna(how='all')
        report['blank_rows_dropped'] = before - len(df)
    else:
        report['blank_rows_dropped'] = 0

    # Drop completely blank columns
    if config.get('drop_blank_cols'):
        before = df.shape[1]
        df = df.dropna(axis=1, how='all')
        report['blank_cols_dropped'] = before - df.shape[1]
    else:
        report['blank_cols_dropped'] = 0

    # Drop duplicates
    if config.get('drop_duplicates'):
        before = len(df)
        df = df.drop_duplicates()
        report['duplicates_dropped'] = before - len(df)
    else:
        report['duplicates_dropped'] = 0

    # Infer numeric/date types
    dtype_changes = {}
    if config.get('infer_types'):
        thresh = float(config.get('date_detect_thresh', 0.5))
        for c in df.columns:
            series = df[c]
            old_dtype = str(series.dtype)

            # skip if empty
            if series.dropna().empty:
                continue

            # Try numeric
            try:
                conv = pd.to_numeric(series, errors='coerce')
                non_null = conv.notna().sum()
                if non_null / max(1, len(series)) >= 0.9:
                    df[c] = conv
                    dtype_changes[c] = {'from': old_dtype, 'to': 'numeric'}
                    continue
            except Exception:
                pass

            # Try datetime
            try:
                # Only attempt to parse as datetime when many values look date-like
                non_na = series.dropna().astype(str)
                if not non_na.empty:
                    # date-like = contains separators like / - . or month names
                    date_like_count = non_na.str.contains(r'[/\\-.]|[A-Za-z]{3,}', regex=True).sum()
                    if date_like_count / max(1, len(non_na)) < thresh:
                        # not enough date-like values; skip datetime detection
                        continue

                conv = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
                non_null = conv.notna().sum()
                if non_null / max(1, len(series)) >= thresh:
                    df[c] = conv
                    dtype_changes[c] = {'from': old_dtype, 'to': 'datetime'}
                    continue
            except Exception:
                pass

    report['dtype_changes'] = dtype_changes

    # Ensure DataFrame is Arrow-compatible for downstream consumers (Streamlit/pyarrow)
    # Convert remaining ambiguous object columns to a safer explicit type:
    # - try numeric if almost all values parse as numbers
    # - try datetime if almost all values parse as datetimes
    # - otherwise convert to pandas' string dtype to avoid pyarrow trying to coerce to int
    def _make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            ser = df[c]
            # Only act on object dtype (ambiguous / mixed types)
            if ser.dtype == 'object':
                # Try numeric if most values can be parsed
                try:
                    conv = pd.to_numeric(ser, errors='coerce')
                    non_null = conv.notna().sum()
                    if non_null / max(1, len(ser)) >= 0.95:
                        df[c] = conv
                        continue
                except Exception:
                    pass

                # Try datetime if most values can be parsed
                try:
                    non_na = ser.dropna().astype(str)
                    date_like = 0
                    if not non_na.empty:
                        date_like = non_na.str.contains(r'[/\\-.]|[A-Za-z]{3,}', regex=True).sum()

                    # require most non-null values to be date-like before coercing
                    if date_like / max(1, len(non_na)) < 0.95:
                        raise ValueError("Not enough date-like values")

                    conv = pd.to_datetime(ser, errors='coerce', infer_datetime_format=True)
                    non_null = conv.notna().sum()
                    if non_null / max(1, len(ser)) >= 0.95:
                        df[c] = conv
                        continue
                except Exception:
                    pass

                # Fallback: convert to pandas string dtype (explicit, works with pyarrow)
                try:
                    df[c] = ser.astype("string")
                except Exception:
                    # As a last resort, coerce to python str
                    df[c] = ser.astype(str)
        return df

    df = _make_arrow_compatible(df)

    # Outlier detection / removal (operates on numeric columns)
    # Config keys: detect_outliers: bool, outlier_method: 'iqr'|'zscore',
    # outlier_threshold: float (multiplier for IQR or z-score cutoff), outlier_action: 'report'|'drop'
    report['outliers'] = {}
    report['outliers_removed'] = 0
    if config.get('detect_outliers'):
        method = config.get('outlier_method', 'iqr')
        thresh = float(config.get('outlier_threshold', 1.5 if method == 'iqr' else 3.0))
        action = config.get('outlier_action', 'report')

        # Exclude identifier-like columns from outlier detection (e.g. 'id', 'user_id', 'id_number')
        # Assumption: columns labeled exactly 'id' or those starting/ending with 'id_' or '_id' should be skipped.
        numeric_cols = [
            c for c in df.select_dtypes(include=['number']).columns
            if not re.search(r'(?i)(?:^id$|_id$|^id_)', str(c))
        ]
        # Keep mask of rows to remove if requested
        removal_mask = pd.Series(False, index=df.index)

        for c in numeric_cols:
            ser = df[c].dropna()
            if ser.empty:
                report['outliers'][c] = {'count': 0, 'percent': 0.0}
                continue

            if method == 'iqr':
                q1 = ser.quantile(0.25)
                q3 = ser.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - thresh * iqr
                upper = q3 + thresh * iqr
                mask = (df[c] < lower) | (df[c] > upper)
            else:  # zscore
                mu = ser.mean()
                sigma = ser.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    mask = pd.Series(False, index=df.index)
                else:
                    z = (df[c] - mu) / sigma
                    mask = z.abs() > thresh

            count = int(mask.sum())
            percent = float(count) / max(1, len(df))
            report['outliers'][c] = {'count': count, 'percent': percent}
            if action == 'drop' and count > 0:
                removal_mask = removal_mask | mask.fillna(False)

        if action == 'drop':
            before = len(df)
            df = df.loc[~removal_mask]
            report['outliers_removed'] = before - len(df)

        # Defensive: ensure identifier-like columns are not included in outlier report
        # (in case any identifier columns were present in report due to previous logic)
        try:
            id_pattern = re.compile(r'(?i)(?:^id$|_id$|^id_)')
            for k in list(report.get('outliers', {}).keys()):
                if id_pattern.search(str(k)):
                    report['outliers'].pop(k, None)
        except Exception:
            # non-fatal if regex or keys operation fails
            pass

    cleaned_shape = df.shape
    report['cleaned_shape'] = cleaned_shape
    report['rows_removed'] = original_shape[0] - cleaned_shape[0]
    report['cols_removed'] = original_shape[1] - cleaned_shape[1]

    return df, report


__all__ = ['clean_dataframe']
