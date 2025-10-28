import io
import pandas as pd
import streamlit as st
from utils.cleaner import clean_dataframe


st.set_page_config(page_title="Excel Cleaner", layout="wide")

st.title("Excel / CSV Cleaner")
st.markdown("A simple app to clean messy Excel/CSV files — remove duplicates, trim whitespace, normalize column names, infer types and more.")

uploaded = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

with st.sidebar:
	st.header("Cleaning options")
	opt_trim = st.checkbox("Trim whitespace (string columns)", value=True)
	opt_drop_dup = st.checkbox("Drop duplicate rows", value=True)
	opt_drop_blank_rows = st.checkbox("Drop completely blank rows", value=True)
	opt_drop_blank_cols = st.checkbox("Drop completely blank columns", value=True)
	opt_normalize_cols = st.checkbox("Normalize column names (lowercase, underscores)", value=True)
	opt_infer_types = st.checkbox("Infer numeric/date types", value=True)
	date_detect_thresh = st.slider("Date detection threshold (% non-null to treat as date)", 0, 100, 50)

	# Null handling options
	null_handling = st.radio("Null handling", options=["none", "drop_rows", "fill"], index=0,
		format_func=lambda x: {
			"none": "Do nothing",
			"drop_rows": "Drop rows with any nulls",
			"fill": "Fill nulls"
		}[x])
	fill_strategy = None
	fill_constant = None
	if null_handling == 'fill':
		fill_strategy = st.selectbox("Fill strategy", options=["mode", "mean", "median", "constant"], index=0)
		if fill_strategy == 'constant':
			fill_constant = st.text_input("Fill constant (applied as string); leave blank for empty string", value="")
	sample_rows = st.number_input("Preview rows", min_value=1, max_value=1000, value=100)

	# Outlier detection options
	detect_outliers = st.checkbox("Detect outliers (numeric columns)", value=False)
	outlier_method = None
	outlier_threshold = None
	outlier_action = None
	if detect_outliers:
		outlier_method = st.selectbox("Method", options=["iqr", "zscore"], index=0)
		if outlier_method == 'iqr':
			outlier_threshold = st.slider("IQR multiplier (outlier cutoff)", 1.0, 3.0, 1.5, step=0.1)
		else:
			outlier_threshold = st.slider("Z-score threshold", 1.0, 6.0, 3.0, step=0.1)
		outlier_action = st.selectbox("When outliers detected", options=["report", "drop"], index=0,
			format_func=lambda x: {"report": "Report only", "drop": "Drop rows that are outliers"}[x])

if uploaded is not None:
	file_bytes = uploaded.read()
	filename = uploaded.name
	try:
		if filename.lower().endswith(('.xls', '.xlsx')):
			df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
		else:
			# try common encodings via pandas auto-detect
			df = pd.read_csv(io.BytesIO(file_bytes))
	except Exception as e:
		st.error(f"Error reading file: {e}")
		st.stop()

	st.subheader("Preview")
	st.dataframe(df.head(sample_rows))

	st.subheader("Columns")
	st.write(list(df.columns))

	st.markdown("---")

	config = {
		"trim_whitespace": opt_trim,
		"drop_duplicates": opt_drop_dup,
		"drop_blank_rows": opt_drop_blank_rows,
		"drop_blank_cols": opt_drop_blank_cols,
		"normalize_columns": opt_normalize_cols,
		"infer_types": opt_infer_types,
		"date_detect_thresh": date_detect_thresh / 100.0,
		# null handling
		"null_handling": None if null_handling == 'none' else null_handling,
		"fill_strategy": fill_strategy,
		"fill_constant": fill_constant,
		# outlier options
		"detect_outliers": detect_outliers,
		"outlier_method": outlier_method,
		"outlier_threshold": outlier_threshold,
		"outlier_action": outlier_action,
	}

	if st.button("Run cleaning"):
		with st.spinner("Cleaning..."):
			cleaned, report = clean_dataframe(df.copy(), config)

		st.success("Cleaning finished")

		st.subheader("Cleaned preview")
		st.dataframe(cleaned.head(sample_rows))

		st.subheader("Report")
		st.json(report)

		# Nicely render nulls-filled summary if present
		if report.get('nulls_filled'):
			st.subheader("Nulls filled (per column)")
			try:
				nf = pd.DataFrame.from_dict(report['nulls_filled'], orient='index', columns=['filled_count'])
				st.table(nf)
			except Exception:
				pass

		# Outlier summary
		if config.get('detect_outliers') and report.get('outliers'):
			st.subheader("Outliers summary (numeric columns)")
			try:
				of = pd.DataFrame.from_dict(report['outliers'], orient='index')
				st.table(of)
			except Exception:
				pass

		# CSV download
		csv_bytes = cleaned.to_csv(index=False).encode('utf-8')
		st.download_button("Download CSV", data=csv_bytes, file_name=f"cleaned_{filename.rsplit('.',1)[0]}.csv", mime='text/csv')

		# Excel download
		towrite = io.BytesIO()
		with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
			cleaned.to_excel(writer, index=False, sheet_name='cleaned')
		towrite.seek(0)
		st.download_button("Download XLSX", data=towrite, file_name=f"cleaned_{filename.rsplit('.',1)[0]}.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

	st.markdown("---")
	st.markdown("Need more control? Clone the repo and extend `utils/cleaner.py` with custom rules.")
