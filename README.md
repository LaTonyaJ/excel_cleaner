# Excel Cleaner

Small Streamlit app to clean messy Excel/CSV files: trims whitespace, drops duplicates, removes blank rows/columns, normalizes column names, and infers types.

Getting started

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

Usage

- Upload a CSV or Excel file.
- Choose cleaning options from the sidebar.
- Click "Run cleaning" to preview results and download cleaned CSV/XLSX.

Files

- `app.py` — Streamlit UI.
- `utils/cleaner.py` — cleaning logic used by the app.

Extend

Add additional cleaning rules to `utils/cleaner.py` and expose them in `app.py`.
