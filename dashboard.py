# dashboard.py
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import pandas as pd

# --- Project Root & Data Paths (100% portable) ---
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"

# Create required folders
(DATA_DIR / "company_list").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "company_cik_mappings").mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(
    filename=STORAGE_DIR / "dashboard.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="CatalystScan Pro", layout="wide", initial_sidebar_state="expanded")
st.title("CatalystScan Pro — AI Catalyst Intelligence")
st.caption("Overnight batch scanning of ASX & SEC 10-Q filings • Market-cap filtered • Zero maintenance")

# ==================== SIDEBAR CONFIG ====================
st.sidebar.header("Scan Configuration")

exchange = st.sidebar.selectbox("Exchange", ["ASX", "SEC"], index=0)

# ==================== SMART COMPANY LIST LOADER ====================
@st.cache_data(show_spinner=False)
def load_company_list(exchange_type):
    if exchange_type == "ASX":
        folder = DATA_DIR / "company_list"
        candidates = list(folder.glob("*.csv"))
        if not candidates:
            st.error(f"""
            No ASX company list found!  
            Please place your CSV file in:  
            `{folder}`  
            Example names: `asx_companies.csv` or `ASX_Listed_Companies_*.csv`
            """)
            st.stop()
        csv_path = max(candidates, key=lambda p: p.stat().st_mtime)
        code_col = "ASX code"

    else:  # SEC
        csv_path = DATA_DIR / "company_cik_mappings" / "sec_companies.csv"
        if not csv_path.exists():
            st.error(f"""
            SEC company list not found!  
            Place your file here:  
            `{csv_path}`
            """)
            st.stop()
        code_col = "cik"

    df = pd.read_csv(csv_path)
    st.sidebar.success(f"Loaded: `{csv_path.name}` • {len(df):,} companies")
    return df, code_col

companies_df, code_col = load_company_list(exchange)
companies_df.head()

# Clean market cap
if "market_cap" in companies_df.columns:
    companies_df["market_cap"] = (
        companies_df["market_cap"]
        .astype(str)
        .str.replace("[^0-9.]", "", regex=True)
        .replace({"": None, "nan": None})
        .astype(float)
    )

# Market cap filter
st.sidebar.subheader("Market Cap Range ($M)")
min_cap = float(companies_df["market_cap"].min() or 10)
max_cap = float(companies_df["market_cap"].max() or 5000)

min_cap_input = st.sidebar.number_input("Min Market Cap", value=min_cap, step=10.0)
max_cap_input = st.sidebar.number_input("Max Market Cap", value=max_cap, step=10.0)

# Document types
if exchange == "ASX":
    asx_doc_types = st.sidebar.multiselect(
        "ASX Reports to Include",
        options=["quarterly", "annual"],
        default=["quarterly"],
        help="Quarterly = Activities Report • Annual = Full Year Report"
    )
    if not asx_doc_types:
        st.sidebar.warning("Select at least one report type")
else:
    asx_doc_types = []

# Time range
time_range = st.sidebar.selectbox(
    "Lookback Period",
    ["Last Week", "Last Month", "Last 3 Months", "Last 6 Months"],
    index=2
)

period_map = {
    "Last Week": ("week", (datetime.now() - timedelta(weeks=1)).date()),
    "Last Month": ("month", (datetime.now() - timedelta(days=30)).date()),
    "Last 3 Months": ("3months", (datetime.now() - timedelta(days=90)).date()),
    "Last 6 Months": ("6months", (datetime.now() - timedelta(days=180)).date()),
}
asx_period, sec_start_date = period_map[time_range]

# Apply filters button
if st.sidebar.button("Apply Filters & Count Companies", type="secondary"):
    filtered = companies_df[
        (companies_df["market_cap"] >= min_cap_input) &
        (companies_df["market_cap"] <= max_cap_input)
    ].copy().reset_index(drop=True)

    st.session_state.filtered_companies = filtered
    st.session_state.total_count = len(filtered)
    st.sidebar.success(f"**{len(filtered):,} companies** ready to scan")

# Stop if no companies selected
if "filtered_companies" not in st.session_state or st.session_state.filtered_companies.empty:
    st.info("Configure filters → Click **Apply Filters & Count Companies**")
    st.stop()

filtered_companies = st.session_state.filtered_companies
total = len(filtered_companies)

st.write(f"### Ready to scan **{total:,} companies**")

# ==================== MAIN BATCH ENGINE ====================
if st.button("Start Catalyst extraction scanning", type="primary", use_container_width=True):
    if exchange == "ASX" and not asx_doc_types:
        st.error("Please select at least one ASX document type")
        st.stop()

    # Lazy imports (faster startup)
    from utils.llm import get_llm
    from ingest.fetch_single_asx import get_asx_announcements
    from pipeline.run_pipeline import run_extraction_pipeline_for_listing
    from ingest.sec_ingest import get_recent_filings, download_filings
    from extractors.sec_10q import SECExtractor

    llm = get_llm(model_name="gpt-5-nano")
    all_catalysts = []
    skipped = []
    failed = []

    progress_bar = st.progress(0.0)
    status = st.empty()
    result_box = st.empty()

    for idx, row in filtered_companies.iterrows():
        code = str(row[code_col]).strip().upper()
        status.info(f"[{idx+1}/{total}] Processing {code}...")

        try:
            found = False

            if exchange == "ASX":
                filings = get_asx_announcements(code, asx_period) or []
                filings = [f for f in filings if f.get("filing_type") in asx_doc_types]

                if not filings:
                    skipped.append(f"{code} (no {', '.join(asx_doc_types)})")
                    status.warning(f"{code} → No reports")
                else:
                    found = True
                    for f in filings:
                        try:
                            result = run_extraction_pipeline_for_listing(
                                exchange="ASX",
                                filing_type=f["filing_type"],
                                ticker=code,
                                file_url=f["pdf_page_url"],
                                filing_date=f["date"],
                            )
                            if result and result.get("items"):
                                all_catalysts.extend(result["items"])
                            time.sleep(1.3)
                        except Exception as e:
                            logging.error(f"ASX {code} filing error: {e}")

            else:  # SEC
                filings = get_recent_filings(code, sec_start_date, datetime.now().date())
                downloaded = download_filings(filings)
                downloaded = [f for f in downloaded if f["filing_type"] == "10-Q"]

                if not downloaded:
                    skipped.append(f"{code} (no 10-Q)")
                    status.warning(f"{code} → No 10-Q")
                else:
                    found = True
                    extractor = SECExtractor(debug=False)
                    for f in downloaded:
                        try:
                            meta = {"doc_id": f"{code}_{f['report_date']}", "date": f["report_date"]}
                            catalysts = extractor.extract(f["local_path"], meta)
                            all_catalysts.extend(catalysts)
                            time.sleep(1.3)
                        except Exception as e:
                            logging.error(f"SEC {code} error: {e}")

            if found:
                status.success(f"{code} → Success")
            else:
                status.warning(f"{code} → Skipped")

        except Exception as e:
            failed.append(code)
            status.error(f"{code} → Failed")
            logging.exception(f"Critical error {code}: {e}")

        progress_bar.progress((idx + 1) / total)

    # ==================== FINAL REPORT ====================
    st.success(f"Batch Complete — Scanned {total:,} companies")

    if skipped:
        with st.expander(f"{len(skipped)} companies skipped (no filings)"):
            for s in skipped[:100]:
                st.write("• " + s)

    if failed:
        st.error(f"{len(failed)} companies failed — check log")

    if all_catalysts:
        # Convert Pydantic models to dicts for DataFrame
        df = pd.DataFrame([c.model_dump() for c in all_catalysts])
        df = df.sort_values(["impact", "score"], ascending=False).reset_index(drop=True)

        if "text" in df.columns:
            df["preview"] = df["text"].apply(lambda x: (str(x)[:400] + "...") if len(str(x)) > 400 else x)

        cols = ["doc_id", "preview", "impact", "tone", "forecast_type", "score", "filing_type", "filing_date"]
        cols = [c for c in cols if c in df.columns]

        st.subheader(f"Extracted {len(df):,} Catalysts")
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV Report", csv, "catalyst_scan.csv", "text/csv")
        st.download_button("Download JSON Report", df.to_json(orient="records", indent=2), "catalyst_scan.json", "application/json")
    else:
        st.warning("No catalysts found")