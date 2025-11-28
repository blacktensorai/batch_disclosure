# pipeline/run_pipeline.py
import os
import io
import json
import sqlite3
import logging
import hashlib
import datetime
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import requests
from .dispatcher import get_extractor_instance
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Storage locations
BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "asx"
ASX_DOWNLOAD_DIR = BASE_DIR / "data" / "asx"
ASX_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB = BASE_DIR / "storage" / "processed_results.db"
SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------
# SQLite helpers
def __init__sqlite():
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            exchange TEXT,
            filing_type TEXT,
            filing_date TEXT,
            source_file TEXT,
            output_json TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def _save_result_sqlite(record_id: str, meta: Dict[str, Any], output_json: Dict[str, Any]):
    __init__sqlite()
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO results
        (id, doc_id, exchange, filing_type, filing_date, source_file, output_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            meta.get("doc_id"),
            meta.get("exchange"),
            meta.get("filing_type"),
            meta.get("filing_date"),
            meta.get("source_file"),
            json.dumps(output_json, ensure_ascii=False),
            datetime.datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

# ---------------------------
# HTTP request with retries
def _safe_request(session: requests.Session, url: str, max_retries: int = 3, **kwargs):
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=15, allow_redirects=True, **kwargs)
            if resp.status_code == 200:
                return resp
        except requests.RequestException as e:
            logger.warning(f"Request failed ({attempt+1}/{max_retries}): {e}")
            time.sleep(1.5)
    return None

# ---------------------------
# Extract PDF from ASX agreement page
def _extract_pdf_from_agreement_page(html_text: str) -> Optional[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    input_tag = soup.find("input", {"name": "pdfURL"})
    if input_tag and input_tag.get("value", "").endswith(".pdf"):
        return input_tag["value"]
    match = re.search(r"https://announcements\.asx\.com\.au/asxpdf/.+?\.pdf", html_text)
    return match.group(0) if match else None

# ---------------------------
# Download PDF locally
def download_local_file(url: str, session: Optional[requests.Session] = None, status_callback: Optional[Callable[[str], None]] = None) -> Path:
    if status_callback:
        status_callback("Downloading PDF...")
    p = Path(url)
    if p.exists():
        return p

    s = session or requests.Session()
    resp = _safe_request(s, url)
    if not resp:
        raise ValueError(f"Could not fetch URL: {url}")

    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type:
        pdf_url = resp.url
    else:
        pdf_url = _extract_pdf_from_agreement_page(resp.text)
        if not pdf_url:
            raise ValueError(f"No PDF URL found in agreement page: {url}")

    pdf_resp = _safe_request(s, pdf_url)
    if not pdf_resp or "pdf" not in pdf_resp.headers.get("Content-Type", "").lower():
        raise ValueError(f"Failed to download PDF from {pdf_url}")

    h = hashlib.sha1(pdf_resp.url.encode("utf-8")).hexdigest()[:10]
    out_path = ASX_DOWNLOAD_DIR / f"asx_{h}.pdf"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    with open(out_path, "wb") as f:
        for chunk in pdf_resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info(f"Saved ASX PDF → {out_path}")
    return out_path

# ---------------------------
# Persist JSON
def persist_output_json(doc_id: str, exchange: str, filing_type: str, filing_date: Optional[str], source_file: str, output: List[Any]) -> Dict[str, Any]:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    record_id = hashlib.sha1(f"{doc_id}{exchange}{filing_type}_{ts}".encode()).hexdigest()
    out_fname = PROCESSED_DIR / f"{doc_id}{filing_type}{ts}.json"

    # Convert to dict for JSON serialization
    serializable = []
    for item in output:
        if hasattr(item, "model_dump"):
            serializable.append(item.model_dump())
        elif hasattr(item, "dict"):
            serializable.append(item.dict())
        elif isinstance(item, dict):
            serializable.append(item)
        else:
            serializable.append(dict(item))

    # Truncate long text fields for storage
    for i in serializable:
        if isinstance(i, dict) and "text" in i and isinstance(i["text"], str):
            i["text"] = i["text"][:400]

    with open(out_fname, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    _save_result_sqlite(
        record_id,
        {
            "doc_id": doc_id,
            "exchange": exchange,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "source_file": source_file
        },
        {"items": serializable, "file": str(out_fname)}
    )

    return {"record_id": record_id, "file_path": str(out_fname)}

# ---------------------------
# Core pipeline
class Pipeline:
    def __init__(self, llm_client: Optional[Any] = None, http_session: Optional[requests.Session] = None):
        self.llm_client = llm_client
        self.session = http_session or requests.Session()

    def process_file(
        self,
        file_url: str,
        exchange: str,
        filing_type: str,
        doc_id: str,
        filing_date: Optional[str] = None,
        source_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        metadata.update({
            "doc_id": doc_id,
            "exchange": exchange,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "source_file": source_file or file_url,
        })

        # Step 1: Download PDF
        if status_callback: status_callback("Fetching & downloading PDF...")
        tmp_path = download_local_file(file_url, session=self.session, status_callback=status_callback)

        # Step 2: Clean / prepare PDF
        if status_callback: status_callback("Cleaning PDF...")
        # optional: insert actual cleaning logic here

        # Step 3: Extractor instance
        extractor = get_extractor_instance(exchange, filing_type)
        if status_callback: status_callback("Getting candidate statements...")

        # Step 4: Run extraction - KEEP AS PYDANTIC OBJECTS
        if status_callback: status_callback("Extracting actual forward-looking statements...")
        items = extractor.run(str(tmp_path), metadata)

        # Cleanup temp file
        try:
            if tmp_path.exists() and tmp_path.is_file():
                tmp_path.unlink()
                logger.info(f"Deleted tmp file: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to delete tmp file: {e}")

        # Step 5: Persist JSON (converts to dict internally)
        if status_callback: status_callback("Saving extraction results...")
        persist_meta = persist_output_json(
            doc_id=doc_id,
            exchange=exchange,
            filing_type=filing_type,
            filing_date=filing_date,
            source_file=metadata.get("source_file"),
            output=items
        )

        if status_callback: status_callback("Extraction completed ✅")

        # RETURN PYDANTIC OBJECTS, NOT DICTS
        return {
            "status": "ok" if items else "no_items",
            "count": len(items),
            "items": items,  # ← KEEP AS PYDANTIC OBJECTS
            "persist": persist_meta,
        }

# ---------------------------
# Convenience wrappers
def _make_doc_id(ticker: Optional[str], filing_date: Optional[str], filing_type: str) -> str:
    t = (ticker or "DOC").upper()
    d = filing_date or datetime.date.today().isoformat()
    safe_ft = str(filing_type).upper().replace(" ", "").replace("/", "")
    return f"{t}{d}{safe_ft}"[:64]

def run_extraction_pipeline_from_url(
    file_url: str,
    exchange: str,
    filing_type: str,
    doc_id: Optional[str] = None,
    filing_date: Optional[str] = None,
    source_file: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if doc_id is None:
        ticker = metadata.get("ticker") if metadata else None
        doc_id = _make_doc_id(ticker, filing_date, filing_type)
    p = Pipeline()
    return p.process_file(
        file_url=file_url,
        exchange=exchange,
        filing_type=filing_type,
        doc_id=doc_id,
        filing_date=filing_date,
        source_file=source_file or file_url,
        metadata=metadata,
        status_callback=status_callback
    )

def run_extraction_pipeline_for_listing(
    exchange: str,
    filing_type: str,
    ticker: str,
    file_url: str,
    filing_date: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    doc_id = _make_doc_id(ticker, filing_date, filing_type)
    return run_extraction_pipeline_from_url(
        file_url=file_url,
        exchange=exchange,
        filing_type=filing_type,
        doc_id=doc_id,
        filing_date=filing_date,
        source_file=file_url,
        metadata={"ticker": ticker},
        status_callback=status_callback
    )

def process_file_request(payload: Dict[str, Any], status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Convenience wrapper to run the pipeline from a dict payload.
    Payload must include:
        - file_url
        - exchange
        - filing_type
        - doc_id
        - filing_date (optional)
        - source_file (optional)
    """
    required = ["file_url", "exchange", "filing_type", "doc_id"]
    for r in required:
        if r not in payload:
            raise ValueError(f"Missing required field: {r}")

    p = Pipeline()
    return p.process_file(
        file_url=payload["file_url"],
        exchange=payload["exchange"],
        filing_type=payload["filing_type"],
        doc_id=payload["doc_id"],
        filing_date=payload.get("filing_date"),
        source_file=payload.get("source_file"),
        metadata=payload.get("metadata"),
        status_callback=status_callback
    )