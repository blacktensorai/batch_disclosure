# fetch_single_asx.py
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from urllib.parse import urlparse

ASX_BASE_URL = "https://www.asx.com.au"
ASX_ENDPOINT = "https://www.asx.com.au/asx/v2/statistics/announcements.do"

PERIOD_MAP = {
    "week": ("D", "W"),    # last week
    "month": ("D", "M"),   # last month
    "3months": ("D", "M3"),
    "6months": ("D", "M6"),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": ASX_ENDPOINT,
    "Accept-Language": "en-US,en;q=0.9",
}

# ----------------------------------------------------------------------
def _normalize_date(date_text: str):
    """Try to parse various date formats returned by ASX."""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(date_text.strip(), fmt)
        except Exception:
            continue
    return datetime.now()

# ----------------------------------------------------------------------
def clean_asx_pdf_url(url: str) -> str:
    """
    Convert ASX 'announcement.do' or 'asxpdf' links into clean, downloadable PDF URLs.
    Removes invalid query params like '?display=pdf&idsId=xxxx'.
    """
    if not url:
        return url

    parsed = urlparse(url)
    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    # Ensure correct format ends with .pdf
    if clean_url.endswith(".do"):
        clean_url = clean_url.replace(".do", ".pdf")

    return clean_url

# ----------------------------------------------------------------------
def classify_filing(title: str):
    """Rough heuristic classification of ASX announcement titles."""
    t = title.lower()
    if "quarterly" in t and "report" in t:
        return "quarterly"
    if "investor presentation" in t:
        return "investor"
    if "annual report" in t:
        return "annual"
    return "other"

# ----------------------------------------------------------------------
def get_asx_announcements(ticker: str, period_key: str):
    """
    Fetch ASX announcements for a ticker and time period.
    Only returns 'quarterly', 'annual', and 'investor' filings.
    """
    timeframe, period = PERIOD_MAP[period_key]

    session = requests.Session()
    session.headers.update(HEADERS)

    # Step 1: Visit base page to initialize session/cookies
    _ = session.get(ASX_ENDPOINT)
    time.sleep(0.3)

    # Step 2: Actual query with parameters
    params = {
        "by": "asxCode",
        "asxCode": ticker.upper(),
        "timeframe": timeframe,
        "period": period,
    }

    resp = session.get(ASX_ENDPOINT, params=params)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code} from ASX")

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tbody > tr")

    filings = []
    if not rows:
        print("⚠️ No rows found — possibly blocked or invalid response.")
        return []

    for tr in rows:
        tds = [td.get_text(strip=True) for td in tr.select("td")]
        if len(tds) < 3:
            continue

        date_txt, code, title, *_ = tds
        date = _normalize_date(date_txt)

        link = tr.find("a", href=True)
        if not link:
            continue

        pdf_page_url = ASX_BASE_URL + link["href"]
        pdf_actual_url = clean_asx_pdf_url(pdf_page_url)
        filing_type = classify_filing(title)

        filings.append({
            "date": str(date.date()),
            "ticker": ticker.upper(),
            "title": title,
            "filing_type": filing_type,
            "pdf_page_url": pdf_page_url,
            "pdf_actual_url": pdf_actual_url,
        })

    # Filter: only keep relevant filings
    allowed = {"quarterly", "annual", "investor"}
    filings = [f for f in filings if f["filing_type"] in allowed]

    return filings
