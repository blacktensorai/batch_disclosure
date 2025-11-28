# extractors/sec_10q.py
import re
import json
import logging
from pathlib import Path
from typing import List, Any, Optional
from bs4 import BeautifulSoup
from flashtext import KeywordProcessor
import spacy

from .base_extractor import BaseExtractor
from models.catalyst_disclosure import (
    CatalystDisclosure,
    Entity,
    Impact,
    Tone,
    ForecastType,
    FilingType
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# SEC-specific keyword groups
SEC_CORE_KEYWORDS = {
    "Timing & Immediacy": [
        "imminent", "near-term", "upcoming", "expected shortly", "anticipated",
        "targeted for", "inbound interest", "execution phase"
    ],
    "Contractual Catalysts": [
        "agreement", "binding", "term sheets", "contracts", "pending", "negotiation",
        "renewal", "submitted", "proposal", "acquisition", "partnership", "discussions",
        "expected to commence", "finalizing", "tender"
    ],
    "Forward-Looking Hints": [
        "anticipate", "expect", "outlook", "projected", "forecasted", "opportunities",
        "pipeline", "strategic review", "advanced discussions", "expansion", "significant"
    ],
    "Regulatory & Compliance": [
        "clearance", "licensing", "approval", "deployment", "advanced",
        "submission", "regulatory approval", "FDA", "TGA", "review", "assay results"
    ]
}

IMPORTANT_SECTIONS = [
    "risk factors", "management's discussion and analysis", "md&a", "results of operations",
    "forward-looking statements", "business", "regulation fd disclosure", "other events",
    "outlook", "item 1.01", "item 2.01", "item 2.02", "item 5.02"
]

# FlashText processor
keyword_processor = KeywordProcessor(case_sensitive=False)
for group, keywords in SEC_CORE_KEYWORDS.items():
    for kw in keywords:
        keyword_processor.add_keyword(kw, group)


class SECExtractor(BaseExtractor):
    def __init__(self, model_name: str = "gpt-5-nano", llm_client: Optional[Any] = None, debug: bool = True):
        super().__init__(model_name=model_name, llm_client=llm_client)
        self.debug = debug
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def _parse_sections_from_html(self, html_text: str) -> List[dict]:
        soup = BeautifulSoup(html_text, "lxml")
        for tag in soup.find_all(["table", "figure", "script", "style", "img"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

        # Find SEC Item headers
        item_pattern = re.compile(r"^\s*(item\s+\d+[A-Za-z]?\.?\s+.*)$", re.IGNORECASE | re.MULTILINE)
        sig_pattern = re.compile(r"^\s*SIGNATURES?\s*$", re.IGNORECASE | re.MULTILINE)

        matches = list(item_pattern.finditer(text))
        sig_match = sig_pattern.search(text)

        sections = []
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else (sig_match.start() if sig_match else len(text))
            content = text[start:end].strip()
            if content and any(sec in title.lower() for sec in IMPORTANT_SECTIONS):
                sections.append({"title": title, "text": content})

        if not sections:
            sections = [{"title": "FULL_DOCUMENT", "text": " ".join(text.split()[:5000])}]

        return sections

    def _extract_candidates(self, text: str) -> List[str]:
        doc = self.nlp(text)
        candidates = []
        seen = set()
        for sent in doc.sents:
            s = sent.text.strip()
            if s and len(s) > 20 and keyword_processor.extract_keywords(s) and s not in seen:
                seen.add(s)
                candidates.append(s)
        return candidates

    def _prompt_pass2(self, numbered_items: str) -> str:
        allowed = ", ".join(f'"{ft.value}"' for ft in ForecastType)
        return f"""
You are an expert financial analyst analyzing SEC filings (10-Q, 10-K, 8-K).
You will receive a numbered list of candidate sentences.

Task:
- KEEP only true forward-looking statements (future plans, projections, deals, regulatory actions, milestones, timelines, approvals, etc.)
- DROP anything that is historical, current status, vague, or not actionable.
- For each kept sentence, return a JSON object with:
  - text: original sentence
  - impact: "LOW" | "MED" | "HIGH"
  - tone: "positive" | "neutral" | "cautious"
  - forecast_type: one of [{allowed}]
  - score: 1-10 (confidence)
  - entities: list of short strings

Output ONLY a valid JSON array. No markdown, no explanation.
Input:
{numbered_items}
""".strip()

    def extract(self, file_path: str, metadata: dict) -> List[CatalystDisclosure]:
        try:
            content = Path(file_path).read_bytes()
            html_text = content.decode("utf-8", errors="ignore")
        except Exception as e:
            if self.debug:
                print(f"Failed to read {file_path}: {e}")
            return []

        sections = self._parse_sections_from_html(html_text)
        candidates = []
        for sec in sections:
            candidates.extend(self._extract_candidates(sec["text"]))

        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            if self.debug:
                print("No candidates found in SEC filing.")
            return []

        if self.debug:
            print(f"\n=== SEC PASS-1: {len(candidates)} candidates ===")
            for c in candidates[:20]:
                print(" -", c)
            if len(candidates) > 20:
                print(f" ... and {len(candidates)-20} more")
            print("========================================\n")

        # ===================================================================
        # YOUR EXACT BATCHING LOGIC — NOW IN SEC EXTRACTOR
        # ===================================================================
        n = len(candidates)

        if n <= 10:
            num_batches = 1
        elif n < 30:
            num_batches = 2
        elif n < 50:
            num_batches = 3
        elif n < 70:
            num_batches = 5
        elif n < 80:
            num_batches = 6
        elif n < 90:
            num_batches = 7
        elif n < 100:
            num_batches = 8
        else:
            num_batches = 9

        batch_size = max(1, (n + num_batches - 1) // num_batches)
        batches = [candidates[i:i + batch_size] for i in range(0, n, batch_size)]
        batches = batches[:num_batches]

        if self.debug:
            print(f"→ {n} candidates → {len(batches)} batches: {[len(b) for b in batches]}")

        # ===================================================================

        results = []
        global_idx = 1

        for batch_num, batch in enumerate(batches, start=1):
            if not batch:
                continue

            if self.debug:
                print(f"\n--- SEC Batch {batch_num}/{len(batches)} ({len(batch)} sentences) ---")

            numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
            resp = self._ask_llm(self._prompt_pass2(numbered))
            json_block = self._extract_json_block(resp)
            items = self._safe_json_load(json_block) or []

            for item in items:
                try:
                    raw_type = item.get("forecast_type", "").lower()
                    if "contract" in raw_type:
                        ftype = ForecastType.CONTRACTUAL
                    elif "regul" in raw_type:
                        ftype = ForecastType.REGULATORY
                    elif "time" in raw_type or "sched" in raw_type:
                        ftype = ForecastType.TIMING
                    else:
                        ftype = ForecastType.HINTS

                    model = CatalystDisclosure(
                        doc_id=metadata.get("doc_id") or Path(file_path).stem,
                        exchange="SEC",
                        filing_type=FilingType.SEC_10Q,
                        filing_date=metadata.get("date"),
                        source_file=file_path,
                        sentence_id=f"s{global_idx}",
                        text=item.get("text", ""),
                        forward_looking=True,
                        forecast_type=ftype,
                        tone=Tone(item.get("tone", "neutral")),
                        impact=Impact(item.get("impact", "MED")),
                        score=int(item.get("score", 5)),
                        categories_matched=[],
                        entities=[Entity(type="entity", value=e, text=e) for e in item.get("entities", [])],
                    )
                    results.append(model)
                    global_idx += 1

                except Exception as e:
                    if self.debug:
                        print(f"Error parsing SEC item: {e}")

        print(f"SEC Extraction Complete → {len(results)} catalysts found")
        return results


if __name__ == "__main__":
    extractor = SECExtractor(debug=True)
    sample_path = r"D:\Moiz\Projects\batch disclosure\data\sec\1227500_10-Q_2025-11-03.html"
    meta = {"doc_id": "1227500_2025-11-03", "date": "2025-11-03", "cik": 1227500}
    results = extractor.extract(sample_path, meta)
    print("JSON:\n\n")
    print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
