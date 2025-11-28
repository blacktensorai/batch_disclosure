# /extractors/asx_annual.py
import re
import logging
from pathlib import Path
from typing import List, Any, Optional
from unstructured.partition.pdf import partition_pdf

import spacy
from flashtext import KeywordProcessor

from .base_extractor import BaseExtractor
from models.catalyst_disclosure import (
    CatalystDisclosure,
    Impact,
    Tone,
    ForecastType,
    FilingType,
    Entity,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------------------
# Load spaCy blank NLP for sentence segmentation
NLP = spacy.blank("en")
NLP.add_pipe("sentencizer")  # essential for proper sentence splitting

# ----------------------------------------
# CONFIG — STOP & DROP
STOP_AFTER_PATTERNS = [
    r"Auditor.?s?.? Independence Declaration",
    r"Notes to the Financial Statements",
    r"Notes to the Consolidated Financial Statements",
    r"Independent Auditor.?s?.? Report",
    r"Corporate Governance Statement"
]
STOP_AFTER = [re.compile(p, re.IGNORECASE) for p in STOP_AFTER_PATTERNS]

DROP_HEADING_PATTERNS = [
    r"Corporate\s+Directory",
    r"^Directors$",
    r"Company\s+Secretar(y|ies)",
    r"Registered\s+Office",
    r"Auditors?",
    r"Share\s+Registry",
    r"Website",
    r"Information\s+on\s+Directors",
    r"Information\s+on\s+Company\s+Secretaries",
    r"Board\s+of\s+Directors",
    r"Remuneration\s+Report",
    r"Non[- ]audit\s+services",
    r"Proceedings\s+on\s+behalf\s+of\s+Company",
    r"Indemnification",
    r"Insurance\s+premiums",
    r"^Share\s+options$",
    r"^Options$",
    r"Warrants",
    r"Rounding\s+of\s+amounts",
    r"Meetings\s+of\s+Directors",
    r"Loan\s+from\s+Directors",
    r"Number\s+of\s+shares\s+held",
    r"Number\s+of\s+listed\s+options",
    r"^Performance\s+Rights$",
    r"Incentive|Sale\s+Bonus\s+Pool|Termination",
    r"Voting.*Annual\s+General\s+Meeting"
]
DROP_HEADINGS = [re.compile(p, re.IGNORECASE) for p in DROP_HEADING_PATTERNS]

# ----------------------------------------
# CATEGORY MAP
CATEGORY_MAP = {
    "Intent Verbs": ForecastType.INTENT,
    "Timeline": ForecastType.TIMING,
    "Guidance": ForecastType.GUIDANCE,
    "Milestones": ForecastType.HINTS,
    "Deals": ForecastType.CONTRACTUAL,
    "Strategy": ForecastType.STRATEGY,
}

# ----------------------------------------
# KEYWORDS for FlashText
FORWARD_KEYS = {
    "Intent Verbs": [
        "expected shortly", "expected to", "expect to", "plan to", "planned to",
        "intends to", "intend to", "inbound interest",
        "anticipates", "anticipate", "anticipated",
        "targeting", "targets", "targeted", "finalizing"
    ],
    "Timeline": [
        "over the next", "in fy", "in h1", "in h2", "year end",
        "during", "near-term", "imminent", "upcoming", "expected in",
        "targeted for", "scheduled for", "execution phase"
    ],
    "Guidance": [
        "guidance", "forecast", "outlook", "projected", "opportunities",
        "cash flow positive", "capital raise", "funding secured", "forecasted"
    ],
    "Milestones": [
        "licensing", "clearance", "approval",
        "resource upgrade", "fda", "deployment", "advanced",
        "regulatory approval", "submission", "tender", "review"
    ],
    "Deals": [
        "agreement expected", "term sheets", "binding", "contracts",
        "mou", "jv", "partnership", "submitted", "pending", "acquisition",
        "negotiation", "commercial launch", "renewal", "proposal", "discussions",
        "expected to commence"
    ],
    "Strategy": [
        "strategy", "strategic review", "significant", "advanced discussions",
        "expansion", "growth strategy", "finalizing", "pipeline"
    ]
}

# Build FlashText keyword matcher
keyword_processor = KeywordProcessor(case_sensitive=False)
for group, kws in FORWARD_KEYS.items():
    for kw in kws:
        keyword_processor.add_keyword(kw, group)

# ============================================================
# ASXAnnualExtractor
# ============================================================
class ASXAnnualExtractor(BaseExtractor):
    def __init__(self, model_name: str = "gpt-5-nano", llm_client: Optional[Any] = None, debug: bool = True):
        super().__init__(model_name=model_name, llm_client=llm_client)
        self.debug = debug

    # --------------------------------------------------------
    # Parse PDF → sections
    # --------------------------------------------------------
    def _parse_sections(self, file_path: str):
        elems = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False,
        )

        sections, current, heading = [], [], "Unknown"

        for elem in elems:
            if elem.category == "Title":
                if current:
                    sections.append({"heading": heading, "text": "\n".join(current)})
                    current = []
                heading = elem.text.strip()
            elif elem.category in {"NarrativeText", "ListItem"}:
                t = elem.text.strip()
                if t:
                    current.append(t)

        if current:
            sections.append({"heading": heading, "text": "\n".join(current)})

        # Filter STOP_AFTER and dropped headings
        cleaned = []
        for sec in sections:
            h = sec["heading"].strip()
            if any(p.search(h) for p in STOP_AFTER):
                break
            if not any(p.search(h) for p in DROP_HEADINGS):
                cleaned.append(sec)
        return cleaned

    # --------------------------------------------------------
    # Pass-1: spaCy + FlashText
    # --------------------------------------------------------
    def _extract_candidates(self, sections: List[dict]) -> List[str]:
        all_text = "\n".join(s["text"] for s in sections)
        doc = NLP(all_text)

        candidates = []
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            matches = keyword_processor.extract_keywords(s)
            if matches:
                candidates.append(s)

        # Dedupe
        candidates = list(dict.fromkeys(candidates))

        return candidates

    # --------------------------------------------------------
    # Pass-2: Prompt for LLM
    # --------------------------------------------------------
    def _prompt_pass2(self, numbered_items: str) -> str:
        allowed = ", ".join(f'"{ft.value}"' for ft in ForecastType)
        return f"""
You are an expert financial analyst. 
You will receive a numbered list of candidate sentences extracted from a company's annual report.

Task:
- From the input sentences, KEEP ONLY those that are true forward-looking statements (plans, projections, forecasts, upcoming actions, regulatory submissions, pending deals, milestones, deployments, approvals, or explicitly scheduled future events).
- DROP sentences that only describe present or past facts, are vague, or offer no actionable forward-looking insight.
- For each KEPT sentence, output a JSON object with the following fields:
  - text: the original sentence (string)
  - impact: one of ["LOW","MED","HIGH"]
  - tone: one of ["positive","neutral","cautious"]
  - forecast_type: one of [{allowed}]
  - score: integer between 1 and 10
  - entities: a list of short strings
  - categories_matched: list of strings

Requirements:
- Output MUST be a single JSON array of objects.
- No explanations, no markdown.
- Keep the sentence text EXACTLY as in input.

Input sentences:
{numbered_items}

Return ONLY the JSON array.
""".strip()

    # --------------------------------------------------------
    # Main extraction pipeline
    # --------------------------------------------------------
    def extract(self, file_path: str, metadata: dict) -> List[CatalystDisclosure]:
        sections = self._parse_sections(file_path)
        if not sections:
            if self.debug:
                print("No relevant sections found in PDF.")
            return []

        candidates = self._extract_candidates(sections)
        
        if self.debug:
            print(f"\n=== PASS-1 CANDIDATES (ASX Annual) - {len(candidates)} ===")
            for c in candidates:
                print(" -", c)
            print("==========================================================\n")

        if not candidates:
            if self.debug:
                print("No candidates found in PDF.")
            return []

        # ===================================================================
        # BATCHING LOGIC
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
        batches = [
            candidates[i:i + batch_size]
            for i in range(0, n, batch_size)
        ]
        batches = batches[:num_batches]

        if self.debug:
            sizes = [len(b) for b in batches]
            print(f"→ {n} candidates → {len(batches)} batches: {sizes}")

        # ===================================================================

        parsed_all = []
        global_idx = 1

        for batch_num, batch in enumerate(batches, start=1):
            if not batch:
                continue

            if self.debug:
                print(f"\n--- Batch {batch_num}/{len(batches)} ({len(batch)} sentences) ---")

            numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(batch))
            resp = self._ask_llm(self._prompt_pass2(numbered))
            json_block = self._extract_json_block(resp)
            items = self._safe_json_load(json_block) or []

            for item in items:
                try:
                    raw_type = item.get("forecast_type", "").lower()
                    if "contract" in raw_type:
                        forecast = ForecastType.CONTRACTUAL
                    elif "regul" in raw_type:
                        forecast = ForecastType.REGULATORY
                    elif "time" in raw_type or "sched" in raw_type:
                        forecast = ForecastType.TIMING
                    elif "guidance" in raw_type:
                        forecast = ForecastType.GUIDANCE
                    elif "strategy" in raw_type:
                        forecast = ForecastType.STRATEGY
                    else:
                        forecast = ForecastType.HINTS

                    model = CatalystDisclosure(
                        doc_id=metadata.get("doc_id") or Path(file_path).stem,
                        exchange="ASX",
                        filing_type=FilingType.ASX_ANNUAL,
                        filing_date=metadata.get("date"),
                        source_file=file_path,
                        sentence_id=f"s{global_idx}",
                        text=item.get("text", ""),
                        forward_looking=True,
                        forecast_type=forecast,
                        tone=Tone(item.get("tone", "neutral")),
                        impact=Impact(item.get("impact", "MED")),
                        score=int(item.get("score", 5)),
                        categories_matched=item.get("categories_matched", []),
                        entities=[Entity(type="entity", value=e, text=e) for e in item.get("entities", [])],
                    )
                    parsed_all.append(model)
                    global_idx += 1

                except Exception as e:
                    if self.debug:
                        print(f"Error parsing item in batch {batch_num}: {e}")
                    continue

        print(f"Total forward-looking statements extracted: {len(parsed_all)}")
        if self.debug:
            print(f"\n=== Extraction completed for {metadata.get('doc_id', Path(file_path).stem)} ===\n")

        return parsed_all