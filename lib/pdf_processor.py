from __future__ import annotations
"""PDF processing module for BMO Annual Report.

Downloads the PDF, extracts structured elements via Unstructured.io,
and segments content into tagged chunks with rich metadata for vector indexing.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

# Suppress noisy pdfminer color-space warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import requests
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title,
    NarrativeText,
    Table,
    ListItem,
    Image,
    FigureCaption,
    Header,
)

PDF_URL = "https://www.bmo.com/ir/archive/en/bmo_ar2025_MDA.pdf"
DATA_DIR = Path(__file__).parent.parent / "data"
PDF_PATH = DATA_DIR / "bmo_ar2025_MDA.pdf"
CHUNKS_CACHE_PATH = DATA_DIR / "chunks.json"
SOURCE_FILE = "bmo_ar2025_MDA.pdf"

# Schema version — bump this to force re-index when metadata schema changes
SCHEMA_VERSION = 4

# Minimum chunk length to avoid noise
MIN_CHUNK_LENGTH = 50

# Maximum chunk length before splitting
MAX_CHUNK_LENGTH = 2000

# Regex patterns for time period extraction
_TIME_PERIOD_PATTERNS = [
    re.compile(r"\bFY\s?(\d{4})\b", re.IGNORECASE),        # FY2023, FY 2023
    re.compile(r"\bQ([1-4])\s+(\d{4})\b"),                   # Q3 2023
    re.compile(r"\bQ([1-4])(\d{4})\b"),                       # Q32023
    re.compile(r"\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(20[1-3]\d)\b"),                           # standalone year 2010-2039
]

_QUARTER_WORD_MAP = {"first": "1", "second": "2", "third": "3", "fourth": "4"}


def download_pdf() -> Path:
    """Download the BMO MDA PDF if not already cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists():
        return PDF_PATH

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    for attempt in range(3):
        try:
            response = requests.get(PDF_URL, timeout=120, headers=headers, stream=True)
            response.raise_for_status()
            PDF_PATH.write_bytes(response.content)
            return PDF_PATH
        except requests.exceptions.ReadTimeout:
            if attempt == 2:
                raise
            continue

    return PDF_PATH


def _extract_time_periods(text: str) -> list[str]:
    """Extract time period references from text using regex.

    Returns deduplicated list like ["FY2023", "Q3 2023", "2024"].
    """
    periods = set()

    # FY patterns
    for m in _TIME_PERIOD_PATTERNS[0].finditer(text):
        periods.add(f"FY{m.group(1)}")

    # Q + year patterns
    for pattern in _TIME_PERIOD_PATTERNS[1:3]:
        for m in pattern.finditer(text):
            periods.add(f"Q{m.group(1)} {m.group(2)}")

    # Quarter word patterns (first quarter of 2023)
    for m in _TIME_PERIOD_PATTERNS[3].finditer(text):
        q = _QUARTER_WORD_MAP[m.group(1).lower()]
        periods.add(f"Q{q} {m.group(2)}")

    # Standalone years
    for m in _TIME_PERIOD_PATTERNS[4].finditer(text):
        periods.add(m.group(1))

    return sorted(periods)


def _map_element_type(element) -> str:
    """Map an Unstructured element to our chunk_type schema."""
    if isinstance(element, Table):
        return "table"
    elif isinstance(element, (Image, FigureCaption)):
        return "image"
    else:
        return "text"


def _split_long_chunk(text: str, max_length: int = MAX_CHUNK_LENGTH) -> list[str]:
    """Split a long chunk into smaller pieces at sentence boundaries."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_length and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = current + " " + sentence if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def _is_cache_valid() -> bool:
    """Check if chunks cache exists and matches current schema version."""
    if not CHUNKS_CACHE_PATH.exists():
        return False
    try:
        with open(CHUNKS_CACHE_PATH) as f:
            data = json.load(f)
        if not data:
            return False
        first = data[0]
        required_fields = {"source_file", "chunk_type", "heading_top", "heading_sub", "time_period_str"}
        if not required_fields.issubset(first.keys()):
            return False
        if first.get("_schema_version") != SCHEMA_VERSION:
            return False
        return True
    except (json.JSONDecodeError, KeyError, IndexError):
        return False


def extract_chunks(pdf_path: Path | None = None) -> list[dict]:
    """Extract structured chunks from the PDF with rich metadata.

    Returns a list of dicts with keys: id, text, source_file, page_number,
    chunk_type, heading, heading_top, heading_sub, time_period_str, _schema_version.
    """
    if _is_cache_valid():
        with open(CHUNKS_CACHE_PATH) as f:
            return json.load(f)

    if pdf_path is None:
        pdf_path = download_pdf()

    # Use Unstructured.io to partition the PDF with layout detection
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        include_page_breaks=False,
        infer_table_structure=True,
    )

    chunks = []
    current_heading = "Document Start"
    current_subheading = ""

    # Buffer for accumulating body text under a heading
    text_buffer = ""
    buffer_chunk_type = "text"
    buffer_page = 1

    def _flush_buffer():
        nonlocal text_buffer, buffer_chunk_type, buffer_page
        if text_buffer.strip() and len(text_buffer.strip()) >= MIN_CHUNK_LENGTH:
            heading_path = current_heading
            if current_subheading:
                heading_path = f"{current_heading} > {current_subheading}"

            heading_top = heading_path.split(" > ")[0]
            heading_sub = heading_path.split(" > ")[-1] if " > " in heading_path else ""
            time_periods = _extract_time_periods(text_buffer)

            for idx, piece in enumerate(_split_long_chunk(text_buffer.strip())):
                chunk_id = hashlib.md5(
                    f"{heading_path}:{buffer_page}:{idx}:{piece}".encode()
                ).hexdigest()
                chunks.append({
                    "id": chunk_id,
                    "text": piece,
                    "source_file": SOURCE_FILE,
                    "page_number": buffer_page,
                    "chunk_type": buffer_chunk_type,
                    "heading": heading_path,
                    "heading_top": heading_top,
                    "heading_sub": heading_sub,
                    "time_period_str": ",".join(time_periods),
                    "_schema_version": SCHEMA_VERSION,
                })

        text_buffer = ""
        buffer_chunk_type = "text"

    for element in elements:
        page_num = element.metadata.page_number or 1
        element_text = str(element).strip()

        if not element_text:
            continue

        if isinstance(element, Title):
            _flush_buffer()

            depth = getattr(element.metadata, "category_depth", None)
            if depth is not None and depth > 0:
                current_subheading = element_text
            else:
                current_heading = element_text
                current_subheading = ""

            buffer_page = page_num

        elif isinstance(element, Header):
            # Running headers — skip
            continue

        elif isinstance(element, (Table, Image, FigureCaption)):
            _flush_buffer()

            chunk_type = _map_element_type(element)
            heading_path = current_heading
            if current_subheading:
                heading_path = f"{current_heading} > {current_subheading}"

            heading_top = heading_path.split(" > ")[0]
            heading_sub = heading_path.split(" > ")[-1] if " > " in heading_path else ""

            display_text = element_text if chunk_type != "image" else f"[Image on page {page_num}]"
            time_periods = _extract_time_periods(display_text)

            chunk_id = hashlib.md5(
                f"{chunk_type}:{heading_path}:{page_num}:{display_text}".encode()
            ).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": display_text,
                "source_file": SOURCE_FILE,
                "page_number": page_num,
                "chunk_type": chunk_type,
                "heading": heading_path,
                "heading_top": heading_top,
                "heading_sub": heading_sub,
                "time_period_str": ",".join(time_periods),
                "_schema_version": SCHEMA_VERSION,
            })

        else:
            # NarrativeText, ListItem, and other text elements — buffer them
            buffer_page = page_num
            text_buffer += " " + element_text

    # Flush remaining buffer
    _flush_buffer()

    # Cache chunks to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_CACHE_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    return chunks


def get_headings(chunks: list[dict]) -> list[str]:
    """Return unique top-level headings from chunks."""
    seen = set()
    headings = []
    for chunk in chunks:
        top = chunk.get("heading_top") or chunk["heading"].split(" > ")[0]
        if top not in seen:
            seen.add(top)
            headings.append(top)
    return headings


def get_all_headings(chunks: list[dict]) -> list[str]:
    """Return all unique heading paths from chunks."""
    seen = set()
    headings = []
    for chunk in chunks:
        h = chunk["heading"]
        if h not in seen:
            seen.add(h)
            headings.append(h)
    return headings


def get_all_time_periods(chunks: list[dict]) -> list[str]:
    """Return all unique time periods across all chunks."""
    periods = set()
    for chunk in chunks:
        tp = chunk.get("time_period_str", "")
        if tp:
            for p in tp.split(","):
                p = p.strip()
                if p:
                    periods.add(p)
    return sorted(periods)
