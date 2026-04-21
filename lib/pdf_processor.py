from __future__ import annotations
"""PDF processing module for BMO Annual Report.

Downloads the PDF, extracts text by page, detects headings via font analysis,
and segments content into tagged chunks for vector indexing.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import requests

PDF_URL = "https://www.bmo.com/ir/archive/en/bmo_ar2025_MDA.pdf"
DATA_DIR = Path(__file__).parent.parent / "data"
PDF_PATH = DATA_DIR / "bmo_ar2025_MDA.pdf"
CHUNKS_CACHE_PATH = DATA_DIR / "chunks.json"

# Font size thresholds calibrated for BMO 2025 MDA report:
# 24pt non-bold = major section titles ("About BMO", "Corporate Events")
# 14pt non-bold = section headings ("Our Strategic Priorities", "Efficiency Ratio")
# 9pt bold = "MD&A" page markers (skip these)
# 8.2pt bold = subheadings ("Credit Risk", "TABLE 1")
# 7.5pt = body text
HEADING_MIN_FONT_SIZE = 12.0
SUBHEADING_MIN_FONT_SIZE = 8.0

# Page marker text to ignore as headings
IGNORE_HEADINGS = {"MD&A", "md&a"}

# Minimum chunk length to avoid noise
MIN_CHUNK_LENGTH = 50

# Maximum chunk length before splitting
MAX_CHUNK_LENGTH = 2000


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


def _classify_block(block: dict, median_font_size: float) -> str:
    """Classify a text block as heading, subheading, or body text.

    BMO PDF structure:
      - 14pt+ (bold or not) → heading (major sections)
      - 8.0–12pt bold → subheading (sub-sections, table labels)
      - 7.5pt body text
      - 9pt bold "MD&A" markers → ignored
    """
    if block.get("type") == 1:  # image block
        return "image"

    lines = block.get("lines", [])
    if not lines:
        return "body"

    # Get the dominant font size and flags for this block
    sizes = []
    bold_count = 0
    total_spans = 0
    full_text = ""
    for line in lines:
        for span in line.get("spans", []):
            sizes.append(span["size"])
            total_spans += 1
            full_text += span["text"]
            if span["flags"] & 2**4:  # bold flag
                bold_count += 1

    if not sizes:
        return "body"

    # Skip known page markers like "MD&A"
    stripped = full_text.strip()
    if stripped in IGNORE_HEADINGS:
        return "body"
    # Normalize curly apostrophes for comparison
    norm = stripped.replace("\u2019", "'").replace("\u2018", "'")
    # Skip running page headers like "MANAGEMENT'S DISCUSSION AND ANALYSIS"
    if norm.startswith("MANAGEMENT'S DISCUSSION AND ANALYSIS"):
        # Only skip if it's JUST the header; keep if it has a real title appended
        remainder = norm.replace("MANAGEMENT'S DISCUSSION AND ANALYSIS", "").strip()
        if not remainder:
            return "body"

    avg_size = sum(sizes) / len(sizes)
    is_mostly_bold = bold_count > total_spans * 0.5

    # Large text (14pt+) = heading regardless of bold
    if avg_size >= HEADING_MIN_FONT_SIZE:
        return "heading"
    # Bold text above subheading threshold = subheading
    elif avg_size >= SUBHEADING_MIN_FONT_SIZE and is_mostly_bold:
        return "subheading"
    return "body"


def _extract_block_text(block: dict) -> str:
    """Extract plain text from a block dict."""
    if block.get("type") == 1:
        return "[Image]"
    lines = block.get("lines", [])
    text_parts = []
    for line in lines:
        spans_text = " ".join(span["text"] for span in line.get("spans", []))
        text_parts.append(spans_text)
    return " ".join(text_parts).strip()


def _detect_table_heuristic(text: str) -> bool:
    """Simple heuristic to detect if a text block is likely a table row."""
    # Tables tend to have multiple numbers separated by spaces/tabs
    num_count = len(re.findall(r"\b[\d,]+\.?\d*\b", text))
    has_dollar = "$" in text
    # If there are 3+ numbers in a single block, likely a table
    return (num_count >= 3) or (has_dollar and num_count >= 2)


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


def extract_chunks(pdf_path: Path | None = None) -> list[dict]:
    """Extract structured chunks from the PDF with heading hierarchy and metadata.

    Returns a list of dicts:
        {
            "id": str,           # unique chunk id
            "text": str,         # chunk content
            "heading": str,      # section heading (e.g., "Risk Management > Credit Risk")
            "content_type": str, # "text", "table", or "image"
            "page_number": int,  # 1-indexed page number
        }
    """
    if CHUNKS_CACHE_PATH.exists():
        with open(CHUNKS_CACHE_PATH) as f:
            return json.load(f)

    if pdf_path is None:
        pdf_path = download_pdf()

    doc = fitz.open(pdf_path)
    chunks = []

    # First pass: compute median font size across document
    all_sizes = []
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    all_sizes.append(span["size"])
    median_font_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 10.0

    # Second pass: extract and classify
    current_heading = "Document Start"
    current_subheading = ""

    # Prefix to strip from headings that contain a real title after it
    MDA_PREFIX = "MANAGEMENT'S DISCUSSION AND ANALYSIS"
    MDA_PREFIX_CURLY = "MANAGEMENT\u2019S DISCUSSION AND ANALYSIS"

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        page_text_buffer = ""
        page_content_type = "text"

        for block in blocks:
            block_type = _classify_block(block, median_font_size)
            block_text = _extract_block_text(block)

            if not block_text.strip():
                continue

            if block_type == "heading":
                # Flush previous buffer
                if page_text_buffer.strip() and len(page_text_buffer.strip()) >= MIN_CHUNK_LENGTH:
                    heading_path = current_heading
                    if current_subheading:
                        heading_path = f"{current_heading} > {current_subheading}"

                    for piece in _split_long_chunk(page_text_buffer.strip()):
                        chunk_id = hashlib.md5(
                            f"{heading_path}:{page_num}:{piece[:50]}".encode()
                        ).hexdigest()
                        chunks.append(
                            {
                                "id": chunk_id,
                                "text": piece,
                                "heading": heading_path,
                                "content_type": page_content_type,
                                "page_number": page_num,
                            }
                        )

                current_heading = block_text.strip()
                # Clean up MDA prefix from headings like
                # "MANAGEMENT'S DISCUSSION AND ANALYSIS About BMO"
                for prefix in (MDA_PREFIX, MDA_PREFIX_CURLY):
                    if prefix in current_heading:
                        remainder = current_heading.replace(prefix, "").strip()
                        if remainder:
                            current_heading = remainder
                        else:
                            # Pure header with no real title — don't treat as heading
                            continue
                        break
                current_subheading = ""
                page_text_buffer = ""
                page_content_type = "text"

            elif block_type == "subheading":
                # Flush buffer under previous subheading
                if page_text_buffer.strip() and len(page_text_buffer.strip()) >= MIN_CHUNK_LENGTH:
                    heading_path = current_heading
                    if current_subheading:
                        heading_path = f"{current_heading} > {current_subheading}"

                    for piece in _split_long_chunk(page_text_buffer.strip()):
                        chunk_id = hashlib.md5(
                            f"{heading_path}:{page_num}:{piece[:50]}".encode()
                        ).hexdigest()
                        chunks.append(
                            {
                                "id": chunk_id,
                                "text": piece,
                                "heading": heading_path,
                                "content_type": page_content_type,
                                "page_number": page_num,
                            }
                        )

                current_subheading = block_text.strip()
                page_text_buffer = ""
                page_content_type = "text"

            elif block_type == "image":
                heading_path = current_heading
                if current_subheading:
                    heading_path = f"{current_heading} > {current_subheading}"
                chunk_id = hashlib.md5(
                    f"img:{heading_path}:{page_num}".encode()
                ).hexdigest()
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": f"[Image on page {page_num}]",
                        "heading": heading_path,
                        "content_type": "image",
                        "page_number": page_num,
                    }
                )

            else:
                # Body text — detect tables
                if _detect_table_heuristic(block_text):
                    page_content_type = "table"
                page_text_buffer += " " + block_text

        # Flush remaining buffer at end of page
        if page_text_buffer.strip() and len(page_text_buffer.strip()) >= MIN_CHUNK_LENGTH:
            heading_path = current_heading
            if current_subheading:
                heading_path = f"{current_heading} > {current_subheading}"

            for piece in _split_long_chunk(page_text_buffer.strip()):
                chunk_id = hashlib.md5(
                    f"{heading_path}:{page_num}:{piece[:50]}".encode()
                ).hexdigest()
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": piece,
                        "heading": heading_path,
                        "content_type": page_content_type,
                        "page_number": page_num,
                    }
                )

    doc.close()

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
        top = chunk["heading"].split(" > ")[0]
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
