#!/usr/bin/env python3
"""
Script: step_1_Docling.py

Purpose:
  Minimal HTML parsing using Docling.
  We rely on docling.DocumentConverter for text extraction.

Features:
- Directly feeds each .html file to docling.DocumentConverter
- Exports the text
- Minimal line cleaning & naive paragraph splitting
- Outputs a minimal JSON: { doc_id, filename, raw_text, paragraphs }

Usage:
    python step_1_Docling.py [input_dir] [output_dir]
Example:
    python step_1_Docling.py data data_cleaned/docling_bs
"""
import os
import re
import json
import time
import logging
import argparse
import unicodedata
import hashlib
from pathlib import Path

from docling.document_converter import DocumentConverter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
converter = DocumentConverter()


def normalize_lines(text: str) -> str:
    """
    Normalize unicode, strip each line, and remove empty lines.
    """
    lines = [
        unicodedata.normalize("NFC", ln.strip())
        for ln in text.splitlines()
        if ln.strip()
    ]
    return "\n".join(lines)


def naive_paragraph_split(text: str) -> list[str]:
    """
    Same paragraph logic as in the BS script:
    Split on blank lines, fallback single line if only 1 chunk.
    """
    paras = re.split(r"\n\s*\n+", text)
    paras = [p.strip() for p in paras if p.strip()]
    if len(paras) <= 1:
        paras = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return paras


def parse_docling_only(html_file: Path) -> dict:
    """
    Pure docling approach:
      - converter.convert() on the .html file
      - doc_text = doc_data.document.export_to_text()
      - unify lines -> paragraphs
      - doc_id assigned in main
    """
    doc_data = converter.convert(str(html_file))
    doc_text = doc_data.document.export_to_text()

    raw_text = normalize_lines(doc_text)
    paragraphs = naive_paragraph_split(raw_text)

    return {
        "filename": html_file.name,
        "raw_text": raw_text,
        "paragraphs": paragraphs
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal HTML parse => JSON using Docling.")
    parser.add_argument("input_dir", help="Folder with .html files")
    parser.add_argument("output_dir", help="Folder to store minimal JSON output")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    count = 0

    for html_file in in_dir.rglob("*.html"):
        if not html_file.is_file():
            continue
        rel = html_file.relative_to(in_dir).as_posix()
        doc_id = hashlib.sha1(rel.encode('utf-8')).hexdigest()
        logging.info(f"[DOCLING-STEP1] Processing {html_file} as doc_id={doc_id}")

        try:
            record = parse_docling_only(html_file)
            record["doc_id"] = doc_id
        except Exception as e:
            logging.error(f"[DOCLING-STEP1] Error parsing {html_file} with Docling: {e}")
            continue

        out_path = out_dir / Path(rel).with_suffix('.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        count += 1
        logging.info(f"[DOCLING-STEP1] Wrote => {out_path}")

    duration = time.time() - start_time
    logging.info(f"[DOCLING-STEP1] Finished {count} files in {duration:.2f}s")


if __name__ == '__main__':
    main()
