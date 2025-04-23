#!/usr/bin/env python3
"""
Script: step_1_BeautifulSoup.py

Purpose:
  Minimal HTML parsing using BeautifulSoup for text extraction.

Features:
- Removes <script>, <style>, <header>, <footer>, <nav>, <aside>, <img>, <figure> tags
- Extracts the resulting text
- Splits into paragraphs (naive approach)
- Outputs a minimal JSON: { doc_id, filename, raw_text, paragraphs }

Usage:
    python step_1_BeautifulSoup.py [input_dir] [output_dir]
Example:
    python step_1_BeautifulSoup.py data data_cleaned/basic_bs
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

from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
    Naive paragraph splitting: split on blank lines, fallback to single lines if only 1 chunk.
    """
    paras = re.split(r"\n\s*\n+", text)
    paras = [p.strip() for p in paras if p.strip()]
    if len(paras) <= 1:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        paras = lines
    return paras


def parse_single_html(file_path: Path) -> dict:
    """
    Minimal parse with BeautifulSoup:
      - Decompose script/style/header/footer/nav/aside/img/figure
      - Extract text
      - Return text fields (doc_id assigned in main)
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html_text = f.read()

    soup = BeautifulSoup(html_text, "lxml")

    # Remove typical boilerplate tags
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "img", "figure"]):
        tag.decompose()

    # Extract raw text
    raw_text = soup.get_text(separator="\n")

    # Normalize and split paragraphs
    clean_text = normalize_lines(raw_text)
    paragraphs = naive_paragraph_split(clean_text)

    return {
        "filename": file_path.name,
        "raw_text": clean_text,
        "paragraphs": paragraphs
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal HTML parse => JSON using BeautifulSoup.")
    parser.add_argument("input_dir", help="Directory containing .html files.")
    parser.add_argument("output_dir", help="Directory to write minimal JSON files.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("[BS-STEP1] Starting minimal BeautifulSoup parse...")
    start_time = time.time()
    count = 0

    for html_file in in_dir.rglob("*.html"):
        if not html_file.is_file():
            continue

        # Compute deterministic unique ID based on file's relative path
        rel = html_file.relative_to(in_dir).as_posix()
        doc_id = hashlib.sha1(rel.encode('utf-8')).hexdigest()
        logging.info(f"[BS-STEP1] Parsing {html_file} as doc_id={doc_id}")

        try:
            record = parse_single_html(html_file)
            record["doc_id"] = doc_id
        except Exception as e:
            logging.error(f"[BS-STEP1] Error reading {html_file}: {e}")
            continue

        # Write JSON
        out_path = out_dir / Path(rel).with_suffix('.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        count += 1
        logging.info(f"[BS-STEP1] Wrote => {out_path}")

    dur = time.time() - start_time
    logging.info(f"[BS-STEP1] Completed {count} HTML files in {dur:.2f}s.")


if __name__ == '__main__':
    main()