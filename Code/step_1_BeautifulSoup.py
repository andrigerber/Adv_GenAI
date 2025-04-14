#!/usr/bin/env python3
"""
Script: step_1_BeautifulSoup.py

Refactored for improved HTML content extraction using BeautifulSoup,
including runtime logging (both total time and optional per-file time).

Features:
- Utilizes BeautifulSoup to parse HTML and remove boilerplate elements (scripts, styles, navigation, footer, etc.).
- Normalizes Unicode characters and collapses excess whitespace for clean text.
- Uses Lingua for accurate language detection (English, German, French, Italian).
- Extracts and standardizes publication dates with dateparser, falling back to folder names if needed.
- Integrates spaCy (English, German) for named entity recognition.
- Integrates YAKE for keyword extraction of top key phrases.
- Generates a basic summary from the first two sentences of the content.
- Outputs structured JSON with fields:
    - filename
    - language
    - title
    - date
    - source
    - main_content
    - named_entities
    - keywords
    - summary
    - semantic_chunk_hints (placeholder)
    - embedding_vector (placeholder)
- Includes logging of total script runtime and (optionally) individual file times.

Future Extensibility:
- The placeholders (semantic_chunk_hints, embedding_vector) allow for subsequent integration
  of advanced chunking or vector-based indexing, facilitating retrieval in a RAG system.
"""

import os
import re
import json
import time
import logging
import argparse
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder
import dateparser
import spacy
import yake

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Initialize language detector (restrict to relevant languages for efficiency)
detector = (
    LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN
    )
    .with_preloaded_language_models()
    .build()
)

# Cache for spaCy models and YAKE extractors to avoid re-loading for each file
spacy_models = {}
yake_extractors = {}

def detect_language(text: str) -> str:
    """Detect language code of text using Lingua detector."""
    if not text or text.isspace():
        return ""
    language = detector.detect_language_of(text)
    if language is None:
        return ""
    # Use ISO 639-1 two-letter code in lowercase (e.g., 'en', 'de')
    lang_code = language.iso_code_639_1.name.lower()
    return lang_code

def extract_date(date_text: str, file_path: Path) -> str:
    """
    Parse date from text using dateparser, fallback to year-month from path
    if no date is found in the text.
    Returns date in 'YYYY-MM-DD' format or '' if none can be derived.
    """
    if date_text:
        dt = dateparser.parse(date_text, languages=["en", "de", "fr", "it"])
        if dt:
            return dt.strftime("%Y-%m-%d")
    # Fallback: look for year and month in file path
    year, month = None, None
    for part in file_path.parts:
        if re.fullmatch(r"\d{4}", part):
            year = part
        if re.fullmatch(r"\d{1,2}", part) or re.fullmatch(r"(0?[1-9]|1[0-2])", part):
            # Only use as month if year already found
            if year:
                month = part.zfill(2)
    if year:
        if month:
            return f"{year}-{month}-01"
        else:
            return f"{year}-01-01"
    return ""

def get_spacy_model(lang: str):
    """
    Load or retrieve cached spaCy model for the given language code.
    Returns None if the language isn't supported or loading fails.
    """
    model_name_map = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        # "fr": "fr_core_news_sm",
        # "it": "it_core_news_sm",
    }
    if lang not in model_name_map:
        return None
    if lang in spacy_models:
        return spacy_models[lang]
    try:
        nlp = spacy.load(model_name_map[lang])
    except Exception as e:
        logging.error(f"spaCy model load failed for language '{lang}': {e}")
        return None
    spacy_models[lang] = nlp
    return nlp

def extract_entities(text: str, lang: str):
    """Extract named entities from text using spaCy for the specified language code."""
    nlp = get_spacy_model(lang)
    if not nlp:
        return []
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        ent_text = ent.text.strip()
        ent_label = ent.label_
        if ent_text and ent_text not in seen:
            seen.add(ent_text)
            entities.append({"text": ent_text, "label": ent_label})
    return entities

def extract_keywords(text: str, lang: str, top_k: int = 10):
    """
    Extract top keywords from text using YAKE (3-gram).
    Returns a list of keyword strings.
    """
    if not text or text.isspace():
        return []
    lang_code = lang if lang in ["en", "de", "fr", "it"] else "en"
    # Initialize YAKE extractor for the language if not already
    if lang_code not in yake_extractors:
        try:
            yake_extractors[lang_code] = yake.KeywordExtractor(lan=lang_code, n=3, top=top_k)
        except Exception as e:
            logging.error(f"YAKE initialization failed for lang '{lang_code}': {e}")
            yake_extractors[lang_code] = yake.KeywordExtractor(lan="en", n=3, top=top_k)
    extractor = yake_extractors[lang_code]
    try:
        keywords_with_scores = extractor.extract_keywords(text)
    except Exception as e:
        logging.error(f"YAKE keyword extraction failed: {e}")
        return []
    # Sort keywords by ascending score => best first
    keywords_with_scores.sort(key=lambda tup: tup[1])
    keywords = [kw for kw, score in keywords_with_scores[:top_k]]
    return keywords

def summarize_text(text: str, lang: str) -> str:
    """Create a simple summary by returning the first two sentences of the text."""
    if not text:
        return ""
    nlp = get_spacy_model(lang)
    summary = ""
    if nlp:
        # Use spaCy sentence segmentation if available
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if len(sentences) >= 2:
            summary = sentences[0] + " " + sentences[1]
        elif len(sentences) == 1:
            summary = sentences[0]
    else:
        # Fallback: naive approach splitting on sentence terminators
        parts = re.split(r'(?<=[.!?]) +', text)
        if len(parts) >= 2:
            summary = parts[0].strip() + " " + parts[1].strip()
        elif len(parts) == 1:
            summary = parts[0].strip()
    return summary

def process_html_file(file_path: Path) -> dict:
    """
    Process a single HTML file and return the structured data as a dict.
    This function:
     - Reads and cleans HTML with BeautifulSoup
     - Extracts title, date
     - Removes boilerplate
     - Normalizes whitespace
     - Detects language
     - Extracts NER, keywords, summary
     - Returns result
    """
    html_text = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "lxml")
    # Remove boilerplate tags
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "img", "figure"]):
        tag.decompose()

    # Extract title
    title_tag = soup.find(["h1", "title"])
    title_text = title_tag.get_text().strip() if title_tag else ""
    # If it's an <h1>, remove from content
    if title_tag and title_tag.name.lower() == "h1":
        title_tag.decompose()
    # Remove a possible category list after the title
    if title_tag:
        next_node = title_tag.next_sibling if title_tag else None
        if next_node and hasattr(next_node, "name") and next_node.name in ["ul", "ol"]:
            next_node.decompose()

    # Attempt to find date text (e.g. '12.05.2022') and remove it from soup
    date_text = ""
    date_node = soup.find(string=re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}"))
    if date_node:
        m = re.search(r"\d{1,2}\.\d{1,2}\.\d{4}", date_node)
        if m:
            date_text = m.group(0)
        # Remove the entire parent element, if any
        try:
            parent = date_node.find_parent()
            if parent:
                parent.decompose()
        except Exception:
            pass

    # Remove breadcrumbs if contain "You are here"
    breadcrumb = soup.find(string=re.compile("You are here"))
    if breadcrumb:
        try:
            bc_parent = breadcrumb.find_parent("div")
            if bc_parent:
                bc_parent.decompose()
            else:
                breadcrumb.replace_with("")
        except Exception:
            pass

    # Convert the remaining HTML to text
    text = soup.get_text(separator="\n")
    # Normalize and clean whitespace
    lines = [unicodedata.normalize("NFC", line.strip()) for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    # Language detection
    lang_code = detect_language(clean_text)
    # Standardize date from text or fallback path
    date_standard = extract_date(date_text, file_path)

    # Named entities
    entities = extract_entities(clean_text, lang_code)
    # Keywords
    keywords = extract_keywords(clean_text, lang_code, top_k=10)
    # Summary
    summary = summarize_text(clean_text, lang_code)

    # Prepare final result
    result = {
        "filename": file_path.name,
        "language": lang_code,
        "title": title_text,
        "date": date_standard,
        "source": "ETH News",
        "main_content": clean_text,
        "named_entities": entities,
        "keywords": keywords,
        "summary": summary,
        "semantic_chunk_hints": [],
        "embedding_vector": []
    }
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Clean HTML files with BeautifulSoup and extract structured data, plus runtime logging."
    )
    parser.add_argument("input_dir", help="Path to input directory containing HTML files")
    parser.add_argument("output_dir", help="Path to output directory for JSON files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logging.error(f"Input directory '{input_dir}' does not exist.")
        return

    # Track total script runtime
    start_time = time.time()

    file_count = 0
    # Optionally track per-file time
    for html_file in input_dir.rglob("*.html"):
        file_start = time.time()
        rel_path = html_file.relative_to(input_dir)
        out_file = output_dir / rel_path.with_suffix(".json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = process_html_file(html_file)
        except Exception as e:
            logging.error(f"Error processing file {html_file}: {e}")
            continue

        # Write JSON output
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        file_count += 1
        file_elapsed = time.time() - file_start
        logging.info(f"[BS4] Processed {html_file} -> {out_file} in {file_elapsed:.3f} sec")

    total_elapsed = time.time() - start_time
    logging.info(f"Completed processing {file_count} files in {total_elapsed:.3f} seconds.")

if __name__ == "__main__":
    main()
