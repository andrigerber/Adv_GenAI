#!/usr/bin/env python3
"""
Script: step_1_Docling.py

- Uses Docling's DocumentConverter to parse HTML files into a unified structured format.
- Employs Docling to extract the main textual content (after converting to text) from each HTML, while still performing custom boilerplate removal for accuracy.
- Normalizes Unicode and cleans up whitespace in the extracted text.
- Utilizes Lingua for language detection across multilingual content.
- Extracts and standardizes dates using dateparser (with folder name fallback for missing day).
- Uses spaCy (English/German models) for named entity extraction.
- Uses YAKE to extract top keywords from the content.
- Generates a simple summary (first two sentences) from the content.
- Outputs a JSON file per input HTML with fields: filename, language, title, date, source, main_content, named_entities, keywords, summary, plus placeholders for semantic_chunk_hints and embedding_vector.
- Structured for clarity with functions and logging, making future extensions (like content chunking or vector embedding integration) straightforward.
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

# Import Docling DocumentConverter
from docling.document_converter import DocumentConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Initialize Docling converter once (downloads models on first use if not cached)
converter = DocumentConverter()

# Initialize language detector
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN
).with_preloaded_language_models().build()

# Cache for spaCy models and YAKE extractors
spacy_models = {}
yake_extractors = {}

def detect_language(text: str) -> str:
    """Detect language code of text using Lingua detector."""
    if not text or text.isspace():
        return ""
    language = detector.detect_language_of(text)
    if language is None:
        return ""
    return language.iso_code_639_1.name.lower()

def extract_date(date_text: str, file_path: Path) -> str:
    """Parse date string to standard YYYY-MM-DD using dateparser, with fallback to path."""
    if date_text:
        dt = dateparser.parse(date_text, languages=["en", "de", "fr", "it"])
        if dt:
            return dt.strftime("%Y-%m-%d")
    year, month = None, None
    for part in file_path.parts:
        if re.fullmatch(r"\d{4}", part):
            year = part
        if re.fullmatch(r"(0?[1-9]|1[0-2])", part):
            if year:
                month = part.zfill(2)
    if year:
        if month:
            return f"{year}-{month}-01"
        else:
            return f"{year}-01-01"
    return ""

def get_spacy_model(lang: str):
    """Load or retrieve spaCy NLP model for given language code."""
    model_name_map = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        # "fr": "fr_core_news_sm",
        # "it": "it_core_news_sm"
    }
    if lang not in model_name_map:
        return None
    if lang in spacy_models:
        return spacy_models[lang]
    try:
        nlp = spacy.load(model_name_map[lang])
    except Exception as e:
        logging.error(f"Could not load spaCy model for '{lang}': {e}")
        return None
    spacy_models[lang] = nlp
    return nlp

def extract_entities(text: str, lang: str):
    """Extract unique named entities from text using spaCy."""
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
    """Extract top keywords using YAKE."""
    if not text or text.isspace():
        return []
    lang_code = lang if lang in ["en", "de", "fr", "it"] else "en"
    if lang_code not in yake_extractors:
        try:
            yake_extractors[lang_code] = yake.KeywordExtractor(lan=lang_code, n=3, top=top_k)
        except Exception as e:
            logging.error(f"YAKE init failed for lang '{lang_code}': {e}")
            yake_extractors[lang_code] = yake.KeywordExtractor(lan="en", n=3, top=top_k)
    extractor = yake_extractors[lang_code]
    try:
        kw_scores = extractor.extract_keywords(text)
    except Exception as e:
        logging.error(f"YAKE extraction error: {e}")
        return []
    kw_scores.sort(key=lambda tup: tup[1])
    keywords = [kw for kw, score in kw_scores[:top_k]]
    return keywords

def summarize_text(text: str, lang: str):
    """Return first two sentences of text as a simple summary."""
    if not text:
        return ""
    nlp = get_spacy_model(lang)
    summary = ""
    if nlp:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if len(sentences) >= 2:
            summary = sentences[0] + " " + sentences[1]
        elif len(sentences) == 1:
            summary = sentences[0]
    else:
        parts = re.split(r'(?<=[.!?]) +', text)
        if len(parts) >= 2:
            summary = parts[0].strip() + " " + parts[1].strip()
        elif parts:
            summary = parts[0].strip()
    return summary

def process_html_file(file_path: Path) -> dict:
    """Process a single HTML file with Docling and return structured data."""
    html_content = file_path.read_text(encoding="utf-8", errors="ignore")
    # Use BeautifulSoup minimally to extract title and remove obvious boilerplate before Docling
    soup = BeautifulSoup(html_content, "lxml")
    title_tag = soup.find("h1")
    title_text = title_tag.get_text().strip() if title_tag else (soup.title.string.strip() if soup.title else "")
    # Remove title from content if present
    if title_tag:
        title_tag.decompose()
    # Remove boilerplate tags to simplify content for Docling
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "img", "figure"]):
        tag.decompose()
    # Remove category list (ul/ol) after title
    # (title_tag is removed, so find any leading category list at top)
    first_list = soup.find(["ul", "ol"])
    if first_list:
        # Only remove if it contains likely category links (heuristic: small list near top)
        first_list.decompose()
    # Identify and capture date text, then remove its container
    date_text = ""
    date_node = soup.find(string=re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}"))
    if date_node:
        match = re.search(r"\d{1,2}\.\d{1,2}\.\d{4}", date_node)
        if match:
            date_text = match.group(0)
        parent = date_node.find_parent()
        if parent:
            parent.decompose()
    # Remove breadcrumb if present
    breadcrumb = soup.find(string=re.compile("You are here"))
    if breadcrumb:
        bc_parent = breadcrumb.find_parent("div")
        if bc_parent:
            bc_parent.decompose()
        else:
            breadcrumb.replace_with("")
    # Convert the cleaned HTML content to text using Docling
    try:
        # We supply the HTML content as a temporary file for Docling to parse
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp.write(str(soup).encode("utf-8"))
            tmp.flush()
            tmp_path = tmp.name
        conv_result = converter.convert(tmp_path)
    finally:
        # Remove temporary file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    # Export document to plain text
    try:
        full_text = conv_result.document.export_to_text()
    except Exception as e:
        logging.error(f"Docling export_to_text failed for {file_path}: {e}")
        full_text = soup.get_text(separator="\n")  # fallback to BeautifulSoup text
    # Normalize Unicode and whitespace
    lines = [unicodedata.normalize("NFC", line.strip()) for line in full_text.splitlines() if line.strip()]
    # Remove any lingering title or breadcrumb lines from text if present
    clean_lines = []
    for line in lines:
        # Drop title line if exactly matches title_text
        if title_text and line == title_text:
            continue
        # Drop known navigation breadcrumbs or category lines
        low = line.lower()
        if low in ("homepage", "news & events", "news & events", "eth news", "all articles", "alle artikel"):
            continue
        # Drop standalone year or month names if likely breadcrumb
        if re.fullmatch(r"\d{4}", line):
            continue
        if line in ["January","February","March","April","May","June","July","August","September","October","November","December",
                    "Januar","Februar","März","April","Mai","Juni","Juli","August","September","Oktober","November","Dezember"]:
            continue
        clean_lines.append(line)
    clean_text = "\n".join(clean_lines)
    # Detect language
    lang_code = detect_language(clean_text)
    # Standardize date string
    date_standard = extract_date(date_text, file_path)
    # Extract entities, keywords, summary
    entities = extract_entities(clean_text, lang_code)
    keywords = extract_keywords(clean_text, lang_code, top_k=10)
    summary = summarize_text(clean_text, lang_code)
    # Prepare result dictionary
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
        description="Parse HTML files with Docling and extract structured data."
    )
    parser.add_argument("input_dir", help="Path to input directory with HTML files")
    parser.add_argument("output_dir", help="Path to output directory for JSON files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logging.error(f"Input directory '{input_dir}' not found.")
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
            logging.error(f"Failed to process {html_file}: {e}")
            continue

        # Write JSON output
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        file_count += 1
        file_elapsed = time.time() - file_start
        logging.info(f"[Docling] Processed {html_file} -> {out_file} in {file_elapsed:.3f} sec")

    total_elapsed = time.time() - start_time
    logging.info(f"Completed processing {file_count} files in {total_elapsed:.3f} seconds.")

if __name__ == "__main__":
    main()
