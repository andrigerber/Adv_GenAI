#!/usr/bin/env python3
"""
Script: step_1_hybrid.py

- Uses BeautifulSoup to pre-clean HTML (removing boilerplate navigation, scripts, footers, etc.) and extract key metadata (title and date).
- Feeds the cleaned HTML content into Docling's DocumentConverter for robust parsing of the main content.
- This combination ensures that boilerplate is removed (via BeautifulSoup) while leveraging Docling's structured parsing for the core content.
- Normalizes Unicode and collapses whitespace in the resulting text.
- Uses Lingua for language detection to handle multilingual content accurately.
- Extracts and standardizes dates using dateparser (with fallback to folder names when necessary).
- Applies spaCy (English/German models) for named entity extraction from content.
- Uses YAKE to retrieve top keywords from the content.
- Creates a simple summary using the first two sentences of the content.
- Outputs one JSON per input HTML file, containing: filename, language, title, date, source, main_content, named_entities, keywords, summary, plus placeholders for semantic_chunk_hints and embedding_vector.
- The code is organized for clarity and maintainability, making it easy to integrate future enhancements such as content chunking or embedding generation (for which placeholders are provided).
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

from docling.document_converter import DocumentConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Initialize Docling converter once
converter = DocumentConverter()

# Initialize Lingua language detector for relevant languages
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.ITALIAN
).with_preloaded_language_models().build()

# Cache for spaCy models and YAKE extractors (to avoid repeated loads)
spacy_models = {}
yake_extractors = {}

def detect_language(text: str) -> str:
    """Detect language code using Lingua."""
    if not text or text.isspace():
        return ""
    language = detector.detect_language_of(text)
    if language is None:
        return ""
    return language.iso_code_639_1.name.lower()

def extract_date(date_text: str, file_path: Path) -> str:
    """Standardize date string to 'YYYY-MM-DD' format, or fallback to year-month from path."""
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
        return f"{year}-{month if month else '01'}-01"
    return ""

def get_spacy_model(lang: str):
    """Get or load spaCy model for the given language code."""
    model_map = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        # "fr": "fr_core_news_sm",
        # "it": "it_core_news_sm"
    }
    if lang not in model_map:
        return None
    if lang in spacy_models:
        return spacy_models[lang]
    try:
        nlp = spacy.load(model_map[lang])
    except Exception as e:
        logging.error(f"spaCy load failed for '{lang}': {e}")
        return None
    spacy_models[lang] = nlp
    return nlp

def extract_entities(text: str, lang: str):
    """Extract named entities using spaCy NER."""
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
    """Extract top keywords with YAKE."""
    if not text or text.isspace():
        return []
    lang_code = lang if lang in ["en", "de", "fr", "it"] else "en"
    if lang_code not in yake_extractors:
        try:
            yake_extractors[lang_code] = yake.KeywordExtractor(lan=lang_code, n=3, top=top_k)
        except Exception as e:
            logging.error(f"YAKE init error for '{lang_code}': {e}")
            yake_extractors[lang_code] = yake.KeywordExtractor(lan="en", n=3, top=top_k)
    extractor = yake_extractors[lang_code]
    try:
        kw_scores = extractor.extract_keywords(text)
    except Exception as e:
        logging.error(f"YAKE keyword extraction error: {e}")
        return []
    kw_scores.sort(key=lambda tup: tup[1])
    return [kw for kw, score in kw_scores[:top_k]]

def summarize_text(text: str, lang: str):
    """Generate a brief summary (first two sentences of the text)."""
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
    """Process a single HTML file using the hybrid BeautifulSoup + Docling approach."""
    html_content = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_content, "lxml")
    # Extract title
    title_tag = soup.find("h1")
    title_text = title_tag.get_text().strip() if title_tag else (soup.title.string.strip() if soup.title else "")
    if title_tag:
        title_tag.decompose()
    # Remove boilerplate tags and elements
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "img", "figure"]):
        tag.decompose()
    # Remove category list after title (if any)
    first_list = soup.find(["ul", "ol"])
    if first_list:
        first_list.decompose()
    # Extract date text and remove its element
    date_text = ""
    date_node = soup.find(string=re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}"))
    if date_node:
        m = re.search(r"\d{1,2}\.\d{1,2}\.\d{4}", date_node)
        if m:
            date_text = m.group(0)
        parent = date_node.find_parent()
        if parent:
            parent.decompose()
    # Remove breadcrumb if exists
    breadcrumb = soup.find(string=re.compile("You are here"))
    if breadcrumb:
        bc_parent = breadcrumb.find_parent("div")
        if bc_parent:
            bc_parent.decompose()
        else:
            breadcrumb.replace_with("")
    # Use Docling to parse the cleaned HTML content
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp.write(str(soup).encode("utf-8"))
            tmp.flush()
            tmp_path = tmp.name
        conv_result = converter.convert(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    try:
        text_content = conv_result.document.export_to_text()
    except Exception as e:
        logging.error(f"Docling export_to_text failed for {file_path}: {e}")
        text_content = soup.get_text(separator="\n")
    # Normalize Unicode and cleanup whitespace
    lines = [unicodedata.normalize("NFC", line.strip()) for line in text_content.splitlines() if line.strip()]
    # Remove any leftover title or nav lines from text
    clean_lines = []
    for line in lines:
        if title_text and line == title_text:
            continue
        low = line.lower()
        if low in ("homepage", "news & events", "eth news", "all articles", "alle artikel"):
            continue
        if re.fullmatch(r"\d{4}", line):
            continue
        if line in ["January","February","March","April","May","June","July","August","September","October","November","December",
                    "Januar","Februar","März","April","Mai","Juni","Juli","August","September","Oktober","November","Dezember"]:
            continue
        clean_lines.append(line)
    clean_text = "\n".join(clean_lines)
    # Language detection
    lang_code = detect_language(clean_text)
    # Standardize date
    date_standard = extract_date(date_text, file_path)
    # Named entities, keywords, summary
    entities = extract_entities(clean_text, lang_code)
    keywords = extract_keywords(clean_text, lang_code, top_k=10)
    summary = summarize_text(clean_text, lang_code)
    # Compile result dictionary
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
        description="Hybrid HTML parser using BeautifulSoup and Docling."
    )
    parser.add_argument("input_dir", help="Directory with HTML files to process")
    parser.add_argument("output_dir", help="Directory to save cleaned JSON files")
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
            logging.error(f"Error processing {html_file}: {e}")
            continue

        # Write JSON output
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        file_count += 1
        file_elapsed = time.time() - file_start
        logging.info(f"[BS&D] Processed {html_file} -> {out_file} in {file_elapsed:.3f} sec")

    total_elapsed = time.time() - start_time
    logging.info(f"Completed processing {file_count} files in {total_elapsed:.3f} seconds.")

if __name__ == "__main__":
    main()
