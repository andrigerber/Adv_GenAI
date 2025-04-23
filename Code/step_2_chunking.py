#!/usr/bin/env python3
"""
Script: step_2_chunking.py

Purpose:
- Reads JSON files produced by any of the step_1 scripts (BeautifulSoup, Docling, Hybrid).
- For each JSON (representing a parsed doc), loads 'main_content' and:
    1) Creates fixed-size chunks by character count (e.g., chunk_size=512, overlap=50).
    2) Creates true semantic chunks by measuring adjacency sentence embedding similarity (embedding threshold).

- Appends "chunks_fixed" and "chunks_semantic" fields to the JSON record, then saves it
  to a specified output directory, preserving subfolders if desired.

Features:
- Offers CLI arguments for chunk size, overlap, and adjacency similarity threshold.
- Uses spaCy for sentence splitting and SentenceTransformer for generating embeddings.
- Allows quick re-running of the chunking step if new chunking parameters are desired,
  without re-parsing the original HTML.

Usage:
    python step_2_chunking.py [input_json_dir] [output_json_dir]
    --fixed_chunk_size 512 --fixed_overlap 50 --similarity_threshold 0.7
"""

import os
import re
import json
import time
import logging
import argparse
import unicodedata
from pathlib import Path
from typing import List

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def chunk_text_fixed_chars(text: str, chunk_size=512, overlap=50) -> List[str]:
    """
    Generate fixed-size overlapping chunks of a given text.

    Args:
        text (str): The entire content of an article/document.
        chunk_size (int): The maximum number of characters in each chunk.
        overlap (int): The number of characters each subsequent chunk overlaps
                       with the previous one to avoid abrupt context breaks.

    Returns:
        List[str]: A list of chunk strings.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break

    return chunks


def chunk_text_semantic_embeddings(
        text: str,
        nlp_model: spacy.Language,
        embed_model: SentenceTransformer,
        similarity_threshold: float = 0.7
) -> List[str]:
    """
    Create semantic chunks by splitting text into sentences (spaCy),
    then merging adjacent sentences while their cosine similarity is above a threshold.

    Steps:
    1) Sentence-split the text using the spaCy model.
    2) Embed each sentence with a SentenceTransformer model.
    3) For each adjacent pair, compute cosine similarity.
    4) If similarity < threshold, start a new chunk.

    Args:
        text (str): The entire document/article text.
        nlp_model (spacy.Language): A loaded spaCy model for sentence tokenization.
        embed_model (SentenceTransformer): The sentence embedding model.
        similarity_threshold (float): If adjacency similarity < this, create a new chunk boundary.

    Returns:
        List[str]: A list of semantically coherent chunk strings.
    """
    # Sentence-splitting
    doc = nlp_model(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sentences:
        return []

    # Encode each sentence into a vector
    sent_embeddings = embed_model.encode(sentences, show_progress_bar=False)

    chunks = []
    current_chunk = [sentences[0]]
    for i in range(len(sentences) - 1):
        emb_i = sent_embeddings[i]
        emb_j = sent_embeddings[i + 1]
        dot = np.dot(emb_i, emb_j)
        norm = (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)) + 1e-9
        similarity = dot / norm

        if similarity < similarity_threshold:
            # new chunk boundary
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    # leftover sentences in the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate fixed-size and semantic chunks for each JSON doc.")
    parser.add_argument("input_dir", help="Directory containing JSON files from Step1")
    parser.add_argument("output_dir", help="Directory to write the new chunked JSON files")
    parser.add_argument("--fixed_chunk_size", type=int, default=512, help="Character size for fixed chunking")
    parser.add_argument("--fixed_overlap", type=int, default=50,
                        help="Overlap in characters for consecutive fixed chunks")
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                        help="Adjacency sentence similarity threshold for semantic chunking")

    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading spaCy 'en_core_web_sm' for sentence splitting (semantic chunking)...")
    nlp_en = spacy.load("en_core_web_sm")

    logging.info("Loading SentenceTransformer embedding model for semantic adjacency checks...")
    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    start_time = time.time()
    count = 0

    for json_file in in_dir.rglob("*.json"):
        # For each JSON record
        with open(json_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        main_text = record.get("main_content", "")
        lang = record.get("language", "en")  # fallback if not specified

        # 1) Fixed-size chunking
        fixed_chunks = chunk_text_fixed_chars(
            main_text,
            chunk_size=args.fixed_chunk_size,
            overlap=args.fixed_overlap
        )

        # 2) True semantic chunking if English
        #    (If you want to handle German or other languages,
        #     you'd load additional spaCy models accordingly.)
        if lang == "en":
            semantic_chunks = chunk_text_semantic_embeddings(
                main_text,
                nlp_en,
                embed_model,
                similarity_threshold=args.similarity_threshold
            )
        else:
            # fallback: entire text as single chunk if not English
            semantic_chunks = [main_text]

        # Save these chunk lists back into the record
        record["chunks_fixed"] = fixed_chunks
        record["chunks_semantic"] = semantic_chunks

        # Write out new JSON
        rel = json_file.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(record, wf, indent=2, ensure_ascii=False)

        count += 1
        logging.info(f"[STEP2-CHUNKING] Wrote chunked JSON for {json_file} -> {out_path}")

    dur = time.time() - start_time
    logging.info(f"Completed chunking on {count} JSON files in {dur:.2f} seconds total.")


if __name__ == "__main__":
    main()
