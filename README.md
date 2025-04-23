# Adv_GenAI

A repository for advanced generative AI projects focused on building a Retrieval-Augmented Generation (RAG) system.

## Directory Structure

### Code Directory
- **Code from Professor:**
  - **README.txt** – Introductory guide for the codebase.
  - **config_v2.txt** – Azure AI configuration (server URL, API key, deployment details).
  - **digester_cmd_v2.py** – Processes PDFs (using Tesseract for OCR) and saves embeddings in `./chroma_db`.
  - **openaicalls_cmd_v2.py** – Sets up LLM chat and embedding tasks via API calls.
  - **rag_retrieve_cmd_v2.py** – Implements the RAG pipeline for Q&A.
  - **reqs.txt** – Lists required Python libraries.
- **Code from Project:**
  - **step_1_BeautifulSoup.py** – Parses HTML files using BeautifulSoup.
  - **step_1_Docling.py** – Parses HTML files using Docling.
  - **step_1_hybrid.py** – Hybrid approach combining BeautifulSoup and Docling.

### Benchmark Directory
- **BenchmarkQuestions.pdf** – Set of benchmark questions.
- **BenchmarkQuestionsAnswers.pdf** – Answers for validating system performance.

### Data Directory
- Stores datasets used for preprocessing, training, testing, and benchmarking the RAG system.

### Project Requirements
- **Project Requirements.pdf** – Detailed specifications covering:
  - Data preparation (HTML parsing, cleaning, metadata enrichment)
  - RAG system development (pre-retrieval strategies, re-ranking, hybrid retrieval)
  - Evaluation (automated metrics and human assessment)

## Usage
**Code from Professor:**
1. **Setup:**
   ```bash
   pip install -r Code/reqs.txt
    ```

2. Update `config_v2.txt` with your Azure AI credentials.

**Generate Embeddings:**

```bash
python Code/digester_cmd_v2.py <PDF_directory>
```

3. **Run the RAG Pipeline:**

```bash
python Code/rag_retrieve_cmd_v2.py "Your query here"
```

## Overview
**Data Preparation:** Extract and clean text from HTML files and enrich with metadata.

**RAG System:** Leverages BM25, semantic search, GraphRAG, and hybrid retrieval techniques.

**Evaluation:** Utilizes automated metrics (e.g., Precision@k, Recall, MRR) and human assessments.


-----------------------
# --> Acutal documentation starts here

# Advanced Generative AI Project: Retrieval-Augmented Generation (RAG) System

**Note:** Copy the **HKNews** data in the data folder:
![img.png](pictures/img.png)

## Step 1 Data Preparation – Structuring the Dataset
**Objective:** Prepare a structured dataset of German and English news articles by extracting and cleaning text from HTML files, enriching it with language metadata and sources, and storing it in a format suitable for retrieval in a RAG system.
**Loading, Parsing, and Cleaning HTML Files (5 Points)**

**Environment Setup:**
```bash
.venv\Scripts\activate
```

**Install:**
```bash
.venv\Scripts\activate
# Core libs
pip install bs4
pip install docling
pip install dateparser
pip install yake
pip install lingua-language-detector
pip install spacy
pip install nltk
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab


# For spaCy NER in German & English, also download the models:
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

1. Use [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) to extract raw text from .html files while removing unnecessary elements such as JavaScript, CSS, or HTML tags.

**Run:**
```bash
python Code/step_1_BeautifulSoup.py data data_cleaned/data_cleaned_BS
```
2. Use [Docling](https://github.com/docling-project/docling) for advanced document parsing, especially if handling potentially complex layouts, tables or non-standard structures

**Run:**
```bash
python python Code/step_1_Docling.py data data_cleaned/data_cleaned_D
```

3. Implement a hybrid approach (BeautifulSoup + Docling) to compare the effectiveness of both tools and optimize extraction quality.

**Run:**
```bash
python Code/step_1_hybrid.py data data_cleaned/data_cleaned_BSD
```

**Multilingual Text Preprocessing and Cleaning (5 points)**

  1. Perform necessary text preprocessing (e.g., removing extra spaces and redundant line breaks, normalizing Unicode characters, standardizing date formats from different sources), and handle German-specific text processing (e.g., compound words, umlaut normalization if needed) .
  2. Store the cleaned text and its metadata in a structured format suitable for retrieval (e.g., JSON, CSV, or a database) with fields such as language, title, date, source, main content, named entities, topics, keywords, summary.
  3. Create additional rich metadata that can support future semantic search and context filtering to enhance document retrieval later, and store metadata in a structured database (e.g., SQLite, Pandas DataFrame, or JSON format) for efficient access.

We add metadata to the cleaned data, including:
```json
data_dict = {
        "filename": file_path.name,   # the name of the file without the path (end with ".html")
        "language": lang_code,        # decided by lingua e.g. "de", "en"
        "title": title_text,          # not filled yet could maybe take file name without ".html"
        "date": date_standard,        # took from the file path
        "source": "ETH News",         # same for all
        "main_content": clean_text,   # the main content of the article
        "named_entities": entities,   # with spaCy
        "keywords": keywords,         # with YAKE
        "summary": summary,           # at the moment only the first two sentences of "main_content"
        "semantic_chunk_hints": [],   # not filled yet
        "embedding_vector": []        # not filled yet
    }

```