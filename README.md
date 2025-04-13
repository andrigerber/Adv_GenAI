# Adv_GenAI

A repository for advanced generative AI projects focused on building a Retrieval-Augmented Generation (RAG) system.

## Directory Structure

### Code Directory
- **README.txt** – Introductory guide for the codebase.
- **config_v2.txt** – Azure AI configuration (server URL, API key, deployment details).
- **digester_cmd_v2.py** – Processes PDFs (using Tesseract for OCR) and saves embeddings in `./chroma_db`.
- **openaicalls_cmd_v2.py** – Sets up LLM chat and embedding tasks via API calls.
- **rag_retrieve_cmd_v2.py** – Implements the RAG pipeline for Q&A.
- **reqs.txt** – Lists required Python libraries.

### Benchmark Directory
- **BenchmarkQuestions.pdf** – Set of benchmark questions.
- **BenchmarkQuestionsAnswers.pdf** – Answers for validating system performance.

### Data Directory
- Stores datasets used for training, testing, and benchmarking the RAG system.

### Project Requirements
- **Project Requirements.pdf** – Detailed specifications covering:
  - Data preparation (HTML parsing, cleaning, metadata enrichment)
  - RAG system development (pre-retrieval strategies, re-ranking, hybrid retrieval)
  - Evaluation (automated metrics and human assessment)  
  :contentReference[oaicite:0]{index=0}

## Usage

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