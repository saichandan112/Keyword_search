# Keyword_search
This repo is for keyword based search results from a folder of ".pdf", ".docx", ".txt", ".xlsx", ".pptx".

This Python-based CLI tool enables users to search and summarize content from multiple document formats within a specified folder. It performs file-only, extractive summarization, ensuring that every result shown is directly taken from the source documents (no paraphrasing).
Supported file formats:

PDF (.pdf)
Word (.docx)
Text (.txt)
Excel (.xlsx)
PowerPoint (.pptx)

<img width="1904" height="858" alt="Screenshot 2026-05-11 065926" src="https://github.com/user-attachments/assets/4610c9ce-6a8a-4101-a288-849512c4e387" />

With output as 

<img width="1897" height="978" alt="Screenshot 2026-05-11 065950" src="https://github.com/user-attachments/assets/9916ac49-372f-4cbb-ab54-c69400033406" />

⚙️ How It Works
1. Text Extraction
Each file type is processed using appropriate libraries:

PyPDF2 → PDFs
python-docx → Word
pandas → Excel
python-pptx → PowerPoint

Text is then cleaned and normalized.

2. Document Loading

Reads files from the given folder.
Filters out short or empty content (<15 words).


3. Chunking

Splits text into overlapping word chunks:

Default: 180 words per chunk
40-word overlap




4. Index Creation

Builds TF-IDF vectors for all chunks.
Stores metadata (file path, chunk ID, text).


5. Search Process

Converts user keywords into a query.
Computes cosine similarity with indexed chunks.
Filters results based on:

Score threshold
Keyword presence (AND/OR logic)




6. Sentence Ranking
Extracted sentences are ranked using:

Similarity to query (65%)
Word frequency importance (20%)
Query term coverage (15%)


7. Output Generation
Results are formatted into:

TL;DR section
Key bullet points
Evidence snippets
Per-document summaries

Each sentence includes a citation reference:
**[file_name | chunk_id]**
