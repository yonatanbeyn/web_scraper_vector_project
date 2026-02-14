# LangChain Version Guide

## Overview

`scrape_and_store_langchain.py` is a LangChain-powered version of the original `scrape_and_store.py` script. It provides the same functionality with improved modularity and integration with the LangChain ecosystem.

## Key Differences from Original

| Feature | Original (`scrape_and_store.py`) | LangChain Version |
|---------|----------------------------------|-------------------|
| **Text Splitting** | Custom token-based chunking with tiktoken | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | Direct OpenAI API calls | `OpenAIEmbeddings` wrapper |
| **Vector Store** | Manual PostgreSQL/psycopg2 | `PGVector` integration |
| **Document Model** | Plain strings | LangChain `Document` objects with metadata |
| **Code Complexity** | ~130 lines, manual DB operations | ~110 lines, abstracted operations |

## Prerequisites

1. **PostgreSQL with pgvector extension** (running via Docker)
2. **Python 3.8+** with virtual environment
3. **OpenAI API Key**

## Installation Steps

### 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- `langchain` - Core LangChain framework
- `langchain-openai` - OpenAI integrations
- `langchain-community` - Community integrations (PGVector)
- `langchain-text-splitters` - Text splitting utilities
- `pgvector` - PostgreSQL vector extension Python client

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

Or export directly:

```bash
export OPENAI_API_KEY=your_api_key_here
```

### 4. Start PostgreSQL with pgvector

```bash
docker-compose up -d
```

### 5. Initialize Database (First Time Only)

Connect to the database:

```bash
docker exec -it pgvector_db psql -U postgres -d embeddings_db
```

Enable pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Note:** The LangChain version will automatically create the required tables, so you don't need to manually create them!

## Running the Script

```bash
python scrape_and_store_langchain.py
```

### Expected Output

```
[INFO] Scraping https://en.wikipedia.org/wiki/Natural_language_processing...
[INFO] Scraped 56000 characters from https://en.wikipedia.org/wiki/Natural_language_processing
[INFO] Split into 75 chunks using LangChain text splitter
[INFO] Storing chunks in PostgreSQL with pgvector...
[INFO] Successfully stored 75 chunks in vector store

[QUERY] Searching for: 'What is natural language processing?'

[RESULTS] Top 3 similar chunks:
--- Result 1 ---
Content: Natural language processing (NLP) is a subfield of linguistics, computer science...
Metadata: {'source': 'https://en.wikipedia.org/wiki/Natural_language_processing'}

--- Result 2 ---
Content: The history of NLP generally started in the 1950s...
Metadata: {'source': 'https://en.wikipedia.org/wiki/Natural_language_processing'}

--- Result 3 ---
Content: Techniques used in NLP include parsing, semantic analysis...
Metadata: {'source': 'https://en.wikipedia.org/wiki/Natural_language_processing'}

[RESULTS WITH SCORES] Top 3 similar chunks with distance scores:
--- Result 1 (Distance: 0.1234) ---
Content: Natural language processing (NLP) is a subfield...

[INFO] Script completed successfully!
```

## Configuration Options

You can modify these settings in the script:

```python
# URL to scrape
URL = "https://en.wikipedia.org/wiki/Natural_language_processing"

# Chunking settings
CHUNK_SIZE = 800  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Database connection
CONNECTION_STRING = "postgresql+psycopg2://postgres:password@localhost:5432/embeddings_db"

# Collection name (table name in PostgreSQL)
collection_name = "langchain_documents"
```

## Advantages of LangChain Version

1. **Automatic Table Management**: LangChain PGVector automatically creates and manages tables
2. **Built-in Deduplication**: Handles duplicate checking internally
3. **Rich Metadata**: Document objects can store extensive metadata
4. **Easy Integration**: Works seamlessly with other LangChain components (chains, agents, etc.)
5. **Multiple Search Methods**:
   - `similarity_search()` - Returns documents
   - `similarity_search_with_score()` - Returns documents with similarity scores
   - `max_marginal_relevance_search()` - Diverse results
6. **Abstraction**: Less boilerplate code, more focus on business logic

## Querying Existing Data

You can create a separate query script:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import os
from dotenv import load_dotenv

load_dotenv()

CONNECTION_STRING = "postgresql+psycopg2://postgres:password@localhost:5432/embeddings_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to existing vector store
vectorstore = PGVector(
    collection_name="langchain_documents",
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

# Query
results = vectorstore.similarity_search("What is NLP?", k=5)
for doc in results:
    print(doc.page_content)
```

## Troubleshooting

### Issue: "relation does not exist"
**Solution**: The table will be created automatically on first run. If issues persist, check database connection.

### Issue: "OpenAI API key not found"
**Solution**: Ensure `.env` file exists with valid `OPENAI_API_KEY`

### Issue: "Could not connect to database"
**Solution**: Verify PostgreSQL is running: `docker ps | grep pgvector`

## Next Steps

With LangChain, you can easily:
- Build RAG (Retrieval-Augmented Generation) applications
- Create question-answering systems
- Integrate with other LLMs (Anthropic, Cohere, etc.)
- Add conversational memory
- Build agent-based systems

## Comparison Command

Run both scripts to compare:

```bash
# Original version
python scrape_and_store.py

# LangChain version
python scrape_and_store_langchain.py
```
