import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ----------------------
# Config
# ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PostgreSQL connection string for LangChain PGVector
CONNECTION_STRING = "postgresql+psycopg2://postgres:password@localhost:5432/embeddings_db"

URL = "https://en.wikipedia.org/wiki/Natural_language_processing"

# Chunking settings
CHUNK_SIZE = 800  # Approximate characters (roughly ~200 tokens)
CHUNK_OVERLAP = 200  # Overlap to maintain context

# ----------------------
# Helper functions
# ----------------------
def clean_text(text: str) -> str:
    """Remove [1], [2], etc. citations and normalize whitespace."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ----------------------
# Step 1: Scrape page
# ----------------------
print(f"[INFO] Scraping {URL}...")
response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.text, 'html.parser')
raw_text = " ".join([p.get_text() for p in soup.find_all('p')])
text = clean_text(raw_text)

print(f"[INFO] Scraped {len(text)} characters from {URL}")

# ----------------------
# Step 2: Create Document and Chunk text with LangChain
# ----------------------
# Create a LangChain Document object
doc = Document(page_content=text, metadata={"source": URL})

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

# Split the document into chunks
chunks = text_splitter.split_documents([doc])
print(f"[INFO] Split into {len(chunks)} chunks using LangChain text splitter")

# ----------------------
# Step 3: Initialize OpenAI Embeddings
# ----------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# ----------------------
# Step 4: Store in PostgreSQL using PGVector
# ----------------------
print(f"[INFO] Storing chunks in PostgreSQL with pgvector...")

# Create or connect to PGVector store
# This will automatically create the table if it doesn't exist
# and handle embeddings generation
vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="langchain_documents",
    connection_string=CONNECTION_STRING,
    pre_delete_collection=False,  # Set to True to clear existing data
)

print(f"[INFO] Successfully stored {len(chunks)} chunks in vector store")

# ----------------------
# Step 5: Perform Similarity Search
# ----------------------
query_text = "What is natural language processing?"
print(f"\n[QUERY] Searching for: '{query_text}'")

# Perform similarity search (returns top k similar documents)
similar_docs = vectorstore.similarity_search(query_text, k=3)

print("\n[RESULTS] Top 3 similar chunks:")
for idx, doc in enumerate(similar_docs, start=1):
    print(f"\n--- Result {idx} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")

# ----------------------
# Optional: Similarity search with scores
# ----------------------
print("\n[RESULTS WITH SCORES] Top 3 similar chunks with distance scores:")
similar_docs_with_scores = vectorstore.similarity_search_with_score(query_text, k=3)

for idx, (doc, score) in enumerate(similar_docs_with_scores, start=1):
    print(f"\n--- Result {idx} (Distance: {score:.4f}) ---")
    print(f"Content: {doc.page_content[:200]}...")

print("\n[INFO] Script completed successfully!")
