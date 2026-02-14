import os
import re
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import psycopg2
from tiktoken import get_encoding
from dotenv import load_dotenv

load_dotenv()

# ----------------------
# Config
# ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PG_PARAMS = {
    "dbname": "embeddings_db",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": 5432
}
URL = "https://en.wikipedia.org/wiki/Natural_language_processing"

# Chunking settings
MAX_TOKENS = 200
OVERLAP_TOKENS = 50
ENCODING_NAME = "cl100k_base"  # matches OpenAI models

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------
# Helper functions
# ----------------------
def clean_text(text: str) -> str:
    """Remove [1], [2], etc. citations and normalize whitespace."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS):
    """Token-based chunking with overlap."""
    encoding = get_encoding(ENCODING_NAME)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        print("chunk text ",chunk_text)
        chunks.append(chunk_text)
    print("chunks" ,chunks)
    return chunks

# ----------------------
# Step 1: Scrape page
# ----------------------
response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.text, 'html.parser')
raw_text = " ".join([p.get_text() for p in soup.find_all('p')])
text = clean_text(raw_text)

print(f"[INFO] Scraped {len(text)} characters from {URL}")

# ----------------------
# Step 2: Chunk text
# ----------------------
chunks = chunk_text(text)
print(f"[INFO] Split into {len(chunks)} chunks")

# ----------------------
# Step 3: Connect to PostgreSQL
# ----------------------
conn = psycopg2.connect(**PG_PARAMS)
cur = conn.cursor()

# ----------------------
# Step 4: Insert chunks with embeddings (avoid duplicates)
# ----------------------
for idx, chunk in enumerate(chunks, start=1):
    # Check for duplicate
    cur.execute("SELECT id FROM document_embeddings WHERE text_chunk = %s;", (chunk,))
    exists = cur.fetchone()
    if exists:
        print(f"[SKIP] Chunk {idx} already exists with ID {exists[0]}")
        continue

    # Generate embedding
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    )
    vector = response.data[0].embedding  # List of floats

    # Insert
    sql = "INSERT INTO document_embeddings (text_chunk, embedding) VALUES (%s, %s) RETURNING id;"
    cur.execute(sql, (chunk, vector))
    inserted_id = cur.fetchone()[0]
    conn.commit()

    print(f"[INFO] Inserted chunk {idx}/{len(chunks)} with ID {inserted_id}")

# ----------------------
# Step 5: Query top 3 similar chunks
# ----------------------
query_text = "What is natural language processing?"
query_text_clean = clean_text(query_text)
query_embedding = client.embeddings.create(
    input=query_text_clean,
    model="text-embedding-3-small"
).data[0].embedding

sql_query = """
SELECT id, text_chunk
FROM document_embeddings
ORDER BY embedding <#> %s::vector
LIMIT 3;
"""

cur.execute(sql_query, (query_embedding,))
results = cur.fetchall()

print("\n[RESULTS] Top 3 similar chunks:")
for r in results:
    print(f"ID {r[0]}: {r[1][:200]}...")  # preview

# ----------------------
# Step 6: Cleanup
# ----------------------
cur.close()
conn.close()
