import os
import requests
from bs4 import BeautifulSoup
import openai
import psycopg2
from tiktoken import get_encoding
from dotenv import load_dotenv

load_dotenv()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# PostgreSQL connection
conn = psycopg2.connect(
    dbname="embeddings_db",
    user="postgres",
    password="password",
    host="localhost",
    port=5432
)
cur = conn.cursor()

# ---- Step 1: Scrape a web page ----
URL = "https://en.wikipedia.org/wiki/Natural_language_processing"
response = requests.get(
    URL,
    headers={"User-Agent": "Mozilla/5.0"}
)
soup = BeautifulSoup(response.text, 'html.parser')
text = " ".join([p.get_text() for p in soup.find_all('p')])

print(f"[INFO] Scraped {len(text)} characters from {URL}")

# ---- Step 2: Chunk text ----
# Simple chunking: 500 characters per chunk with 50 char overlap
chunk_size = 500
overlap = 50

chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

print(f"[INFO] Split into {len(chunks)} chunks")

# ---- Step 3: Generate embeddings for each chunk ----
for idx, chunk in enumerate(chunks):
    print("chunks for embeddings ",chunk)
    response = openai.Embedding.create(
        input=chunk,
        model="text-embedding-3-small"
    )
    vector = response['data'][0]['embedding']  # List of floats

    # ---- Step 4: Insert into PostgreSQL ----
    sql = "INSERT INTO document_embeddings (text_chunk, embedding) VALUES (%s, %s) RETURNING id;"
    cur.execute(sql, (chunk, vector))
    inserted_id = cur.fetchone()[0]
    conn.commit()
    print(f"[INFO] Inserted chunk {idx+1}/{len(chunks)} with ID {inserted_id}")
    print(f"[SQL] {cur.mogrify(sql, (chunk, vector)).decode('utf-8')}")

# ---- Step 5: Example: query top 3 similar chunks ----
query_text = "What is natural language processing?"
query_embedding = openai.Embedding.create(
    input=query_text,
    model="text-embedding-3-small"
)['data'][0]['embedding']

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
    print(f"ID {r[0]}: {r[1][:100]}...")

cur.close()
conn.close()
