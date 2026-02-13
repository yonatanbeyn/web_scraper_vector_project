docker exec -it pgvector_db psql -U postgres -d embeddings_db


-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE document_embeddings (
id SERIAL PRIMARY KEY,
text_chunk TEXT NOT NULL,
embedding vector(1536) NOT NULL
);
set env for key
export OPENAI_API_KEY=

docker-compose up -d
pip install -r requirements.txt
python scrape_and_store.py

python3.13 -m venv venv
source venv/bin/activate



[INFO] Scraped 56000 characters from https://...
[INFO] Split into 112 chunks
[INFO] Inserted chunk 1/112 with ID 1
[SQL] INSERT INTO document_embeddings (text_chunk, embedding) VALUES ('Natural language processing...', '[0.0123, -0.45,...]')
...
[RESULTS] Top 3 similar chunks:
ID 5: Natural language processing (NLP) is a subfield of linguistics, computer science, ...
ID 8: The history of NLP generally started in the 1950s...
ID 3: Techniques used in NLP include parsing, semantic analysis, and...