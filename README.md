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


q That quits the pager and returns you to the psql prompt.

🟡 To scroll instead of exit
While you’re in (END):

Space → next page

Enter → one line down

b → back one page

g → go to top

G → go to bottom
SELECT id, text_chunk
FROM document_embeddings
ORDER BY embedding <#> %s::vector
LIMIT 3;

SELECT oprname, oprleft::regtype, oprright::regtype
FROM pg_operator
WHERE oprname IN ('<#>', '<->');Q

ANN INDEX
CREATE INDEX ON document_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Check data
SELECT COUNT(*) FROM document_embeddings;

-- Inspect samples
SELECT id, LEFT(text_chunk, 100)
FROM document_embeddings
LIMIT 3;

-- Vector search
SELECT id, text_chunk
FROM document_embeddings
ORDER BY embedding <#> :query_vector::vector
LIMIT 5;

