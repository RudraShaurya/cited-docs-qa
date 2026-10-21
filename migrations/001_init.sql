CREATE TABLE IF NOT EXISTS documents (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title       text NOT NULL,
    source_url  text,
    licence     text,
    sha256      text UNIQUE NOT NULL,
    ingested_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index int  NOT NULL,
    page        int,
    section     text,
    text        text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding   vector(384)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
