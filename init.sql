-- Extensões
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Datasets
CREATE TABLE IF NOT EXISTS rag_datasets (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        VARCHAR(255) NOT NULL UNIQUE,
    domain      VARCHAR(100) NOT NULL DEFAULT 'general',
    source_url  TEXT,
    version     VARCHAR(50)  NOT NULL DEFAULT '1.0.0',
    description TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Versões de dataset
CREATE TABLE IF NOT EXISTS rag_dataset_versions (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id  UUID NOT NULL REFERENCES rag_datasets(id) ON DELETE CASCADE,
    version     VARCHAR(50) NOT NULL,
    source_url  TEXT,
    notes       TEXT,
    created_by  VARCHAR(100) NOT NULL DEFAULT 'system',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (dataset_id, version)
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_ds
    ON rag_dataset_versions (dataset_id, created_at DESC);

-- Arquivos de dados
CREATE TABLE IF NOT EXISTS data_files (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id   UUID REFERENCES rag_datasets(id) ON DELETE SET NULL,
    layer        VARCHAR(20)  NOT NULL,
    bucket       VARCHAR(255) NOT NULL,
    object_key   TEXT         NOT NULL,
    file_format  VARCHAR(20)  NOT NULL DEFAULT 'csv',
    size_bytes   BIGINT,
    row_count    INTEGER,
    checksum     VARCHAR(64),
    status       VARCHAR(20)  NOT NULL DEFAULT 'done',
    error_msg    TEXT,
    processed_at TIMESTAMPTZ,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (bucket, object_key)
);

CREATE INDEX IF NOT EXISTS idx_data_files_dataset
    ON data_files (dataset_id, status);

-- Documentos / chunks indexados
CREATE TABLE IF NOT EXISTS documents (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_file_id UUID REFERENCES data_files(id) ON DELETE SET NULL,
    milvus_id    BIGINT UNIQUE NOT NULL,
    content      TEXT         NOT NULL,
    chunk_index  INTEGER      NOT NULL DEFAULT 0,
    token_count  INTEGER,
    metadata     JSONB        NOT NULL DEFAULT '{}',
    embed_model  VARCHAR(255),
    embed_version VARCHAR(50),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_data_file
    ON documents (data_file_id);

-- Runs RAG
CREATE TABLE IF NOT EXISTS rag_runs (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mlflow_run_id  VARCHAR(255),
    query          TEXT         NOT NULL,
    retrieved_docs JSONB        NOT NULL DEFAULT '[]',
    prompt_used    TEXT,
    response       TEXT,
    llm_model      VARCHAR(255),
    embed_model    VARCHAR(255),
    top_k          INTEGER,
    latency_ms     INTEGER,
    user_feedback  SMALLINT,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Auditoria
CREATE TABLE IF NOT EXISTS audit_log (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity     VARCHAR(100) NOT NULL,
    entity_id  UUID,
    action     VARCHAR(100) NOT NULL,
    actor      VARCHAR(100) NOT NULL DEFAULT 'system',
    details    JSONB        NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_entity
    ON audit_log (entity, entity_id, created_at DESC);