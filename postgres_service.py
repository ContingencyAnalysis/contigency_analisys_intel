"""Serviço PostgreSQL para metadados, auditoria e runs RAG."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import settings


class PostgresService:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.engine: Engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            future=True,
        )
        self._dataset_versions_ready = False

    @staticmethod
    def _uuid_or_none(value: str | None) -> str | None:
        if not value:
            return None
        try:
            return str(uuid.UUID(str(value)))
        except ValueError:
            return None

    @staticmethod
    def _is_equal(left: Any, right: Any) -> bool:
        return (left or None) == (right or None)

    def _register_dataset_version(
        self,
        conn,
        dataset_id: str,
        version: str,
        source_url: str | None = None,
        notes: str | None = None,
        created_by: str = "system",
    ) -> bool:
        if not self._dataset_versions_ready:
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS rag_dataset_versions (
                        id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        dataset_id  UUID NOT NULL REFERENCES rag_datasets(id) ON DELETE CASCADE,
                        version     VARCHAR(50) NOT NULL,
                        source_url  TEXT,
                        notes       TEXT,
                        created_by  VARCHAR(100) NOT NULL DEFAULT 'system',
                        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (dataset_id, version)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_dataset_versions_ds
                    ON rag_dataset_versions (dataset_id, created_at DESC)
                    """
                )
            )
            self._dataset_versions_ready = True

        result = conn.execute(
            text(
                """
                INSERT INTO rag_dataset_versions (
                    dataset_id, version, source_url, notes, created_by
                ) VALUES (
                    :dataset_id, :version, :source_url, :notes, :created_by
                )
                ON CONFLICT (dataset_id, version) DO NOTHING
                RETURNING id
                """
            ),
            {
                "dataset_id": dataset_id,
                "version": version,
                "source_url": source_url,
                "notes": notes,
                "created_by": created_by,
            },
        ).mappings().first()
        return bool(result)

    def ensure_dataset(
        self,
        name: str,
        domain: str = "general",
        source_url: str | None = None,
        version: str = "1.0.0",
        description: str | None = None,
    ) -> str:
        """Cria/atualiza dataset lógico e retorna `dataset_id`."""
        dataset_id: str
        version_inserted = False
        created = False
        changed = False
        with self.engine.begin() as conn:
            existing = conn.execute(
                text(
                    """
                    SELECT id, domain, source_url, version, description
                    FROM rag_datasets
                    WHERE name = :name
                    LIMIT 1
                    """
                ),
                {"name": name},
            ).mappings().first()

            if existing:
                dataset_id = str(existing["id"])
                changed = not (
                    self._is_equal(existing["domain"], domain)
                    and self._is_equal(existing["source_url"], source_url)
                    and self._is_equal(existing["version"], version)
                    and self._is_equal(existing["description"], description)
                )
                if changed:
                    conn.execute(
                        text(
                            """
                            UPDATE rag_datasets
                            SET domain = :domain,
                                source_url = :source_url,
                                version = :version,
                                description = :description,
                                updated_at = NOW()
                            WHERE id = :id
                            """
                        ),
                        {
                            "id": dataset_id,
                            "domain": domain,
                            "source_url": source_url,
                            "version": version,
                            "description": description,
                        },
                    )
                version_inserted = self._register_dataset_version(
                    conn=conn,
                    dataset_id=dataset_id,
                    version=version,
                    source_url=source_url,
                    notes=description,
                )
            else:
                created = True
                dataset_id = str(uuid.uuid4())
                conn.execute(
                    text(
                        """
                        INSERT INTO rag_datasets (
                            id, name, domain, source_url, version, description
                        ) VALUES (
                            :id, :name, :domain, :source_url, :version, :description
                        )
                        """
                    ),
                    {
                        "id": dataset_id,
                        "name": name,
                        "domain": domain,
                        "source_url": source_url,
                        "version": version,
                        "description": description,
                    },
                )
                version_inserted = self._register_dataset_version(
                    conn=conn,
                    dataset_id=dataset_id,
                    version=version,
                    source_url=source_url,
                    notes=description,
                )

        if created:
            self.log_audit(
                entity="rag_datasets",
                entity_id=dataset_id,
                action="dataset_created",
                details={"name": name, "domain": domain, "version": version, "source_url": source_url},
            )
        elif changed:
            self.log_audit(
                entity="rag_datasets",
                entity_id=dataset_id,
                action="dataset_updated",
                details={"name": name, "domain": domain, "version": version, "source_url": source_url},
            )

        if version_inserted:
            self.log_audit(
                entity="rag_dataset_versions",
                entity_id=dataset_id,
                action="dataset_version_registered",
                details={"name": name, "version": version, "source_url": source_url},
            )

        return dataset_id

    def register_data_file(
        self,
        dataset_id: str | None,
        layer: str,
        bucket: str,
        object_key: str,
        file_format: str = "jsonl",
        size_bytes: int | None = None,
        row_count: int | None = None,
        checksum: str | None = None,
        status: str = "done",
        error_msg: str | None = None,
    ) -> str:
        """Upsert de arquivo em `data_files` com status e métricas."""
        dataset_uuid = self._uuid_or_none(dataset_id)
        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO data_files (
                        dataset_id, layer, bucket, object_key, file_format,
                        size_bytes, row_count, checksum, status, error_msg, processed_at
                    ) VALUES (
                        :dataset_id, :layer, :bucket, :object_key, :file_format,
                        :size_bytes, :row_count, :checksum, :status, :error_msg,
                        CASE WHEN :status = 'done' THEN NOW() ELSE NULL END
                    )
                    ON CONFLICT (bucket, object_key)
                    DO UPDATE SET
                        dataset_id = EXCLUDED.dataset_id,
                        layer = EXCLUDED.layer,
                        file_format = EXCLUDED.file_format,
                        size_bytes = EXCLUDED.size_bytes,
                        row_count = EXCLUDED.row_count,
                        checksum = EXCLUDED.checksum,
                        status = EXCLUDED.status,
                        error_msg = EXCLUDED.error_msg,
                        processed_at = CASE
                            WHEN EXCLUDED.status = 'done' THEN NOW()
                            ELSE data_files.processed_at
                        END
                    RETURNING id
                    """
                ),
                {
                    "dataset_id": dataset_uuid,
                    "layer": layer,
                    "bucket": bucket,
                    "object_key": object_key,
                    "file_format": file_format,
                    "size_bytes": size_bytes,
                    "row_count": row_count,
                    "checksum": checksum,
                    "status": status,
                    "error_msg": error_msg,
                },
            ).mappings().first()
            return str(result["id"])

    def get_data_file(self, bucket: str, object_key: str) -> dict[str, Any] | None:
        """Retorna metadados do arquivo registrado em `data_files`."""
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT id, dataset_id, layer, bucket, object_key, row_count, checksum, status
                    FROM data_files
                    WHERE bucket = :bucket AND object_key = :object_key
                    LIMIT 1
                    """
                ),
                {"bucket": bucket, "object_key": object_key},
            ).mappings().first()
            return dict(row) if row else None

    def log_audit(
        self,
        entity: str,
        action: str,
        entity_id: str | None = None,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Registra evento em `audit_log`."""
        entity_uuid = self._uuid_or_none(entity_id)
        details_json = json.dumps(details or {}, ensure_ascii=False)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO audit_log (entity, entity_id, action, actor, details)
                    VALUES (:entity, :entity_id, :action, :actor, CAST(:details AS JSONB))
                    """
                ),
                {
                    "entity": entity,
                    "entity_id": entity_uuid,
                    "action": action,
                    "actor": actor,
                    "details": details_json,
                },
            )

    def save_document(
        self,
        data_file_id: str | None,
        milvus_id: int,
        content: str,
        chunk_index: int,
        token_count: int,
        metadata: dict[str, Any] | None,
        embed_model: str,
        embed_version: str = "1.0.0",
    ) -> None:
        """Persiste vínculo chunk↔embedding em `documents`."""
        data_file_uuid = self._uuid_or_none(data_file_id)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO documents (
                        data_file_id, milvus_id, content, chunk_index, token_count,
                        metadata, embed_model, embed_version
                    ) VALUES (
                        :data_file_id, :milvus_id, :content, :chunk_index, :token_count,
                        CAST(:metadata AS JSONB), :embed_model, :embed_version
                    )
                    ON CONFLICT (milvus_id) DO NOTHING
                    """
                ),
                {
                    "data_file_id": data_file_uuid,
                    "milvus_id": milvus_id,
                    "content": content,
                    "chunk_index": chunk_index,
                    "token_count": token_count,
                    "metadata": metadata_json,
                    "embed_model": embed_model,
                    "embed_version": embed_version,
                },
            )

    def save_rag_run_sync(
        self,
        run_id: str,
        mlflow_run_id: str,
        query: str,
        retrieved_docs: list[dict],
        prompt_used: str,
        response: str,
        llm_model: str,
        latency_ms: int,
        top_k: int | None = None,
        embed_model: str | None = None,
    ) -> None:
        run_uuid = self._uuid_or_none(run_id) or str(uuid.uuid4())
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO rag_runs (
                        id, mlflow_run_id, query, retrieved_docs, prompt_used,
                        response, llm_model, embed_model, top_k, latency_ms
                    ) VALUES (
                        :id, :mlflow_run_id, :query, CAST(:retrieved_docs AS JSONB), :prompt_used,
                        :response, :llm_model, :embed_model, :top_k, :latency_ms
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                {
                    "id": run_uuid,
                    "mlflow_run_id": mlflow_run_id,
                    "query": query,
                    "retrieved_docs": json.dumps(retrieved_docs, ensure_ascii=False),
                    "prompt_used": prompt_used,
                    "response": response,
                    "llm_model": llm_model,
                    "embed_model": embed_model,
                    "top_k": top_k,
                    "latency_ms": latency_ms,
                },
            )

        self.log_audit(
            entity="rag_runs",
            entity_id=run_uuid,
            action="query_executed",
            details={
                "mlflow_run_id": mlflow_run_id,
                "llm_model": llm_model,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "retrieved_docs": len(retrieved_docs),
            },
        )

    async def save_rag_run(
        self,
        run_id: str,
        mlflow_run_id: str,
        query: str,
        retrieved_docs: list[dict],
        prompt_used: str,
        response: str,
        llm_model: str,
        latency_ms: int,
        top_k: int | None = None,
        embed_model: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.save_rag_run_sync,
            run_id,
            mlflow_run_id,
            query,
            retrieved_docs,
            prompt_used,
            response,
            llm_model,
            latency_ms,
            top_k,
            embed_model,
        )

    def update_feedback_sync(self, run_id: str, feedback: int) -> None:
        run_uuid = self._uuid_or_none(run_id)
        if not run_uuid:
            logger.warning("run_id inválido para feedback: {}", run_id)
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE rag_runs
                    SET user_feedback = :feedback
                    WHERE id = :id
                    """
                ),
                {"feedback": feedback, "id": run_uuid},
            )

        self.log_audit(
            entity="rag_runs",
            entity_id=run_uuid,
            action="feedback_updated",
            details={"feedback": feedback},
        )

    async def update_feedback(self, run_id: str, feedback: int) -> None:
        await asyncio.to_thread(self.update_feedback_sync, run_id, feedback)

    def list_data_files(
    self,
    dataset_id: str,
    status: str | None = None,
) -> list[dict[str, Any]]:
        """Retorna todos os arquivos de um dataset, opcionalmente filtrados por status."""
        dataset_uuid = self._uuid_or_none(dataset_id)
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, dataset_id, layer, bucket, object_key,
                        row_count, checksum, status
                    FROM data_files
                    WHERE dataset_id = :dataset_id
                    AND (:status IS NULL OR status = :status)
                    ORDER BY processed_at DESC
                    """
                ),
                {"dataset_id": dataset_uuid, "status": status},
            ).mappings().all()
            return [dict(row) for row in rows]
