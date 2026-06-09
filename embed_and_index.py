"""
Pipeline — Embedding & Indexação
Lê arquivos .csv, .xls, .xlsx dentro de data/
e indexa no Milvus. Pula arquivos já indexados.
"""

import os
import uuid
import sys
from pathlib import Path
import math

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from milvus_service import MilvusService
from postgres_service import PostgresService


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
DATASET_VERSION = os.getenv("DATASET_VERSION", "1.0.0")

# Coluna que será usada como conteúdo principal para embedding.
# Se None, concatena todas as colunas em "col: valor | col: valor"
CONTENT_COLUMN = os.getenv("CONTENT_COLUMN", None)

SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


logger.info("Carregando modelo de embeddings {}", EMBED_MODEL)
model = SentenceTransformer(EMBED_MODEL, device="cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        device="cpu",
    )
    return embeddings.tolist()


def read_file(filepath: Path) -> pd.DataFrame:
    ext = filepath.suffix.lower()
    if ext == ".csv":
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]:
            try:
                df = pd.read_csv(filepath, encoding=encoding, on_bad_lines="skip")
                if not df.empty:
                    return df
            except (UnicodeDecodeError, ValueError):
                continue
        # último recurso: ignora bytes inválidos
        return pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip", encoding_errors="replace")
    elif ext in {".xls", ".xlsx"}:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Extensão não suportada: {filepath.suffix}")


def sanitize_value(val):
    """Converte valores inválidos para JSON em tipos seguros."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, (int, float, bool, str)):
        return val
    return str(val)


def sanitize_metadata(meta: dict) -> dict:
    """Limpa todo o dicionário de metadata para ser JSON-safe."""
    return {
        str(k): sanitize_value(v)
        for k, v in meta.items()
    }


def row_to_text(row: pd.Series, content_col: str | None) -> str:
    if content_col and content_col in row.index:
        val = row[content_col]
        if pd.isna(val):
            return ""
        return str(val).strip()
    return " | ".join(
        f"{col}: {val}"
        for col, val in row.items()
        if pd.notna(val) and str(val).strip()
    )


def list_local_files(data_dir: Path) -> list[Path]:
    return [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def get_indexed_object_keys(db: PostgresService, dataset_id: str) -> set[str]:
    rows = db.list_data_files(dataset_id=dataset_id, status="done")
    return {row["object_key"] for row in rows}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run() -> None:
    logger.info("=== Embedding & Indexação iniciada ===")
    logger.info("Lendo arquivos de: {}", DATA_DIR.resolve())

    milvus = MilvusService()
    db = PostgresService()

    dataset_name = DATA_DIR.resolve().name

    dataset_id = db.ensure_dataset(
        name=dataset_name,
        version=DATASET_VERSION,
        description=f"Dataset {dataset_name} indexado no Milvus",
    )

    already_indexed = get_indexed_object_keys(db, dataset_id)
    logger.info("{} arquivo(s) já indexado(s) serão ignorados.", len(already_indexed))

    local_files = list_local_files(DATA_DIR)
    logger.info("{} arquivo(s) encontrado(s) em {}", len(local_files), DATA_DIR)

    total_files = 0
    total_skipped = 0
    total_indexed = 0

    for filepath in local_files:
        object_key = str(filepath.relative_to(DATA_DIR))

        if object_key in already_indexed:
            logger.info("Pulando (já indexado): {}", object_key)
            total_skipped += 1
            continue

        logger.info("Processando {}", object_key)

        try:
            df = read_file(filepath)
        except Exception as e:
            logger.error("Erro ao ler {}: {}", object_key, e)
            continue

        if df.empty:
            logger.warning("Arquivo vazio: {}", object_key)
            continue

        # Remove linhas completamente vazias
        df = df.dropna(how="all").reset_index(drop=True)

        logger.info("{} linhas carregadas de {}", len(df), object_key)

        data_file_id = db.register_data_file(
            dataset_id=dataset_id,
            layer="gold",
            bucket=str(DATA_DIR),
            object_key=object_key,
            row_count=len(df),
            size_bytes=filepath.stat().st_size,
            checksum=None,
            status="done",
        )

        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i:i + BATCH_SIZE]

            texts = [
                row_to_text(row, CONTENT_COLUMN)
                for _, row in batch_df.iterrows()
            ]

            # ignora linhas que resultaram em texto vazio
            valid = [(t, r) for t, (_, r) in zip(texts, batch_df.iterrows()) if t.strip()]
            if not valid:
                continue
            texts, rows = zip(*valid)
            texts = list(texts)

            embeddings = embed_texts(texts)

            doc_ids   = [str(uuid.uuid4()) for _ in texts]
            contents = texts
            metadatas = [sanitize_metadata(r.to_dict()) for r in rows]  # ← sanitizado

            milvus_ids = milvus.insert_batch(
                doc_ids=doc_ids,
                contents=contents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            for idx, (milvus_id, row_meta, content) in enumerate(
                zip(milvus_ids, metadatas, contents)
            ):
                db.save_document(
                    data_file_id=data_file_id,
                    milvus_id=milvus_id,
                    content=content,
                    chunk_index=i + idx,
                    token_count=len(content.split()),
                    metadata=row_meta,
                    embed_model=EMBED_MODEL,
                    embed_version=DATASET_VERSION,
                )

            total_indexed += len(batch_df)

        total_files += 1

        db.log_audit(
            entity="data_files",
            entity_id=data_file_id,
            action="indexed_milvus",
            details={
                "object_key": object_key,
                "rows_indexed": len(df),
                "embed_model": EMBED_MODEL,
            },
        )

    milvus.flush()

    logger.info(
        "=== Indexação concluída: {} linhas em {} arquivo(s) novo(s), {} ignorado(s) ===",
        total_indexed, total_files, total_skipped,
    )


if __name__ == "__main__":
    run()