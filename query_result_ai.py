import os
import json
import sqlite3
import boto3
import re
import unicodedata
from typing import Iterator
from pathlib import Path

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import dotenv_values, load_dotenv

load_dotenv()

# =====================================================
# CONFIG
# =====================================================

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "motor_estatistico.db")
DATASET = os.getenv("AGENT_DATASET", "emendas").strip().lower()
DATA_SOURCE = os.getenv("AGENT_SOURCE", "sqlite").strip().lower()
SQLITE_TABLE = os.getenv(
    "SQLITE_TABLE",
    "Compras" if DATASET in {"compras", "compras_publicas"} else "EmendasParlamentares",
)
# The bulk pipeline exposes group_id, while the legacy emendas table uses
# codigo_da_emenda.  A column can be forced when a source has an ambiguous
# schema; otherwise the first existing candidate is selected automatically.
ENTITY_ID_COLUMN = os.getenv("ENTITY_ID_COLUMN", "").strip()
ENTITY_ID_CANDIDATES = tuple(
    value.strip()
    for value in os.getenv(
        "ENTITY_ID_CANDIDATES",
        "group_id,codigo_da_compra,numero_processo,identificador,codigo_da_emenda,id",
    ).split(",")
    if value.strip()
)
PROCESS_LIMIT = int(os.getenv("AGENT_LIMIT", "0"))
PROCESS_OFFSET = int(os.getenv("AGENT_OFFSET", "0"))
# Safety guard for batch runs. Zero means no additional cap.
MAX_PROCESSED = int(os.getenv("AGENT_MAX_PROCESSED", "0"))
REQUESTED_GROUP_IDS = tuple(
    value.strip() for value in os.getenv("AGENT_GROUP_IDS", "").split(",") if value.strip()
)
BACKEND_ENV_PATH = Path(__file__).resolve().parent.parent / "contigency_analisys_backend" / ".env"
POSTGRES_URL = (
    os.getenv("AGENT_DATABASE_URL")
    or dotenv_values(BACKEND_ENV_PATH).get("DATABASE_URL")
    or os.getenv("DATABASE_URL", "")
)
POSTGRES_TABLE = os.getenv("AGENT_POSTGRES_TABLE", "baseline_organization_metrics")
POSTGRES_SOURCE_SYSTEM = os.getenv("AGENT_SOURCE_SYSTEM", "Compras")
POSTGRES_DATASET = os.getenv("AGENT_POSTGRES_DATASET", "compras")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_documents")
TOP_K = int(os.getenv("TOP_K", "5"))
MILVUS_CANDIDATES = max(TOP_K * 5, 20)

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "openai.gpt-oss-20b-1:0")

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# =====================================================
# SQLITE
# =====================================================

def _quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier after validating it came from configuration."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Identificador SQLite inválido: {identifier!r}")
    return f'"{identifier}"'


def load_transactions() -> Iterator[tuple[str, dict]]:
    """Read transactions from the configured dataset, including Compras."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        table = _quote_identifier(SQLITE_TABLE)
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if not columns:
            raise Exception(f"Tabela de dados não encontrada: {SQLITE_TABLE}")
        id_column = ENTITY_ID_COLUMN or next(
            (candidate for candidate in ENTITY_ID_CANDIDATES if candidate in columns), None
        )
        if not id_column:
            raise Exception(
                f"Nenhuma coluna identificadora encontrada em {SQLITE_TABLE}. "
                f"Colunas disponíveis: {', '.join(columns)}"
            )
        query = f"SELECT rowid AS _agent_rowid, * FROM {table} WHERE {_quote_identifier(id_column)} IS NOT NULL"
        parameters: list[int] = []
        if PROCESS_LIMIT:
            query += " LIMIT ? OFFSET ?"
            parameters.extend([PROCESS_LIMIT, PROCESS_OFFSET])
        elif PROCESS_OFFSET:
            query += " LIMIT -1 OFFSET ?"
            parameters.append(PROCESS_OFFSET)
        for row in conn.execute(query, parameters):
            data = dict(row)
            entity_id = str(data.get(id_column)).strip()
            data.pop("_agent_rowid", None)
            yield entity_id, data
    finally:
        conn.close()


def load_postgres_transactions() -> Iterator[tuple[str, dict]]:
    """Read the Benford metrics published for the dashboard's Compras view."""
    if not POSTGRES_URL:
        raise RuntimeError("DATABASE_URL é obrigatório quando AGENT_SOURCE=postgres")
    if POSTGRES_TABLE != "baseline_organization_metrics":
        raise ValueError("AGENT_POSTGRES_TABLE não permitido")

    query = f"""
        SELECT run_id, source_system, dataset, orgao_id, group_id, group_name,
               sample_size, mad, jsd, z_max, last_digit_test,
               risk_p50, risk_p90, risk_p95, alerted, eligible,
               eligibility_reason, mad_threshold, jsd_threshold, z_threshold,
               last_digit_threshold, mad_normalized, jsd_normalized,
               z_normalized, last_digit_normalized, config_version,
               analysis_variant, first_two_sample_size, mad_2d,
               mad_2d_threshold, mad_2d_normalized, mad_2d_flag, jsd_2d,
               jsd_2d_threshold, jsd_2d_normalized, jsd_2d_flag,
               measure_name, grouping_level, period_key, source_partition,
               input_fingerprint, schema_fingerprint,
               analysis_contract_fingerprint, evidence_strength
        FROM {POSTGRES_TABLE}
        WHERE source_system = %s AND dataset = %s
          AND is_active = TRUE
          AND (mad IS NOT NULL OR jsd IS NOT NULL OR z_max IS NOT NULL
               OR last_digit_test IS NOT NULL OR mad_2d IS NOT NULL
               OR jsd_2d IS NOT NULL)
        {'AND group_id = ANY(%s)' if REQUESTED_GROUP_IDS else ''}
        ORDER BY period_key DESC NULLS LAST, group_id, measure_name
        LIMIT %s OFFSET %s
    """
    query_parameters: list[object] = [POSTGRES_SOURCE_SYSTEM, POSTGRES_DATASET]
    if REQUESTED_GROUP_IDS:
        query_parameters.append(list(REQUESTED_GROUP_IDS))
    query_parameters.extend([PROCESS_LIMIT or 100000, PROCESS_OFFSET])
    with psycopg2.connect(POSTGRES_URL) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, query_parameters)
            rows = cursor.fetchall()
    if not rows:
        raise RuntimeError(
            f"Nenhum registro encontrado para {POSTGRES_SOURCE_SYSTEM}/{POSTGRES_DATASET}"
        )
    for row in rows:
        data = dict(row)
        entity_id = ":".join(
            str(data.get(field) or "").strip()
            for field in ("group_id", "orgao_id", "measure_name", "grouping_level", "period_key")
        ).strip(":")
        if entity_id:
            yield entity_id, data


def iter_transactions() -> Iterator[tuple[str, dict]]:
    if DATA_SOURCE == "postgres":
        return load_postgres_transactions()
    if DATA_SOURCE != "sqlite":
        raise ValueError("AGENT_SOURCE deve ser 'sqlite' ou 'postgres'")
    return load_transactions()


def init_result_table(conn):
    cursor = conn.cursor()
    # The source index is useful for reruns, but is optional because some
    # datasets (notably Compras) use a different identifier column.
    source_columns = [row[1] for row in cursor.execute(
        f"PRAGMA table_info({_quote_identifier(SQLITE_TABLE)})"
    )]
    source_id = ENTITY_ID_COLUMN or next(
        (candidate for candidate in ENTITY_ID_CANDIDATES if candidate in source_columns), None
    )
    if source_id:
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_agent_entity_id ON {_quote_identifier(SQLITE_TABLE)} ({_quote_identifier(source_id)})"
        )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS AnaliseResultados (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emenda_id TEXT NOT NULL,
            criado_em TEXT DEFAULT CURRENT_TIMESTAMP,
            nivel_anomalia TEXT NOT NULL DEFAULT 'INDEFINIDO',
            documents_json TEXT NOT NULL,
            analysis TEXT NOT NULL,
            resultado_completo TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analise_emenda_id
        ON AnaliseResultados (emenda_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analise_nivel_anomalia
        ON AnaliseResultados (nivel_anomalia)
        """
    )

    conn.commit()


def save_result_to_sqlite(conn, emenda_id, result):
    cursor = conn.cursor()
    nivel_anomalia = extract_nivel_anomalia(result["analysis"])

    cursor.execute(
        """
        INSERT INTO AnaliseResultados (
            emenda_id,
            nivel_anomalia,
            documents_json,
            analysis,
            resultado_completo
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            emenda_id,
            nivel_anomalia,
            json.dumps(result["documents"], ensure_ascii=False),
            result["analysis"],
            json.dumps(result, ensure_ascii=False),
        )
    )

    conn.commit()
    return cursor.lastrowid

# =====================================================
# HUGGINGFACE EMBEDDING
# =====================================================

embedding_model = SentenceTransformer(HF_EMBED_MODEL)

def generate_embedding(text: str):
    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embedding.tolist()

# =====================================================
# MILVUS
# =====================================================

def _search_tokens(value):
    normalized = unicodedata.normalize("NFKD", str(value or "")).encode(
        "ascii", "ignore"
    ).decode().lower()
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9_.-]{2,}", normalized)
        if token not in {"true", "false", "null", "none"}
    }


def search_documents(transaction):
    """Retrieve and rerank legal documents using the transaction's identity/context."""
    query_text = " | ".join(
        f"{key}: {value}"
        for key, value in transaction.items()
        if value is not None and str(value).strip()
    )
    transaction_tokens = _search_tokens(query_text)
    client = MilvusClient(uri=MILVUS_URI)
    embedding = generate_embedding(query_text)

    results = client.search(
        collection_name=MILVUS_COLLECTION,
        data=[embedding],
        limit=MILVUS_CANDIDATES,
        output_fields=["content", "metadata"]
    )

    docs = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        content = entity.get("content", "")
        metadata = entity.get("metadata") or {}
        document_tokens = _search_tokens(
            content + " " + " ".join(str(value) for value in metadata.values())
        )
        overlap = transaction_tokens & document_tokens
        # Exact contextual matches outrank generic semantic matches. This avoids
        # returning unrelated audit reports merely because they discuss fraud.
        relevance = len(overlap) * 0.15
        for field in ("group_name", "group_id", "orgao_id", "period_key", "dataset"):
            value_tokens = _search_tokens(transaction.get(field))
            if value_tokens and value_tokens <= document_tokens:
                relevance += 2.0
        docs.append({
            "score": hit["distance"],
            "relevance": relevance,
            "content": content,
            "metadata": metadata,
        })

    docs.sort(key=lambda item: (item["relevance"], item["score"]), reverse=True)
    return docs[:TOP_K]

# =====================================================
# PROMPT
# =====================================================

SYSTEM_PROMPT = """
Você é um analista de auditoria pública especializado em explicar resultados de detecção de anomalias.

OBJETIVO:
Explicar exclusivamente os possíveis motivos que levaram o motor estatístico a classificar um registro como suspeito, e classificar o nível de severidade dessa anomalia.

REGRAS:

- Não conclua que existe fraude.
- Não afirme irregularidades sem evidências.
- Explique apenas indícios, inconsistências ou padrões atípicos.
- Utilize somente os documentos recuperados.
- Se os documentos recuperados não forem relacionados diretamente ao caso, informe isso explicitamente.
- Não faça resumo dos documentos.
- Não liste informações irrelevantes.
- Considere obrigatoriamente os campos estatísticos da transação (sample_size,
  eligible, alerted e eligibility_reason). Se eligible for falso ou alerted for
  falso, deixe claro que não há evidência estatística suficiente de anomalia;
  não trate a falta de amostra como prova de suspeita.
- Não invente métricas para campos nulos e não atribua aos documentos relação
  com a transação quando ela não existir.
- Classifique o NÍVEL DA ANOMALIA como BAIXO, MÉDIO ou CRÍTICO, considerando:
  * BAIXO: inconsistências pontuais, sem indícios de padrão de irregularidade ou valores discrepantes relevantes.
  * MÉDIO: inconsistências relevantes e/ou repetição de padrões atípicos, mas sem evidência direta de fraude/desvio.
  * CRÍTICO: múltiplas inconsistências graves, valores muito discrepantes, ou forte similaridade com padrões de fraude/desvio observados nos documentos recuperados.

Sua resposta deve conter SOMENTE:

1. Motivos que podem ter contribuído para a classificação de risco.
2. Evidências observadas na transação.
3. Evidências encontradas nos documentos recuperados.
4. Nível da anomalia (Baixo, Médio ou Crítico).
5. Nível de confiança da análise (Alto, Médio ou Baixo).

Formato obrigatório:

MOTIVOS IDENTIFICADOS

- motivo 1
- motivo 2
- motivo 3

EVIDÊNCIAS

- evidência 1
- evidência 2

NÍVEL DA ANOMALIA

Baixo/Médio/Crítico

CONFIANÇA DA ANÁLISE

Baixa/Média/Alta

JUSTIFICATIVA

Texto curto de até 5 linhas.
"""

def build_prompt(transaction, docs):
    transaction_json = json.dumps(transaction, indent=2, ensure_ascii=False)
    context = "\n\n".join(
        [f"[Documento {i+1}]\n{d['content']}" for i, d in enumerate(docs)]
    )

    return f"""
TRANSAÇÃO ANALISADA

{transaction_json}

DOCUMENTOS RECUPERADOS

{context}

Pergunta:

        Explique somente os possíveis motivos que levaram o motor estatístico a considerar esta transação suspeita. Se os campos
        `eligible` ou `alerted` indicarem que não houve alerta, corrija a premissa:
        informe que a análise não confirmou suspeita e explique apenas as limitações observadas.

Não determine fraude.

Não faça recomendações.

Não descreva todos os documentos.

Mostre apenas os indícios relevantes.
"""

# =====================================================
# HUGGINGFACE / BEDROCK LLM
# =====================================================

def generate_answer(prompt):
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    body = {
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["choices"][0]["message"]["content"]

def extract_nivel_anomalia(answer: str) -> str:
    match = re.search(
        r"NÍVEL DA ANOMALIA\s*\n+\s*(Baixo|Médio|Medio|Crítico|Critico)",
        answer,
        re.IGNORECASE
    )

    if not match:
        return "INDEFINIDO"

    nivel = match.group(1).upper()
    nivel = nivel.replace("É", "E").replace("Í", "I")
    return nivel

# =====================================================
# MAIN
# =====================================================

def run_single(emenda_id, transaction):
    print(f"\n--- Processando Emenda: {emenda_id} ---")

    print("[1] Buscando documentos no Milvus...")
    docs = search_documents(transaction)
    print(f"    {len(docs)} documentos recuperados.")

    print("[2] Montando prompt e consultando LLM...")
    prompt = build_prompt(transaction, docs)
    answer = generate_answer(prompt)

    print("\n" + "=" * 60)
    print(answer)
    print("=" * 60)

    return {
        "transaction": transaction,
        "documents": docs,
        "analysis": answer
    }

# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FRAUD RAG AGENT - PROCESSAMENTO EM LOTE")
    print("=" * 60)

    # Conecta ao SQLite e garante a criação da tabela de resultados
    conn = sqlite3.connect(SQLITE_DB_PATH)
    init_result_table(conn)

    total_processados = 0

    try:
        loader = iter_transactions()
        for emenda_id, transaction in loader:
            if MAX_PROCESSED and total_processados >= MAX_PROCESSED:
                print(
                    f"\n[PARADA] Limite AGENT_MAX_PROCESSED={MAX_PROCESSED} atingido."
                )
                break
            try:
                result = run_single(emenda_id, transaction)
                inserted_id = save_result_to_sqlite(conn, emenda_id, result)
                print(f"[SUCESSO] Salvo no banco AnaliseResultados (id={inserted_id}, emenda_id={emenda_id})")
                total_processados += 1
            except Exception as e:
                print(f"[ERRO] Falha ao processar emenda {emenda_id}: {str(e)}")
    finally:
        conn.close()

    print(
        "\n[FINALIZADO] Processamento concluído. "
        f"Total de registros analisados: {total_processados}"
    )
