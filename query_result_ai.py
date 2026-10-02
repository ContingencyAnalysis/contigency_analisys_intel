import os
import json
import sqlite3
import boto3
import re
import random
import unicodedata
from collections import Counter
from typing import Any, Iterator
from pathlib import Path

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import dotenv_values, load_dotenv
from bedrock_web_search import (
    run as run_bedrock_web_search,
    select_analytical_fields,
)

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
COMPRAS_PARQUET_DIR = Path(
    os.getenv(
        "COMPRAS_PARQUET_DIR",
        str(Path(__file__).resolve().parent / "transacao_compras"),
    )
)
MAX_TRANSACTION_CONTEXT_ROWS = 20
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
        print(
            f"[FIM] Nenhum registro encontrado para "
            f"{POSTGRES_SOURCE_SYSTEM}/{POSTGRES_DATASET} "
            f"com offset {PROCESS_OFFSET}."
        )
        return
    for row in rows:
        data = dict(row)
        # The application identifies published entities by group_id (falling
        # back to orgao_id). Keep the same identifier in AnaliseResultados so
        # a dashboard selection can open the saved analysis directly.
        entity_id = str(data.get("group_id") or data.get("orgao_id") or "").strip()
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
            json.dumps(result["documents"], ensure_ascii=False, default=str),
            result["analysis"],
            json.dumps(result, ensure_ascii=False, default=str),
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


def _transaction_group_column(grouping_level: Any) -> str | None:
    level = unicodedata.normalize("NFKD", str(grouping_level or ""))
    level = "".join(character for character in level if not unicodedata.combining(character))
    level = re.sub(r"[^a-z0-9]+", "_", level.lower()).strip("_")
    return {
        "orgao_superior": "codigo_orgao_superior",
        "unidade_gestora": "codigo_ug",
        "orgao": "codigo_orgao",
        "contratado": "codigo_contratado",
        "fornecedor": "codigo_contratado",
        "empresa": "codigo_contratado",
        "favorecido": "codigo_contratado",
    }.get(level)


def _parse_transaction_period(period_key: Any) -> dict[str, int] | None:
    value = str(period_key or "").strip()
    match = re.fullmatch(
        r"(?P<year>\d{4})(?:[-/](?P<month>\d{1,2})(?:[-/](?P<day>\d{1,2}))?|(?P<compact_month>\d{2}))?",
        value,
    )
    if not match:
        return None
    parts = {"year": int(match.group("year"))}
    month = match.group("month") or match.group("compact_month")
    if month:
        parts["month"] = int(month)
        if not 1 <= parts["month"] <= 12:
            return None
    if match.group("day"):
        parts["day"] = int(match.group("day"))
        if not 1 <= parts["day"] <= 31:
            return None
    return parts


def load_compras_transaction_context(record: dict[str, Any]) -> dict[str, Any]:
    """Load a bounded, period- and group-matched view from local purchase Parquet."""
    base_context = {
        "source": COMPRAS_PARQUET_DIR.name,
        "source_system": record.get("source_system"),
        "dataset": record.get("dataset"),
        "group_id": str(record.get("group_id") or record.get("orgao_id") or "").strip(),
        "group_name": record.get("group_name"),
        "grouping_level": record.get("grouping_level"),
        "period_key": record.get("period_key"),
        "measure_name": record.get("measure_name"),
    }
    if not COMPRAS_PARQUET_DIR.is_dir():
        return {**base_context, "status": "dataset_missing", "matched_transaction_count": 0}
    if not any(COMPRAS_PARQUET_DIR.rglob("*.parquet")):
        return {**base_context, "status": "no_parquet_files", "matched_transaction_count": 0}

    group_id = base_context["group_id"]
    group_column = _transaction_group_column(record.get("grouping_level"))
    if not group_id or not group_column:
        return {
            **base_context,
            "status": "unsupported_grouping_level",
            "matched_transaction_count": 0,
        }

    period = _parse_transaction_period(record.get("period_key"))
    if period is None:
        return {
            **base_context,
            "status": "unsupported_period_key",
            "matched_transaction_count": 0,
        }

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError(
            "Pacote pyarrow ausente para ler transacao_compras. "
            "Instale as dependências de contigency_analisys_intel."
        ) from error

    dataset = ds.dataset(str(COMPRAS_PARQUET_DIR), format="parquet", partitioning="hive")
    schema_names = set(dataset.schema.names)
    if group_column not in schema_names or "data_assinatura_contrato" not in schema_names:
        return {
            **base_context,
            "status": "incompatible_parquet_schema",
            "matched_transaction_count": 0,
            "required_columns": [group_column, "data_assinatura_contrato"],
        }

    date_field = ds.field("data_assinatura_contrato")
    date_type = dataset.schema.field("data_assinatura_contrato").type
    if pa.types.is_string(date_type):
        date_field = pc.strptime(date_field, format="%Y-%m-%d", unit="s")
    elif not (
        pa.types.is_date(date_type)
        or pa.types.is_timestamp(date_type)
    ):
        return {
            **base_context,
            "status": "unsupported_date_column",
            "matched_transaction_count": 0,
        }

    filter_expression = ds.field(group_column) == group_id
    filter_expression = filter_expression & (pc.year(date_field) == period["year"])
    if "month" in period:
        if "mes" in schema_names:
            filter_expression = filter_expression & (ds.field("mes") == period["month"])
        filter_expression = filter_expression & (pc.month(date_field) == period["month"])
    if "day" in period:
        filter_expression = filter_expression & (pc.day(date_field) == period["day"])

    measure_name = str(record.get("measure_name") or "").strip()
    measure_column = measure_name if measure_name in schema_names else None
    if measure_column:
        measure_type = dataset.schema.field(measure_column).type
        if pa.types.is_integer(measure_type) or pa.types.is_floating(measure_type) or pa.types.is_decimal(measure_type):
            filter_expression = filter_expression & (ds.field(measure_column) > 0)
        else:
            measure_column = None

    context_columns = (
        "numero_do_contrato",
        "objeto",
        "fundamento_legal",
        "modalidade_compra",
        "situacao_contrato",
        "codigo_orgao_superior",
        "nome_orgao_superior",
        "codigo_orgao",
        "nome_orgao",
        "codigo_ug",
        "nome_ug",
        "data_assinatura_contrato",
        "data_publicacao_dou",
        "data_inicio_vigencia",
        "data_fim_vigencia",
        "codigo_contratado",
        "nome_contratado",
        "valor_inicial_compra",
        "valor_final_compra",
        "numero_licitacao",
        "modalidade_compra_licitacao",
        "mes",
    )
    columns = [column for column in context_columns if column in schema_names]
    if group_column not in columns:
        columns.append(group_column)
    if measure_column and measure_column not in columns:
        columns.append(measure_column)

    scanner = dataset.scanner(
        columns=columns,
        filter=filter_expression,
        batch_size=1024,
    )
    sample: list[dict[str, Any]] = []
    randomizer = random.Random(
        f"{group_id}|{record.get('period_key')}|{measure_name}"
    )
    matched_count = 0
    amount_count = 0
    amount_total = 0.0
    amount_min: float | None = None
    amount_max: float | None = None
    contractor_codes: set[str] = set()
    modality_counts: Counter[str] = Counter()

    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            matched_count += 1
            if row.get("codigo_contratado"):
                contractor_codes.add(str(row["codigo_contratado"]))
            if row.get("modalidade_compra"):
                modality_counts[str(row["modalidade_compra"])] += 1
            if measure_column and row.get(measure_column) is not None:
                amount = float(row[measure_column])
                amount_count += 1
                amount_total += amount
                amount_min = amount if amount_min is None else min(amount_min, amount)
                amount_max = amount if amount_max is None else max(amount_max, amount)

            if row.get("objeto") is not None:
                row["objeto"] = str(row["objeto"])[:600]
            if row.get("fundamento_legal") is not None:
                row["fundamento_legal"] = str(row["fundamento_legal"])[:400]

            if len(sample) < MAX_TRANSACTION_CONTEXT_ROWS:
                sample.append(row)
            else:
                replacement = randomizer.randrange(matched_count)
                if replacement < MAX_TRANSACTION_CONTEXT_ROWS:
                    sample[replacement] = row

    status = "matched" if measure_column else "matched_context_only"
    context = {
        **base_context,
        "status": status if matched_count else "no_matches",
        "matched_transaction_count": matched_count,
        "matching_rule": {
            "group_column": group_column,
            "date_column": "data_assinatura_contrato",
            "measure_column": measure_column,
            "period": period,
            "positive_measure_only": bool(measure_column),
        },
        "distinct_contractor_count": len(contractor_codes),
        "modality_counts": dict(modality_counts),
        "sample_row_count": len(sample),
        "sample_is_illustrative": True,
        "sampled_transactions": sample,
    }
    if measure_column and amount_count:
        context["measure_summary"] = {
            "column": measure_column,
            "count": amount_count,
            "sum": amount_total,
            "min": amount_min,
            "max": amount_max,
            "mean": amount_total / amount_count,
        }
    if matched_count > len(sample):
        context["omitted_transaction_count"] = matched_count - len(sample)
    return context


def search_documents(transaction):
    """Retrieve and rerank legal documents using the transaction's identity/context."""
    transaction_context = " | ".join(
        f"{key}: {value}"
        for key, value in transaction.items()
        if value is not None and str(value).strip()
    )
    # The legal corpus normally will not contain the purchase ID or exact
    # amount. Keep procurement-law concepts in the query so applicable norms
    # can be retrieved as contextual evidence.
    query_text = (
        "compras públicas licitação contrato administrativo fornecedor "
        "Lei 14.133/2021 Lei 8.666/1993 Decreto 11.129/2022 "
        "integridade controle fiscalização prestação de contas | "
        + transaction_context
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
        legal_terms = _search_tokens(
            "licitação contrato administrativo fornecedor lei decreto "
            "integridade controle fiscalização prestação contas"
        )
        legal_overlap = legal_terms & document_tokens
        # Exact contextual matches outrank generic semantic matches. This avoids
        # returning unrelated audit reports merely because they discuss fraud.
        relevance = len(overlap) * 0.15
        relevance += len(legal_overlap) * 0.08
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
- Os documentos podem servir como base jurídica contextual mesmo quando não
  mencionarem o ID ou o valor exato da transação. Relacione somente normas,
  decisões, auditorias ou orientações que tratem de compras públicas,
  licitações, contratos, integridade, controle ou fiscalização aplicáveis ao
  caso. Se nenhum documento tiver esse conteúdo, informe isso explicitamente.
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
3. Base jurídica/contextual encontrada nos documentos recuperados, citando a
   lei, decreto, decisão ou orientação quando ela aparecer no conteúdo.
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
    configured_source_system = (
        POSTGRES_SOURCE_SYSTEM if DATA_SOURCE == "postgres" else ""
    )
    configured_dataset = POSTGRES_DATASET if DATA_SOURCE == "postgres" else ""
    source_system = str(
        transaction.get("source_system") or configured_source_system
    ).strip().lower()
    dataset = str(transaction.get("dataset") or configured_dataset).strip().lower()
    if (
        source_system == "compras"
        and dataset in {"compras", "compras_publicas"}
    ):
        return run_compras_single(emenda_id, transaction)

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


def run_compras_single(group_id, transaction):
    """Explain Compras metrics together with matching local contract records."""
    print(f"\n--- Processando registro de Compras: {group_id} ---")
    record = select_analytical_fields(transaction)
    transaction_context = load_compras_transaction_context(transaction)

    print(
        "[1] Contexto transacional: "
        f"{transaction_context['status']} "
        f"({transaction_context.get('matched_transaction_count', 0)} registros encontrados)."
    )
    print("[2] Pesquisando fontes e gerando a justificativa...")
    search_result = run_bedrock_web_search(record, transaction_context)
    research = search_result["web_research_summary"]
    sources = search_result["web_search_sources"]
    print(f"    {len(sources)} fontes recuperadas.")

    analysis = search_result["analysis"]

    # Keep both transaction evidence and cited web research in documents_json.
    documents = [
        {
            "content": json.dumps(transaction_context, ensure_ascii=False, default=str),
            "metadata": {
                "source": "transacao_compras Parquet",
                "status": transaction_context["status"],
                "matched_transaction_count": transaction_context.get(
                    "matched_transaction_count", 0
                ),
            },
        },
        {
            "content": research,
            "metadata": {
                "source": "Amazon Bedrock Web Search",
                "web_search_sources": sources,
            },
        },
    ]
    print("\n" + "=" * 60)
    print(analysis)
    print("=" * 60)

    return {
        "transaction": transaction,
        "documents": documents,
        "analysis": analysis,
        "web_search_model": search_result["web_search_model"],
        "analysis_model": search_result["analysis_model"],
        "web_research_summary": research,
        "web_search_sources": sources,
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
