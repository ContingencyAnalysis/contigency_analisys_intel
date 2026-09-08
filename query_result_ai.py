import os
import json
import sqlite3
import boto3
import re

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "motor_estatistico.db")
SQLITE_TABLE = os.getenv("SQLITE_TABLE", "EmendasParlamentares") 
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_documents")
TOP_K = int(os.getenv("TOP_K", "5"))

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "openai.gpt-oss-20b-1:0")

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# =====================================================
# SQLITE
# =====================================================

def load_transactions():
    """Gerador que retorna todas as transações da tabela."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {SQLITE_TABLE}")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise Exception("Nenhuma transação encontrada na tabela")

    for row in rows:
        data = dict(row)
        emenda_key = data.get("codigo_da_emenda")
        if not emenda_key:
            print(f"[AVISO] Registro ID {data.get('id', 'N/A')} ignorado: sem 'codigo_da_emenda'")
            continue
        yield emenda_key, data


def init_result_table(conn):
    cursor = conn.cursor()

    cursor.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_emendas_codigo
        ON {SQLITE_TABLE} (codigo_da_emenda)
        """
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

def search_documents(query_text):
    client = MilvusClient(uri=MILVUS_URI)
    embedding = generate_embedding(query_text)

    results = client.search(
        collection_name=MILVUS_COLLECTION,
        data=[embedding],
        limit=TOP_K,
        output_fields=["content"]
    )

    docs = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        docs.append(
            {
                "score": hit["distance"],
                "content": entity.get("content", "")
            }
        )

    return docs

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

Explique somente os possíveis motivos que levaram o motor estatístico a considerar esta transação suspeita.

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

    query_text = (
        "fraude auditoria irregularidade "
        + " ".join([f"{k} {v}" for k, v in transaction.items() if v is not None])
    )

    print("[1] Buscando documentos no Milvus...")
    docs = search_documents(query_text)
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
        for emenda_id, transaction in load_transactions():
            try:
                result = run_single(emenda_id, transaction)
                inserted_id = save_result_to_sqlite(conn, emenda_id, result)
                print(f"[SUCESSO] Salvo no banco AnaliseResultados (id={inserted_id}, emenda_id={emenda_id})")
                total_processados += 1
            except Exception as e:
                print(f"[ERRO] Falha ao processar emenda {emenda_id}: {str(e)}")
    finally:
        conn.close()

    print(f"\n[FINALIZADO] Processamento concluído. Total de registros analisados: {total_processados}")