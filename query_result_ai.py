import os
import json
import sqlite3
from openai import OpenAI

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================

SQLITE_DB_PATH = os.getenv(
    "SQLITE_DB_PATH",
    "motor_estatistico.db"
)

SQLITE_TABLE = os.getenv(
    "SQLITE_TABLE",
    "EmendasParlamentares"
)

MILVUS_URI = os.getenv(
    "MILVUS_URI",
    "http://localhost:19530"
)

MILVUS_COLLECTION = os.getenv(
    "MILVUS_COLLECTION",
    "rag_documents"
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "5"
    )
)

HF_TOKEN = os.getenv(
    "HUGGINGFACE_API_TOKEN",
    ""
)

HF_EMBED_MODEL = os.getenv(
    "HF_EMBED_MODEL",
    "BAAI/bge-base-en-v1.5"
)

HF_LLM_MODEL = os.getenv(
    "HF_LLM_MODEL",
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini"
)

client = OpenAI(
    api_key=OPENAI_API_KEY
)

# =====================================================
# SQLITE
# =====================================================

def load_transaction():

    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT *
        FROM {SQLITE_TABLE}
        LIMIT 1
        """
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        raise Exception("Nenhuma transação encontrada")

    return dict(row)

# =====================================================
# HUGGINGFACE EMBEDDING
# =====================================================

embedding_model = SentenceTransformer(
    HF_EMBED_MODEL
)

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

    client = MilvusClient(
        uri=MILVUS_URI
    )

    embedding = generate_embedding(
        query_text
    )

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
                "content": entity.get(
                    "content",
                    ""
                )
            }
        )

    return docs

# =====================================================
# PROMPT
# =====================================================

SYSTEM_PROMPT = """
Você é um especialista em:

- auditoria pública
- detecção de fraudes
- análise de emendas parlamentares

Regras:

- Utilize apenas os documentos fornecidos.
- Não invente informações.
- Cite [Documento X].
- Se não houver evidência suficiente, informe isso.
"""

def build_prompt(transaction, docs):

    transaction_json = json.dumps(
        transaction,
        indent=2,
        ensure_ascii=False
    )

    context = "\n\n".join(
        [
            f"[Documento {i+1}]\n{d['content']}"
            for i, d in enumerate(docs)
        ]
    )

    return f"""
Dados analisados:

{transaction_json}

Documentos recuperados:

{context}

Explique:

1. Quais características chamam atenção.
2. Quais padrões encontrados nos documentos são semelhantes.
3. Possíveis irregularidades associadas.
4. Justifique usando apenas os documentos recuperados.

Cite [Documento X].
"""

# =====================================================
# HUGGINGFACE LLM
# =====================================================

def generate_answer(prompt):

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.output_text

# =====================================================
# MAIN
# =====================================================

def run():

    print("=" * 60)
    print("FRAUD RAG AGENT")
    print("=" * 60)

    print("\n[1] Carregando transação...")
    transaction = load_transaction()

    query_text = (
        "fraude auditoria irregularidade "
        + " ".join(
            [
                f"{k} {v}"
                for k, v in transaction.items()
                if v is not None
            ]
        )
    )

    print("\n[2] Buscando documentos...")

    docs = search_documents(
        query_text
    )

    print(
        f"{len(docs)} documentos recuperados"
    )

    print("\n[3] Montando prompt...")

    prompt = build_prompt(
        transaction,
        docs
    )

    print("\n[4] Consultando LLM...")

    answer = generate_answer(
        prompt
    )

    print("\n")
    print("=" * 60)
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

    result = run()

    with open(
        "fraud_analysis_result.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\nResultado salvo em fraud_analysis_result.json"
    )