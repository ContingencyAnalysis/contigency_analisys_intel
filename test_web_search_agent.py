#!/usr/bin/env python3
"""Exercise Amazon Bedrock Web Search and Bedrock analysis, without Milvus."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ENV_PATH = Path(__file__).resolve().with_name(".env")
try:
    from dotenv import dotenv_values

    DOTENV_CONFIG = dotenv_values(ENV_PATH)
except ImportError as error:
    raise RuntimeError(
        "Pacote python-dotenv ausente. Instale as dependências com: pip install -r requirements.txt"
    ) from error


def configured_value(name: str, default: str = "") -> str:
    return str(DOTENV_CONFIG.get(name) or os.getenv(name) or default)


AWS_REGION = configured_value("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = configured_value("BEDROCK_MODEL_ID", "openai.gpt-oss-20b-1:0")
# Bedrock Web Search currently supports GPT 5.4/5.5/5.6 through the Mantle endpoint.
BEDROCK_WEB_SEARCH_MODEL = configured_value(
    "BEDROCK_WEB_SEARCH_MODEL", "openai.gpt-5.6-terra"
)
REQUEST_TIMEOUT = int(
    configured_value("BEDROCK_TIMEOUT_SECONDS")
    or configured_value("OPENAI_TIMEOUT_SECONDS", "120")
)

# This synthetic example is deliberately labeled; it is not a real alert.
SAMPLE_RECORD: dict[str, Any] = {
    "source_system": "Compras",
    "dataset": "compras",
    "measure_name": "valor_total",
    "grouping_level": "fornecedor",
    "period_key": "2025",
    "sample_size": 380,
    "eligible": True,
    "alerted": True,
    "eligibility_reason": "amostra elegível; sinal acima do limiar configurado",
    "mad": 0.012,
    "mad_threshold": 0.006,
    "mad_normalized": 2.0,
    "jsd": 0.021,
    "jsd_threshold": 0.010,
    "jsd_normalized": 2.1,
    "z_max": 3.2,
    "z_threshold": 2.5,
    "z_normalized": 1.28,
}

ANALYTICAL_FIELDS = {
    "source_system",
    "dataset",
    "measure_name",
    "grouping_level",
    "period_key",
    "sample_size",
    "eligible",
    "alerted",
    "eligibility_reason",
    "mad",
    "mad_threshold",
    "mad_normalized",
    "jsd",
    "jsd_threshold",
    "jsd_normalized",
    "z_max",
    "z_threshold",
    "z_normalized",
    "last_digit_test",
    "last_digit_threshold",
    "last_digit_normalized",
    "mad_2d",
    "mad_2d_threshold",
    "mad_2d_normalized",
    "mad_2d_flag",
    "jsd_2d",
    "jsd_2d_threshold",
    "jsd_2d_normalized",
    "jsd_2d_flag",
}

SEARCH_INSTRUCTIONS = """Pesquise na web fontes oficiais brasileiras para contextualizar este alerta estatístico.

Use exclusivamente fontes dos domínios oficiais .gov.br, especialmente Planalto, TCU e CGU. Considere a vigência no período informado. Procure a Lei nº 14.133/2021 (governança, competição, gestão de riscos e controle) e orientações do TCU sobre a Lei de Benford como ferramenta de triagem.

Retorne um resumo curto com o diploma e artigo quando disponíveis. Use citações para as fontes consultadas. Não afirme que Benford é exigido por lei nem que um desvio prova fraude.
"""

ANALYSIS_INSTRUCTIONS = """Você explica alertas estatísticos agregados de compras públicas para revisão humana.

Regras:
- Um alerta de Benford é um sinal de triagem sobre a distribuição de uma população; não prova fraude, manipulação ou infração legal.
- O registro contém métricas agregadas por grupo e período, não uma transação ou fornecedor individual.
- Explique quais métricas fornecidas contribuíram para o alerta e apresente causas operacionais legítimas como hipóteses, sem inventar dados.
- Use como base jurídica somente o resumo e as fontes oficiais retornadas pela busca do Bedrock. Informe diploma e artigo quando disponíveis.
- A base legal contextualiza governança, competição, gestão de riscos e controle. Não afirme que Benford é exigido por lei nem que o resultado, por si só, viola uma norma.
- Distinga evidência estatística de contexto jurídico e explicite limitações. Se eligible ou alerted não for true, diga que o registro não está marcado como alerta elegível.
- Responda em português brasileiro, de forma concisa, usando as seções: MOTIVOS POSSÍVEIS, EVIDÊNCIAS ESTATÍSTICAS, CONTEXTO LEGAL, LIMITAÇÕES.
"""


def load_record(path: str | None) -> dict[str, Any]:
    if path is None:
        return SAMPLE_RECORD.copy()

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("O JSON da transação deve conter um objeto na raiz")
    return payload


def select_analytical_fields(record: dict[str, Any]) -> dict[str, Any]:
    selected = {key: value for key, value in record.items() if key in ANALYTICAL_FIELDS}
    metric_fields = {"mad", "jsd", "z_max", "last_digit_test", "mad_2d", "jsd_2d"}
    if not any(selected.get(key) is not None for key in metric_fields):
        raise ValueError(
            "O registro não contém métricas reconhecidas (mad, jsd, z_max, "
            "last_digit_test, mad_2d ou jsd_2d)"
        )
    return selected


def build_search_input(record: dict[str, Any]) -> str:
    return (
        "Pesquise fontes legais oficiais que contextualizem este alerta estatístico. "
        "As métricas servem apenas para direcionar a pesquisa; não conclua fraude.\n\n"
        "REGISTRO:\n"
        + json.dumps(record, ensure_ascii=False, indent=2, default=str)
    )


def extract_response_text(response_data: dict[str, Any]) -> str:
    chunks = []
    for item in response_data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def extract_web_search_sources(response_data: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in response_data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            for citation in content.get("annotations", []):
                if citation.get("type") != "url_citation":
                    continue
                url = str(citation.get("url") or "")
                hostname = (urlparse(url).hostname or "").lower()
                if not url.startswith("https://") or not hostname.endswith(".gov.br"):
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                sources.append(
                    {
                        "title": str(citation.get("title") or "Fonte oficial consultada"),
                        "url": url,
                    }
                )
    return sources


def create_aws_session():
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError(
            "Pacote boto3 ausente. Instale as dependências com: pip install -r requirements.txt"
        ) from error

    access_key = configured_value("AWS_ACCESS_KEY_ID")
    secret_key = configured_value("AWS_SECRET_ACCESS_KEY")
    session_token = configured_value("AWS_SESSION_TOKEN")
    if bool(access_key) != bool(secret_key):
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID e AWS_SECRET_ACCESS_KEY precisam ser configuradas juntas"
        )

    session_options: dict[str, str] = {"region_name": AWS_REGION}
    if access_key and secret_key:
        session_options.update(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        if session_token:
            session_options["aws_session_token"] = session_token
    session = boto3.Session(**session_options)
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("Credenciais AWS não encontradas no .env, ambiente ou perfil AWS")
    return session, credentials.get_frozen_credentials()


def search_legal_sources(
    record: dict[str, Any], credentials: Any
) -> tuple[str, list[dict[str, str]]]:
    try:
        from openai import OpenAI
        from openai.providers import bedrock
    except ImportError as error:
        raise RuntimeError(
            "Instale openai[bedrock] para usar SigV4 com Amazon Bedrock Mantle"
        ) from error

    # The OpenAI-compatible SDK is pointed at Bedrock Mantle and signed with AWS IAM.
    # api_key=None disables both OpenAI API-key and Bedrock bearer-token authentication.
    provider = bedrock(
        base_url=f"https://bedrock-mantle.{AWS_REGION}.api.aws/openai/v1",
        region=AWS_REGION,
        api_key=None,
        access_key_id=credentials.access_key,
        secret_access_key=credentials.secret_key,
        session_token=credentials.token,
    )
    client = OpenAI(provider=provider, timeout=REQUEST_TIMEOUT)
    response = client.responses.create(
        model=BEDROCK_WEB_SEARCH_MODEL,
        instructions=SEARCH_INSTRUCTIONS,
        input=build_search_input(record),
        tools=[
            {
                "type": "web_search",
                "search_context_size": "medium",
                "external_web_access": False,
            }
        ],
        tool_choice="required",
        store=False,
    )
    response_data = response.model_dump(mode="json")
    research = extract_response_text(response_data)
    sources = extract_web_search_sources(response_data)
    if not research:
        raise RuntimeError("A busca web do Bedrock retornou uma resposta vazia")
    if not sources:
        raise RuntimeError("A busca web não retornou citações de fontes oficiais .gov.br")
    return research, sources


def generate_analysis(
    record: dict[str, Any],
    research: str,
    sources: list[dict[str, str]],
    session: Any,
) -> str:
    prompt = json.dumps(
        {
            "aggregate_metrics": record,
            "web_research_summary": research,
            "official_sources": sources,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    body = {
        "messages": [
            {"role": "system", "content": ANALYSIS_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 3000,
        "temperature": 0.1,
    }
    response = session.client("bedrock-runtime", region_name=AWS_REGION).invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
    )
    response_body = json.loads(response["body"].read())
    choice = response_body.get("choices", [{}])[0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("Resposta do modelo Bedrock truncada")
    content = (choice.get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("O modelo Bedrock não retornou uma explicação textual")
    content = re.sub(
        r"<\s*reasoning\b[^>]*>[\s\S]*?<\s*/\s*reasoning\s*>",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()
    if "<reasoning" in content.lower():
        raise RuntimeError("O modelo Bedrock retornou um bloco reasoning incompleto")
    return content


def run(record: dict[str, Any]) -> dict[str, Any]:
    session, credentials = create_aws_session()
    research, sources = search_legal_sources(record, credentials)
    analysis = generate_analysis(record, research, sources, session)
    return {
        "web_search_model": BEDROCK_WEB_SEARCH_MODEL,
        "analysis_model": BEDROCK_MODEL_ID,
        "analysis": analysis,
        "web_search_sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Testa o Web Search do Amazon Bedrock e a análise no modelo Bedrock configurado, "
            "sem consultar Milvus. Envia à AWS apenas os campos analíticos reconhecidos."
        )
    )
    parser.add_argument(
        "--transaction-json",
        help="Arquivo JSON com uma linha de métricas; sem esta opção usa um exemplo sintético.",
    )
    parser.add_argument(
        "--output",
        help="Grava a resposta JSON completa neste caminho; por padrão imprime no terminal.",
    )
    args = parser.parse_args()

    try:
        record = select_analytical_fields(load_record(args.transaction_json))
        result = run(record)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"Resultado salvo em {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
