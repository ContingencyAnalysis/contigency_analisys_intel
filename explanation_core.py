"""Grounded Benford explanations backed by curated Brazilian public sources."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LEGAL_SOURCES_PATH = ROOT / "data" / "legal_sources.json"
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "openai.gpt-oss-20b-1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

METRIC_FIELDS = {
    "mad": ("MAD 1D", "mad_threshold", "mad_normalized", None),
    "jsd": ("JSD 1D", "jsd_threshold", "jsd_normalized", None),
    "z_max": ("Z máximo", "z_threshold", "z_normalized", None),
    "last_digit_test": (
        "Teste do último dígito",
        "last_digit_threshold",
        "last_digit_normalized",
        None,
    ),
    "mad_2d": ("MAD 2D", "mad_2d_threshold", "mad_2d_normalized", "mad_2d_flag"),
    "jsd_2d": ("JSD 2D", "jsd_2d_threshold", "jsd_2d_normalized", "jsd_2d_flag"),
}

SYSTEM_PROMPT = """Você explica alertas estatísticos de compras públicas para revisão humana.

Regras obrigatórias:
- A análise Benford é um sinal de triagem da distribuição de uma população; nunca é prova de fraude, manipulação ou infração legal.
- O registro recebido é uma linha de métricas agregadas por grupo, medida e período. Não o descreva como uma compra individual.
- Use apenas as métricas fornecidas. Distinga valores acima e abaixo dos limiares; não invente dado ausente.
- Para cada interpretação, use apenas um identificador de métrica permitido e explique o significado do resultado sem repetir números.
- Dê hipóteses operacionais como hipóteses, incluindo causas legítimas possíveis. Não atribua culpa nem afirme descumprimento legal.
- Os trechos legais são contexto para gestão de riscos e controle. Não dizem que Benford é obrigatório nem que um desvio viola a lei.
- Se eligible ou alerted não for true, diga que o registro não representa um alerta elegível e classifique a severidade como INDEFINIDO.
- A confiança descreve a qualidade desta explicação com os dados disponíveis, não a probabilidade de fraude.
- Responda no JSON definido pelo schema, em português brasileiro. Não inclua cadeia de raciocínio, texto fora do JSON, citações ou números nas interpretações.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "metric_interpretations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "metric": {"type": "string", "enum": list(METRIC_FIELDS)},
                    "interpretation": {"type": "string"},
                },
                "required": ["metric", "interpretation"],
            },
        },
        "possible_operational_explanations": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
        "limitations": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "review_actions": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "severity": {
            "type": "string",
            "enum": ["BAIXO", "MEDIO", "CRITICO", "INDEFINIDO"],
        },
        "confidence": {"type": "string", "enum": ["BAIXA", "MEDIA", "ALTA"]},
    },
    "required": [
        "summary",
        "metric_interpretations",
        "possible_operational_explanations",
        "limitations",
        "review_actions",
        "severity",
        "confidence",
    ],
}


def load_legal_sources() -> list[dict[str, Any]]:
    """Load the versioned official-source excerpts shipped with this service."""
    payload = json.loads(LEGAL_SOURCES_PATH.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Nenhuma fonte legal configurada em {LEGAL_SOURCES_PATH}")

    seen: set[str] = set()
    for source in sources:
        for required in ("source_id", "title", "citation", "source_url", "content"):
            if not source.get(required):
                raise ValueError(f"Fonte legal sem campo obrigatório {required!r}")
        if source["source_id"] in seen:
            raise ValueError(f"source_id duplicado: {source['source_id']}")
        seen.add(source["source_id"])
    return sources


def build_prompt(transaction: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    source_packets = [
        {
            "source_id": source["source_id"],
            "title": source["title"],
            "citation": source["citation"],
            "scope": source["scope"],
            "official_excerpt": source["content"],
        }
        for source in sources
    ]
    return json.dumps(
        {
            "record_type": "benford_alert_aggregate",
            "aggregate_metrics": transaction,
            "official_sources": source_packets,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _clean_model_content(content: Any) -> str:
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not isinstance(content, str):
        raise ValueError("Resposta do modelo sem conteúdo textual")
    # AWS documents that GPT-OSS InvokeModel responses can prefix the answer
    # with a <reasoning> block. Never expose or persist that block.
    cleaned = re.sub(
        r"<\s*reasoning\b[^>]*>[\s\S]*?<\s*/\s*reasoning\s*>",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()
    if "<reasoning" in cleaned.lower():
        raise ValueError("O modelo retornou um bloco reasoning incompleto")
    return cleaned


def _validate_structured_answer(answer: Any, transaction: dict[str, Any]) -> dict[str, Any]:
    expected = set(OUTPUT_SCHEMA["required"])
    if not isinstance(answer, dict) or set(answer) != expected:
        raise ValueError("Resposta do modelo não corresponde ao schema de explicabilidade")

    for key in ("summary",):
        if not isinstance(answer[key], str) or not answer[key].strip():
            raise ValueError(f"Campo obrigatório vazio: {key}")
    for key in ("possible_operational_explanations", "limitations", "review_actions"):
        if not isinstance(answer[key], list) or not answer[key]:
            raise ValueError(f"Lista obrigatória vazia ou inválida: {key}")
        if any(not isinstance(item, str) or not item.strip() for item in answer[key]):
            raise ValueError(f"Item vazio ou inválido em {key}")

    interpretations = answer["metric_interpretations"]
    if not isinstance(interpretations, list) or not interpretations:
        raise ValueError("A resposta precisa interpretar ao menos uma métrica")
    seen_metrics: set[str] = set()
    for item in interpretations:
        if not isinstance(item, dict) or set(item) != {"metric", "interpretation"}:
            raise ValueError("Interpretação de métrica inválida")
        metric = item["metric"]
        if metric not in METRIC_FIELDS or metric in seen_metrics:
            raise ValueError(f"Identificador de métrica inválido ou repetido: {metric!r}")
        if transaction.get(metric) is None:
            raise ValueError(f"O modelo interpretou métrica ausente: {metric}")
        if not isinstance(item["interpretation"], str) or not item["interpretation"].strip():
            raise ValueError(f"Interpretação vazia para {metric}")
        # Metric values and thresholds are rendered deterministically below.
        if re.search(r"\d", item["interpretation"]):
            raise ValueError(f"Interpretação contém números não validados: {metric}")
        seen_metrics.add(metric)

    if answer["severity"] not in {"BAIXO", "MEDIO", "CRITICO", "INDEFINIDO"}:
        raise ValueError("Severidade inválida")
    if answer["confidence"] not in {"BAIXA", "MEDIA", "ALTA"}:
        raise ValueError("Confiança inválida")
    if transaction.get("eligible") is not True or transaction.get("alerted") is not True:
        answer["severity"] = "INDEFINIDO"
    return answer


def generate_answer(prompt: str, transaction: dict[str, Any]) -> dict[str, Any]:
    """Invoke Bedrock with schema-constrained JSON and reject incomplete answers."""
    import boto3

    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": int(os.getenv("AGENT_MAX_COMPLETION_TOKENS", "3000")),
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "benford_explanation", "schema": OUTPUT_SCHEMA},
        },
    }
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    response_body = json.loads(response["body"].read())
    choice = response_body.get("choices", [{}])[0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Resposta do modelo truncada; nenhuma análise foi salva")
    message = choice.get("message") or {}
    content = _clean_model_content(message.get("content"))
    if not content:
        raise ValueError("O modelo não produziu uma explicação final; nada foi salvo")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("O modelo não retornou JSON válido; nada foi salvo") from error
    return _validate_structured_answer(parsed, transaction)


def extract_nivel_anomalia(structured_answer: dict[str, Any]) -> str:
    """Read severity only from a schema-validated field, never from prose."""
    return structured_answer.get("severity", "INDEFINIDO")


def _format_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
    return f"{value:.6g}"


def render_analysis(
    transaction: dict[str, Any],
    sources: list[dict[str, Any]],
    answer: dict[str, Any],
) -> str:
    """Render metrics, model interpretation, and deterministic citations."""
    lines = [
        "## Contexto do alerta",
        (
            f"Resultado agregado de {transaction.get('measure_name') or 'medida não informada'} "
            f"para {transaction.get('grouping_level') or 'grupo não informado'} "
            f"no período {transaction.get('period_key') or 'não informado'}; "
            f"amostra de {transaction.get('sample_size') or 'tamanho não informado'} registros."
        ),
        "Este registro não identifica uma compra individual.",
        "",
        "## Evidências estatísticas",
        (
            f"Elegível: {transaction.get('eligible')}; alerta ativo: "
            f"{transaction.get('alerted')}; motivo de elegibilidade: "
            f"{transaction.get('eligibility_reason') or 'não informado'}."
        ),
    ]

    for metric, (label, threshold_key, normalized_key, flag_key) in METRIC_FIELDS.items():
        value = transaction.get(metric)
        if value is None:
            continue
        threshold = transaction.get(threshold_key)
        comparison = ""
        if threshold is not None:
            try:
                relation = "acima" if float(value) > float(threshold) else "não supera"
                comparison = f"; {relation} o limiar {_format_number(threshold)}"
            except (TypeError, ValueError):
                comparison = f"; limiar {_format_number(threshold)}"
        normalized = transaction.get(normalized_key)
        normalized_text = (
            f"; normalizado {_format_number(normalized)}" if normalized is not None else ""
        )
        flag = transaction.get(flag_key) if flag_key else None
        flag_text = f"; sinalizador {flag}" if flag else ""
        lines.append(
            f"- {label}: {_format_number(value)}{comparison}{normalized_text}{flag_text}."
        )

    lines.extend(["", "## Interpretação dos sinais"])
    lines.append(answer["summary"].strip())
    lines.extend(
        f"- {item['metric']}: {item['interpretation'].strip()}"
        for item in answer["metric_interpretations"]
    )

    lines.extend(["", "## Contexto legal e metodológico"])
    sources_by_id = {source["source_id"]: source for source in sources}
    cite = lambda source_id: (
        f"[{sources_by_id[source_id]['citation']}]"
        f"({sources_by_id[source_id]['source_url']})"
    )
    lines.append(
        "- A Lei nº 14.133/2021 inclui compras em seu âmbito e trata de objetivos "
        "licitatórios, governança, gestão de riscos e controle preventivo "
        f"({cite('LEI-14133-ART-2-II')}, {cite('LEI-14133-ART-11')}, "
        f"{cite('LEI-14133-ART-169')})."
    )
    lines.append(
        "- O TCU descreve Benford como ferramenta de triagem e trata desvios como "
        "red flags/candidatos a auditoria; isso não transforma o resultado em prova "
        f"de fraude ou infração ({cite('TCU-BENFORD-AUDITORIA-2016')})."
    )
    lines.append(
        "- Essas normas oferecem contexto para revisão humana; não prescrevem "
        "Benford nem permitem inferir, só pelas métricas, violação da Lei. Confirme "
        "o regime jurídico aplicável e consulte os documentos do processo específico."
    )

    lines.extend(["", "## Hipóteses operacionais a verificar"])
    lines.extend(f"- {item.strip()}" for item in answer["possible_operational_explanations"])
    lines.extend(["", "## Limitações"])
    lines.extend(f"- {item.strip()}" for item in answer["limitations"])
    if transaction.get("eligible") is not True or transaction.get("alerted") is not True:
        lines.append("- O registro não está marcado como alerta elegível; não há base para tratá-lo como suspeito.")
    else:
        lines.append("- O alerta estatístico é não causal e exige validação nos documentos da compra.")

    lines.extend(["", "## Ações de revisão"])
    lines.extend(f"- {item.strip()}" for item in answer["review_actions"])
    lines.extend(
        [
            "",
            "## Classificação do agente",
            f"Severidade da anomalia: {answer['severity']}",
            f"Confiança na explicação: {answer['confidence']}",
        ]
    )
    return "\n".join(lines)
