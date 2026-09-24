"""API HTTP para consulta das análises produzidas pelo agente estatístico."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("SQLITE_DB_PATH", BASE_DIR / "motor_estatistico.db"))


class AnalysisSummary(BaseModel):
    id: int
    emenda_id: str
    criado_em: str | None
    nivel_anomalia: str
    analysis: str


class Analysis(AnalysisSummary):
    model_config = ConfigDict(from_attributes=True)
    transaction: dict[str, Any] | None = None
    documents: list[dict[str, Any]] = Field(default_factory=list)
    resultado_completo: dict[str, Any] | None = None


app = FastAPI(
    title="Contigency Analysis Intel API",
    version="1.0.0",
    description="Consulta das análises geradas pelo agente de detecção de anomalias.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _connect() -> sqlite3.Connection:
    if not DATABASE_PATH.exists():
        raise HTTPException(status_code=503, detail="Banco de análises não encontrado")
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _to_analysis(row: sqlite3.Row) -> Analysis:
    complete = _parse_json(row["resultado_completo"], {})
    return Analysis(
        id=row["id"],
        emenda_id=row["emenda_id"],
        criado_em=row["criado_em"],
        nivel_anomalia=row["nivel_anomalia"],
        analysis=row["analysis"],
        transaction=complete.get("transaction") if isinstance(complete, dict) else None,
        documents=_parse_json(row["documents_json"], []),
        resultado_completo=complete if isinstance(complete, dict) else None,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/analises", response_model=list[AnalysisSummary])
def list_analyses(
    nivel_anomalia: str | None = Query(default=None, description="BAIXO, MEDIO ou CRITICO"),
    emenda_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[AnalysisSummary]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if nivel_anomalia:
        clauses.append("UPPER(nivel_anomalia) = UPPER(?)")
        parameters.append(nivel_anomalia)
    if emenda_id:
        clauses.append("emenda_id = ?")
        parameters.append(emenda_id)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT id, emenda_id, criado_em, nivel_anomalia, analysis
        FROM AnaliseResultados
        {where}
        ORDER BY datetime(criado_em) DESC, id DESC
        LIMIT ? OFFSET ?
    """
    parameters.extend([limit, offset])
    with _connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [AnalysisSummary(**dict(row)) for row in rows]


@app.get("/api/analises/{emenda_id}", response_model=Analysis)
def get_analysis(emenda_id: str) -> Analysis:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, emenda_id, criado_em, nivel_anomalia, analysis,
                   documents_json, resultado_completo
            FROM AnaliseResultados
            WHERE emenda_id = ?
            ORDER BY datetime(criado_em) DESC, id DESC
            LIMIT 1
            """,
            (emenda_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Análise não encontrada")
    return _to_analysis(row)
