"""
parquet_to_sqlite.py
Lê um ou mais arquivos .parquet e insere o conteúdo em um banco SQLite local.
"""

import sys
from pathlib import Path

import pandas as pd
import sqlite3
from loguru import logger


PARQUET_INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./motor_estatistico_results/emendas_parlamentares")
SQLITE_PATH   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("motor_estatistico.db")


def parquet_files(source: Path) -> list[Path]:
    if source.is_file() and source.suffix == ".parquet":
        return [source]
    return sorted(source.rglob("*.parquet"))


def table_name_from(filepath: Path) -> str:
    """Usa o nome do arquivo (sem extensão) como nome da tabela."""
    return filepath.stem.lower().replace("-", "_").replace(" ", "_")


def load_parquet(filepath: Path) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    # Normaliza colunas
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
    )
    return df


def to_sqlite(df: pd.DataFrame, table: str, conn: sqlite3.Connection) -> int:
    df.to_sql(
        name=table,
        con=conn,
        if_exists="replace",   # troca: use "append" para acumular
        index=False,
        chunksize=1000,
    )
    return len(df)


TABLE_NAME = "EmendasParlamentares"

def run():
    files = parquet_files(PARQUET_INPUT)

    if not files:
        logger.error("Nenhum arquivo .parquet encontrado em: {}", PARQUET_INPUT)
        sys.exit(1)

    logger.info("{} arquivo(s) encontrado(s)", len(files))

    conn = sqlite3.connect(SQLITE_PATH)

    try:
        total = 0
        first = True
        for filepath in files:
            logger.info("Lendo {} → tabela '{}'", filepath.name, TABLE_NAME)
            try:
                df = load_parquet(filepath)
                df.to_sql(
                    name=TABLE_NAME,
                    con=conn,
                    if_exists="replace" if first else "append",  # recria na primeira, acumula nas demais
                    index=False,
                    chunksize=1000,
                )
                first = False
                logger.info("✓ {} linhas inseridas em '{}'", len(df), TABLE_NAME)
                total += len(df)
            except Exception as e:
                logger.error("Erro em {}: {}", filepath.name, e)

        conn.commit()
        logger.info("=== Concluído: {} linhas no total em {} ===", total, SQLITE_PATH)

    finally:
        conn.close()


if __name__ == "__main__":
    run()