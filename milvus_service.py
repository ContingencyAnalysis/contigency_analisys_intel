"""
Serviço Milvus: gerencia coleção, indexação e busca vetorial.
Usa MilvusClient (API moderna — PyMilvus >= 2.4).
"""
from loguru import logger
from pymilvus import MilvusClient, DataType

from config import settings

# BAAI/bge-base-en-v1.5 → 768 dimensões
EMBED_DIMENSION = int(settings.__dict__.get("EMBED_DIMENSION", 768))


class MilvusService:
    def __init__(self):
        self.collection_name = settings.MILVUS_COLLECTION
        self.client = MilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        )
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(self.collection_name):
            logger.info("Coleção já existe: {}", self.collection_name)
            return

        logger.info("Criando coleção Milvus: {}", self.collection_name)

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id",       DataType.INT64,        is_primary=True)
        schema.add_field("doc_uuid", DataType.VARCHAR,      max_length=64)
        schema.add_field("content",  DataType.VARCHAR,      max_length=8192)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIMENSION)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info("Coleção e índice criados.")

    def insert(
        self,
        doc_uuid: str,
        content: str,
        embedding: list[float],
        metadata: dict = {},
    ) -> int:
        res = self.client.insert(
            collection_name=self.collection_name,
            data=[{
                "doc_uuid":  doc_uuid,
                "content":   content[:8192],
                "embedding": embedding,
                "metadata":  metadata,
            }],
        )
        return res["ids"][0]

    def insert_batch(
        self,
        doc_ids: list[str],
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> list[int]:
        rows = [
            {
                "doc_uuid":  doc_id,
                "content":   content[:8192],
                "embedding": embedding,
                "metadata":  metadata,
            }
            for doc_id, content, embedding, metadata
            in zip(doc_ids, contents, embeddings, metadatas)
        ]

        res = self.client.insert(
            collection_name=self.collection_name,
            data=rows,
        )
        return res["ids"]

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["doc_uuid", "content", "metadata"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "id":       hit["id"],
                "content":  hit["entity"].get("content"),
                "score":    hit["distance"],
                "metadata": hit["entity"].get("metadata") or {},
            })
        return hits

    def flush(self) -> None:
        """Força persistência dos segmentos."""
        self.client.flush(self.collection_name)

    def delete_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.warning("Coleção {} removida.", self.collection_name)