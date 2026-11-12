"""Эмбеддинги (bge-m3) и запись узлов в Qdrant через LlamaIndex."""
import logging
import uuid
from functools import lru_cache

from .blocks import Chunk
from .config import SETTINGS

log = logging.getLogger("pathrag.index")

# Эти поля не входят в текст для эмбеддинга (шум для поиска), но лежат в payload
EXCLUDE_FROM_EMBED = [
    "source_id", "doc_id", "file", "url", "page_start", "page_end", "chunk_index",
    "chunk_total", "char_len", "language", "content_type", "ingested_at", "sha",
]
# Эти поля не показываем LLM (для цитирования нужны book_title, страницы, url)
EXCLUDE_FROM_LLM = ["doc_id", "chunk_index", "chunk_total", "char_len", "ingested_at", "sha"]


@lru_cache(maxsize=1)
def get_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=SETTINGS.embed_model,
        device=SETTINGS.device,
        embed_batch_size=SETTINGS.embed_batch_size,
        max_length=SETTINGS.embed_max_tokens,
    )


@lru_cache(maxsize=1)
def get_client():
    from qdrant_client import QdrantClient

    if SETTINGS.qdrant_path:
        return QdrantClient(path=SETTINGS.qdrant_path)
    return QdrantClient(url=SETTINGS.qdrant_url, api_key=SETTINGS.qdrant_api_key)


def get_vector_store():
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    return QdrantVectorStore(client=get_client(), collection_name=SETTINGS.collection)


def chunks_to_nodes(chunks: list[Chunk], doc_id: str):
    from llama_index.core.schema import TextNode

    nodes = []
    for c in chunks:
        node_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{c.metadata['chunk_index']}"))
        n = TextNode(id_=node_id, text=c.text, metadata={**c.metadata, "doc_id": doc_id})
        n.excluded_embed_metadata_keys = EXCLUDE_FROM_EMBED
        n.excluded_llm_metadata_keys = EXCLUDE_FROM_LLM
        nodes.append(n)
    return nodes


def delete_doc(doc_id: str) -> None:
    """Удаляет старые чанки документа (если файл изменился и число чанков стало другим)."""
    from qdrant_client import models

    try:
        get_client().delete(
            collection_name=SETTINGS.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )
    except Exception as e:  # коллекции может ещё не быть
        log.debug("delete_doc skipped: %s", e)


def ensure_payload_indexes() -> None:
    from qdrant_client import models

    for field in ("source_id", "doc_id", "language", "book_title", "content_type"):
        try:
            get_client().create_payload_index(
                SETTINGS.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            log.debug("payload index %s: %s", field, e)


def upsert_chunks(chunks: list[Chunk], doc_id: str) -> int:
    from llama_index.core import StorageContext, VectorStoreIndex

    nodes = chunks_to_nodes(chunks, doc_id)
    delete_doc(doc_id)
    storage = StorageContext.from_defaults(vector_store=get_vector_store())
    VectorStoreIndex(nodes=nodes, storage_context=storage, embed_model=get_embed_model(), show_progress=True)
    ensure_payload_indexes()
    return len(nodes)


def collection_count() -> int | None:
    try:
        return get_client().count(SETTINGS.collection, exact=True).count
    except Exception:
        return None
