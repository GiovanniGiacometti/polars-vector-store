import os
from typing import Any

import loguru
import numpy as np
from polars_vector_store.base import VectorStore
import chromadb

from polars_vector_store.loader.parquet import ParquetLoader


class ChromaDB(VectorStore):
    """
    Wrapper of ChromaDB client
    """

    DEFAULT_PATH = os.path.join("data", "chromadb")
    DEFAULT_COLLECTION_NAME = "embeddings"

    # chromadb has a limit of 5461 documents per batch
    MAX_BATCH_SIZE = 5461

    def __init__(
        self,
        db_path: str = DEFAULT_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: dict | None = None,
        **kwargs,
    ) -> None:
        self.client = chromadb.PersistentClient(db_path, **kwargs)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata=collection_metadata
        )

    @classmethod
    def from_parquet(cls, loader: ParquetLoader, **kwargs):
        """
        Initialize ChromaDB from a Parquet file
        """

        db = ChromaDB(**kwargs)

        # Insert data into the collection.

        ids = loader.get_ids()
        texts = loader.get_texts()
        embeddings = loader.get_embeddings()
        metadata = loader.get_metadata()

        # Chroma doesn't allow metadata to be None
        for i in range(len(metadata)):
            to_pop = []
            for k, v in metadata[i].items():  # type: ignore
                if v is None:
                    to_pop.append(k)
            for k in to_pop:
                metadata[i].pop(k)  # type: ignore

        # There's a limit of 5461 documents per batch
        for i in range(0, len(ids), ChromaDB.MAX_BATCH_SIZE):
            loguru.logger.info(
                "Upserting batch {start_idx} to {end_idx}",
                start_idx=i,
                end_idx=i + ChromaDB.MAX_BATCH_SIZE,
            )
            db.collection.upsert(
                ids=ids[i : i + ChromaDB.MAX_BATCH_SIZE],
                documents=texts[i : i + ChromaDB.MAX_BATCH_SIZE],
                embeddings=embeddings[i : i + ChromaDB.MAX_BATCH_SIZE],
                metadatas=metadata[i : i + ChromaDB.MAX_BATCH_SIZE],  # type: ignore
            )

        return db

    def similarity_search_by_vector(
        self,
        vector: np.ndarray,
        k: int,
        filters: Any | None = None,
        **kwargs,
    ):
        """
        Similarity search by vector
        """

        return self.collection.query(
            query_embeddings=vector,
            n_results=k,
            where=filters,
            **kwargs,
        )
