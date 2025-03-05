from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from polars_vector_store.loader.parquet import ParquetLoader


class VectorStore(ABC):
    """
    Base class for a VectorStore
    """

    @abstractmethod
    def similarity_search_by_vector(
        self,
        vector: np.ndarray,
        k: int,
        filters: Any | None = None,
        **kwargs,
    ):
        """
        Abstract method for similarity search by vector
        """

    @classmethod
    @abstractmethod
    def from_parquet(cls, loader: ParquetLoader, **kwargs):
        """
        Abstract method for initializing the vector store from a Parquet file
        """
