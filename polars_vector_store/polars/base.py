import os
import polars as pl
from polars_vector_store.base import VectorStore
from polars_vector_store.loader.parquet import ParquetLoader


class PolarsVectorStore(VectorStore):
    """
    Base class for PolarsVectorStore

    We want to provide different implementations
    and this base class provides the interface.

    The reason to have multiple implementations is two-fold:
    - Benchmarking
    - Different use cases might require different implementations
    """

    def __init__(self, loader: ParquetLoader) -> None:
        """
        Initialize the PolarsVectorStore

        Parameters
        ----------
        loader: ParquetLoader
            The loader to load the data
        """

        self.loader = loader
