import os
from typing import Any
import numpy as np
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

    def __init__(self, path_to_file: str, **kwargs) -> None:
        """
        Initialize the PolarsVectorStore

        Parameters
        ----------
        path_to_file : str
            Path to the file that contains the data. Allowed types are CSV and Parquet.
        kwargs : dict
            Additional keyword arguments that are passed to the Polars read_csv or read_parquet function.
        """
        self.path_to_file = path_to_file

        # Read file as a LazyFrame according to the file extension
        match os.path.splitext(path_to_file)[1]:  # noqa
            case ".csv":
                self.df = pl.scan_csv(path_to_file, **kwargs)
            case ".parquet":
                self.df = pl.scan_parquet(path_to_file, **kwargs)
            case _:
                raise ValueError("Only CSV and Parquet files are supported.")

    @staticmethod
    def from_parquet_file(loader: ParquetLoader, **kwargs):
        return PolarsVectorStore(loader.path_to_file, **kwargs)

    def similarity_search_by_vector(
        self,
        vector: np.ndarray,
        k: int,
        filter: Any | None = None,
        **kwargs,
    ):
        """
        Similarity search by vector
        """
