from polars_vector_store.base import VectorStore
from polars_vector_store.loader.parquet import ParquetLoader


class PolarsVectorStore(VectorStore):
    """
    Base class for PolarsVectorStore

    This class represents the interface of Polars
    based VectorStores.

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
            The loader to load the data from
            a Parquet file
        """

        self.loader = loader

    @classmethod
    def from_parquet(cls, loader: ParquetLoader, **kwargs):
        """
        Initialize the PolarsVectorStore from a Parquet file

        Parameters
        ----------
        loader: ParquetLoader
            The loader to load the data from
            a Parquet file
        """

        return cls(loader)
