from typing import Any
import polars as pl
import numpy as np
from polars_vector_store.loader.parquet import ParquetLoader
from polars_vector_store.polars.base import PolarsVectorStore


class PolarsTopKVectorStore(PolarsVectorStore):
    """
    Polars based Vector Store that uses
    pl.Lazyframe.top_k to sort similarities

    All operations are computed on the Lazyframe
    """

    def similarity_search_by_vector(
        self,
        vector: np.ndarray,
        k: int,
        filters: Any | None = None,
        **kwargs,
    ):
        """
        similarity search by vector
        """

        lazy_df = self.loader.get_lazy_df()

        # If we have filters
        if filters is not None:
            lazy_df = lazy_df.filter(filters)

        return (
            # Add column with the query vector
            # as repeated element, so that we can compute
            # numpy vectorized operations
            # I couldn't find a way to make this better
            lazy_df.with_columns(
                query=pl.lit(vector.reshape(-1).tolist()).cast(
                    pl.Array(pl.Float64, shape=vector.shape[1])
                ),
            )
            .with_columns(
                sim=np.dot(  # type: ignore
                    pl.col("embedding"),
                    pl.col("query"),
                ).arr.sum()
            )
            .top_k(k, by="sim")
            .select(self.loader.get_info_columns())
            .collect()
        )
