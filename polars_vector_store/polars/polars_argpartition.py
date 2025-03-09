from typing import Any
import polars as pl
import numpy as np
from polars_vector_store.loader.parquet import ParquetLoader
from polars_vector_store.polars.base import PolarsVectorStore
from polars_argpartition import argpartition


class PolarsArgPartitionVectorStore(PolarsVectorStore):
    """
    Polars based Vector Store that uses
    argpartition to sort similarities
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
            # I couldn't find a way to make this better, there
            # must be one.
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
            .with_columns(
                idxs=argpartition(
                    -pl.col(  # Use negative so that we don't need to reverse the array
                        "sim"
                    ),
                    k=k,
                )
            )
            .with_row_index(name="row_index")
            .filter(pl.col("row_index").is_in(pl.col("idxs").slice(0, k)))
            .select(self.loader.get_info_columns())
            .collect()
        )

    @classmethod
    def from_parquet(cls, loader: ParquetLoader, **kwargs):
        """
        Create a PolarsVectorStore from a parquet file
        """
        return cls(loader, **kwargs)
