from typing import Any
import polars as pl
import numpy as np
from polars_vector_store.loader.parquet import ParquetLoader
from polars_vector_store.polars.base import PolarsVectorStore


class NumpyBasedPolarsVectorStore(PolarsVectorStore):
    """
    Numpy Based PolarsVectorStore.

    It materializes the data as a numpy array, computes
    the vector operations using numpy and then retrieves
    the result in the original dataframe.

    Since we need to materialize the data to extract the
    embeddings, the efficiency of this implementation
    depends on whether the data is already materialized.
    In general, that's not the case, since embeddings might
    not fit in memory.
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
        materialized_df = self.loader.materialized_df

        embedding_col = self.loader.embedding_column_name

        # If we have filters
        if filters is not None:
            # We also filter the lazy df since later we want to
            # retrieve the other columns using row index. Maybe
            # it's not optimal to do it even if we already have
            # the materialized df, but it should not impact too much.
            lazy_df = lazy_df.filter(filters)

            # If we have already the materialized df, we filter it
            if materialized_df is not None:
                materialized_df = materialized_df.filter(filters)
            else:
                materialized_df = lazy_df.select([embedding_col]).collect()

        # If we don't have filters and the data is not materialized,
        # we materialize it
        elif materialized_df is None:
            materialized_df = lazy_df.select([embedding_col]).collect()

        vector_store_embeds = materialized_df[embedding_col].to_numpy()

        # Compute cosine similarity.
        # Since the embeddings are normalized, this is equivalent to the dot product.
        cosine_similarities = np.einsum("ij,ij->i", vector_store_embeds, vector)

        # Get the indices of the k smallest cosine similarities
        # Notice that argpartition gives no guarantee on the order
        # of the k smallest elements, which is why we need
        # an extra sorting step after the partitioning.
        closest_indices = np.argpartition(cosine_similarities, -k)[-k:]

        # Sort the k closest indices by cosine similarity
        idx = closest_indices[np.argsort(cosine_similarities[closest_indices])[::-1]]

        return (
            lazy_df.with_row_index()
            .filter(pl.col("index").is_in(idx))
            .select(self.loader.get_info_columns())
            .collect()
        )
