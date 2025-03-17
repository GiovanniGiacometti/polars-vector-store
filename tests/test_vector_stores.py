import numpy as np
import pytest
import polars as pl
from polars.testing import assert_frame_equal
from polars_vector_store.chroma import ChromaDB
from polars_vector_store.polars.numpy_based import NumpyBasedPolarsVectorStore
from polars_vector_store.polars.polars_argpartition import PolarsArgPartitionVectorStore
from polars_vector_store.polars.polars_top_k import PolarsTopKVectorStore


def get_closest_indices(
    embeddings: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    """
    Get the indices of the k closest embeddings to the query
    """
    similarities = embeddings @ query.reshape(-1)

    return similarities.argsort()[-k:]


class TestVectorStore:
    """
    Test vector stores
    """

    def test_query_chroma_vector_store(self, test_data, tmp_path):
        """
        Test querying a ChromaDB vector store without filters
        """
        parquet_loader, query = test_data

        vector_store = ChromaDB.from_parquet(parquet_loader, db_path=str(tmp_path))

        materialized_df = parquet_loader.get_lazy_df().collect()

        embeddings = materialized_df[parquet_loader.embedding_column_name].to_numpy()

        # compute 3 closest embeddings to the query
        closest_indices = get_closest_indices(embeddings, query, k=3)

        # query the vector store

        result = vector_store.similarity_search_by_vector(query, k=3)

        # We leverage that ids are just the indexes
        # Chroma might not return the exact same order of the closest indices
        # so we sort the ids before comparing
        assert sorted(result["ids"][0]) == sorted(map(str, closest_indices.tolist()))

    @pytest.mark.parametrize(
        "vector_store",
        [
            NumpyBasedPolarsVectorStore,
            PolarsArgPartitionVectorStore,
            PolarsTopKVectorStore,
        ],
    )
    def test_query_polars_based_vector_store(self, vector_store, test_data):
        """
        Test querying a Polars based vector store without filters
        """
        parquet_loader, query = test_data

        vector_store = vector_store(parquet_loader)

        materialized_df = parquet_loader.get_lazy_df().collect()

        embeddings = materialized_df[parquet_loader.embedding_column_name].to_numpy()

        # compute 3 closest embeddings to the query
        closest_indices = get_closest_indices(embeddings, query, k=3)

        # query the vector store

        result = vector_store.similarity_search_by_vector(query, k=3)

        # We leverage that ids are just the indexes

        # Each vector store has a slightly different behaviour
        # with respect to how top elements are returned. Let's make no
        # distinction here for simplicity, just check ids are the same
        assert_frame_equal(
            result.select(parquet_loader.id_column_name).sort(
                parquet_loader.id_column_name
            ),
            pl.DataFrame(
                {
                    parquet_loader.id_column_name: list(
                        map(str, np.sort(closest_indices).tolist())
                    )
                }
            ),
        )
