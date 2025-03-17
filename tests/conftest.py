import os
import numpy as np
import polars as pl
import pytest
from polars_vector_store.loader.parquet import ParquetLoader


def generate_embedding(
    n_embeddings: int,
    n_dimensions: int,
    dtype: np.dtype,
    seed: int = 42,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random embeddings
    """
    np.random.seed(seed)
    embeddings = np.random.rand(n_embeddings, n_dimensions).astype(dtype)

    if normalize:
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings


def generate_loader(
    path_to_file: str,
    n_rows: int,
    n_dimensions: int,
    dtype: np.dtype,
    seed: int = 42,
    metadata_possible_values: list[list[str | int]] | None = None,
    n_metadata: int = 2,
) -> ParquetLoader:
    """
    Generate a ParquetLoader with random data
    """

    # Generate random embeddings
    embeddings = generate_embedding(
        n_embeddings=n_rows, n_dimensions=n_dimensions, seed=seed, dtype=dtype
    )

    if n_metadata > 0 and metadata_possible_values is None:
        raise ValueError("No possible values for metadata provided")

    if len(metadata_possible_values) != n_metadata:
        raise ValueError("Not enough possible values for metadata provided")

    pl.DataFrame(
        {
            "id": np.arange(n_rows).astype(np.str_).tolist(),
            "text": [f"random_text_{i}" for i in range(n_rows)],
            "embedding": embeddings,
            **{
                f"metadata_{i}": np.random.choice(metadata_possible_values[i], n_rows)
                for i in range(n_metadata)
            },
        }
    ).write_parquet(path_to_file)

    return ParquetLoader(
        path_to_file=path_to_file,
        id_column_name="id",
        text_column_name="text",
        embedding_column_name="embedding",
        metadata_columns_names=[f"metadata_{i}" for i in range(n_metadata)],
    )


@pytest.fixture
def test_data(tmp_path):
    """
    Generate a ParquetLoader with random data that gets cleaned up after tests.

    Returns:
        ParquetLoader: A loader for test data
    """
    n_rows = 100
    n_dimensions = 128
    dtype = np.float32
    seed = 42
    n_metadata = 2
    metadata_possible_values = [
        ["category_A", "category_B", "category_C"],
        [1, 2, 3, 4, 5],
    ]

    # Generate random embeddings
    embeddings = generate_embedding(
        n_embeddings=n_rows, n_dimensions=n_dimensions, seed=seed, dtype=dtype
    )

    # Create path for the parquet file
    parquet_path = os.path.join(tmp_path, "test_embeddings.parquet")

    # Create the dataframe
    pl.DataFrame(
        {
            "id": np.arange(n_rows).astype(np.str_).tolist(),
            "text": [f"random_text_{i}" for i in range(n_rows)],
            "embedding": embeddings,
            **{
                f"metadata_{i}": np.random.choice(metadata_possible_values[i], n_rows)
                for i in range(n_metadata)
            },
        }
    ).write_parquet(parquet_path)

    # Create and return the loader
    loader = ParquetLoader(
        path_to_file=parquet_path,
        id_column_name="id",
        text_column_name="text",
        embedding_column_name="embedding",
        metadata_columns_names=[f"metadata_{i}" for i in range(n_metadata)],
    )

    # Return loader and the query vector
    return loader, generate_embedding(
        n_embeddings=1, n_dimensions=n_dimensions, seed=seed + 1, dtype=dtype
    )
