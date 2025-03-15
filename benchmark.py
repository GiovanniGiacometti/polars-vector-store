from collections import defaultdict
import os
import sys
import timeit
import loguru
import numpy as np
import polars as pl
from polars_vector_store.base import VectorStore
from polars_vector_store.chroma import ChromaDB
from polars_vector_store.loader.parquet import ParquetLoader
from polars_vector_store.polars.numpy_based import NumpyBasedPolarsVectorStore
from polars_vector_store.polars.polars_argpartition import PolarsArgPartitionVectorStore
from polars_vector_store.polars.polars_top_k import PolarsTopKVectorStore

logger = loguru.logger
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {message}",
)


# --- Constants ---

FOLDER = "benchmark_results"
DATA_FOLDER = os.path.join(FOLDER, "data")
RESULTS_FOLDER = os.path.join(FOLDER, "results")

# Embedding dimensions
# Just fixed to 700 for now, it should not matter too much
EMBEDDING_DIM = 700
N_METADATA = 3

# Possible values for metadata

METADATA_POSSIBLE_VALUES = [
    ["A", "B", "C", "D", "E"],
    ["X", "Y", "Z"],
    [1, 2, 3, 4, 5],
]

# Number of times to run each benchmark for more reliable results
NUM_RUNS = 5

# Number of results to retrieve in queries
TOP_K = 5

# Vector type

VECTOR_DTYPES: list[tuple[np.dtype, str]] = [
    (np.float16, "float16"),
    (np.float32, "float32"),
    (np.float64, "float64"),
]

# -----------------


# --- Helper functions ---


def generate_embedding(
    n_embeddings: int,
    n_dimensions: int,
    dtype: np.dtype,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random embeddings
    """
    np.random.seed(seed)
    return np.random.rand(n_embeddings, n_dimensions).astype(dtype)


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


def instantiate_vector_store(
    vector_store_class: type[VectorStore],
    loader: ParquetLoader,
    **kwargs,
) -> VectorStore:
    """
    Instantiate a VectorStore class
    """
    # Actually create the instance for returning
    vector_store = vector_store_class.from_parquet(loader=loader, **kwargs)

    return vector_store


def benchmark_query(
    vector_store: VectorStore,
    query_vector: np.ndarray,
    top_k: int = TOP_K,
    filter_conditions: dict = None,
    num_runs: int = NUM_RUNS,
) -> float:
    """
    Benchmark vector store query operation using timeit
    """
    # Create a local namespace for timeit to access the objects
    local_namespace = {
        "vector_store": vector_store,
        "query_vector": query_vector,
        "top_k": top_k,
        "filters": filter_conditions,
    }

    if filter_conditions:
        stmt_code = "vector_store.similarity_search_by_vector(query_vector, k=top_k, filters=filter_conditions)"
    else:
        stmt_code = "vector_store.similarity_search_by_vector(query_vector, k=top_k)"

    # Run the benchmark
    query_time = timeit.repeat(
        stmt=stmt_code,
        globals=local_namespace,
        repeat=3,  # number of repetitions of the timer
        number=num_runs,  # number of executions per run
    )

    # The docstring of timeit.repeat says that the best time is the minimum
    # of the list of times

    return min(query_time)


# -----------------


if __name__ == "__main__":
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Define the VectorStore classes to benchmark
    vectorstores_classes = [
        ChromaDB,
        NumpyBasedPolarsVectorStore,
        PolarsArgPartitionVectorStore,
        PolarsTopKVectorStore,
    ]

    # Define the dimensions to benchmark with.
    # I'm aware these are not enough, but it's a start.
    # Also, ChromaDB takes some time to insert data,
    # so I'm keeping the number of vectors low for now.
    vector_store_cardinality = [500, 1000, 10_000, 50_000, 100_000]

    for (
        dtype,
        dtype_name,
    ) in VECTOR_DTYPES:
        results = defaultdict(list)

        for n_rows in vector_store_cardinality:
            results["n_rows"].append(n_rows)

            logger.info("=====================================")
            logger.info(
                "Starting benchmark with number of vectors = {n_rows} and vectors dtype = {type}",
                n_rows=n_rows,
                type=dtype_name,
            )

            # Generate random data
            path_to_file = os.path.join(
                DATA_FOLDER, f"benchmark_{n_rows}_{dtype_name}.parquet"
            )

            loader = generate_loader(
                path_to_file=path_to_file,
                n_rows=n_rows,
                n_dimensions=EMBEDDING_DIM,
                metadata_possible_values=METADATA_POSSIBLE_VALUES,
                n_metadata=N_METADATA,
                dtype=dtype,
            )

            # generate query vector
            query_vector = generate_embedding(
                n_embeddings=1, n_dimensions=EMBEDDING_DIM, seed=42, dtype=dtype
            )

            # We query once without metadata and once with all metadata

            for vector_store_class in vectorstores_classes:
                logger.info("-------------------------------------")
                logger.info(
                    "Benchmarking {vector_store_class}",
                    vector_store_class=vector_store_class.__name__,
                )

                # Instantiate the VectorStore

                start_time = timeit.default_timer()
                vector_store = instantiate_vector_store(
                    vector_store_class=vector_store_class,
                    loader=loader,
                    db_path=os.path.join(  # only for ChromaDB
                        DATA_FOLDER, f"chromadb-{n_rows}-{dtype_name}"
                    ),
                )

                result_time = round(timeit.default_timer() - start_time, 5)

                results[f"{vector_store_class.__name__} - instantiation"].append(
                    result_time
                )

                logger.info(
                    "Time to instantiate VectorStore: {time} seconds",
                    time=result_time,
                )

                # Query without metadata

                logger.info("Querying without metadata")

                query_time = benchmark_query(
                    vector_store=vector_store,
                    query_vector=query_vector,
                    top_k=TOP_K,
                    num_runs=NUM_RUNS,
                )

                results[f"{vector_store_class.__name__} - query"].append(query_time)

                logger.info(
                    "Time to query without metadata: {time} seconds",
                    time=round(query_time, 5),
                )

        # Save results to a CSV file
        results_df = pl.DataFrame(results)
        results_df.write_csv(
            os.path.join(RESULTS_FOLDER, f"benchmark_results_{dtype_name}.csv")
        )
        logger.info("Results saved")
        logger.info(
            results_df.select(
                [c for c in results_df.columns if "query" in c or "n_rows" in c]
            )
        )
