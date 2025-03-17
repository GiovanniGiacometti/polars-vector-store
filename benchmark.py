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
from polars_vector_store.polars.base import PolarsVectorStore
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
EMBEDDING_DIMS = [128, 384, 768, 1024]

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
    # (np.float64, "float64"),
]

# -----------------


# --- Helper functions ---


def generate_embeddings(
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
    embeddings = generate_embeddings(
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
    filters: dict | pl.Expr | None = None,
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
        "filters": filters,
    }

    if filters is not None:
        stmt_code = "vector_store.similarity_search_by_vector(query_vector, k=top_k, filters=filters)"
    else:
        stmt_code = "vector_store.similarity_search_by_vector(query_vector, k=top_k)"

    # Run the benchmark
    query_time = timeit.repeat(
        stmt=stmt_code,
        globals=local_namespace,
        repeat=3,  # number of repetitions of the timer
        number=num_runs,  # number of executions per run
    )

    # The docstring of timeit.repeat says that the best time 7
    # to consider is the minimum
    # of the list of times

    return min(query_time)


def make_metadata_filters(
    vector_store: VectorStore,
    loader: ParquetLoader,
    metadata_possible_values: list[list[str | int]],
    n_filters: int,
) -> dict | pl.Expr:
    """
    Make metadata filters according to the VectorStore class.
    One filter per metadata column
    """

    if n_filters > len(loader.metadata_columns_names):
        raise ValueError("Number of filters exceeds number of metadata columns")

    if n_filters == 0:
        raise ValueError("Number of filters must be greater than 0")

    filter_cnt = 0

    if isinstance(vector_store, ChromaDB):
        filters_cond = []

        for i, col in enumerate(loader.metadata_columns_names):
            if isinstance(metadata_possible_values[i][0], str):
                filters_cond.append({col: {"$in": metadata_possible_values[i][:3]}})

            elif isinstance(metadata_possible_values[i][0], int):
                filters_cond.append(
                    {
                        col: {
                            "$gt": metadata_possible_values[i][1],
                        }
                    }
                )

            else:
                raise ValueError("Metadata type not recognized")

            filter_cnt += 1

            if filter_cnt == n_filters:
                break

        if len(filters_cond) == 1:
            return filters_cond[0]

        return {
            "$and": filters_cond,
        }

    elif isinstance(vector_store, PolarsVectorStore):
        filters = pl.lit(True)

        for i, col in enumerate(loader.metadata_columns_names):
            if isinstance(metadata_possible_values[i][0], str):
                filters &= pl.col(col).is_in(metadata_possible_values[i][:3])

            elif isinstance(metadata_possible_values[i][0], int):
                filters &= pl.col(col).gt(metadata_possible_values[i][1])

            else:
                raise ValueError("Metadata type not recognized")

            filter_cnt += 1

            if filter_cnt == n_filters:
                break

        return filters

    else:
        raise ValueError("VectorStore class not recognized")


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
    vector_store_cardinality = [500, 1000, 10_000, 50_000]

    for (
        dtype,
        dtype_name,
    ) in VECTOR_DTYPES:
        for n_dimensions in EMBEDDING_DIMS:
            results = defaultdict(list)

            for n_rows in vector_store_cardinality:
                results["n_rows"].append(n_rows)

                logger.info("=====================================")
                logger.info(
                    "Starting benchmark with number of vectors = {n_rows}, vectors dtype = {type}, vectors dimensions = {n_dimensions}",
                    n_dimensions=n_dimensions,
                    n_rows=n_rows,
                    type=dtype_name,
                )

                # Generate random data
                path_to_file = os.path.join(
                    DATA_FOLDER,
                    f"benchmark_{n_rows}_{dtype_name}_{n_dimensions}.parquet",
                )

                loader = generate_loader(
                    path_to_file=path_to_file,
                    n_rows=n_rows,
                    n_dimensions=n_dimensions,
                    metadata_possible_values=METADATA_POSSIBLE_VALUES,
                    n_metadata=N_METADATA,
                    dtype=dtype,
                )

                # generate query vector
                query_vector = generate_embeddings(
                    n_embeddings=1, n_dimensions=n_dimensions, seed=42, dtype=dtype
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
                            DATA_FOLDER,
                            f"chromadb-{n_rows}-{dtype_name}-{n_dimensions}",
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

                    # Query with metadata

                    if N_METADATA > 0:
                        logger.info("Querying with filters on 1 metadata")

                        # we need to make specific filters according to the vector store.
                        # Let's query with 1 metadata first

                        filters = make_metadata_filters(
                            vector_store=vector_store,
                            metadata_possible_values=METADATA_POSSIBLE_VALUES,
                            loader=loader,
                            n_filters=1,
                        )

                        query_time = benchmark_query(
                            vector_store=vector_store,
                            query_vector=query_vector,
                            top_k=TOP_K,
                            filters=filters,
                            num_runs=NUM_RUNS,
                        )

                        results[
                            f"{vector_store_class.__name__} - query 1 metadata"
                        ].append(query_time)

                        # Query with all metadata

                        logger.info(
                            "Time to query 1 metadata: {time} seconds",
                            time=round(query_time, 5),
                        )

                        logger.info(
                            f"Querying with filters on all ({N_METADATA}) metadata"
                        )

                        filters = make_metadata_filters(
                            vector_store=vector_store,
                            metadata_possible_values=METADATA_POSSIBLE_VALUES,
                            loader=loader,
                            n_filters=N_METADATA,
                        )

                        query_time = benchmark_query(
                            vector_store=vector_store,
                            query_vector=query_vector,
                            top_k=TOP_K,
                            filters=filters,
                            num_runs=NUM_RUNS,
                        )

                        results[
                            f"{vector_store_class.__name__} - query all ({N_METADATA}) metadata"
                        ].append(query_time)

                        logger.info(
                            "Time to query all ({n}) metadata: {time} seconds",
                            time=round(query_time, 5),
                            n=N_METADATA,
                        )

            # Save results to a CSV file
            results_df = pl.DataFrame(results)
            results_df.write_csv(
                os.path.join(
                    RESULTS_FOLDER, f"benchmark_results_{dtype_name}_{n_dimensions}.csv"
                )
            )
            logger.info("Results saved")
            logger.info(
                results_df.select(
                    [c for c in results_df.columns if "query" in c or "n_rows" in c]
                )
            )
