from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
import polars as pl


class ParquetLoader(BaseModel):
    """
    Class wrapping a parquet file
    and providing access to codified columns
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path_to_file: str

    id_column_name: str
    text_column_name: str
    embedding_column_name: str

    metadata_columns_names: list[str] = Field(default_factory=list)

    streaming_mode: bool = False

    _lazy_df: pl.LazyFrame | None = None
    _materialized_df: pl.DataFrame | None = None

    def get_info_columns(
        self,
    ) -> list[str]:
        """
        Get all the columns in the parquet file
        """

        return [
            self.id_column_name,
            self.text_column_name,
        ] + self.metadata_columns_names

    def get_ids(self) -> list:
        """
        Get the id column as a list
        """
        return self._get_column_as_list(self.id_column_name)

    def get_texts(self) -> list:
        """
        Get the text column as a list
        """

        return self._get_column_as_list(self.text_column_name)

    def get_embeddings(self) -> list:
        """
        Get the embedding column as a list
        """
        return self._get_column_as_list(self.embedding_column_name)

    def get_metadata(self) -> list[dict]:
        """
        Get the metadata columns as a list of dictionaries,
        one per row, with the column name as the key
        and the column value as the value
        """
        return self.materialized_df.select(self.metadata_columns_names).to_dicts()

    @property
    def lazy_df(self) -> pl.LazyFrame:
        """
        Get the lazy DataFrame
        """

        if self._lazy_df is None:
            self._lazy_df = pl.scan_parquet(self.path_to_file)

        return self._lazy_df

    @property
    def materialized_df(self) -> pl.DataFrame:
        """
        Get the materialized DataFrame.

        WARNING: This will materialize the DataFrame
        and it might be expensive
        """

        if self._materialized_df is None:
            self._materialized_df = self.lazy_df.collect(streaming=self.streaming_mode)

        return self._materialized_df

    @property
    def has_materialized_df(self) -> bool:
        """
        Check if the materialized DataFrame is available
        """

        return self._materialized_df is not None

    def _get_column_as_list(self, col_name: str) -> list:
        """
        Get a column as a list
        """

        return self.materialized_df[col_name].to_list()
