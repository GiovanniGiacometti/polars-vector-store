from __future__ import annotations
from typing import Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
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

    lazy_df: pl.LazyFrame | None = None
    materialized_df: pl.DataFrame | None = None

    streaming_mode: bool = False

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

    def get_metadata(self) -> dict[str, list]:
        """
        Get the metadata columns as a list of dictionaries,
        one per row, with the column name as the key
        and the column value as the value
        """
        return (
            self._get_materialized_df().select(self.metadata_columns_names).to_dicts()
        )

    def get_lazy_df(self) -> pl.LazyFrame:
        """
        Get the lazy DataFrame
        """

        if self.lazy_df is None:
            self.lazy_df = pl.scan_parquet(self.path_to_file)

        return self.lazy_df

    def _get_materialized_df(self) -> pl.DataFrame:
        """
        Get the materialized DataFrame
        """

        if self.materialized_df is None:
            if self.lazy_df is None:
                self.lazy_df = self.get_lazy_df()
            self.materialized_df = self.lazy_df.collect(streaming=self.streaming_mode)

        return self.materialized_df

    def _get_column_as_list(self, col_name: str) -> list:
        """
        Get a column as a list
        """

        return self._get_materialized_df()[col_name].to_list()
