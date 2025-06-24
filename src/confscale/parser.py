"""Parser module for ConfScale package.

This module provides functionality for reading and converting molecular conformation data
between different formats. It supports efficient handling of large-scale data using Dask
for distributed computing and PyArrow for optimized data storage.

Functions:
    read_smi_to_dask: Reads SMI files and converts them to a Dask DataFrame
    write_dask_to_parquet: Writes a Dask DataFrame to Parquet files with compression
    smi2parquet: Converts SMI files to a PyArrow Dataset
    write_parquet_dataset: Writes a PyArrow Dataset or Table to Parquet format with optimized settings

"""

import dask.dataframe as dd
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import csv


def read_smi_to_dask(input_path: str, blocksize: str | int = "64MB", sep: str = " ") -> dd.DataFrame:
    """
    Reads SMI files from a directory and converts them to a Dask DataFrame.

    Args:
        input_path (str): Path to the directory containing the SMI files.
        blocksize (str): Block size for reading the SMI files.
        sep (str): Separator used in the SMI files.

    Returns:
        dd.DataFrame: Dask DataFrame containing the parsed SMI data.
    """
    return dd.read_table(input_path, blocksize=blocksize, sep=sep)


def write_dask_to_parquet(
    ddf: dd.DataFrame,
    output_path: str,
    compression: str = "zstd",
    level_compression: int = 5,
) -> dd.dask_expr._collection.Scalar:
    """
    Writes a Dask DataFrame to Parquet files.

    Args:
        ddf (dd.DataFrame): Input Dask DataFrame.
        output_path (str): Path to the output directory for Parquet files.
        compression (str): Compression algorithm for the Parquet files.
        level_compression (int): Compression level for the Parquet files.

    Returns:
        dd.dask_expr._collection.Scalar: Dask Scalar object containing parquet writer.
    """
    return ddf.to_parquet(
        output_path,
        engine="pyarrow",
        compression=compression,
        compression_level=level_compression,
        compute=False,
    )


def smi2parquet(dataset_path: str, schema: pa.Schema, sep: str = " ", partitioning: pa.Schema | None = None) -> None:
    """
    Convert SMI (Simplified Molecular Input Line Entry Specification) files to a PyArrow Dataset.

    This function reads SMI files from the given path and converts them to a PyArrow Dataset
    using the specified schema. The SMI files are treated as CSV files with a custom delimiter.

    Args:
        dataset_path (str): Path to the directory containing SMI files.
        schema (pa.Schema): PyArrow schema defining the structure of the dataset.
        sep (str, optional): Field delimiter for the SMI files. Defaults to " " (space).
        partitioning (pa.Schema | None, optional): Schema for partitioning the dataset.
            If provided, the dataset will be read according to this partitioning schema.
            Defaults to None.

    Returns:
        pyarrow.dataset.Dataset: PyArrow Dataset created from the SMI files.

    Note:
        The function treats SMI files as CSV files with custom delimiter and schema.
    """

    if partitioning is None:
        dataset = ds.dataset(dataset_path, format=ds.CsvFileFormat(parse_options=csv.ParseOptions(delimiter=sep)), schema=schema)
    else:
        dataset = ds.dataset(
            dataset_path, format=ds.CsvFileFormat(parse_options=csv.ParseOptions(delimiter=sep)), schema=schema, partitioning=partitioning
        )
    return dataset


def write_parquet_dataset(
    data: ds.Dataset | pa.Table,
    final_output_path: str,
    min_rows_per_group: int = 1e5,
    max_rows_per_group: int = 1e6,
    max_rows_per_file: int = 1e7,
    compression: str = "zstd",
    compression_level: int | None = None,
) -> None:
    """
    Writes a PyArrow Dataset or Table to Parquet format with optimized settings.

    Args:
        data (ds.Dataset | pa.Table): Data to write to parquet format.
        final_output_path (str): Directory to write the dataset to.
        min_rows_per_group (int, optional): Minimum rows per row group. Defaults to 10000.
        max_rows_per_group (int, optional): Maximum rows per row group. Defaults to 100000.
        max_rows_per_file (int, optional): Maximum rows per file. Defaults to 1000000.
        compression (str, optional): Compression algorithm to use. Defaults to "zstd".
        compression_level (int, optional): Compression level. Defaults to None.

    Returns:
        None
    """
    ds.write_dataset(
        data,
        final_output_path,
        format="parquet",
        min_rows_per_group=min_rows_per_group,
        max_rows_per_group=max_rows_per_group,
        max_rows_per_file=max_rows_per_file,
        file_options=ds.ParquetFileFormat().make_write_options(compression=compression, compression_level=compression_level),
    )
