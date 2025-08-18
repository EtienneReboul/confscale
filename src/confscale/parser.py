"""Parser module for ConfScale package.

This module provides functionality for reading and converting molecular conformation data
between different formats. It supports efficient handling of large-scale data using Dask
for distributed computing and PyArrow for optimized data storage.

Functions
---------
read_smi_to_dask : function
    Reads SMI files and converts them to a Dask DataFrame
write_dask_to_parquet : function
    Writes a Dask DataFrame to Parquet files with compression
smi2parquet : function
    Converts SMI files to a PyArrow Dataset
write_parquet_dataset : function
    Writes a PyArrow Dataset or Table to Parquet format with optimized settings
"""

import dask.dataframe as dd
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import csv


def read_smi_to_dask(input_path: str, blocksize: str | int = "64MB", sep: str = " ") -> dd.DataFrame:
    """
    Reads SMI files from a directory and converts them to a Dask DataFrame.

    Parameters
    ----------
    input_path : str
        Path to the directory containing the SMI files.
    blocksize : str or int, optional
        Block size for reading the SMI files. Default is "64MB".
    sep : str, optional
        Separator used in the SMI files. Default is " ".

    Returns
    -------
    dd.DataFrame
        Dask DataFrame containing the parsed SMI data.

    Notes
    -----
    dask read table uses in the backend pandas.read_table which does not support pyarrow schema

    Example
    -------
    >>> import tempfile
    >>> import os
    >>> # Create a temporary file with sample SMI data using space separator
    >>> sample_data = [
    ...     "CCC ethane",
    ...     "CCCC butane",
    ...     "CCCCC pentane"
    ... ]
    >>> with tempfile.NamedTemporaryFile(suffix='.smi', delete=False) as tmp:
    ...     for line in sample_data:
    ...        _= tmp.write((line + "\\n").encode()) # _ suppresses number of bytes written
    ...     tmp_path = tmp.name
    >>> # Parse the file with space separator
    >>> ddf = read_smi_to_dask(tmp_path, sep=' ')
    >>> assert ddf.npartitions == 1
    >>> os.unlink(tmp_path)  # Clean up the temporary file
    """
    return dd.read_table(input_path, blocksize=blocksize, sep=sep)


def write_dask_to_parquet(
    ddf: dd.DataFrame,
    output_path: str,
    compression: str = "zstd",
    level_compression: int = 5,
    schema: pa.Schema | None = None,
) -> dd.dask_expr._collection.Scalar:
    """
    Writes a Dask DataFrame to Parquet files.

    Parameters
    ----------
    ddf : dd.DataFrame
        Input Dask DataFrame.
    output_path : str
        Path to the output directory for Parquet files.
    compression : str, optional
        Compression algorithm for the Parquet files. Default is "zstd".
    level_compression : int, optional
        Compression level for the Parquet files. Default is 5.
    schema : pa.Schema or None, optional
        PyArrow schema to use for the Parquet files. Default is None.

    Returns
    -------
    dd.dask_expr._collection.Scalar
        Dask Scalar object containing parquet writer.

    Example
    -------
    >>> import tempfile
    >>> import os
    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> # Create a sample Dask DataFrame
    >>> df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'col2': [1, 2, 3]})
    >>> ddf = dd.from_pandas(df, npartitions=3)
    >>> # Write to a temporary directory
    >>> with tempfile.TemporaryDirectory() as tmp_dir:
    ...     # Write the DataFrame to Parquet
    ...     task = write_dask_to_parquet(ddf, tmp_dir)
    ...     _ = task.compute()  # Suppress output
    ...     # Verify parquet files were created
    ...     parquet_files = [f for f in os.listdir(tmp_dir) if f.endswith('.parquet')]
    ...     assert len(parquet_files) == 3
    """
    return ddf.to_parquet(
        output_path,
        engine="pyarrow",
        compression=compression,
        compression_level=level_compression,
        compute=False,
        schema=schema,
    )


def smi2parquet(dataset_path: str, schema: pa.Schema, sep: str = " ", partitioning: pa.Schema | None = None) -> ds.Dataset:
    """
    Convert SMI (Simplified Molecular Input Line Entry Specification) files to a PyArrow Dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the directory containing SMI files.
    schema : pa.Schema
        PyArrow schema defining the structure of the dataset.
    sep : str, optional
        Field delimiter for the SMI files. Default is " " (space).
    partitioning : pa.Schema or None, optional
        Schema for partitioning the dataset.
        If provided, the dataset will be read according to this partitioning schema.
        Default is None.

    Returns
    -------
    pyarrow.dataset.Dataset
        PyArrow Dataset created from the SMI files.

    Notes
    -----
    The function treats SMI files as CSV files with custom delimiter and schema.

    Example
    -------
    >>> import tempfile
    >>> import os
    >>> import pyarrow as pa
    >>> # Create a temporary file with sample SMI data
    >>> sample_data = [
    ...     "smiles name",
    ...     "CCC ethane",
    ...     "CCCC butane",
    ...     "CCCCC pentane"
    ... ]
    >>> with tempfile.NamedTemporaryFile(suffix='.smi', delete=False) as tmp:
    ...     for line in sample_data:
    ...        _= tmp.write((line + "\\n").encode())
    ...     tmp_path = tmp.name
    >>> # Define schema
    >>> schema = pa.schema([
    ...     ('smiles', pa.string()),
    ...     ('name', pa.string())
    ... ])
    >>> # Convert to PyArrow Dataset
    >>> dataset = smi2parquet(tmp_path, schema=schema, sep=' ')
    >>> dataset.count_rows()
    3
    >>> os.unlink(tmp_path)  # Clean up the temporary file
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

    Parameters
    ----------
    data : ds.Dataset or pa.Table
        Data to write to parquet format.
    final_output_path : str
        Directory to write the dataset to.
    min_rows_per_group : int, optional
        Minimum rows per row group. Default is 1e5.
    max_rows_per_group : int, optional
        Maximum rows per row group. Default is 1e6.
    max_rows_per_file : int, optional
        Maximum rows per file. Default is 1e7.
    compression : str, optional
        Compression algorithm to use. Default is "zstd".
    compression_level : int or None, optional
        Compression level. Default is None.

    Returns
    -------
    None
        This function doesn't return any value.

    Example
    -------
    >>> import tempfile
    >>> import os
    >>> import pyarrow as pa
    >>> import pyarrow.dataset as ds
    >>> # Create a sample PyArrow Table
    >>> data = pa.table({'col1': ['A', 'B', 'C'], 'col2': [1, 2, 3]})
    >>> # Write to a temporary directory
    >>> with tempfile.TemporaryDirectory() as tmp_dir:
    ...     write_parquet_dataset(data, tmp_dir, compression="zstd", compression_level=3)
    ...     # Verify the parquet dataset was created
    ...     dataset = ds.dataset(tmp_dir, format="parquet")
    ...     assert dataset.count_rows() == 3
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
