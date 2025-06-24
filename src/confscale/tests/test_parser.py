"""
Test module for the parser functionality.
Verifies the correct behavior and edge cases of the parser module implementation.
"""

import tempfile
from pathlib import Path

import dask.dataframe as dd
import pyarrow as pa
import pytest

from confscale.parser import read_smi_to_dask
from confscale.parser import smi2parquet
from confscale.parser import write_dask_to_parquet
from confscale.parser import write_parquet_dataset


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def small_smi_files():
    """Get list of test URLs from data file"""
    data_file = Path("data/raw/smi_files_below_80MB.txt")
    with data_file.open(encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def test_read_smi_to_dask(small_smi_files):
    """Test reading SMI files into Dask DataFrame"""
    for url in small_smi_files:
        ddf = read_smi_to_dask(url)
        assert isinstance(ddf, dd.DataFrame)
        # Check DataFrame has content
        assert ddf.npartitions > 0
        assert len(ddf.columns) > 0


def test_write_dask_to_parquet(temp_dir, small_smi_files):
    """Test writing Dask DataFrame to parquet"""
    url = small_smi_files[0]  # Test with first file
    ddf = read_smi_to_dask(url)

    output_path = Path(temp_dir) / "test.parquet"
    result = write_dask_to_parquet(ddf, str(output_path))
    result.compute()  # Execute the write operation

    # Verify parquet file was created
    assert output_path.exists()

    # Try reading back the parquet file
    ddf_read = dd.read_parquet(str(output_path))
    assert isinstance(ddf_read, dd.DataFrame)
    assert ddf_read.npartitions > 0


def test_smi2parquet(temp_dir, small_smi_files):
    """Test converting SMI files to PyArrow Dataset"""
    # Use the first SMI file from the fixture
    test_smi_path = small_smi_files[0]

    # Define schema for the test data
    schema = pa.schema([pa.field("smiles", pa.string()), pa.field("zinc_id", pa.uint32())])

    # Convert to parquet
    dataset = smi2parquet(test_smi_path, schema=schema)

    # Verify dataset was created correctly
    assert dataset is not None
    # Check schema matches
    assert dataset.schema == schema
    # Check data content
    table = dataset.to_table()
    assert table.num_rows > 0  # Real files will have varying number of rows
    assert table.column_names == ["smiles", "zinc_id"]


def test_write_parquet_dataset(temp_dir):
    """Test writing PyArrow Dataset to Parquet format"""
    import pyarrow.dataset as ds

    # Create a simple table
    data = pa.table({"smiles": pa.array(["CCCC", "CCC", "CCCCC"]), "id": pa.array(["mol1", "mol2", "mol3"])})

    output_path = Path(temp_dir) / "output_dataset"

    # Test with default parameters
    write_parquet_dataset(data, str(output_path))

    # Verify dataset was written correctly
    result_dataset = ds.dataset(str(output_path), format="parquet")
    result_table = result_dataset.to_table()

    assert result_table.num_rows == 3
    assert result_table.column_names == ["smiles", "id"]

    # Verify custom compression settings
    output_path2 = Path(temp_dir) / "output_dataset_custom"
    write_parquet_dataset(data, str(output_path2), min_rows_per_group=10, max_rows_per_group=100, compression="snappy")

    # Verify the dataset exists
    assert output_path2.exists()
