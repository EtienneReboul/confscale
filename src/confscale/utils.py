"""
Utility functions for handling PyArrow datasets and file size formatting.

This module provides helper functions for working with PyArrow datasets,
including calculating total dataset size and formatting byte sizes into
human-readable strings.
"""

import pyarrow.dataset as ds


def compute_dataset_size(dataset: ds.Dataset) -> int:
    """
    Computes the total size of a PyArrow dataset by summing up the sizes of all file fragments.

    Args:
        dataset: A PyArrow dataset object

    Returns:
        int: Total size of the dataset in bytes
    """
    file_system = dataset.filesystem
    total_size = 0

    for fragment in dataset.get_fragments():
        file_size = file_system.get_file_info(fragment.path).size
        total_size += file_size

    return total_size


def format_size(size_bytes: int) -> str:
    """
    Convert a size in bytes to a human-readable string.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    str
        Human-readable size (e.g., '2.5MB')

    Raises
    ------
    ValueError
        If size_bytes is negative

    Examples
    --------
    >>> format_size(1024)
    '1.00KB'
    >>> format_size(1048576)
    '1.00MB'
    >>> format_size(1500)
    '1.46KB'
    >>> format_size(0)
    '0B'
    >>> format_size(2500000)
    '2.38MB'
    """
    if size_bytes < 0:
        raise ValueError("Size must be non-negative")

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    # Format with one decimal place if not bytes
    if unit_index == 0:
        return f"{int(size)}{units[unit_index]}"
    else:
        return f"{size:.2f}{units[unit_index]}"
