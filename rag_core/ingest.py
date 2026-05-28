"""
rag_core.ingest
~~~~~~~~~~~~~~~
Helper utilities to facilitate document ingestion from local file systems,
directories, and other sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

from .pipeline import RAGPipeline

logger = logging.getLogger("rag_core.ingest")


def ingest_directory(
    pipeline: RAGPipeline,
    directory_path: Union[str, Path],
    extensions: tuple[str, ...] = (".pdf", ".txt", ".md"),
    recursive: bool = True,
) -> int:
    """Scan a directory and ingest all matching files into the pipeline.

    Args:
        pipeline: The active RAGPipeline instance to ingest into.
        directory_path: Path to the directory to scan.
        extensions: File extensions to match (e.g., ``(".pdf", ".txt")``).
        recursive: Whether to scan subdirectories recursively.

    Returns:
        The number of files successfully ingested.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory path does not exist or is not a directory: {dir_path}")

    pattern = "**/*" if recursive else "*"
    all_files = list(dir_path.glob(pattern))

    matching_files = [
        f for f in all_files if f.is_file() and f.suffix.lower() in extensions
    ]

    if not matching_files:
        logger.warning(f"No files matching extensions {extensions} found in {dir_path}")
        return 0

    logger.info(f"Found {len(matching_files)} files to ingest from {dir_path}")
    # Convert paths to strings
    file_list = [str(f) for f in matching_files]

    # Ingest in one batch
    pipeline.ingest_files(file_list)

    return len(matching_files)
