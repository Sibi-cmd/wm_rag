"""
examples.ingest_directory
~~~~~~~~~~~~~~~~~~~~~~~~~
An example showcasing how to recursively scan a local directory, batch-ingest all
supported files (.txt, .md, .pdf), and search/retrieve relevant segments.
"""

import os
import sys
from pathlib import Path

# Add the repository root to Python path to import rag_core cleanly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rag_core import RAGPipeline, ingest_directory

load_dotenv()


def main():
    # 1. Resolve configuration path
    config_path = Path(__file__).parent.parent / "configs" / "rag_config.yaml"
    pipeline = RAGPipeline.from_config(config_path)

    # 2. Setup a temporary mock directory to ingest
    mock_dir = Path(__file__).parent / "sample_docs"
    mock_dir.mkdir(exist_ok=True)

    # Write a dummy reference file
    reference_file = mock_dir / "charging_guide.txt"
    with open(reference_file, "w", encoding="utf-8") as fh:
        fh.write(
            "TROUBLESHOOTING: EV Charging Issues\n"
            "If the EV charging process does not start, verify that the connector\n"
            "is firmly plugged in. Lock the doors to engage the connector lock mechanism."
        )

    print(f"Created sample directory for ingestion: {mock_dir.resolve()}")

    # 3. Trigger recursive directory ingestion
    print("\nScanning and ingesting directory...")
    num_files = ingest_directory(
        pipeline=pipeline,
        directory_path=mock_dir,
        extensions=(".txt", ".md", ".pdf"),
        recursive=True,
    )
    print(f"Ingested {num_files} files successfully!")

    # 4. Perform search/retrieval
    query = "charging connector will not lock"
    print(f"\nSearching database for: '{query}'")
    results = pipeline.retrieve(query, top_k=2)

    print("\n--- RETRIEVED SEGMENTS ---")
    for idx, match in enumerate(results, 1):
        print(f"[{idx}] (Score: {match.score:.2f}) Source: {match.metadata.get('source')}")
        print(f"    Content: {match.text.strip()}")
        print("-" * 50)

    # Clean up temporary mock files
    reference_file.unlink(missing_ok=True)
    mock_dir.rmdir()


if __name__ == "__main__":
    main()
