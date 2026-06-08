"""
examples.simple_qa
~~~~~~~~~~~~~~~~~~
A simple working example showcasing how to load the RAGPipeline from a config file,
ingest raw text data, and perform Q&A with references.
"""

import os
import sys
from pathlib import Path

# Add the repository root to Python path to import rag_core cleanly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rag_core import RAGPipeline

# Load environmental variables (.env) containing API keys
load_dotenv()


def main():
    # 1. Resolve configuration path
    config_path = Path(__file__).parent.parent / "configs" / "rag_config.yaml"
    print(f"Loading RAG configuration from: {config_path}")

    # 2. Initialize the pipeline
    pipeline = RAGPipeline.from_config(config_path)

    # 3. Ingest raw text data
    print("\nIngesting documents...")
    documents = [
        {
            "text": (
                "PROCEDURE: Nexon EV battery reset\n"
                "Disconnect the auxiliary 12V battery terminal for 10 minutes.\n"
                "WARNING: Do not touch any high voltage orange cables!"
            ),
            "metadata": {"source": "nexon_ev_manual.pdf", "section": "Battery System"},
        },
        {
            "text": (
                "Our customer return policy allows product returns within 30 days of purchase.\n"
                "Items must be in original packaging to qualify."
            ),
            "metadata": {"source": "return_policy.txt", "section": "Returns"},
        },
    ]

    pipeline.ingest_texts(documents)
    print("Ingestion completed successfully!")

    # 4. Trigger query
    # (Note: To call OpenAI or Cloud models, ensure OPENAI_API_KEY is configured in your .env)
    question = "How do I perform a battery reset on the Nexon EV?"
    print(f"\nAsking question: '{question}'")

    try:
        response = pipeline.ask(question)
        print("\n--- RAG ANSWER ---")
        print(response.text)
        print("------------------")
        print(f"Confidence score: {response.confidence:.2f}")
        print(f"Source references: {response.sources}")
    except ValueError as err:
        print(f"\n[Warning] Could not complete AI generation: {err}")
        print("RAG search still retrieved the following local segments:")
        results = pipeline.retrieve(question)
        for idx, res in enumerate(results, 1):
            print(f"  [{idx}] (Score: {res.score:.2f}) from {res.metadata.get('source')}: {res.text[:100]}...")


if __name__ == "__main__":
    main()
