"""
Simple test script for the Semantic Chunking API.

Run this after starting the API server to test inference.
"""

import json

import requests

API_URL = "http://localhost:8000"


def main():
    # Health check
    response = requests.get(f"{API_URL}/health")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

    # Read document
    with open("scripts/test_doc.txt") as f:
        document = f.read()

    # Call chunk endpoint
    response = requests.post(
        f"{API_URL}/v1/chunk",
        json={"document": document},
        timeout=120,
    )

    # Print result
    if response.status_code == 200:
        result = response.json()
        print(f"Number of chunks: {result['num_chunks']}\n")

        for i, chunk in enumerate(result["chunks"], 1):
            word_count = len(chunk.split())
            print(f"{'─' * 80}")
            print(f"Chunk {i} ({word_count} words)")
            print(f"{'─' * 80}")
            print(f"{chunk}\n")
    else:
        print(f"Error: {response.text}\n")


if __name__ == "__main__":
    main()
