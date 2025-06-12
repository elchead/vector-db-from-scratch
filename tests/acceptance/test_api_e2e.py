import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pytest
from fastapi.testclient import TestClient
from src.main import app

@pytest.mark.skipif(not os.environ.get("COHERE_API_KEY"), reason="COHERE_API_KEY environment variable not set")
def test_api_e2e_search():
    client = TestClient(app)

    # 1. Create a library
    lib_resp = client.post(
        "/libraries/",
        json={"name": "Test Library", "description": "A test library"}
    )
    assert lib_resp.status_code == 201
    library = lib_resp.json()
    library_id = library["id"]

    # 2. Add a document with chunks
    doc_resp = client.post(
        f"/libraries/{library_id}/documents/",
        json={
            "chunks": [
                {"text": "The quick brown fox jumps over the lazy dog."},
                {"text": "A fast brown fox leaps over the sleepy canine."},
                {"text": "Machine learning models process data to make predictions."}
            ],
            "metadata": {"title": "Test Document"}
        }
    )
    assert doc_resp.status_code == 201

    # 3. Search
    search_resp = client.post(
        f"/libraries/{library_id}/search/",
        json={
            "query": "A quick brown animal jumps over a lazy dog.",
            "top_k": 2
        }
    )
    assert search_resp.status_code == 200
    results = search_resp.json()["results"]
    assert len(results) == 2
    assert all("fox" in chunk["text"] for chunk in results)

    print("Top results:")
    for chunk in results:
        print(f"Chunk UUID: {chunk['id']}, Text: {chunk['text']}")
