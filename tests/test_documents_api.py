# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# import pytest
# from fastapi.testclient import TestClient
# from src.main import app

# client = TestClient(app)

# @pytest.fixture
# def library_and_document():
#     # Create a library
#     metadata = {"name": "Docs Test Library", "description": "For document API tests"}
#     lib_resp = client.post("/libraries/", json=metadata)
#     assert lib_resp.status_code == 201
#     library_id = lib_resp.json()["id"]
#     # Create a document
#     doc = {"title": "Test Document", "chunks": [], "metadata": {}}
#     doc_resp = client.post(f"/libraries/{library_id}/documents/", json=doc)
#     assert doc_resp.status_code in (200, 201)
#     doc_id = doc_resp.json()["id"]
#     return library_id, doc_id

# def test_create_and_get_document(library_and_document):
#     library_id, doc_id = library_and_document
#     # Get the created document
#     response = client.get(f"/libraries/{library_id}/documents/{doc_id}")
#     assert response.status_code == 200
#     data = response.json()
#     assert data["metadata"]["title"] == "Test Document"    # Optionally check for empty chunks or metadata as created


# def test_list_documents(library_and_document):
#     library_id, doc_id = library_and_document
#     response = client.get(f"/libraries/{library_id}/documents/")
#     assert response.status_code == 200
#     docs = response.json()
#     assert any(doc["id"] == doc_id for doc in docs)

# def test_delete_document(library_and_document):
#     library_id, doc_id = library_and_document
#     # Delete document
#     response = client.delete(f"/libraries/{library_id}/documents/{doc_id}")
#     assert response.status_code == 204
#     # Confirm deletion
#     response = client.get(f"/libraries/{library_id}/documents/{doc_id}")
#     assert response.status_code == 404
