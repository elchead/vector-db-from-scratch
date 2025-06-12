import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

@pytest.fixture
def library_metadata():
    return {"name": "Test Library", "description": "A test library"}

def test_create_and_get_library(library_metadata):
    # Create a library
    response = client.post("/libraries/", json=library_metadata)
    assert response.status_code == 201
    data = response.json()
    assert data["metadata"]["name"] == library_metadata["name"]
    assert data["metadata"]["description"] == library_metadata["description"]
    library_id = data["id"]

    # Get the created library
    response = client.get(f"/libraries/{library_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["name"] == library_metadata["name"]
    assert data["metadata"]["description"] == library_metadata["description"]

def test_list_libraries(library_metadata):
    # Create a library
    client.post("/libraries/", json=library_metadata)
    # List libraries
    response = client.get("/libraries/")
    assert response.status_code == 200
    data = response.json()
    assert any(lib["metadata"]["name"] == library_metadata["name"] for lib in data)

def test_update_and_delete_library(library_metadata):
    # Create a library
    response = client.post("/libraries/", json=library_metadata)
    library_id = response.json()["id"]
    # Update the library
    update_data = {"name": "Updated Name", "description": "Updated description"}
    response = client.put(f"/libraries/{library_id}", params=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["name"] == update_data["name"]
    assert data["metadata"]["description"] == update_data["description"]
    # Delete the library
    response = client.delete(f"/libraries/{library_id}")
    assert response.status_code == 204
    # Confirm deletion
    response = client.get(f"/libraries/{library_id}")
    assert response.status_code == 404
