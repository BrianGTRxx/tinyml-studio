import pytest
from fastapi.testclient import TestClient

from tinyml_studio.api import create_app

TOKEN = "t" * 32


@pytest.fixture
def client(dataset_dir):
    app = create_app(TOKEN, [dataset_dir])
    return TestClient(app, base_url="http://127.0.0.1:8000")


def auth(tok=TOKEN):
    return {"Authorization": f"Bearer {tok}"}


def test_requires_token(client):
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers=auth("x" * 32)).status_code == 401
    r = client.get("/health", headers=auth())
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_rejects_foreign_host(dataset_dir):
    c = TestClient(create_app(TOKEN, [dataset_dir]), base_url="http://evil.example")
    assert c.get("/health", headers=auth()).status_code == 400


def test_weak_token_refused():
    with pytest.raises(ValueError):
        create_app("corto")


def test_no_docs_exposed(client):
    assert client.get("/docs", headers=auth()).status_code == 404
    assert client.get("/openapi.json", headers=auth()).status_code == 404




def test_profile_outside_root_or_traversal(client, dataset_dir, tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("a\n1\n")
    assert client.post("/profile", headers=auth(), json={"path": str(f)}).status_code == 403
    sneaky = str(dataset_dir / ".." / ".." / "pyproject.toml")
    assert client.post("/profile", headers=auth(), json={"path": sneaky}).status_code == 403


def test_roots_endpoint_allows_new_folder(client, tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("a\n1\n2\n")
    assert client.post("/profile", headers=auth(), json={"path": str(f)}).status_code == 403
    assert client.post("/roots", headers=auth(), json={"path": str(tmp_path)}).status_code == 200
    assert client.post("/profile", headers=auth(), json={"path": str(f)}).json()["rows"] == 2


