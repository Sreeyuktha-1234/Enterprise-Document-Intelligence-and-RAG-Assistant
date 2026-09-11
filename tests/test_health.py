"""Tests for the FastAPI application foundation and health endpoint."""

from unittest.mock import Mock

from fastapi.testclient import TestClient

from app import main


def test_application_metadata() -> None:
    assert main.app.title == main.settings.APP_NAME
    assert main.app.version == "0.1.0"
    assert main.app.debug is main.settings.DEBUG
    assert main.app.description


def test_health_endpoint_returns_healthy_status() -> None:
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_router_is_registered_once() -> None:
    paths = main.app.openapi()["paths"]

    assert "/health" in paths
    assert set(paths["/health"]) == {"get"}


def test_shutdown_disposes_database_engine(monkeypatch) -> None:
    dispose = Mock()
    monkeypatch.setattr(main.engine, "dispose", dispose)

    with TestClient(main.app):
        pass

    dispose.assert_called_once_with()
