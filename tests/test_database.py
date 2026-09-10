"""Tests for the SQLAlchemy database infrastructure."""

from unittest.mock import Mock

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import DeclarativeBase, Session

from app.core import database


def test_database_engine_is_configured() -> None:
    assert isinstance(database.engine, Engine)
    assert database.engine.pool._pre_ping is True


def test_base_uses_sqlalchemy_declarative_base() -> None:
    assert issubclass(database.Base, DeclarativeBase)
    assert database.Base.metadata is not None


def test_session_factory_creates_sqlalchemy_sessions() -> None:
    session = database.SessionLocal()

    try:
        assert isinstance(session, Session)
        assert session.bind is database.engine
    finally:
        session.close()


def test_get_db_closes_session_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    session = Mock(spec=Session)
    monkeypatch.setattr(database, "SessionLocal", Mock(return_value=session))
    dependency = database.get_db()

    assert next(dependency) is session
    dependency.close()

    session.close.assert_called_once_with()


def test_get_db_closes_session_after_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = Mock(spec=Session)
    monkeypatch.setattr(database, "SessionLocal", Mock(return_value=session))
    dependency = database.get_db()
    next(dependency)

    with pytest.raises(RuntimeError, match="request failed"):
        dependency.throw(RuntimeError("request failed"))

    session.close.assert_called_once_with()
