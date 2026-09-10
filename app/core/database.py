"""SQLAlchemy engine, session, and declarative base configuration."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all future SQLAlchemy models."""


def get_db() -> Generator[Session, None, None]:
    """Provide a database session and always close it after the request."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
