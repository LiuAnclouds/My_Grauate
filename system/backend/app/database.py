from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import ensure_runtime_dirs, settings


class Base(DeclarativeBase):
    pass


ensure_runtime_dirs()

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models
    from app.services.security import hash_secret

    Base.metadata.create_all(bind=engine)
    _apply_runtime_migrations()
    if settings.demo_mode and settings.demo_admin_email and settings.demo_admin_password:
        with SessionLocal() as db:
            existing = db.scalar(select(models.User).where(models.User.email == settings.demo_admin_email))
            expected_hash = hash_secret(settings.demo_admin_password)
            if existing is None:
                db.add(
                    models.User(
                        email=settings.demo_admin_email,
                        password_hash=expected_hash,
                        is_admin=True,
                    )
                )
                db.commit()
            else:
                existing.password_hash = expected_hash
                existing.is_admin = True
                db.commit()


def _apply_runtime_migrations() -> None:
    inspector = inspect(engine)
    columns_by_table = {
        table_name: {column["name"] for column in inspector.get_columns(table_name)}
        for table_name in inspector.get_table_names()
    }
    statements: list[str] = []
    if "users" in columns_by_table and "is_admin" not in columns_by_table["users"]:
        statements.append("ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0")
    if "dataset_uploads" in columns_by_table and "summary_json" not in columns_by_table["dataset_uploads"]:
        statements.append("ALTER TABLE dataset_uploads ADD COLUMN summary_json JSON NOT NULL DEFAULT '{}' ")
    if "processing_tasks" in columns_by_table and "summary_json" not in columns_by_table["processing_tasks"]:
        statements.append("ALTER TABLE processing_tasks ADD COLUMN summary_json JSON NOT NULL DEFAULT '{}' ")
    if statements:
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
