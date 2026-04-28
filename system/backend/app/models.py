from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class VerificationCode(Base):
    __tablename__ = "verification_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    code_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    consumed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DatasetUpload(Base):
    __tablename__ = "dataset_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="uploaded", nullable=False)
    mapping_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    nodes: Mapped[list["PersonNode"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    edges: Mapped[list["GraphEdge"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class PersonNode(Base):
    __tablename__ = "person_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset_uploads.id"), index=True, nullable=False)
    node_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(64), nullable=False)
    id_number: Mapped[str] = mapped_column(String(32), nullable=False)
    phone: Mapped[str] = mapped_column(String(32), nullable=False)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    occupation: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    feature_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    dataset: Mapped[DatasetUpload] = relationship(back_populates="nodes")


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset_uploads.id"), index=True, nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    target_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    edge_type: Mapped[str] = mapped_column(String(64), default="relation", nullable=False)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[str | None] = mapped_column(String(64), nullable=True)

    dataset: Mapped[DatasetUpload] = relationship(back_populates="edges")


class ProcessingTask(Base):
    __tablename__ = "processing_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset_uploads.id"), index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_step: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset_uploads.id"), index=True, nullable=False)
    node_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_label: Mapped[str] = mapped_column(String(32), nullable=False)
    explanation_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
