from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field
from typing import Any


class HealthResponse(BaseModel):
    status: str
    app_name: str


class VerificationCodeRequest(BaseModel):
    email: str = Field(min_length=1, max_length=255)
    purpose: str = Field(pattern="^(register|login)$")


class RegisterRequest(BaseModel):
    email: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=4, max_length=12)


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=4, max_length=12)


class AuthResponse(BaseModel):
    user_id: int
    email: str
    message: str


class DatasetSummary(BaseModel):
    id: int
    name: str
    original_filename: str
    row_count: int
    status: str
    created_at: datetime


class MappingResponse(BaseModel):
    dataset_id: int | None = None
    mapping: dict[str, Any]
    method: str
    message: str


class GraphNode(BaseModel):
    id: str
    label: str
    region: str
    occupation: str
    size: int
    color: str
    risk_score: float | None = None
    risk_label: str | None = None


class GraphEdgeItem(BaseModel):
    id: str
    source: str
    target: str
    edge_type: str
    amount: float | None = None
    timestamp: str | None = None


class GraphResponse(BaseModel):
    dataset_id: int
    nodes: list[GraphNode]
    edges: list[GraphEdgeItem]


class TaskResponse(BaseModel):
    id: int
    dataset_id: int
    task_type: str
    status: str
    progress: float
    current_step: str
    message: str


class InferenceResultItem(BaseModel):
    node_id: str
    display_name: str
    id_number: str
    region: str
    occupation: str
    risk_score: float
    risk_label: str
    reason: str
    support_neighbors: list[str] = []
    top_features: list[str] = []


class InferenceRunResponse(BaseModel):
    dataset_id: int
    total_nodes: int
    abnormal_nodes: int
    normal_nodes: int
    message: str
    results: list[InferenceResultItem]
