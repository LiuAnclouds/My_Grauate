from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str


class VerificationCodeRequest(BaseModel):
    email: str = Field(min_length=1, max_length=255)
    purpose: str = Field(pattern="^(register)$")


class LoginCaptchaResponse(BaseModel):
    captcha_id: str
    captcha_text: str
    expires_at: datetime


class RegisterRequest(BaseModel):
    email: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=4, max_length=12)


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=128)
    captcha_id: str = Field(min_length=8, max_length=128)
    captcha_code: str = Field(min_length=4, max_length=12)


class AuthResponse(BaseModel):
    user_id: int
    email: str
    message: str
    is_admin: bool = False
    session_expires_at: datetime | None = None


class DatasetSummary(BaseModel):
    id: int
    name: str
    original_filename: str
    row_count: int
    status: str
    created_at: datetime
    summary: dict[str, Any] = {}


class DatasetUpdateRequest(BaseModel):
    business_name: str = Field(min_length=1, max_length=80)


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
    source_type: str | None = None
    feature_count: int = 0


class GraphEdgeItem(BaseModel):
    id: str
    source: str
    target: str
    edge_type: str
    amount: float | None = None
    timestamp: str | None = None
    highlighted: bool = False


class GraphResponse(BaseModel):
    dataset_id: int
    nodes: list[GraphNode]
    edges: list[GraphEdgeItem]
    summary: dict[str, Any] = {}


class TaskResponse(BaseModel):
    id: int
    dataset_id: int
    task_type: str
    status: str
    progress: float
    current_step: str
    message: str
    summary: dict[str, Any] = {}


class ProcessingEventItem(BaseModel):
    id: int
    stage: str
    step_key: str
    title: str
    detail: str
    progress: float
    focus_node_id: str | None = None
    focus_neighbor_ids: list[str] = []
    top_features: list[str] = []
    metrics: dict[str, Any] = {}
    created_at: datetime


class TaskTimelineResponse(BaseModel):
    dataset_id: int
    task: TaskResponse | None = None
    events: list[ProcessingEventItem] = []


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
    task_id: int | None = None
