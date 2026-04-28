from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str


class VerificationCodeRequest(BaseModel):
    email: EmailStr
    purpose: str = Field(pattern="^(register|login)$")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    code: str = Field(min_length=4, max_length=12)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    code: str = Field(min_length=4, max_length=12)


class AuthResponse(BaseModel):
    user_id: int
    email: EmailStr
    message: str


class DatasetSummary(BaseModel):
    id: int
    name: str
    original_filename: str
    row_count: int
    status: str
    created_at: datetime


class GraphNode(BaseModel):
    id: str
    label: str
    region: str
    occupation: str
    size: int
    color: str


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
