export type DatasetSummary = {
  id: number;
  name: string;
  original_filename: string;
  row_count: number;
  status: string;
  created_at: string;
};

export type GraphNode = {
  id: string;
  label: string;
  region: string;
  occupation: string;
  size: number;
  color: string;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  edge_type: string;
  amount?: number | null;
  timestamp?: string | null;
};

export type GraphResponse = {
  dataset_id: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type TaskResponse = {
  id: number;
  dataset_id: number;
  task_type: string;
  status: string;
  progress: number;
  current_step: string;
  message: string;
};

const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function requestCode(email: string, purpose: "register" | "login") {
  return request<{ message: string }>("/auth/request-code", {
    method: "POST",
    body: JSON.stringify({ email, purpose })
  });
}

export function register(email: string, password: string, code: string) {
  return request<{ user_id: number; email: string; message: string }>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, code })
  });
}

export function login(email: string, password: string, code: string) {
  return request<{ user_id: number; email: string; message: string }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password, code })
  });
}

export function listDatasets() {
  return request<DatasetSummary[]>("/datasets");
}

export function uploadDataset(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<DatasetSummary>("/datasets/upload", {
    method: "POST",
    body: form
  });
}

export function fetchGraph(datasetId: number) {
  return request<GraphResponse>(`/datasets/${datasetId}/graph`);
}

export function createFeatureTask(datasetId: number) {
  return request<TaskResponse>(`/datasets/${datasetId}/feature-task`, { method: "POST" });
}
