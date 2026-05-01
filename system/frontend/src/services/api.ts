export type AuthResponse = {
  user_id: number;
  email: string;
  message: string;
  is_admin: boolean;
  session_expires_at?: string | null;
};

export type LoginCaptchaResponse = {
  captcha_id: string;
  captcha_text: string;
  expires_at: string;
};

export type DatasetSummary = {
  id: number;
  name: string;
  original_filename: string;
  row_count: number;
  status: string;
  created_at: string;
  summary: Record<string, unknown>;
};

export type MappingResponse = {
  dataset_id: number | null;
  mapping: Record<string, unknown>;
  method: string;
  message: string;
};

export type GraphNode = {
  id: string;
  label: string;
  region: string;
  occupation: string;
  size: number;
  color: string;
  risk_score?: number | null;
  risk_label?: string | null;
  source_type?: string | null;
  feature_count: number;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  edge_type: string;
  amount?: number | null;
  timestamp?: string | null;
  highlighted: boolean;
};

export type GraphResponse = {
  dataset_id: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  summary: Record<string, unknown>;
};

export type TaskResponse = {
  id: number;
  dataset_id: number;
  task_type: string;
  status: string;
  progress: number;
  current_step: string;
  message: string;
  summary: Record<string, unknown>;
};

export type ProcessingEventItem = {
  id: number;
  stage: string;
  step_key: string;
  title: string;
  detail: string;
  progress: number;
  focus_node_id?: string | null;
  focus_neighbor_ids: string[];
  top_features: string[];
  metrics: Record<string, unknown>;
  created_at: string;
};

export type TaskTimelineResponse = {
  dataset_id: number;
  task?: TaskResponse | null;
  events: ProcessingEventItem[];
};

export type InferenceResultItem = {
  node_id: string;
  display_name: string;
  id_number: string;
  region: string;
  occupation: string;
  risk_score: number;
  risk_label: string;
  reason: string;
  support_neighbors: string[];
  top_features: string[];
};

export type InferenceRunResponse = {
  dataset_id: number;
  total_nodes: number;
  abnormal_nodes: number;
  normal_nodes: number;
  message: string;
  results: InferenceResultItem[];
  task_id?: number | null;
};

const API_BASE = "/api";

function normalizeApiError(message: string, status?: number) {
  const text = message.trim();
  if (!text || text === "Failed to fetch" || text === "NetworkError when attempting to fetch resource.") {
    return "服务暂时无法连接，请确认后端已启动后重试。";
  }
  if (status === 404) {
    return "服务接口暂时不可用，请刷新页面或重启后端后重试。";
  }
  if (text.includes("email already registered")) {
    return "该邮箱已注册，请直接登录或更换邮箱。";
  }
  if (text.includes("invalid or expired verification code")) {
    return "邮箱验证码错误或已过期，请重新获取。";
  }
  if (text.includes("invalid email or password")) {
    return "邮箱账号或密码不正确，请重新输入。";
  }
  return text;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
      ...init
    });
  } catch (error) {
    throw new Error(normalizeApiError(error instanceof Error ? error.message : ""));
  }
  if (!response.ok) {
    let message = "";
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const payload = (await response.json().catch(() => null)) as { detail?: unknown; message?: unknown } | null;
      message = String(payload?.detail ?? payload?.message ?? "");
    } else {
      message = await response.text();
    }
    throw new Error(normalizeApiError(message || `Request failed: ${response.status}`, response.status));
  }
  return response.json() as Promise<T>;
}

export function requestCode(email: string) {
  return request<{ message: string; code?: string }>("/auth/request-code", {
    method: "POST",
    body: JSON.stringify({ email, purpose: "register" })
  });
}

export function fetchLoginCaptcha() {
  return request<LoginCaptchaResponse>("/auth/login-captcha");
}

export function register(email: string, password: string, code: string) {
  return request<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, code })
  });
}

export function login(email: string, password: string, captchaId: string, captchaCode: string) {
  return request<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password, captcha_id: captchaId, captcha_code: captchaCode })
  });
}

export function listDatasets() {
  return request<DatasetSummary[]>("/datasets");
}

export function uploadDataset(file: File, useLlm = false, networkName = "", eventName = "") {
  const form = new FormData();
  form.append("file", file);
  form.append("use_llm", String(useLlm));
  if (networkName.trim()) {
    form.append("network_name", networkName.trim());
  }
  if (eventName.trim()) {
    form.append("event_name", eventName.trim());
  }
  return request<DatasetSummary>("/datasets/upload", {
    method: "POST",
    body: form
  });
}

export function createDemoDataset(datasetName: string) {
  return request<DatasetSummary>(`/datasets/demo/${datasetName}`, { method: "POST" });
}

export function updateDataset(datasetId: number, businessName: string) {
  return request<DatasetSummary>(`/datasets/${datasetId}`, {
    method: "PATCH",
    body: JSON.stringify({ business_name: businessName })
  });
}

export function deleteDataset(datasetId: number) {
  return request<{ message: string }>(`/datasets/${datasetId}`, { method: "DELETE" });
}

export function fetchGraph(datasetId: number) {
  return request<GraphResponse>(`/datasets/${datasetId}/graph`);
}

export function createGraphTask(datasetId: number) {
  return request<TaskResponse>(`/datasets/${datasetId}/graph-task`, { method: "POST" });
}

export function fetchMapping(datasetId: number) {
  return request<MappingResponse>(`/datasets/${datasetId}/mapping`);
}

export function createFeatureTask(datasetId: number) {
  return request<TaskResponse>(`/datasets/${datasetId}/feature-task`, { method: "POST" });
}

export function fetchTimeline(datasetId: number) {
  return request<TaskTimelineResponse>(`/datasets/${datasetId}/timeline`);
}

export function runInference(datasetId: number) {
  return request<InferenceRunResponse>(`/datasets/${datasetId}/infer`, { method: "POST" });
}

export function listInferenceResults(datasetId: number) {
  return request<InferenceResultItem[]>(`/datasets/${datasetId}/inference-results`);
}
