import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  createFeatureTask,
  fetchTimeline,
  InferenceRunResponse,
  ProcessingEventItem,
  runInference,
  TaskResponse,
  TaskTimelineResponse
} from "../services/api";

type Props = {
  datasetId: number | null;
  onInferenceComplete: () => void;
  onFocusNode: (nodeId: string | null) => void;
};

const stageLabels = ["对象装载", "关系组织", "特征准备", "时序编码", "风险识别"];
const stageTitleMap: Record<string, string> = {
  feature_processing: "特征准备",
  graph_construction: "图谱构建",
  inference: "风险推理",
  ingestion: "数据接入"
};
const metricLabelMap: Record<string, string> = {
  nodes: "对象规模",
  edges: "关系规模",
  avg_feature_count: "平均特征数",
  records: "处理记录数",
  abnormal: "高风险对象",
  normal: "低风险对象",
  ready: "状态",
  risk_score: "风险分数",
  risk_label: "风险标签",
  display_name: "对象名称"
};

const featureProgress = [0.12, 0.3, 0.52, 0.72, 0.78];

function makeLocalEvent(title: string, detail: string, progress: number, stepKey: string): ProcessingEventItem {
  return {
    id: -Math.round(Date.now() + progress * 1000),
    stage: "inference",
    step_key: stepKey,
    title,
    detail,
    progress,
    focus_neighbor_ids: [],
    top_features: [],
    metrics: {},
    created_at: new Date().toISOString()
  };
}

function normalizeFeatureEvents(events: ProcessingEventItem[]) {
  return events.map((event, index) => ({
    ...event,
    progress: featureProgress[index] ?? Math.min(0.78, event.progress * 0.78)
  }));
}

function normalizeInferenceEvents(events: ProcessingEventItem[]) {
  const finalIndex = Math.max(1, events.length - 1);
  return events.map((event, index) => {
    const progress = event.progress >= 1
      ? 1
      : Math.min(0.96, 0.82 + (index / finalIndex) * 0.14);
    return { ...event, progress };
  });
}

function stageIndexForProgress(progress: number) {
  if (progress <= 0) return -1;
  if (progress < 18) return 0;
  if (progress < 38) return 1;
  if (progress < 62) return 2;
  if (progress < 82) return 3;
  return 4;
}

export function PipelinePanel({ datasetId, onInferenceComplete, onFocusNode }: Props) {
  const runTokenRef = useRef(0);
  const [task, setTask] = useState<TaskResponse | null>(null);
  const [timeline, setTimeline] = useState<TaskTimelineResponse | null>(null);
  const [inference, setInference] = useState<InferenceRunResponse | null>(null);
  const [currentEventIndex, setCurrentEventIndex] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("等待选择业务网络并启动风险任务。");

  const currentEvent = timeline?.events?.[currentEventIndex] ?? null;
  const progress = hasStarted ? Math.round((currentEvent?.progress ?? 0) * 100) : 0;

  useEffect(() => {
    runTokenRef.current += 1;
    setTask(null);
    setTimeline(null);
    setInference(null);
    setCurrentEventIndex(0);
    setHasStarted(false);
    setBusy(false);
    onFocusNode(null);
    if (!datasetId) {
      setMessage("等待选择业务网络并启动风险任务。");
      return;
    }
    setMessage("点击启动分析任务后，系统会从对象装载开始演示研判流程。");
  }, [datasetId, onFocusNode]);

  useEffect(() => {
    if (!hasStarted || !currentEvent?.focus_node_id) return;
    onFocusNode(currentEvent.focus_node_id);
  }, [currentEvent, hasStarted, onFocusNode]);

  const metricPairs = useMemo(() => {
    if (!hasStarted) return [];
    const metrics = currentEvent?.metrics ?? task?.summary ?? {};
    return Object.entries(metrics).slice(0, 4);
  }, [currentEvent, hasStarted, task]);

  async function refreshTimeline(targetDatasetId: number) {
    const data = await fetchTimeline(targetDatasetId);
    setTimeline(data);
    setTask(data.task ?? null);
    setCurrentEventIndex(0);
    return data;
  }

  async function playEvents(events: ProcessingEventItem[], stepDelay: number, token: number) {
    if (!datasetId || !events.length) return false;
    setTimeline({ dataset_id: datasetId, task: null, events });
    for (let index = 0; index < events.length; index += 1) {
      if (runTokenRef.current !== token) return false;
      const event = events[index];
      setCurrentEventIndex(index);
      setMessage(event.detail);
      await delay(stepDelay);
    }
    return true;
  }

  async function startPipeline() {
    if (!datasetId || busy) return;
    const token = runTokenRef.current + 1;
    runTokenRef.current = token;
    setBusy(true);
    setHasStarted(true);
    setInference(null);
    setTask(null);
    onFocusNode(null);
    let pulseTimer: number | undefined;
    try {
      const bootEvent = makeLocalEvent("启动研判任务", "正在建立本次研判上下文。", 0.03, "start");
      setTimeline({ dataset_id: datasetId, task: null, events: [bootEvent] });
      setCurrentEventIndex(0);
      setMessage(bootEvent.detail);
      await delay(420);

      const createdTask = await createFeatureTask(datasetId);
      setTask(createdTask);
      const featureTimeline = await refreshTimeline(datasetId);
      const featureEvents = normalizeFeatureEvents(featureTimeline.events);
      const featurePlayed = await playEvents(featureEvents, 620, token);
      if (!featurePlayed) return;

      let pulse = 0;
      pulseTimer = window.setInterval(() => {
        if (runTokenRef.current !== token || !datasetId) return;
        pulse += 1;
        const progressValue = Math.min(0.94, 0.82 + pulse * 0.012);
        const event = makeLocalEvent(
          "风险识别中",
          "正在聚合邻域关系与关键特征，生成对象级风险判断。",
          progressValue,
          "inference_running"
        );
        setTimeline({ dataset_id: datasetId, task: null, events: [event] });
        setCurrentEventIndex(0);
        setMessage(event.detail);
      }, 420);
      const result = await runInference(datasetId);
      if (pulseTimer) window.clearInterval(pulseTimer);
      pulseTimer = undefined;
      if (runTokenRef.current !== token) return;

      const inferenceTimeline = await refreshTimeline(datasetId);
      const inferenceEvents = normalizeInferenceEvents(inferenceTimeline.events);
      await playEvents(inferenceEvents, 380, token);
      if (runTokenRef.current !== token) return;
      setInference(result);
      setMessage(inferenceTimeline.task?.message ?? result.message);
      onInferenceComplete();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "处理失败。");
    } finally {
      if (pulseTimer) window.clearInterval(pulseTimer);
      if (runTokenRef.current === token) setBusy(false);
    }
  }

  const currentStageIndex = stageIndexForProgress(progress);
  const taskCompleted = Boolean(inference);

  return (
    <section className="panel panel-stack analysis-process-panel analysis-visual-panel app-panel">
      <div className="panel-heading aligned-start split-heading">
        <div>
          <p className="eyebrow">Process Center</p>
          <h2>智能研判流程</h2>
          <p className="section-copy">启动后，右侧图谱会同步定位当前推理节点。</p>
        </div>
        <button className="primary analysis-run-button" disabled={!datasetId || busy} onClick={startPipeline}>
          {busy ? "任务执行中" : "启动分析任务"}
        </button>
      </div>

      <div className="analysis-progress-card">
        <div>
          <small>研判进度</small>
          <strong>{progress}%</strong>
        </div>
        <span>{taskCompleted ? "结果已同步" : busy ? "正在研判" : datasetId ? "等待启动" : "未选择网络"}</span>
        <div className="progress-track large dark-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="analysis-stage-lane" style={{ "--stage-progress": `${progress}%` } as CSSProperties}>
        {stageLabels.map((step, index) => (
          <button
            key={step}
            className={[
              "analysis-stage-node",
              progress >= [12, 30, 52, 72, 100][index] ? "active" : "",
              hasStarted && currentStageIndex === index && !taskCompleted ? "current" : ""
            ].filter(Boolean).join(" ")}
            type="button"
          >
            <span>{index + 1}</span>
            <strong>{step}</strong>
          </button>
        ))}
      </div>

      <div className="timeline-card emphasis-card analysis-current-card">
        <div className="panel-subheading compact-heading">
          <h3>当前任务节点</h3>
          <span>{progress}%</span>
        </div>
        <strong>{currentEvent?.title ?? task?.current_step ?? "等待启动"}</strong>
        <p>{currentEvent?.detail ?? task?.message ?? message}</p>
        <div className="event-tags">
          <span className="subtle-tag">{stageTitleMap[currentEvent?.stage ?? task?.task_type ?? ""] ?? "任务阶段"}</span>
          {currentEvent?.top_features?.length ? <span className="subtle-tag">关键特征 {currentEvent.top_features.length}</span> : null}
          {currentEvent?.focus_neighbor_ids?.length ? <span className="subtle-tag">关联对象 {currentEvent.focus_neighbor_ids.length}</span> : null}
        </div>
      </div>

      {metricPairs.length ? (
        <div className="metric-strip four-cols analysis-context-strip">
          {metricPairs.map(([key, value]) => (
            <span key={key}>
              <small>{metricLabelMap[key] ?? key}</small>
              <strong>{typeof value === "object" ? JSON.stringify(value) : String(value)}</strong>
            </span>
          ))}
        </div>
      ) : null}

      {inference ? (
        <div className="analysis-finish-note">
          <strong>研判完成</strong>
          <span>风险对象已生成，可进入“风险对象”页面复核。</span>
        </div>
      ) : null}

      <div className="timeline-list process-list">
        {(timeline?.events ?? []).map((event, index) => (
          <button
            key={event.id}
            className={index === currentEventIndex ? "timeline-item active" : "timeline-item"}
            onClick={() => setCurrentEventIndex(index)}
            type="button"
          >
            <div className="timeline-item-head">
              <strong>{event.title}</strong>
              <span>{Math.round(event.progress * 100)}%</span>
            </div>
            <small>{event.detail}</small>
          </button>
        ))}
      </div>

      <p className="hint">{message}</p>
    </section>
  );
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
