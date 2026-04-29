import { useEffect, useMemo, useState } from "react";
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

export function PipelinePanel({ datasetId, onInferenceComplete, onFocusNode }: Props) {
  const [task, setTask] = useState<TaskResponse | null>(null);
  const [timeline, setTimeline] = useState<TaskTimelineResponse | null>(null);
  const [inference, setInference] = useState<InferenceRunResponse | null>(null);
  const [currentEventIndex, setCurrentEventIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("等待选择业务网络并启动风险任务。");

  const currentEvent = timeline?.events?.[currentEventIndex] ?? null;
  const progress = Math.round((currentEvent?.progress ?? task?.progress ?? 0) * 100);

  useEffect(() => {
    if (!datasetId) {
      setTask(null);
      setTimeline(null);
      setInference(null);
      setCurrentEventIndex(0);
      onFocusNode(null);
      return;
    }
    fetchTimeline(datasetId)
      .then((data) => {
        setTimeline(data);
        setTask(data.task ?? null);
        setCurrentEventIndex(Math.max(0, data.events.length - 1));
      })
      .catch(() => undefined);
  }, [datasetId, onFocusNode]);

  useEffect(() => {
    if (!currentEvent?.focus_node_id) return;
    onFocusNode(currentEvent.focus_node_id);
  }, [currentEvent, onFocusNode]);

  useEffect(() => {
    if (!timeline || timeline.events.length <= 1 || !busy) return;
    const timer = window.setInterval(() => {
      setCurrentEventIndex((value) => {
        if (value >= timeline.events.length - 1) {
          window.clearInterval(timer);
          return value;
        }
        return value + 1;
      });
    }, 900);
    return () => window.clearInterval(timer);
  }, [timeline, busy]);

  const metricPairs = useMemo(() => {
    const metrics = currentEvent?.metrics ?? task?.summary ?? {};
    return Object.entries(metrics).slice(0, 4);
  }, [currentEvent, task]);

  async function refreshTimeline(targetDatasetId: number) {
    const data = await fetchTimeline(targetDatasetId);
    setTimeline(data);
    setTask(data.task ?? null);
    setCurrentEventIndex(0);
    return data;
  }

  async function startPipeline() {
    if (!datasetId || busy) return;
    setBusy(true);
    setInference(null);
    try {
      setMessage("正在准备特征与关系上下文。");
      const createdTask = await createFeatureTask(datasetId);
      setTask(createdTask);
      const featureTimeline = await refreshTimeline(datasetId);
      setMessage(featureTimeline.task?.message ?? "特征准备完成，准备进入风险识别。");
      await delay(Math.max(1200, featureTimeline.events.length * 800));

      setMessage("正在执行风险识别并生成结果台账。");
      const result = await runInference(datasetId);
      setInference(result);
      const inferenceTimeline = await refreshTimeline(datasetId);
      setCurrentEventIndex(0);
      setMessage(inferenceTimeline.task?.message ?? result.message);
      onInferenceComplete();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "处理失败。");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel panel-stack analysis-process-panel app-panel">
      <div className="panel-heading aligned-start split-heading">
        <div>
          <p className="eyebrow">Process Center</p>
          <h2>智能研判流程</h2>
          <p className="section-copy">启动任务后，系统会按阶段展示处理进度和当前定位对象。</p>
        </div>
        <button className="primary" disabled={!datasetId || busy} onClick={startPipeline}>
          {busy ? "任务执行中" : "启动分析任务"}
        </button>
      </div>

      <div className="progress-track large dark-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="step-grid enterprise-steps">
        {stageLabels.map((step, index) => (
          <div key={step} className={progress >= (index + 1) * 18 ? "step active" : "step"}>
            <span>{index + 1}</span>
            {step}
          </div>
        ))}
      </div>

      <div className="timeline-card emphasis-card">
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
        <div className="metric-strip four-cols">
          {metricPairs.map(([key, value]) => (
            <span key={key}>
              <small>{metricLabelMap[key] ?? key}</small>
              <strong>{typeof value === "object" ? JSON.stringify(value) : String(value)}</strong>
            </span>
          ))}
        </div>
      ) : null}

      {inference ? (
        <div className="metric-strip">
          <span>
            <small>对象总量</small>
            <strong>{inference.total_nodes}</strong>
          </span>
          <span>
            <small>高风险对象</small>
            <strong>{inference.abnormal_nodes}</strong>
          </span>
          <span>
            <small>低风险对象</small>
            <strong>{inference.normal_nodes}</strong>
          </span>
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
