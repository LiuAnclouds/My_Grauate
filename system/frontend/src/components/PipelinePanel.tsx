import { useState } from "react";
import { createFeatureTask, InferenceRunResponse, runInference, TaskResponse } from "../services/api";

type Props = {
  datasetId: number | null;
  onInferenceComplete: () => void;
};

const steps = ["数据解析", "图结构构建", "特征归一化", "时间漂移编码", "模型推理"];

export function PipelinePanel({ datasetId, onInferenceComplete }: Props) {
  const [task, setTask] = useState<TaskResponse | null>(null);
  const [inference, setInference] = useState<InferenceRunResponse | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("等待选择数据集。");

  async function startPipeline() {
    if (!datasetId || busy) return;
    setBusy(true);
    setInference(null);
    try {
      setProgress(12);
      setMessage("正在读取数据库记录并构建演示图。");
      const createdTask = await createFeatureTask(datasetId);
      setTask(createdTask);
      setProgress(35);
      setMessage("正在对齐 full 模型使用的特征处理流程。");
      await delay(400);
      setProgress(62);
      setMessage("正在加载 DyRIFT-TGAT full 权重并执行节点推理。");
      const result = await runInference(datasetId);
      setInference(result);
      setProgress(100);
      setMessage(result.message);
      onInferenceComplete();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "处理失败。");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Feature Pipeline</p>
          <h2>特征处理进度</h2>
        </div>
        <button className="primary" disabled={!datasetId || busy} onClick={startPipeline}>
          {busy ? "处理中" : "启动处理"}
        </button>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="step-grid">
        {steps.map((step, index) => (
          <div key={step} className={index <= Math.floor(progress / 22) ? "step active" : "step"}>
            <span>{index + 1}</span>
            {step}
          </div>
        ))}
      </div>
      {inference ? (
        <div className="metric-strip">
          <span>节点 {inference.total_nodes}</span>
          <span>异常 {inference.abnormal_nodes}</span>
          <span>正常 {inference.normal_nodes}</span>
        </div>
      ) : null}
      <p className="hint">{task?.message ? `${task.message} ${message}` : message}</p>
    </section>
  );
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
