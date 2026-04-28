import { useState } from "react";
import { createFeatureTask, TaskResponse } from "../services/api";

type Props = {
  datasetId: number | null;
};

const steps = ["数据解析", "图结构构建", "特征归一化", "时间漂移编码", "模型推理"];

export function PipelinePanel({ datasetId }: Props) {
  const [task, setTask] = useState<TaskResponse | null>(null);
  const progress = task ? Math.round(task.progress * 100) : 0;

  async function startFeatureTask() {
    if (!datasetId) return;
    setTask(await createFeatureTask(datasetId));
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Feature Pipeline</p>
          <h2>特征处理进度</h2>
        </div>
        <button className="primary" disabled={!datasetId} onClick={startFeatureTask}>
          启动处理
        </button>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="step-grid">
        {steps.map((step, index) => (
          <div key={step} className={index <= Math.floor(progress / 25) ? "step active" : "step"}>
            <span>{index + 1}</span>
            {step}
          </div>
        ))}
      </div>
      <p className="hint">{task?.message ?? "后续会接入真实特征构建，并同步动画状态。"}</p>
    </section>
  );
}
