import { useState } from "react";
import { AuthResponse } from "./services/api";
import { AuthPanel } from "./components/AuthPanel";
import { DataUpload } from "./components/DataUpload";
import { GraphWorkspace } from "./components/GraphWorkspace";
import { InferenceResults } from "./components/InferenceResults";
import { PipelinePanel } from "./components/PipelinePanel";

const overviewCards = [
  {
    title: "场景覆盖",
    value: "账户网络 / 交易链路 / 关联对象",
    description: "支持官方样本与自定义 CSV 数据接入，统一沉淀为可分析的数据资产。"
  },
  {
    title: "分析链路",
    value: "接入治理 / 关系建模 / 风险研判",
    description: "从结构化入库到时序关系组织，再到异常识别，形成完整分析闭环。"
  },
  {
    title: "输出能力",
    value: "图谱联动 / 过程追踪 / 结果台账",
    description: "工作台同步呈现关系网络、任务流转与高风险对象明细，便于定位与复核。"
  }
];

export default function App() {
  const [session, setSession] = useState<AuthResponse | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [graphRefreshKey, setGraphRefreshKey] = useState(0);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);
  const [activeTimelineNodeId, setActiveTimelineNodeId] = useState<string | null>(null);

  function handleDatasetSelect(datasetId: number) {
    setSelectedDatasetId(datasetId);
    setHighlightedNodeId(null);
    setActiveTimelineNodeId(null);
  }

  if (!session) {
    return <AuthPanel onAuthed={setSession} />;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <p className="eyebrow">Risk Intelligence Workspace</p>
          <h1 className="brand-title">星枢反欺诈分析平台</h1>
          <p className="topbar-subtitle">面向关系网络风险识别、异常对象研判与任务追踪的统一分析工作台。</p>
        </div>
        <div className="topbar-actions">
          <span className="account-pill">{session.email}</span>
          {session.is_admin ? <span className="role-pill">管理工作台</span> : <span className="role-pill muted">分析账号</span>}
          <button className="ghost-button" onClick={() => setSession(null)}>
            退出登录
          </button>
        </div>
      </header>

      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Enterprise Anti-Fraud</p>
          <h2>将数据接入、关系图谱与风险输出收敛到同一分析界面。</h2>
          <p>
            平台围绕关系网络风控落地场景设计，支持业务样本入库、异常链路追踪、对象级风险定位与过程可视化复核，
            适用于账户网络、交易关系和多对象关联研判。
          </p>
          <div className="hero-tags">
            <span className="hero-tag">数据资产统一接入</span>
            <span className="hero-tag">关系网络主舞台</span>
            <span className="hero-tag">风险任务流程中心</span>
          </div>
        </div>

        <div className="hero-metrics">
          {overviewCards.map((card) => (
            <article key={card.title} className="hero-metric">
              <span>{card.title}</span>
              <strong>{card.value}</strong>
              <p>{card.description}</p>
            </article>
          ))}
        </div>
      </section>

      <div className="dashboard-layout">
        <aside className="sidebar">
          <DataUpload selectedDatasetId={selectedDatasetId} onSelect={handleDatasetSelect} />
          <PipelinePanel
            datasetId={selectedDatasetId}
            onFocusNode={setActiveTimelineNodeId}
            onInferenceComplete={() => setGraphRefreshKey((value) => value + 1)}
          />
        </aside>

        <section className="workspace">
          <GraphWorkspace
            datasetId={selectedDatasetId}
            refreshKey={graphRefreshKey}
            highlightedNodeId={highlightedNodeId}
            timelineNodeId={activeTimelineNodeId}
          />
          <InferenceResults
            datasetId={selectedDatasetId}
            refreshKey={graphRefreshKey}
            onNodeFocus={setHighlightedNodeId}
          />
        </section>
      </div>
    </main>
  );
}
