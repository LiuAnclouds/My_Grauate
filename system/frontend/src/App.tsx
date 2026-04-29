import { ReactNode, useMemo, useState } from "react";
import { AuthResponse } from "./services/api";
import { AuthPanel } from "./components/AuthPanel";
import { DataUpload } from "./components/DataUpload";
import { GraphWorkspace } from "./components/GraphWorkspace";
import { InferenceResults } from "./components/InferenceResults";
import { PipelinePanel } from "./components/PipelinePanel";

type AppPage = "overview" | "data" | "graph" | "pipeline" | "ledger" | "settings";

const navItems: Array<{ key: AppPage; label: string; eyebrow: string }> = [
  { key: "overview", label: "工作台", eyebrow: "Overview" },
  { key: "data", label: "数据资产", eyebrow: "Data" },
  { key: "graph", label: "关系图谱", eyebrow: "Graph" },
  { key: "pipeline", label: "分析流程", eyebrow: "Pipeline" },
  { key: "ledger", label: "风险台账", eyebrow: "Ledger" },
  { key: "settings", label: "系统设置", eyebrow: "Settings" }
];

const workflowSteps = [
  {
    title: "数据接入",
    detail: "上传 CSV 或加载平台样本，完成字段识别、人物信息生成与数据入库。"
  },
  {
    title: "关系建模",
    detail: "将 source、target、金额、时间等字段组织为可交互的关系网络。"
  },
  {
    title: "特征处理",
    detail: "将身份展示字段与模型特征字段分离，生成推理所需的节点输入。"
  },
  {
    title: "模型推理",
    detail: "复用 Full GNN 权重，对节点及邻域上下文进行风险识别。"
  },
  {
    title: "结果台账",
    detail: "沉淀高风险对象、画像信息、风险分数与解释线索。"
  }
];

const dashboardCards = [
  {
    label: "当前数据集",
    value: "按需选择",
    detail: "从数据资产页接入或切换分析对象。"
  },
  {
    label: "分析链路",
    value: "5 步闭环",
    detail: "接入、建图、特征、推理、台账分页面承载。"
  },
  {
    label: "模型模式",
    value: "纯 GNN 推理",
    detail: "系统阶段不使用标签，只复用训练好的权重。"
  },
  {
    label: "展示目标",
    value: "异常对象定位",
    detail: "面向关系网络中的高风险节点发现。"
  }
];

const settingsCards = [
  {
    title: "用户与认证库",
    detail: "保存系统账号、管理员身份、邮箱验证码和登录相关状态。"
  },
  {
    title: "业务分析库",
    detail: "保存数据资产、人物画像、节点特征、关系边、任务事件和推理结果。"
  },
  {
    title: "本地运行配置",
    detail: "邮箱授权码、模型路径等敏感配置仅保存在本地环境文件，不提交到 GitHub。"
  }
];

export default function App() {
  const [session, setSession] = useState<AuthResponse | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [graphRefreshKey, setGraphRefreshKey] = useState(0);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);
  const [activeTimelineNodeId, setActiveTimelineNodeId] = useState<string | null>(null);
  const [activePage, setActivePage] = useState<AppPage>("overview");

  const activeNav = useMemo(() => navItems.find((item) => item.key === activePage) ?? navItems[0], [activePage]);

  function handleDatasetSelect(datasetId: number) {
    setSelectedDatasetId(datasetId);
    setHighlightedNodeId(null);
    setActiveTimelineNodeId(null);
  }

  function openPage(nextPage: AppPage) {
    setActivePage(nextPage);
  }

  if (!session) {
    return <AuthPanel onAuthed={setSession} />;
  }

  return (
    <main className="app-shell enterprise-app-shell">
      <header className="topbar app-topbar">
        <div className="brand-block">
          <p className="eyebrow">StarHubGraph Workspace</p>
          <h1 className="brand-title">星枢反欺诈分析平台</h1>
          <p className="topbar-subtitle">面向数据接入、关系图谱、特征处理、风险推理与结果复核的一体化工作台。</p>
        </div>
        <div className="topbar-actions">
          <span className="dataset-pill">{selectedDatasetId ? `数据集 #${selectedDatasetId}` : "未选择数据集"}</span>
          <span className="account-pill">{session.email}</span>
          {session.is_admin ? <span className="role-pill">管理员</span> : <span className="role-pill muted">分析账号</span>}
          <button className="ghost-button" onClick={() => setSession(null)}>
            退出登录
          </button>
        </div>
      </header>

      <div className="system-frame">
        <aside className="system-nav" aria-label="系统功能导航">
          <div className="nav-card">
            <span className="nav-mark">SG</span>
            <div>
              <strong>功能导航</strong>
              <small>按分析流程组织页面</small>
            </div>
          </div>

          <nav className="nav-list">
            {navItems.map((item, index) => (
              <button
                key={item.key}
                className={item.key === activePage ? "nav-button active" : "nav-button"}
                onClick={() => openPage(item.key)}
                aria-current={item.key === activePage ? "page" : undefined}
                type="button"
              >
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{item.label}</strong>
                <small>{item.eyebrow}</small>
              </button>
            ))}
          </nav>

          <div className="nav-hint">
            <strong>推荐流程</strong>
            <span>先接入数据，再进入图谱和分析流程，最后查看风险台账。</span>
          </div>
        </aside>

        <section className="system-main">
          <div className="page-toolbar">
            <div>
              <p className="eyebrow">{activeNav.eyebrow}</p>
              <h2>{activeNav.label}</h2>
            </div>
            <div className="page-actions">
              <button className="ghost-inline" onClick={() => openPage("data")} type="button">
                接入数据
              </button>
              <button className="secondary" onClick={() => openPage("pipeline")} type="button">
                启动分析
              </button>
            </div>
          </div>

          {activePage === "overview" ? (
            <OverviewPage onOpenPage={openPage} selectedDatasetId={selectedDatasetId} />
          ) : null}

          {activePage === "data" ? (
            <PageSurface
              title="数据资产中心"
              description="负责 CSV 上传、内置样本接入、字段解析和当前分析资产选择。人物信息、关系字段和模型特征会在这里形成清晰边界。"
            >
              <DataUpload selectedDatasetId={selectedDatasetId} onSelect={handleDatasetSelect} />
            </PageSurface>
          ) : null}

          {activePage === "graph" ? (
            <PageSurface
              title="关系图谱主舞台"
              description="集中展示对象关联结构、边方向、风险着色和任务过程中的节点联动。"
            >
              <GraphWorkspace
                datasetId={selectedDatasetId}
                refreshKey={graphRefreshKey}
                highlightedNodeId={highlightedNodeId}
                timelineNodeId={activeTimelineNodeId}
              />
            </PageSurface>
          ) : null}

          {activePage === "pipeline" ? (
            <PageSurface
              title="特征处理与风险推理"
              description="展示从数据装载、关系组织、特征准备到模型推理的全过程，并联动图谱中的当前节点。"
            >
              <div className="split-page">
                <PipelinePanel
                  datasetId={selectedDatasetId}
                  onFocusNode={setActiveTimelineNodeId}
                  onInferenceComplete={() => setGraphRefreshKey((value) => value + 1)}
                />
                <GraphWorkspace
                  datasetId={selectedDatasetId}
                  refreshKey={graphRefreshKey}
                  highlightedNodeId={highlightedNodeId}
                  timelineNodeId={activeTimelineNodeId}
                />
              </div>
            </PageSurface>
          ) : null}

          {activePage === "ledger" ? (
            <PageSurface
              title="风险结果台账"
              description="面向最终复核输出，展示异常对象、身份画像、风险分数、解释线索和关联对象。"
            >
              <InferenceResults datasetId={selectedDatasetId} refreshKey={graphRefreshKey} onNodeFocus={setHighlightedNodeId} />
            </PageSurface>
          ) : null}

          {activePage === "settings" ? <SettingsPage /> : null}
        </section>
      </div>
    </main>
  );
}

function PageSurface({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className="page-surface">
      <div className="page-hero">
        <div>
          <p className="eyebrow">Workspace Module</p>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </div>
      <div className="page-content">{children}</div>
    </section>
  );
}

function OverviewPage({ onOpenPage, selectedDatasetId }: { onOpenPage: (page: AppPage) => void; selectedDatasetId: number | null }) {
  return (
    <section className="overview-page">
      <div className="overview-hero">
        <div>
          <p className="eyebrow">Analysis Command Center</p>
          <h2>把接入、建图、特征处理、推理和结果复核拆成清晰工作区。</h2>
          <p>系统面向演示和落地使用，首页只保留状态总览和快捷入口，具体操作放到对应页面中。</p>
        </div>
        <div className="overview-current">
          <span>当前分析资产</span>
          <strong>{selectedDatasetId ? `数据集 #${selectedDatasetId}` : "尚未选择"}</strong>
          <small>{selectedDatasetId ? "可以进入图谱、分析流程或风险台账继续操作。" : "请先在数据资产页上传 CSV 或加载平台样本。"}</small>
        </div>
      </div>

      <div className="dashboard-card-grid">
        {dashboardCards.map((card) => (
          <article key={card.label} className="dashboard-card">
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <p>{card.detail}</p>
          </article>
        ))}
      </div>

      <div className="workflow-board">
        <div className="board-heading">
          <div>
            <p className="eyebrow">Process Map</p>
            <h3>系统主流程</h3>
          </div>
          <button className="primary" onClick={() => onOpenPage("data")} type="button">
            开始接入数据
          </button>
        </div>
        <div className="workflow-steps">
          {workflowSteps.map((step, index) => (
            <button
              key={step.title}
              className="workflow-step"
              onClick={() => onOpenPage(index === 0 ? "data" : index === 1 ? "graph" : index <= 3 ? "pipeline" : "ledger")}
              type="button"
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{step.title}</strong>
              <small>{step.detail}</small>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

function SettingsPage() {
  return (
    <section className="settings-page">
      <div className="page-hero">
        <div>
          <p className="eyebrow">Runtime & Storage</p>
          <h2>系统设置</h2>
          <p>这里先展示运行配置和数据库分层规划，后续可以继续接入管理员配置表单。</p>
        </div>
      </div>
      <div className="settings-grid">
        {settingsCards.map((card) => (
          <article key={card.title} className="settings-card">
            <span className="status-light" />
            <strong>{card.title}</strong>
            <p>{card.detail}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
