import { ReactNode, useMemo, useState } from "react";
import { AuthResponse } from "./services/api";
import { AuthPanel } from "./components/AuthPanel";
import { DataUpload } from "./components/DataUpload";
import { GraphWorkspace } from "./components/GraphWorkspace";
import { InferenceResults } from "./components/InferenceResults";
import { PipelinePanel } from "./components/PipelinePanel";

type AppPage = "monitor" | "access" | "network" | "analysis" | "cases" | "admin";

const navItems: Array<{ key: AppPage; label: string; eyebrow: string; description: string }> = [
  { key: "monitor", label: "风险总览", eyebrow: "Monitor", description: "查看业务网络状态与待处理风险" },
  { key: "access", label: "业务接入", eyebrow: "Access", description: "接入默认网络或上传业务文件" },
  { key: "network", label: "关系网络", eyebrow: "Network", description: "查看对象与交易关系结构" },
  { key: "analysis", label: "智能研判", eyebrow: "Analysis", description: "执行特征处理与风险推理" },
  { key: "cases", label: "风险名单", eyebrow: "Cases", description: "复核高风险对象与解释线索" },
  { key: "admin", label: "系统管理", eyebrow: "Admin", description: "查看运行状态与配置边界" }
];

const monitorCards = [
  { label: "风险识别模式", value: "纯 GNN", detail: "推理阶段不使用标签，只复用训练完成的模型权重。" },
  { label: "默认业务网络", value: "3 套", detail: "覆盖零售交易、支付链路与综合关系三类演示场景。" },
  { label: "处置闭环", value: "5 步", detail: "接入、建图、处理、研判、名单复核形成业务闭环。" },
  { label: "当前状态", value: "可运行", detail: "后端服务、邮件验证和前端系统均已接入。" }
];

const operationFlow = [
  { title: "选择业务网络", detail: "从默认网络进入，也可以导入业务 CSV 文件。" },
  { title: "查看关系结构", detail: "在关系网络中定位对象、交易方向和异常关联。" },
  { title: "启动智能研判", detail: "执行特征整理、邻域聚合和模型风险识别。" },
  { title: "复核风险名单", detail: "查看高风险对象画像、分数、原因和关联对象。" }
];

const adminCards = [
  { title: "账号认证", detail: "用户账号、密码哈希和邮箱验证码独立保存，不和业务网络内容混用。" },
  { title: "业务网络", detail: "对象画像、关系边、模型输入和推理结果按业务分析流程组织。" },
  { title: "敏感配置", detail: "邮箱授权码、模型路径等运行配置只保存在本地环境文件。" }
];

export default function App() {
  const [session, setSession] = useState<AuthResponse | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [selectedNetworkName, setSelectedNetworkName] = useState("");
  const [graphRefreshKey, setGraphRefreshKey] = useState(0);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);
  const [activeTimelineNodeId, setActiveTimelineNodeId] = useState<string | null>(null);
  const [activePage, setActivePage] = useState<AppPage>("monitor");

  const activeNav = useMemo(() => navItems.find((item) => item.key === activePage) ?? navItems[0], [activePage]);
  const currentNetwork = selectedNetworkName || (selectedDatasetId ? "已接入业务网络" : "尚未接入业务网络");

  function handleBusinessSelect(datasetId: number, networkName?: string) {
    setSelectedDatasetId(datasetId);
    setSelectedNetworkName(networkName ?? "已接入业务网络");
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
          <p className="eyebrow">StarHubGraph RiskOps</p>
          <h1 className="brand-title">星枢反欺诈分析平台</h1>
          <p className="topbar-subtitle">面向业务人员的关系网络风控系统，聚合业务接入、关系分析、智能研判和风险名单复核。</p>
        </div>
        <div className="topbar-actions">
          <span className="dataset-pill">{currentNetwork}</span>
          <span className="account-pill">{session.email}</span>
          {session.is_admin ? <span className="role-pill">管理员</span> : <span className="role-pill muted">分析员</span>}
          <button className="ghost-button" onClick={() => setSession(null)}>
            退出
          </button>
        </div>
      </header>

      <div className="system-frame riskops-frame">
        <aside className="system-nav riskops-nav" aria-label="系统功能导航">
          <div className="nav-card compact-nav-card">
            <span className="nav-mark">星</span>
            <div>
              <strong>风险运营台</strong>
              <small>按业务处置流程组织</small>
            </div>
          </div>

          <nav className="nav-list">
            {navItems.map((item) => (
              <button
                key={item.key}
                className={item.key === activePage ? "nav-button active" : "nav-button"}
                onClick={() => openPage(item.key)}
                aria-current={item.key === activePage ? "page" : undefined}
                type="button"
              >
                <span>{item.eyebrow}</span>
                <strong>{item.label}</strong>
                <small>{item.description}</small>
              </button>
            ))}
          </nav>

          <div className="nav-hint">
            <strong>当前业务网络</strong>
            <span>{currentNetwork}</span>
          </div>
        </aside>

        <section className="system-main riskops-main">
          <div className="page-toolbar riskops-toolbar">
            <div>
              <p className="eyebrow">{activeNav.eyebrow}</p>
              <h2>{activeNav.label}</h2>
              <span>{activeNav.description}</span>
            </div>
            <div className="page-actions">
              <button className="ghost-inline" onClick={() => openPage("access")} type="button">
                选择网络
              </button>
              <button className="secondary" onClick={() => openPage("analysis")} type="button">
                开始研判
              </button>
            </div>
          </div>

          {activePage === "monitor" ? <MonitorPage onOpenPage={openPage} currentNetwork={currentNetwork} hasNetwork={Boolean(selectedDatasetId)} /> : null}

          {activePage === "access" ? (
            <PageSurface title="业务接入" description="选择系统内置业务网络，或导入新的业务文件进入风控分析流程。">
              <DataUpload selectedDatasetId={selectedDatasetId} onSelect={handleBusinessSelect} />
            </PageSurface>
          ) : null}

          {activePage === "network" ? (
            <PageSurface title="关系网络" description="查看对象、交易方向、关联强度和风险着色，定位关键关系链路。">
              <GraphWorkspace
                datasetId={selectedDatasetId}
                refreshKey={graphRefreshKey}
                highlightedNodeId={highlightedNodeId}
                timelineNodeId={activeTimelineNodeId}
              />
            </PageSurface>
          ) : null}

          {activePage === "analysis" ? (
            <PageSurface title="智能研判" description="执行特征整理、邻域聚合和模型推理，并与关系网络中的当前对象联动。">
              <div className="split-page riskops-split">
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

          {activePage === "cases" ? (
            <PageSurface title="风险名单" description="面向业务复核输出异常对象、风险分数、画像信息和研判依据。">
              <InferenceResults datasetId={selectedDatasetId} refreshKey={graphRefreshKey} onNodeFocus={setHighlightedNodeId} />
            </PageSurface>
          ) : null}

          {activePage === "admin" ? <AdminPage /> : null}
        </section>
      </div>
    </main>
  );
}

function PageSurface({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className="page-surface riskops-surface">
      <div className="page-hero riskops-page-hero">
        <div>
          <p className="eyebrow">业务模块</p>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </div>
      <div className="page-content riskops-page-content">{children}</div>
    </section>
  );
}

function MonitorPage({ onOpenPage, currentNetwork, hasNetwork }: { onOpenPage: (page: AppPage) => void; currentNetwork: string; hasNetwork: boolean }) {
  return (
    <section className="monitor-page">
      <div className="monitor-hero">
        <div className="monitor-copy">
          <p className="eyebrow">Risk Operations</p>
          <h2>面向业务风险处置的关系网络分析系统。</h2>
          <p>系统默认提供三套业务网络用于演示和验证，也支持后续扩展新的业务文件接入。业务用户只需要选择网络、启动研判、查看风险名单。</p>
          <div className="monitor-actions">
            <button className="primary" onClick={() => onOpenPage("access")} type="button">
              选择业务网络
            </button>
            <button className="secondary" onClick={() => onOpenPage(hasNetwork ? "analysis" : "access")} type="button">
              {hasNetwork ? "进入智能研判" : "先接入网络"}
            </button>
          </div>
        </div>
        <div className="network-status-card">
          <span>当前业务网络</span>
          <strong>{currentNetwork}</strong>
          <small>{hasNetwork ? "已准备进入关系分析与风险研判。" : "请选择默认业务网络或导入业务文件。"}</small>
        </div>
      </div>

      <div className="dashboard-card-grid riskops-metrics">
        {monitorCards.map((card) => (
          <article key={card.label} className="dashboard-card riskops-metric-card">
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <p>{card.detail}</p>
          </article>
        ))}
      </div>

      <div className="workflow-board riskops-flow-board">
        <div className="board-heading">
          <div>
            <p className="eyebrow">Operational Flow</p>
            <h3>业务使用流程</h3>
          </div>
        </div>
        <div className="workflow-steps riskops-flow">
          {operationFlow.map((step, index) => (
            <button
              key={step.title}
              className="workflow-step riskops-flow-step"
              onClick={() => onOpenPage(index === 0 ? "access" : index === 1 ? "network" : index === 2 ? "analysis" : "cases")}
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

function AdminPage() {
  return (
    <section className="settings-page">
      <div className="page-hero riskops-page-hero">
        <div>
          <p className="eyebrow">System Control</p>
          <h2>系统管理</h2>
          <p>这里保留系统运行状态、数据库边界和本地配置说明，后续可以继续扩展管理员控制项。</p>
        </div>
      </div>
      <div className="settings-grid">
        {adminCards.map((card) => (
          <article key={card.title} className="settings-card riskops-admin-card">
            <span className="status-light" />
            <strong>{card.title}</strong>
            <p>{card.detail}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
