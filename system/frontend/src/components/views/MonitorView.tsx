import { AppPage } from "../../App";

type MonitorCard = { label: string; value: string; detail: string };
type OperationStep = { title: string; detail: string };

type Props = {
  currentNetwork: string;
  hasNetwork: boolean;
  monitorCards: MonitorCard[];
  operationFlow: OperationStep[];
  onOpenPage: (page: AppPage) => void;
};

export function MonitorView({ currentNetwork, hasNetwork, monitorCards, operationFlow, onOpenPage }: Props) {
  const nextAction = hasNetwork
    ? { title: "进入智能研判", detail: "当前网络已就绪，可以直接进入任务编排与风险识别。", page: "analysis" as AppPage }
    : { title: "接入业务网络", detail: "先选择内置网络或导入业务文件，建立后续分析基础。", page: "access" as AppPage };

  return (
    <div className="monitor-workspace">
      <section className="ops-overview-hero">
        <div className="ops-overview-hero__copy">
          <p className="eyebrow">Operational Overview</p>
          <h3>以业务流程驱动的反欺诈分析工作台</h3>
          <p>
            从业务网络接入、关系结构观察、智能研判到风险台账复核，将图谱分析与模型推理组织为统一的系统作业流。
          </p>
          <div className="ops-overview-hero__actions">
            <button className="primary" type="button" onClick={() => onOpenPage("access")}>
              进入业务网络
            </button>
            <button className="secondary" type="button" onClick={() => onOpenPage(nextAction.page)}>
              {nextAction.title}
            </button>
          </div>
        </div>

        <div className="ops-overview-status-card">
          <div className="ops-overview-status-card__header">
            <span>工作台状态</span>
            <em>{hasNetwork ? "已就绪" : "待接入"}</em>
          </div>
          <strong>{currentNetwork}</strong>
          <p>{hasNetwork ? "当前已具备进入图谱浏览、智能研判与风险复核的分析前置条件。" : "请先完成业务网络选择或导入，系统会据此装载关系结构与后续任务数据。"}</p>
          <div className="ops-overview-status-card__tiles">
            <div>
              <span>执行模式</span>
              <strong>图谱驱动</strong>
            </div>
            <div>
              <span>推理方式</span>
              <strong>纯 GNN</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="ops-overview-strip" aria-label="系统概览指标">
        {monitorCards.map((card) => (
          <article key={card.label} className="ops-kpi-card">
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <p>{card.detail}</p>
          </article>
        ))}
      </section>

      <section className="ops-flow-board">
        <div className="ops-section-heading">
          <div>
            <p className="eyebrow">Workflow</p>
            <h3>业务处置流程</h3>
          </div>
          <span>围绕接入 → 建图 → 研判 → 复核构建统一操作路径</span>
        </div>
        <div className="ops-flow-grid">
          {operationFlow.map((step, index) => {
            const target = index === 0 ? "access" : index === 1 ? "network" : index === 2 ? "analysis" : "cases";
            return (
              <button key={step.title} type="button" className="ops-flow-card" onClick={() => onOpenPage(target)}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{step.title}</strong>
                <p>{step.detail}</p>
              </button>
            );
          })}
        </div>
      </section>

      <section className="ops-recommendation-card">
        <div>
          <p className="eyebrow">Next Action</p>
          <h3>{nextAction.title}</h3>
          <p>{nextAction.detail}</p>
        </div>
        <button className="primary" type="button" onClick={() => onOpenPage(nextAction.page)}>
          立即处理
        </button>
      </section>
    </div>
  );
}
