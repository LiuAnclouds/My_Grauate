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

const riskObjects = [
  { name: "星枢零售科技有限公司", type: "企业", level: "高风险", count: "128", status: "待复核", tone: "danger" },
  { name: "李某某", type: "自然人", level: "高风险", count: "86", status: "待复核", tone: "danger" },
  { name: "尾号 8888 的银行卡", type: "银行卡", level: "中风险", count: "54", status: "处理中", tone: "warning" },
  { name: "139****5678", type: "手机号", level: "中风险", count: "42", status: "待复核", tone: "warning" },
  { name: "上海荣景贸易公司", type: "企业", level: "低风险", count: "18", status: "待复核", tone: "success" }
];

const operationRecords = [
  { time: "2025-05-20 10:15:32", operator: "张伟", action: "启动智能研判", result: "成功" },
  { time: "2025-05-20 09:42:18", operator: "王敏", action: "标记风险对象", result: "成功" },
  { time: "2025-05-20 09:21:07", operator: "李强", action: "调整风险等级", result: "成功" },
  { time: "2025-05-20 08:55:14", operator: "张伟", action: "查看关系图谱", result: "成功" }
];

const analysisRecords = [
  { time: "2025-05-20 10:30:45", scope: "星枢金融业务网络", high: "36", medium: "89", analyst: "张伟" },
  { time: "2025-05-19 22:10:11", scope: "星枢金融业务网络", high: "28", medium: "76", analyst: "张伟" },
  { time: "2025-05-19 16:05:33", scope: "星枢企业业务网络", high: "31", medium: "92", analyst: "王敏" },
  { time: "2025-05-19 10:20:54", scope: "星枢金融业务网络", high: "25", medium: "68", analyst: "系统" }
];

function WorkbenchIcon({ tone }: { tone: string }) {
  return (
    <span className={`workbench-icon workbench-icon--${tone}`} aria-hidden="true">
      <svg viewBox="0 0 24 24">
        {tone === "network" ? (
          <>
            <circle cx="6" cy="7" r="2.7" />
            <circle cx="18" cy="7" r="2.7" />
            <circle cx="12" cy="17" r="3.1" />
            <path d="m8.1 8.8 2.6 5.4" />
            <path d="m15.8 8.8-2.6 5.4" />
          </>
        ) : tone === "risk" ? (
          <>
            <circle cx="12" cy="7.2" r="3.2" />
            <path d="M5.8 19c.8-3.9 3-5.8 6.2-5.8s5.4 1.9 6.2 5.8" />
          </>
        ) : tone === "task" ? (
          <>
            <path d="M7 4.5h10v15H7z" />
            <path d="m9.5 9.5 1.4 1.4 3.6-3.8" />
            <path d="M9.5 15h5" />
          </>
        ) : (
          <>
            <path d="M12 4.5 18 7v5.3c0 3.5-2.4 6.2-6 7.2-3.6-1-6-3.7-6-7.2V7z" />
            <path d="m9.2 12.1 1.9 1.9 3.8-4" />
          </>
        )}
      </svg>
    </span>
  );
}

export function MonitorView({ currentNetwork, hasNetwork, monitorCards, operationFlow, onOpenPage }: Props) {
  const nextAction = hasNetwork
    ? { title: "进入智能研判", detail: "当前网络已就绪，可以直接进入任务编排与风险识别。", page: "analysis" as AppPage }
    : { title: "接入业务网络", detail: "先选择内置网络或导入业务文件，建立后续分析基础。", page: "access" as AppPage };
  const statusText = hasNetwork ? "高风险" : "待接入";
  const statusTone = hasNetwork ? "danger" : "pending";
  const metricTones = ["network", "risk", "task", "health"];
  const actionPages: AppPage[] = ["access", "network", "analysis", "cases"];

  return (
    <div className="monitor-workspace">
      <section className="risk-workbench-hero">
        <div>
          <p className="eyebrow">Risk Operations</p>
          <h2>风险工作台</h2>
          <p>风险态势全局感知，风险对象高效处置。</p>
        </div>
        <button className="secondary risk-workbench-hero__button" type="button" onClick={() => onOpenPage(nextAction.page)}>
          {nextAction.title}
        </button>
      </section>

      <section className="risk-context-strip" aria-label="当前业务网络状态">
        <div className="risk-context-item">
          <WorkbenchIcon tone="network" />
          <div>
            <span>网络名称</span>
            <strong>{currentNetwork}</strong>
          </div>
        </div>
        <div className="risk-context-item">
          <WorkbenchIcon tone="task" />
          <div>
            <span>最近分析时间</span>
            <strong>{hasNetwork ? "2025-05-20 10:30:45" : "等待接入"}</strong>
          </div>
        </div>
        <div className="risk-context-item">
          <WorkbenchIcon tone={statusTone} />
          <div>
            <span>风险状态</span>
            <strong className={`risk-context-status risk-context-status--${statusTone}`}>{statusText}</strong>
          </div>
        </div>
        <div className="risk-context-item">
          <WorkbenchIcon tone="health" />
          <div>
            <span>处理进度</span>
            <strong>{hasNetwork ? "68%" : "0%"}</strong>
          </div>
        </div>
      </section>

      <section className="risk-kpi-grid" aria-label="风险工作台指标">
        {monitorCards.map((card, index) => (
          <article key={card.label} className="risk-kpi-card">
            <WorkbenchIcon tone={metricTones[index] ?? "network"} />
            <div>
              <span>{card.label}</span>
              <strong>{card.value}</strong>
              <p>{card.detail}</p>
            </div>
          </article>
        ))}
      </section>

      <section className="risk-workbench-main">
        <article className="risk-board-panel">
          <div className="risk-board-heading">
            <div>
              <p className="eyebrow">Pending Risks</p>
              <h3>待处理风险</h3>
            </div>
            <button type="button" className="text-link-button" onClick={() => onOpenPage("cases")}>查看全部</button>
          </div>
          <div className="risk-object-table">
            <div className="risk-object-table__head">
              <span>风险对象</span>
              <span>风险等级</span>
              <span>关联数量</span>
              <span>处理状态</span>
            </div>
            {riskObjects.map((item) => (
              <button key={item.name} type="button" className="risk-object-row" onClick={() => onOpenPage("cases")}>
                <span className="risk-object-profile">
                  <span className={`risk-avatar risk-avatar--${item.tone}`} aria-hidden="true" />
                  <span>
                    <strong>{item.name}</strong>
                    <small>{item.type}</small>
                  </span>
                </span>
                <span className={`risk-badge risk-badge--${item.tone}`}>{item.level}</span>
                <span>{item.count}</span>
                <span className={`risk-state risk-state--${item.status === "处理中" ? "active" : "waiting"}`}>{item.status}</span>
              </button>
            ))}
          </div>
        </article>

        <article className="risk-board-panel">
          <div className="risk-board-heading">
            <div>
              <p className="eyebrow">Next Steps</p>
              <h3>下一步操作</h3>
            </div>
          </div>
          <div className="risk-action-list">
            <button type="button" className="risk-action-card" onClick={() => onOpenPage(nextAction.page)}>
              <WorkbenchIcon tone={hasNetwork ? "task" : "network"} />
              <span>
                <strong>{nextAction.title}</strong>
                <small>{nextAction.detail}</small>
              </span>
            </button>
          {operationFlow.map((step, index) => {
              const target = actionPages[index];
            return (
                <button key={step.title} type="button" className="risk-action-card" onClick={() => onOpenPage(target)}>
                  <WorkbenchIcon tone={metricTones[index] ?? "network"} />
                  <span>
                    <strong>{step.title}</strong>
                    <small>{step.detail}</small>
                  </span>
              </button>
            );
          })}
          </div>
        </article>
      </section>

      <section className="risk-record-grid">
        <article className="risk-board-panel">
          <div className="risk-board-heading">
            <div>
              <p className="eyebrow">Operation Logs</p>
              <h3>最近操作记录</h3>
            </div>
          </div>
          <div className="risk-mini-table">
            {operationRecords.map((record) => (
              <div key={`${record.time}-${record.action}`}>
                <span>{record.time}</span>
                <span>{record.operator}</span>
                <strong>{record.action}</strong>
                <em>{record.result}</em>
              </div>
            ))}
          </div>
        </article>
        <article className="risk-board-panel">
          <div className="risk-board-heading">
            <div>
              <p className="eyebrow">Analysis Logs</p>
              <h3>最近分析记录</h3>
            </div>
            <button type="button" className="text-link-button" onClick={() => onOpenPage("analysis")}>查看全部</button>
          </div>
          <div className="risk-mini-table risk-mini-table--analysis">
            {analysisRecords.map((record) => (
              <div key={`${record.time}-${record.high}`}>
                <span>{record.time}</span>
                <strong>{record.scope}</strong>
                <span>高风险 {record.high} / 中风险 {record.medium}</span>
                <em>{record.analyst}</em>
              </div>
            ))}
          </div>
        </article>
      </section>
    </div>
  );
}
