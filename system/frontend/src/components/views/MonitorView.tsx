import { useEffect, useMemo, useState } from "react";
import { AppPage } from "../../App";
import {
  fetchTimeline,
  InferenceResultItem,
  listDatasets,
  listInferenceResults,
  ProcessingEventItem,
  TaskTimelineResponse
} from "../../services/api";

type OperationStep = { title: string; detail: string };

type Props = {
  currentNetwork: string;
  selectedDatasetId: number | null;
  hasNetwork: boolean;
  operationFlow: OperationStep[];
  onOpenPage: (page: AppPage) => void;
};

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

function statusLabel(status?: string) {
  const map: Record<string, string> = {
    pending: "待处理",
    running: "进行中",
    completed: "已完成",
    failed: "失败"
  };
  return map[status ?? ""] ?? "未开始";
}

function riskTone(score: number) {
  if (score >= 0.75) return "danger";
  if (score >= 0.55) return "warning";
  return "success";
}

function riskLabel(score: number, label: string) {
  if (label === "suspicious" || score >= 0.75) return "高风险";
  if (score >= 0.55) return "中风险";
  return "低风险";
}

function formatTime(value?: string) {
  if (!value) return "暂无记录";
  return value.replace("T", " ").slice(0, 19);
}

export function MonitorView({ currentNetwork, selectedDatasetId, hasNetwork, operationFlow, onOpenPage }: Props) {
  const [networkCount, setNetworkCount] = useState(0);
  const [timeline, setTimeline] = useState<TaskTimelineResponse | null>(null);
  const [results, setResults] = useState<InferenceResultItem[]>([]);
  const [loadMessage, setLoadMessage] = useState("");

  const nextAction = hasNetwork
    ? { title: "进入智能研判", detail: "网络已就绪，开始识别。", page: "analysis" as AppPage }
    : { title: "接入业务网络", detail: "先上传业务记录。", page: "access" as AppPage };
  const actionPages: AppPage[] = ["access", "network", "analysis", "cases"];
  const metricTones = ["network", "risk", "task", "health"];
  const suspiciousCount = results.filter((item) => item.risk_label === "suspicious" || item.risk_score >= 0.75).length;
  const latestEvent = timeline?.events?.[timeline.events.length - 1];
  const progress = timeline?.task ? `${Math.round(timeline.task.progress * 100)}%` : hasNetwork ? "0%" : "0%";
  const statusText = !hasNetwork ? "待接入" : suspiciousCount ? "存在风险" : timeline?.task?.status === "completed" ? "已完成" : "待研判";
  const statusTone = !hasNetwork ? "pending" : suspiciousCount ? "danger" : "health";

  const metrics = useMemo(
    () => [
      { label: "接入网络数", value: `${networkCount} 个`, detail: networkCount ? "已接入" : "暂无网络" },
      { label: "待复核风险对象", value: `${suspiciousCount} 个`, detail: results.length ? "待处理" : "待生成" },
      { label: "最近研判状态", value: statusLabel(timeline?.task?.status), detail: timeline?.task ? `${progress} 完成` : "暂无任务" },
      { label: "系统可用状态", value: "正常", detail: loadMessage || "连接正常" }
    ],
    [loadMessage, networkCount, progress, results.length, suspiciousCount, timeline]
  );

  useEffect(() => {
    let canceled = false;
    listDatasets()
      .then((items) => {
        if (!canceled) setNetworkCount(items.length);
      })
      .catch((error) => {
        if (!canceled) setLoadMessage(error instanceof Error ? error.message : "网络统计加载失败");
      });
    return () => {
      canceled = true;
    };
  }, []);

  useEffect(() => {
    let canceled = false;
    setTimeline(null);
    setResults([]);
    if (!selectedDatasetId) return;
    Promise.all([
      fetchTimeline(selectedDatasetId).catch(() => null),
      listInferenceResults(selectedDatasetId).catch(() => [])
    ]).then(([timelineResult, resultItems]) => {
      if (canceled) return;
      setTimeline(timelineResult);
      setResults(resultItems);
    });
    return () => {
      canceled = true;
    };
  }, [selectedDatasetId]);

  return (
    <div className="monitor-workspace">
      <section className="risk-workbench-hero">
        <div>
          <p className="eyebrow">Risk Operations</p>
          <h2>风险工作台</h2>
          <p>风险态势与处置入口。</p>
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
            <strong>{formatTime(latestEvent?.created_at)}</strong>
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
            <strong>{progress}</strong>
          </div>
        </div>
      </section>

      <section className="risk-kpi-grid" aria-label="风险工作台指标">
        {metrics.map((card, index) => (
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
          {results.length ? (
            <div className="risk-object-table">
              <div className="risk-object-table__head">
                <span>风险对象</span>
                <span>风险等级</span>
                <span>风险分数</span>
                <span>处理状态</span>
              </div>
              {results.slice(0, 5).map((item) => {
                const tone = riskTone(item.risk_score);
                return (
                  <button key={item.node_id} type="button" className="risk-object-row" onClick={() => onOpenPage("cases")}>
                    <span className="risk-object-profile">
                      <span className={`risk-avatar risk-avatar--${tone}`} aria-hidden="true" />
                      <span>
                        <strong>{item.display_name}</strong>
                        <small>{item.region || "人员对象"}</small>
                      </span>
                    </span>
                    <span className={`risk-badge risk-badge--${tone}`}>{riskLabel(item.risk_score, item.risk_label)}</span>
                    <span>{item.risk_score.toFixed(3)}</span>
                    <span className="risk-state risk-state--waiting">待复核</span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="empty-network-state">
              <strong>暂无待处理风险</strong>
              <span>{hasNetwork ? "研判后显示风险对象。" : "请先接入业务网络。"}</span>
            </div>
          )}
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
        <TimelinePanel
          title="最近操作记录"
          eyebrow="Operation Logs"
          events={timeline?.events ?? []}
          emptyText={hasNetwork ? "暂无操作记录，启动处理任务后自动生成。" : "接入业务网络后显示操作记录。"}
        />
        <TimelinePanel
          title="最近分析记录"
          eyebrow="Analysis Logs"
          events={(timeline?.events ?? []).filter((item) => item.stage === "inference" || item.stage === "feature")}
          emptyText={hasNetwork ? "任务启动后生成。" : "接入业务网络后显示记录。"}
          action={() => onOpenPage("analysis")}
        />
      </section>
    </div>
  );
}

function TimelinePanel({
  title,
  eyebrow,
  events,
  emptyText,
  action
}: {
  title: string;
  eyebrow: string;
  events: ProcessingEventItem[];
  emptyText: string;
  action?: () => void;
}) {
  const displayEvents = [...events].slice(-4).reverse();
  return (
    <article className="risk-board-panel">
      <div className="risk-board-heading">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
        </div>
        {action ? <button type="button" className="text-link-button" onClick={action}>查看全部</button> : null}
      </div>
      {displayEvents.length ? (
        <div className="risk-mini-table">
          {displayEvents.map((event) => (
            <div key={event.id}>
              <span>{formatTime(event.created_at)}</span>
              <span>{event.stage}</span>
              <strong>{event.title}</strong>
              <em>{Math.round(event.progress * 100)}%</em>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-network-state compact-empty-state">
          <strong>暂无记录</strong>
          <span>{emptyText}</span>
        </div>
      )}
    </article>
  );
}
