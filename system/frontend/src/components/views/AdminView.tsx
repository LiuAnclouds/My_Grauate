import { useEffect, useMemo, useState } from "react";
import { AnalystSummary, AuthResponse, listAnalysts } from "../../services/api";

type Props = {
  session: AuthResponse;
  onLogout: () => void;
};

function formatDate(value?: string | null) {
  if (!value) return "暂无记录";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

export function AdminView({ session, onLogout }: Props) {
  const [analysts, setAnalysts] = useState<AnalystSummary[]>([]);
  const [selectedAnalystId, setSelectedAnalystId] = useState<number | null>(null);
  const [message, setMessage] = useState("正在读取分析员信息...");

  const selectedAnalyst = useMemo(() => {
    return analysts.find((item) => item.user_id === selectedAnalystId) ?? analysts[0] ?? null;
  }, [analysts, selectedAnalystId]);

  const totals = useMemo(() => {
    return analysts.reduce(
      (acc, item) => ({
        analysts: acc.analysts + 1,
        networks: acc.networks + item.network_count,
        nodes: acc.nodes + item.node_count,
        edges: acc.edges + item.edge_count
      }),
      { analysts: 0, networks: 0, nodes: 0, edges: 0 }
    );
  }, [analysts]);

  async function refresh() {
    try {
      const items = await listAnalysts(session.user_id);
      setAnalysts(items);
      setSelectedAnalystId((current) => current ?? items[0]?.user_id ?? null);
      setMessage(items.length ? "分析员信息已同步。" : "当前还没有分析员账号。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "分析员信息读取失败。");
    }
  }

  useEffect(() => {
    refresh();
  }, [session.user_id]);

  return (
    <main className="app-shell enterprise-app-shell nextgen-shell admin-portal-shell">
      <section className="admin-portal-frame">
        <header className="admin-portal-header glass-panel">
          <div>
            <p className="eyebrow">Administrator</p>
            <h1>管理员控制台</h1>
            <span>查看分析员账号与其接入的业务网络。</span>
          </div>
          <div className="admin-portal-user">
            <span className="account-pill">{session.email}</span>
            <span className="role-pill">管理员</span>
            <button className="ghost-button app-logout-button" type="button" onClick={onLogout}>
              退出
            </button>
          </div>
        </header>

        <section className="admin-kpi-grid">
          <article className="admin-rule-card">
            <span className="admin-rule-card__dot" />
            <strong>{totals.analysts}</strong>
            <p>分析员账号</p>
          </article>
          <article className="admin-rule-card">
            <span className="admin-rule-card__dot" />
            <strong>{totals.networks}</strong>
            <p>接入网络</p>
          </article>
          <article className="admin-rule-card">
            <span className="admin-rule-card__dot" />
            <strong>{totals.nodes}</strong>
            <p>对象总量</p>
          </article>
          <article className="admin-rule-card">
            <span className="admin-rule-card__dot" />
            <strong>{totals.edges}</strong>
            <p>关系总量</p>
          </article>
        </section>

        <section className="admin-management-grid">
          <div className="admin-directory-panel app-panel">
            <div className="app-section-heading">
              <div>
                <p className="eyebrow">Analysts</p>
                <h2>分析员列表</h2>
                <p>{message}</p>
              </div>
              <button className="network-row-action" type="button" onClick={refresh}>
                刷新
              </button>
            </div>

            <div className="admin-analyst-list">
              {analysts.map((analyst) => (
                <button
                  key={analyst.user_id}
                  type="button"
                  className={analyst.user_id === selectedAnalyst?.user_id ? "admin-analyst-row active" : "admin-analyst-row"}
                  onClick={() => setSelectedAnalystId(analyst.user_id)}
                >
                  <span>
                    <strong>{analyst.email}</strong>
                    <small>注册时间 {formatDate(analyst.created_at)}</small>
                  </span>
                  <em>{analyst.network_count} 个网络</em>
                </button>
              ))}
              {!analysts.length ? (
                <div className="empty-network-state">
                  <strong>暂无分析员</strong>
                  <span>分析员注册后会显示在这里。</span>
                </div>
              ) : null}
            </div>
          </div>

          <div className="admin-detail-panel app-panel">
            {selectedAnalyst ? (
              <>
                <div className="app-section-heading">
                  <div>
                    <p className="eyebrow">Account Detail</p>
                    <h2>{selectedAnalyst.email}</h2>
                    <p>最近接入：{selectedAnalyst.latest_network_name ?? "暂无业务网络"}</p>
                  </div>
                  <span className="app-inline-badge">{selectedAnalyst.is_active ? "正常" : "停用"}</span>
                </div>

                <div className="admin-note-panel">
                  <div>
                    <span>账号角色</span>
                    <strong>分析员</strong>
                  </div>
                  <div>
                    <span>网络数量</span>
                    <strong>{selectedAnalyst.network_count}</strong>
                  </div>
                  <div>
                    <span>最近接入</span>
                    <strong>{formatDate(selectedAnalyst.latest_network_at)}</strong>
                  </div>
                </div>

                <div className="admin-network-list">
                  {selectedAnalyst.networks.map((network) => (
                    <article key={network.id} className="network-row admin-network-row">
                      <div className="network-row-main">
                        <span className="network-id"><b>网络ID</b>{network.business_id}</span>
                        <strong>{network.business_name}</strong>
                        <small>{formatDate(network.created_at)}</small>
                      </div>
                      <div className="network-row-stats">
                        <span className="network-status-lamp status-normalized">{network.status}</span>
                        <div className="network-row-metrics">
                          <strong>{network.node_count} 人</strong>
                          <small>{network.edge_count} 条关系</small>
                        </div>
                      </div>
                    </article>
                  ))}
                  {!selectedAnalyst.networks.length ? (
                    <div className="empty-network-state">
                      <strong>尚未接入业务网络</strong>
                      <span>该分析员上传业务 CSV 后会在这里生成记录。</span>
                    </div>
                  ) : null}
                </div>
              </>
            ) : (
              <div className="empty-network-state">
                <strong>请选择分析员</strong>
                <span>左侧列表会显示已注册的分析员账号。</span>
              </div>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}
