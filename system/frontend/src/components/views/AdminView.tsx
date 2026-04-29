type AdminCard = { title: string; detail: string };

type Props = {
  cards: AdminCard[];
};

export function AdminView({ cards }: Props) {
  return (
    <div className="admin-workspace">
      <section className="admin-governance-card">
        <div>
          <p className="eyebrow">Governance</p>
          <h3>系统边界与运行说明</h3>
          <p>当前页面用于说明认证数据、业务网络、敏感配置和本地运行边界，便于后续扩展治理类能力。</p>
        </div>
        <span className="app-inline-badge">本地环境</span>
      </section>

      <section className="admin-grid">
        {cards.map((card) => (
          <article key={card.title} className="admin-rule-card">
            <span className="admin-rule-card__dot" aria-hidden="true" />
            <strong>{card.title}</strong>
            <p>{card.detail}</p>
          </article>
        ))}
      </section>

      <section className="admin-note-panel">
        <div>
          <span>数据边界</span>
          <strong>认证信息、业务网络与本地配置分别管理</strong>
        </div>
        <div>
          <span>扩展方向</span>
          <strong>后续可继续挂接系统状态、运行日志与管理员动作面板</strong>
        </div>
      </section>
    </div>
  );
}
