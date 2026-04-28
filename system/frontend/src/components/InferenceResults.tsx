const demoRows = [
  { id: "node_1024", name: "Li Wei", score: "0.91", reason: "高频交易 + 异常邻居聚合" },
  { id: "node_2048", name: "Wang Min", score: "0.87", reason: "时间窗口漂移 + 资金流入异常" },
  { id: "node_4096", name: "Zhang Tao", score: "0.82", reason: "二跳邻居风险集中" }
];

export function InferenceResults() {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Inference Output</p>
          <h2>异常节点结果</h2>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>节点 ID</th>
            <th>用户</th>
            <th>异常概率</th>
            <th>推理依据</th>
          </tr>
        </thead>
        <tbody>
          {demoRows.map((row) => (
            <tr key={row.id}>
              <td>{row.id}</td>
              <td>{row.name}</td>
              <td>{row.score}</td>
              <td>{row.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">下一阶段接入 full 模型权重后，这里会展示真实推理结果。</p>
    </section>
  );
}
