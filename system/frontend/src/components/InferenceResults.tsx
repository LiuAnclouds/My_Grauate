import { useEffect, useState } from "react";
import { InferenceResultItem, listInferenceResults } from "../services/api";

type Props = {
  datasetId: number | null;
  refreshKey: number;
  onNodeFocus: (nodeId: string) => void;
};

export function InferenceResults({ datasetId, refreshKey, onNodeFocus }: Props) {
  const [rows, setRows] = useState<InferenceResultItem[]>([]);
  const [message, setMessage] = useState("完成推理后展示异常节点明细。");

  useEffect(() => {
    if (!datasetId) {
      setRows([]);
      return;
    }
    listInferenceResults(datasetId)
      .then((items) => {
        setRows(items.slice(0, 20));
        setMessage(items.length ? `已载入 ${items.length} 条推理结果。` : "当前数据集还没有推理结果。");
      })
      .catch((error) => setMessage(error.message));
  }, [datasetId, refreshKey]);

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
            <th>地区</th>
            <th>异常概率</th>
            <th>推理依据</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.node_id} onClick={() => onNodeFocus(row.node_id)}>
              <td>{row.node_id}</td>
              <td>{row.display_name}</td>
              <td>{row.region}</td>
              <td>{row.risk_score.toFixed(4)}</td>
              <td>
                <div className="reason-cell">
                  <span>{row.reason}</span>
                  <small>
                    邻居：{row.support_neighbors.slice(0, 3).join(", ") || "-"}；特征：
                    {row.top_features.slice(0, 3).join(", ") || "-"}
                  </small>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">{message}</p>
    </section>
  );
}
