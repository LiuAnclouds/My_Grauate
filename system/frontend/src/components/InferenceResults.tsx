import { useEffect, useMemo, useState } from "react";
import { InferenceResultItem, listInferenceResults } from "../services/api";

type Props = {
  datasetId: number | null;
  refreshKey: number;
  onNodeFocus: (nodeId: string) => void;
};

export function InferenceResults({ datasetId, refreshKey, onNodeFocus }: Props) {
  const [rows, setRows] = useState<InferenceResultItem[]>([]);
  const [message, setMessage] = useState("风险识别完成后，这里将生成对象级风险台账。\n");
  const [keyword, setKeyword] = useState("");

  useEffect(() => {
    if (!datasetId) {
      setRows([]);
      return;
    }
    listInferenceResults(datasetId)
      .then((items) => {
        setRows(items);
        setMessage(items.length ? `已生成 ${items.length} 条风险记录。` : "当前分析资产尚未生成风险记录。");
      })
      .catch((error) => setMessage(error.message));
  }, [datasetId, refreshKey]);

  const filteredRows = useMemo(() => {
    if (!keyword.trim()) {
      return rows.slice(0, 30);
    }
    const text = keyword.trim().toLowerCase();
    return rows
      .filter((row) => {
        return [row.node_id, row.display_name, row.region, row.occupation, row.risk_label]
          .join(" ")
          .toLowerCase()
          .includes(text);
      })
      .slice(0, 30);
  }, [keyword, rows]);

  const abnormalCount = rows.filter((row) => row.risk_label === "suspicious").length;
  const normalCount = rows.length - abnormalCount;

  return (
    <section className="panel panel-stack result-panel">
      <div className="panel-heading aligned-start split-heading">
        <div>
          <p className="eyebrow">Risk Ledger</p>
          <h2>风险结果台账</h2>
          <p className="section-copy">沉淀对象级风险评分、画像信息与模型解释线索，支持快速定位与复核。</p>
        </div>
        <div className="result-summary enterprise-summary">
          <span>总记录 {rows.length}</span>
          <span>高风险 {abnormalCount}</span>
          <span>低风险 {normalCount}</span>
        </div>
      </div>

      <input
        className="search-input"
        value={keyword}
        onChange={(event) => setKeyword(event.target.value)}
        placeholder="按对象编号、姓名、区域、职业或风险标签筛选"
      />

      <div className="table-wrap enterprise-table-wrap">
        <table>
          <thead>
            <tr>
              <th>对象编号</th>
              <th>对象信息</th>
              <th>区域 / 职业</th>
              <th>风险分数</th>
              <th>风险等级</th>
              <th>研判依据</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row) => (
              <tr key={row.node_id} onClick={() => onNodeFocus(row.node_id)}>
                <td>{row.node_id}</td>
                <td>
                  <div className="reason-cell">
                    <span>{row.display_name}</span>
                    <small>{row.id_number}</small>
                  </div>
                </td>
                <td>
                  <div className="reason-cell">
                    <span>{row.region}</span>
                    <small>{row.occupation}</small>
                  </div>
                </td>
                <td>{row.risk_score.toFixed(4)}</td>
                <td>
                  <span className={row.risk_label === "suspicious" ? "risk-chip danger" : "risk-chip success"}>
                    {row.risk_label === "suspicious" ? "高风险" : "低风险"}
                  </span>
                </td>
                <td>
                  <div className="reason-cell">
                    <span>{row.reason || "模型已完成关系聚合并生成风险判断。"}</span>
                    <small>
                      关联对象：{row.support_neighbors.slice(0, 4).join("、") || "-"}；关键特征：
                      {row.top_features.slice(0, 4).join("、") || "-"}
                    </small>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">{message.trim()}</p>
    </section>
  );
}
