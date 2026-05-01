import { useEffect, useMemo, useState } from "react";
import { InferenceResultItem, listInferenceResults } from "../services/api";

type Props = {
  datasetId: number | null;
  refreshKey: number;
  onNodeFocus: (nodeId: string) => void;
};

const featureNames = [
  "近期交易活跃度",
  "交易金额波动",
  "邻域风险关联",
  "资金流入流出偏移",
  "交易时间规律",
  "账户行为稳定性",
  "跨区域流转强度",
  "对手方集中度",
  "短期交易频次变化",
  "资金链路深度",
  "账户画像偏移",
  "历史行为一致性",
  "社群关联密度",
  "异常金额占比",
  "短期行为突增",
  "风险邻居暴露"
];

function riskLevel(row: InferenceResultItem) {
  if (row.risk_label === "suspicious" || row.risk_score >= 0.75) {
    return { key: "suspicious", label: "高风险", className: "danger" };
  }
  if (row.risk_label === "no_risk" || row.risk_score < 0.35) {
    return { key: "no_risk", label: "无风险", className: "neutral" };
  }
  return { key: "low_risk", label: "低风险", className: "warning" };
}

function featureDisplayName(value: string) {
  const text = value.trim();
  const lower = text.toLowerCase();
  const digits = lower.match(/(\d+)$/)?.[1];
  if (/^(core_)?feature_\d+$/.test(lower) && digits) {
    return featureNames[Number(digits) % featureNames.length];
  }
  if (/[\u4e00-\u9fff]/.test(text)) return text;
  const keywordMap: Array<[string[], string]> = [
    [["amount", "amt", "value", "money", "balance"], "交易金额特征"],
    [["count", "freq", "frequency", "degree", "num"], "交易频次特征"],
    [["time", "hour", "day", "window", "period"], "时间行为特征"],
    [["region", "city", "area", "geo"], "地域流转特征"],
    [["neighbor", "edge", "graph", "relation"], "关系网络特征"],
    [["in", "out", "flow"], "资金流向特征"],
    [["risk", "fraud", "label"], "历史风险特征"],
    [["device", "ip", "account"], "账户环境特征"]
  ];
  return keywordMap.find(([keys]) => keys.some((key) => lower.includes(key)))?.[1] ?? "业务行为特征";
}

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
        setMessage(items.length ? `已生成 ${items.length} 条风险记录。` : "当前业务网络尚未生成风险记录。");
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
        const level = riskLevel(row);
        return [row.node_id, row.display_name, row.region, row.occupation, row.risk_label, level.label]
          .join(" ")
          .toLowerCase()
          .includes(text);
      })
      .slice(0, 30);
  }, [keyword, rows]);

  const abnormalCount = rows.filter((row) => riskLevel(row).key === "suspicious").length;
  const lowRiskCount = rows.filter((row) => riskLevel(row).key === "low_risk").length;
  const noRiskCount = rows.filter((row) => riskLevel(row).key === "no_risk").length;

  return (
    <section className="panel panel-stack result-panel risk-ledger-panel app-panel">
      <div className="ledger-header">
        <div>
          <p className="eyebrow">Risk Ledger</p>
          <h2>风险名单</h2>
          <p>查看待复核对象、风险等级和关联线索。</p>
        </div>
        <div className="result-summary enterprise-summary">
          <span>总记录 {rows.length}</span>
          <span>高风险 {abnormalCount}</span>
          <span>低风险 {lowRiskCount}</span>
          <span>无风险 {noRiskCount}</span>
        </div>
      </div>

      <div className="ledger-tools">
        <input
          className="search-input"
          value={keyword}
          onChange={(event) => setKeyword(event.target.value)}
          placeholder="按对象编号、姓名、区域、职业或风险标签筛选"
        />
      </div>

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
            {filteredRows.map((row) => {
              const level = riskLevel(row);
              const featureText = row.top_features.slice(0, 4).map(featureDisplayName).join("、");
              return (
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
                  <span className={`risk-chip ${level.className}`}>
                    {level.label}
                  </span>
                </td>
                <td>
                  <div className="reason-cell">
                    <span>{row.reason || "模型已完成关系聚合并生成风险判断。"}</span>
                    <small>
                      关联对象：{row.support_neighbors.slice(0, 4).join("、") || "-"}；关键特征：
                      {featureText || "-"}
                    </small>
                  </div>
                </td>
              </tr>
              );
            })}
            {!filteredRows.length ? (
              <tr>
                <td colSpan={6}>
                  <div className="empty-ledger-state">
                    <strong>暂无可展示的风险记录</strong>
                    <span>{rows.length ? "没有匹配当前筛选条件的记录。" : "请先完成智能研判任务。"}</span>
                  </div>
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      <p className="hint">{message.trim()}</p>
    </section>
  );
}
