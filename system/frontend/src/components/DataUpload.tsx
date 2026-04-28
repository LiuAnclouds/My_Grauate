import { useEffect, useMemo, useState } from "react";
import {
  createDemoDataset,
  DatasetSummary,
  fetchMapping,
  listDatasets,
  MappingResponse,
  uploadDataset
} from "../services/api";

type Props = {
  selectedDatasetId: number | null;
  onSelect: (datasetId: number) => void;
};

const demoOptions = [
  { key: "xinye_dgraph", label: "零售交易网络 A", description: "高频账户往来样本，用于关联聚合与异常识别分析。" },
  { key: "elliptic_transactions", label: "链路交易网络 B", description: "跨主体资金转移样本，适合观察时序关系与链路扩散。" },
  { key: "ellipticpp_transactions", label: "综合关系网络 C", description: "增强多维关系样本，适合展示多特征风控分析过程。" }
];

function formatStatus(status: string) {
  const map: Record<string, string> = {
    normalized: "已完成入库",
    official_validation: "已接入样本",
    feature_ready: "已完成特征准备",
    inference_completed: "已生成风险结果"
  };
  return map[status] ?? status;
}

export function DataUpload({ selectedDatasetId, onSelect }: Props) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [useLlm, setUseLlm] = useState(false);
  const [mapping, setMapping] = useState<MappingResponse | null>(null);
  const [message, setMessage] = useState("可直接接入业务 CSV，也可一键加载平台内置样本以快速进入关系分析流程。");
  const [busy, setBusy] = useState(false);

  const selectedDataset = useMemo(
    () => datasets.find((item) => item.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId]
  );

  async function refresh() {
    setDatasets(await listDatasets());
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    setBusy(true);
    try {
      const dataset = await uploadDataset(file, useLlm);
      const mappingResult = await fetchMapping(dataset.id);
      setMapping(mappingResult);
      setMessage(`数据资产已接入：${dataset.original_filename}，共 ${dataset.row_count} 条记录。`);
      await refresh();
      onSelect(dataset.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "CSV 上传失败。");
    } finally {
      setBusy(false);
    }
  }

  async function handleDemo(datasetName: string) {
    setBusy(true);
    try {
      const dataset = await createDemoDataset(datasetName);
      const mappingResult = await fetchMapping(dataset.id);
      setMapping(mappingResult);
      setMessage(`内置样本已接入：${dataset.name}，共 ${dataset.row_count} 个对象。`);
      await refresh();
      onSelect(dataset.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "内置样本加载失败。");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refresh().catch((error) => setMessage(error.message));
  }, []);

  return (
    <section className="panel panel-stack">
      <div className="panel-heading aligned-start">
        <div>
          <p className="eyebrow">Data Access</p>
          <h2>数据资产中心</h2>
          <p className="section-copy">统一管理接入样本、内置关系网络与当前分析对象。</p>
        </div>
      </div>

      <div className="upload-card emphasis-card">
        <div className="panel-subheading compact-heading">
          <h3>接入业务数据</h3>
          <span className="subtle-tag">CSV</span>
        </div>
        <label className="upload-label">
          <span>上传待分析的结构化数据文件</span>
          <input type="file" accept=".csv" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} disabled={busy} />
        </label>
        <label className="inline-check">
          <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
          启用智能字段识别，自动匹配节点、关系与特征字段
        </label>
      </div>

      <div className="resource-grid">
        {demoOptions.map((option) => (
          <button key={option.key} className="resource-card" onClick={() => handleDemo(option.key)} disabled={busy}>
            <small>平台样本</small>
            <strong>{option.label}</strong>
            <span>{option.description}</span>
          </button>
        ))}
      </div>

      {mapping ? (
        <div className="mapping-preview product-preview">
          <div className="panel-subheading compact-heading">
            <h3>接入解析结果</h3>
            <span className="subtle-tag">{mapping.method}</span>
          </div>
          <span>{mapping.message}</span>
          <small>对象主键字段：{String(mapping.mapping.node_id ?? "-")}</small>
          <small>
            关系字段：{String(mapping.mapping.source_id ?? "-")} → {String(mapping.mapping.target_id ?? "-")}
          </small>
          <small>识别特征字段数：{Array.isArray(mapping.mapping.feature_columns) ? mapping.mapping.feature_columns.length : 0}</small>
        </div>
      ) : null}

      {selectedDataset ? (
        <div className="dataset-summary-card emphasis-card">
          <div className="panel-subheading compact-heading">
            <h3>当前分析资产</h3>
            <span className={`status-badge status-${selectedDataset.status}`}>{formatStatus(selectedDataset.status)}</span>
          </div>
          <strong>{String(selectedDataset.summary?.business_name ?? selectedDataset.name)}</strong>
          <small>{String(selectedDataset.summary?.source_description ?? selectedDataset.original_filename)}</small>
          <div className="stat-grid compact">
            <div>
              <span>对象规模</span>
              <strong>{String(selectedDataset.summary?.node_count ?? selectedDataset.row_count)}</strong>
            </div>
            <div>
              <span>关系规模</span>
              <strong>{String(selectedDataset.summary?.edge_count ?? "-")}</strong>
            </div>
            <div>
              <span>特征维度</span>
              <strong>{String(selectedDataset.summary?.feature_count ?? "-")}</strong>
            </div>
          </div>
        </div>
      ) : null}

      <p className="hint">{message}</p>

      <div className="dataset-list">
        {datasets.map((dataset) => (
          <button
            key={dataset.id}
            className={dataset.id === selectedDatasetId ? "dataset-item active" : "dataset-item"}
            onClick={() => onSelect(dataset.id)}
          >
            <div>
              <span>{String(dataset.summary?.business_name ?? dataset.name)}</span>
              <small>{String(dataset.summary?.source_description ?? dataset.original_filename)}</small>
            </div>
            <div className="dataset-meta">
              <small>{formatStatus(dataset.status)}</small>
              <small>{String(dataset.summary?.edge_count ?? "-")} relations</small>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}
