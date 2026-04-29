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
  onSelect: (datasetId: number, networkName?: string) => void;
};

const networkProfiles: Record<string, { label: string; description: string; scene: string }> = {
  xinye_dgraph: {
    label: "星链零售网络",
    description: "面向零售交易场景的账户关联网络，适合观察高频往来与异常聚集。",
    scene: "零售交易风控"
  },
  elliptic_transactions: {
    label: "清算支付网络",
    description: "面向跨主体支付链路的业务网络，适合观察资金路径和链路扩散。",
    scene: "支付链路监测"
  },
  ellipticpp_transactions: {
    label: "枢纽综合网络",
    description: "面向多关系混合场景的综合业务网络，适合展示多维风险分析过程。",
    scene: "综合关系研判"
  }
};

const defaultNetworks = [
  { key: "xinye_dgraph", ...networkProfiles.xinye_dgraph },
  { key: "elliptic_transactions", ...networkProfiles.elliptic_transactions },
  { key: "ellipticpp_transactions", ...networkProfiles.ellipticpp_transactions }
];

function formatStatus(status: string) {
  const map: Record<string, string> = {
    normalized: "已入库",
    official_validation: "可分析",
    feature_ready: "已完成处理",
    inference_completed: "已生成风险名单"
  };
  return map[status] ?? "处理中";
}

function resolveNetworkKey(item: DatasetSummary) {
  const text = [item.name, item.original_filename, String(item.summary?.technical_name ?? "")].join(" ");
  return Object.keys(networkProfiles).find((key) => text.includes(key)) ?? "";
}

function displayNameFor(item: DatasetSummary) {
  const key = resolveNetworkKey(item);
  return key ? networkProfiles[key].label : String(item.summary?.business_name ?? item.name);
}

function descriptionFor(item: DatasetSummary) {
  const key = resolveNetworkKey(item);
  return key ? networkProfiles[key].description : String(item.summary?.source_description ?? item.original_filename);
}

export function DataUpload({ selectedDatasetId, onSelect }: Props) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [useLlm, setUseLlm] = useState(false);
  const [mapping, setMapping] = useState<MappingResponse | null>(null);
  const [message, setMessage] = useState("请选择默认业务网络，或导入新的业务文件进入关系分析流程。");
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
      setMessage(`业务文件已接入：${dataset.original_filename}，共 ${dataset.row_count} 条记录。`);
      await refresh();
      onSelect(dataset.id, displayNameFor(dataset));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "业务文件导入失败。");
    } finally {
      setBusy(false);
    }
  }

  async function handleDefaultNetwork(networkKey: string, label: string) {
    setBusy(true);
    try {
      const dataset = await createDemoDataset(networkKey);
      const mappingResult = await fetchMapping(dataset.id);
      setMapping(mappingResult);
      setMessage(`${label} 已接入，可以进入关系网络或智能研判。`);
      await refresh();
      onSelect(dataset.id, displayNameFor(dataset));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "默认业务网络加载失败。");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refresh().catch((error) => setMessage(error.message));
  }, []);

  return (
    <section className="panel panel-stack business-access-panel">
      <div className="panel-heading aligned-start">
        <div>
          <p className="eyebrow">Business Access</p>
          <h2>业务接入中心</h2>
          <p className="section-copy">选择默认业务网络快速进入风控流程，也可以导入新的业务文件进行关系分析。</p>
        </div>
      </div>

      <div className="default-network-grid">
        {defaultNetworks.map((option) => (
          <button
            key={option.key}
            className="resource-card default-network-card"
            onClick={() => handleDefaultNetwork(option.key, option.label)}
            disabled={busy}
            type="button"
          >
            <small>{option.scene}</small>
            <strong>{option.label}</strong>
            <span>{option.description}</span>
          </button>
        ))}
      </div>

      <div className="upload-card emphasis-card">
        <div className="panel-subheading compact-heading">
          <h3>导入业务文件</h3>
          <span className="subtle-tag">CSV</span>
        </div>
        <label className="upload-label">
          <span>上传待分析的业务结构化文件</span>
          <input type="file" accept=".csv" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} disabled={busy} />
        </label>
        <label className="inline-check">
          <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
          启用智能字段识别，自动匹配对象、关系与特征字段
        </label>
      </div>

      {selectedDataset ? (
        <div className="dataset-summary-card emphasis-card selected-network-card">
          <div className="panel-subheading compact-heading">
            <h3>当前业务网络</h3>
            <span className={`status-badge status-${selectedDataset.status}`}>{formatStatus(selectedDataset.status)}</span>
          </div>
          <strong>{displayNameFor(selectedDataset)}</strong>
          <small>{descriptionFor(selectedDataset)}</small>
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

      {mapping ? (
        <div className="mapping-preview product-preview">
          <div className="panel-subheading compact-heading">
            <h3>接入解析结果</h3>
            <span className="subtle-tag">{mapping.method}</span>
          </div>
          <span>{mapping.message}</span>
          <small>对象主键字段：{String(mapping.mapping.node_id ?? "-")}</small>
          <small>
            关系字段：{String(mapping.mapping.source_id ?? "-")} {"->"} {String(mapping.mapping.target_id ?? "-")}
          </small>
          <small>模型输入字段数：{Array.isArray(mapping.mapping.feature_columns) ? mapping.mapping.feature_columns.length : 0}</small>
        </div>
      ) : null}

      <p className="hint">{message}</p>

      <div className="network-list">
        {datasets.map((dataset) => (
          <button
            key={dataset.id}
            className={dataset.id === selectedDatasetId ? "dataset-item active" : "dataset-item"}
            onClick={() => onSelect(dataset.id, displayNameFor(dataset))}
            type="button"
          >
            <div>
              <span>{displayNameFor(dataset)}</span>
              <small>{descriptionFor(dataset)}</small>
            </div>
            <div className="dataset-meta">
              <small>{formatStatus(dataset.status)}</small>
              <small>{String(dataset.summary?.edge_count ?? "-")} 条关系</small>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}
