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
  onOpenPage?: (page: "network" | "analysis") => void;
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

function networkIdFor(item: DatasetSummary) {
  return `BN-${String(item.id).padStart(4, "0")}`;
}

function countText(value: unknown, fallback = "-") {
  return value === null || value === undefined ? fallback : String(value);
}

export function DataUpload({ selectedDatasetId, onSelect, onOpenPage }: Props) {
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
    <section className="business-network-page app-module-page">
      <div className="business-network-layout">
        <div className="network-directory-panel app-panel">
          <div className="business-module-heading app-section-heading">
            <div>
              <p className="eyebrow">Business Networks</p>
              <h2>业务网络目录</h2>
              <p>统一管理演示网络、已接入数据集与导入入口，为关系建图和后续研判提供分析底座。</p>
            </div>
            <span className="directory-count">{datasets.length || defaultNetworks.length} 个网络</span>
          </div>

          <div className="network-directory-list" aria-label="业务网络列表">
            {datasets.length ? (
              datasets.map((dataset) => (
                <article key={dataset.id} className={dataset.id === selectedDatasetId ? "network-row active" : "network-row"}>
                  <button className="network-row-main" onClick={() => onSelect(dataset.id, displayNameFor(dataset))} type="button">
                    <span className="network-id">{networkIdFor(dataset)}</span>
                    <strong>{displayNameFor(dataset)}</strong>
                    <small>{descriptionFor(dataset)}</small>
                  </button>
                  <div className="network-row-stats">
                    <span>{formatStatus(dataset.status)}</span>
                    <strong>{countText(dataset.summary?.edge_count)} 条关系</strong>
                  </div>
                  <button className="network-row-action" onClick={() => onSelect(dataset.id, displayNameFor(dataset))} type="button">
                    选择
                  </button>
                </article>
              ))
            ) : (
              <div className="empty-network-state">
                <strong>暂无已接入的业务网络</strong>
                <span>可以先注册一个默认网络，或从右侧导入业务文件。</span>
              </div>
            )}
          </div>

          <div className="network-registry app-panel subtle-panel">
            <div className="registry-heading">
              <strong>可注册业务网络</strong>
              <span>测试阶段可直接接入系统内置网络</span>
            </div>
            <div className="registry-list">
              {defaultNetworks.map((option) => (
                <button
                  key={option.key}
                  className="registry-item"
                  onClick={() => handleDefaultNetwork(option.key, option.label)}
                  disabled={busy}
                  type="button"
                >
                  <span>{option.scene}</span>
                  <strong>{option.label}</strong>
                  <small>{option.description}</small>
                </button>
              ))}
            </div>
          </div>
        </div>

        <aside className="network-side-stack app-context-rail">
          <div className="selected-network-card app-panel emphasis-panel">
            <div className="side-card-heading">
              <span>当前网络</span>
              {selectedDataset ? <em className={`status-badge status-${selectedDataset.status}`}>{formatStatus(selectedDataset.status)}</em> : null}
            </div>
            {selectedDataset ? (
              <>
                <strong>{displayNameFor(selectedDataset)}</strong>
                <small>{networkIdFor(selectedDataset)}</small>
                <p>{descriptionFor(selectedDataset)}</p>
                <div className="selected-network-stats">
                  <div>
                    <span>对象</span>
                    <strong>{countText(selectedDataset.summary?.node_count ?? selectedDataset.row_count)}</strong>
                  </div>
                  <div>
                    <span>关系</span>
                    <strong>{countText(selectedDataset.summary?.edge_count)}</strong>
                  </div>
                </div>
                <div className="selected-network-actions">
                  <button className="secondary" onClick={() => onOpenPage?.("network")} type="button">
                    查看关系
                  </button>
                  <button className="primary" onClick={() => onOpenPage?.("analysis")} type="button">
                    开始研判
                  </button>
                </div>
              </>
            ) : (
              <p>请先从左侧选择或注册业务网络。</p>
            )}
          </div>

          <div className="network-import-card app-panel">
            <div className="side-card-heading">
              <span>导入业务文件</span>
              <em>CSV</em>
            </div>
            <label className="upload-label compact-upload">
              <span>选择待分析文件</span>
              <input type="file" accept=".csv" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} disabled={busy} />
            </label>
            <label className="inline-check compact-check">
              <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
              自动识别文件结构
            </label>
          </div>

          {mapping ? (
            <div className="network-import-result app-panel success-panel">
              <div className="side-card-heading">
                <span>文件已识别</span>
                <em>{mapping.method}</em>
              </div>
              <p>{mapping.message}</p>
              <small>已完成对象、关系和分析字段匹配。</small>
            </div>
          ) : null}

          <p className="network-feedback">{message}</p>
        </aside>
      </div>
    </section>
  );
}
