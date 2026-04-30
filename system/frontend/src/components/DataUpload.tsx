import { useEffect, useMemo, useState } from "react";
import {
  DatasetSummary,
  deleteDataset,
  fetchMapping,
  listDatasets,
  MappingResponse,
  updateDataset,
  uploadDataset
} from "../services/api";

type Props = {
  selectedDatasetId: number | null;
  onSelect: (datasetId: number | null, networkName?: string) => void;
  onOpenPage?: (page: "network" | "analysis") => void;
};

const fraudEvents = [
  "异常资金快进快出",
  "多账户协同套现",
  "虚假商户交易团伙",
  "高频小额试探交易",
  "跨区域异常转移",
  "账号接管风险事件",
  "疑似洗钱链路扩散",
  "空壳企业关联交易",
  "设备集群异常注册",
  "黑灰产中介撮合"
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

function displayNameFor(item: DatasetSummary) {
  return String(item.summary?.business_name ?? item.name);
}

function descriptionFor(item: DatasetSummary) {
  const eventName = item.summary?.fraud_event ? ` · ${String(item.summary.fraud_event)}` : "";
  return `${String(item.summary?.source_description ?? item.original_filename)}${eventName}`;
}

function networkIdFor(item: DatasetSummary) {
  return String(item.summary?.business_id ?? `BN-${String(item.id).padStart(4, "0")}`);
}

function countText(value: unknown, fallback = "-") {
  return value === null || value === undefined ? fallback : String(value);
}

export function DataUpload({ selectedDatasetId, onSelect, onOpenPage }: Props) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [query, setQuery] = useState("");
  const [networkName, setNetworkName] = useState("");
  const [eventName, setEventName] = useState(fraudEvents[0]);
  const [editName, setEditName] = useState("");
  const [useLlm, setUseLlm] = useState(false);
  const [mapping, setMapping] = useState<MappingResponse | null>(null);
  const [message, setMessage] = useState("当前还没有业务网络，请先导入 CSV 文件完成接入。");
  const [busy, setBusy] = useState(false);

  const selectedDataset = useMemo(
    () => datasets.find((item) => item.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId]
  );

  const filteredDatasets = useMemo(() => {
    const text = query.trim().toLowerCase();
    if (!text) return datasets;
    return datasets.filter((item) => {
      const searchable = [
        displayNameFor(item),
        networkIdFor(item),
        item.original_filename,
        String(item.summary?.fraud_event ?? "")
      ]
        .join(" ")
        .toLowerCase();
      return searchable.includes(text);
    });
  }, [datasets, query]);

  async function refresh() {
    const items = await listDatasets();
    setDatasets(items);
    return items;
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    setBusy(true);
    try {
      const dataset = await uploadDataset(file, useLlm, networkName, eventName);
      const mappingResult = await fetchMapping(dataset.id);
      setMapping(mappingResult);
      setMessage(`业务网络已接入：${displayNameFor(dataset)}，共 ${dataset.row_count} 条记录。`);
      await refresh();
      onSelect(dataset.id, displayNameFor(dataset));
      setNetworkName("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "业务文件导入失败。");
    } finally {
      setBusy(false);
    }
  }

  async function handleRename() {
    if (!selectedDataset || !editName.trim()) return;
    setBusy(true);
    try {
      const dataset = await updateDataset(selectedDataset.id, editName);
      setMessage(`业务网络已重命名为：${displayNameFor(dataset)}。`);
      await refresh();
      onSelect(dataset.id, displayNameFor(dataset));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "业务网络重命名失败。");
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(dataset: DatasetSummary) {
    if (!window.confirm(`确认删除业务网络“${displayNameFor(dataset)}”？`)) return;
    setBusy(true);
    try {
      await deleteDataset(dataset.id);
      const items = await refresh();
      setMessage(`业务网络已删除：${displayNameFor(dataset)}。`);
      if (selectedDatasetId === dataset.id) {
        const next = items[0] ?? null;
        onSelect(next?.id ?? null, next ? displayNameFor(next) : "");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "业务网络删除失败。");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refresh().catch((error) => setMessage(error.message));
  }, []);

  useEffect(() => {
    setEditName(selectedDataset ? displayNameFor(selectedDataset) : "");
  }, [selectedDataset]);

  return (
    <section className="business-network-page app-module-page">
      <div className="business-network-layout">
        <div className="network-directory-panel app-panel">
          <div className="business-module-heading app-section-heading">
            <div>
              <p className="eyebrow">Business Networks</p>
              <h2>业务网络目录</h2>
              <p>上传 CSV 后生成业务网络、人员身份、关系结构与分析字段，作为后续图谱和研判的统一入口。</p>
            </div>
            <span className="directory-count">{datasets.length} 个网络</span>
          </div>

          <label className="network-search-box">
            <span>检索业务网络</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入网络名称、ID 或事件类型" />
          </label>

          <div className="network-directory-list network-directory-list--scroll" aria-label="业务网络列表">
            {filteredDatasets.length ? (
              filteredDatasets.map((dataset) => (
                <article key={dataset.id} className={dataset.id === selectedDatasetId ? "network-row active" : "network-row"}>
                  <button className="network-row-main" onClick={() => onSelect(dataset.id, displayNameFor(dataset))} type="button">
                    <span className="network-id">{networkIdFor(dataset)}</span>
                    <strong>{displayNameFor(dataset)}</strong>
                    <small>{descriptionFor(dataset)}</small>
                  </button>
                  <div className="network-row-stats">
                    <span>{formatStatus(dataset.status)}</span>
                    <strong>{countText(dataset.summary?.node_count ?? dataset.row_count)} 人</strong>
                    <small>{countText(dataset.summary?.edge_count)} 条关系</small>
                  </div>
                  <div className="network-row-actions">
                    <button className="network-row-action" onClick={() => onSelect(dataset.id, displayNameFor(dataset))} type="button">
                      选择
                    </button>
                    <button className="network-row-action danger" onClick={() => handleDelete(dataset)} disabled={busy} type="button">
                      删除
                    </button>
                  </div>
                </article>
              ))
            ) : (
              <div className="empty-network-state">
                <strong>{datasets.length ? "没有匹配的业务网络" : "暂无已接入的业务网络"}</strong>
                <span>{datasets.length ? "请调整检索条件。" : "请从右侧上传 CSV 文件，系统会自动生成可分析的人员关系网络。"}</span>
              </div>
            )}
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
                    <span>人员</span>
                    <strong>{countText(selectedDataset.summary?.node_count ?? selectedDataset.row_count)}</strong>
                  </div>
                  <div>
                    <span>关系</span>
                    <strong>{countText(selectedDataset.summary?.edge_count)}</strong>
                  </div>
                </div>
                <label className="network-edit-field">
                  <span>网络名称</span>
                  <input value={editName} onChange={(event) => setEditName(event.target.value)} />
                </label>
                <div className="selected-network-actions">
                  <button className="secondary" onClick={handleRename} disabled={busy || !editName.trim()} type="button">
                    保存名称
                  </button>
                  <button className="secondary" onClick={() => onOpenPage?.("network")} type="button">
                    查看关系
                  </button>
                  <button className="primary" onClick={() => onOpenPage?.("analysis")} type="button">
                    开始研判
                  </button>
                </div>
              </>
            ) : (
              <p>当前没有选中的业务网络。请先上传 CSV 文件完成接入。</p>
            )}
          </div>

          <div className="network-import-card app-panel">
            <div className="side-card-heading">
              <span>接入业务网络</span>
              <em>CSV</em>
            </div>
            <label className="network-edit-field">
              <span>网络名称</span>
              <input value={networkName} onChange={(event) => setNetworkName(event.target.value)} placeholder="例如：华东支付异常事件" />
            </label>
            <label className="network-edit-field">
              <span>欺诈事件映射</span>
              <select value={eventName} onChange={(event) => setEventName(event.target.value)}>
                {fraudEvents.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>
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
              <small>系统已生成业务网络 ID、人员身份、团伙归属和分析字段。</small>
            </div>
          ) : null}

          <p className="network-feedback">{message}</p>
        </aside>
      </div>
    </section>
  );
}
