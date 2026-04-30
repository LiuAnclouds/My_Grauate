import { useEffect, useMemo, useState } from "react";
import {
  DatasetSummary,
  deleteDataset,
  listDatasets,
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

function statusClassFor(status: string) {
  const normalized = status.replace(/[^a-z0-9_-]/gi, "-").toLowerCase();
  return `network-status-lamp status-${normalized}`;
}

function displayNameFor(item: DatasetSummary) {
  return String(item.summary?.business_name ?? networkIdFor(item));
}

function descriptionFor(item: DatasetSummary) {
  const eventName = item.summary?.fraud_event ? String(item.summary.fraud_event) : "待标记";
  return `欺诈类型：${eventName}`;
}

function networkIdFor(item: DatasetSummary) {
  return String(item.summary?.business_id ?? `BN-${String(item.id).padStart(4, "0")}`);
}

function countText(value: unknown, fallback = "-") {
  return value === null || value === undefined ? fallback : String(value);
}

export function DataUpload({ selectedDatasetId, onSelect }: Props) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [query, setQuery] = useState("");
  const [networkName, setNetworkName] = useState("");
  const [eventName, setEventName] = useState(fraudEvents[0]);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [useLlm, setUseLlm] = useState(false);
  const [lastImported, setLastImported] = useState<DatasetSummary | null>(null);
  const [message, setMessage] = useState("选择文件并填写网络名称后，即可接入新的业务网络。");
  const [busy, setBusy] = useState(false);

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

  async function handleImport() {
    if (!pendingFile) {
      setMessage("请先选择需要接入的 CSV 文件。");
      return;
    }
    if (!networkName.trim()) {
      setMessage("请先填写业务网络名称，方便后续识别。");
      return;
    }
    setBusy(true);
    try {
      const dataset = await uploadDataset(pendingFile, useLlm, networkName, eventName);
      setLastImported(dataset);
      setMessage(`接入完成：${displayNameFor(dataset)}，已生成 ${countText(dataset.summary?.node_count ?? dataset.row_count)} 个风险对象。`);
      await refresh();
      onSelect(dataset.id, displayNameFor(dataset));
      setNetworkName("");
      setPendingFile(null);
      setFileInputKey((value) => value + 1);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "业务文件导入失败。");
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
      if (lastImported?.id === dataset.id) {
        setLastImported(null);
      }
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
                    <span className="network-id"><b>网络ID</b>{networkIdFor(dataset)}</span>
                    <strong>{displayNameFor(dataset)}</strong>
                    <small>{descriptionFor(dataset)}</small>
                  </button>
                  <div className="network-row-stats">
                    <span className={statusClassFor(dataset.status)}>{formatStatus(dataset.status)}</span>
                    <div className="network-row-metrics">
                      <strong>{countText(dataset.summary?.node_count ?? dataset.row_count)} 人</strong>
                      <small>{countText(dataset.summary?.edge_count)} 条关系</small>
                    </div>
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
          <div className="network-import-card app-panel">
            <div className="side-card-heading">
              <span>接入业务网络</span>
              <em>新建</em>
            </div>
            <div className="network-import-intro">
              <h3>上传业务记录，生成可分析的人员关系网络</h3>
              <p>每次接入都会生成独立的网络 ID，可在左侧目录中选择、查看或删除。</p>
            </div>
            <label className="network-edit-field">
              <span>网络名称</span>
              <input value={networkName} onChange={(event) => setNetworkName(event.target.value)} placeholder="例如：华东支付异常事件" />
            </label>
            <label className="network-edit-field">
              <span>欺诈类型</span>
              <select value={eventName} onChange={(event) => setEventName(event.target.value)}>
                {fraudEvents.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>
            <label className="network-file-picker">
              <span>选择待分析文件</span>
              <input key={fileInputKey} type="file" accept=".csv" onChange={(event) => setPendingFile(event.target.files?.[0] ?? null)} disabled={busy} />
              <strong>{pendingFile ? "已选择 1 个 CSV 文件" : "点击选择 CSV 文件"}</strong>
            </label>
            <label className="inline-check compact-check">
              <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
              自动整理字段
            </label>

            <button className="primary network-import-submit" onClick={handleImport} disabled={busy || !pendingFile || !networkName.trim()} type="button">
              {busy ? "正在接入..." : "接入业务网络"}
            </button>

            {lastImported ? (
              <div className="network-import-status">
                <span>最近接入</span>
                <strong>{displayNameFor(lastImported)}</strong>
                <small>{networkIdFor(lastImported)} · {descriptionFor(lastImported)}</small>
                <div className="network-import-status__stats">
                  <em>{countText(lastImported.summary?.node_count ?? lastImported.row_count)} 人</em>
                  <em>{countText(lastImported.summary?.edge_count)} 条关系</em>
                </div>
              </div>
            ) : null}

            <p className="network-feedback">{message}</p>
          </div>
        </aside>
      </div>
    </section>
  );
}
