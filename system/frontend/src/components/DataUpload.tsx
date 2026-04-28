import { useEffect, useState } from "react";
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

export function DataUpload({ selectedDatasetId, onSelect }: Props) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [useLlm, setUseLlm] = useState(false);
  const [mapping, setMapping] = useState<MappingResponse | null>(null);
  const [message, setMessage] = useState("上传 CSV 后会写入数据库，并生成演示用人物信息。");

  async function refresh() {
    setDatasets(await listDatasets());
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    const dataset = await uploadDataset(file, useLlm);
    const mappingResult = await fetchMapping(dataset.id);
    setMapping(mappingResult);
    setMessage(`已入库：${dataset.original_filename}，共 ${dataset.row_count} 行。`);
    await refresh();
    onSelect(dataset.id);
  }

  async function handleDemo(datasetName: string) {
    const dataset = await createDemoDataset(datasetName);
    const mappingResult = await fetchMapping(dataset.id);
    setMapping(mappingResult);
    setMessage(`已载入官方验证集：${dataset.original_filename}，共 ${dataset.row_count} 个节点。`);
    await refresh();
    onSelect(dataset.id);
  }

  useEffect(() => {
    refresh().catch((error) => setMessage(error.message));
  }, []);

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Data Intake</p>
          <h2>数据上传与入库</h2>
        </div>
      </div>
      <input type="file" accept=".csv" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} />
      <label className="inline-check">
        <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
        LLM 辅助识别字段
      </label>
      <div className="demo-buttons">
        <button onClick={() => handleDemo("xinye_dgraph")}>XinYe 验证集</button>
        <button onClick={() => handleDemo("elliptic_transactions")}>ET 验证集</button>
        <button onClick={() => handleDemo("ellipticpp_transactions")}>EPP 验证集</button>
      </div>
      {mapping ? (
        <div className="mapping-preview">
          <strong>{mapping.method}</strong>
          <span>{mapping.message}</span>
          <small>节点字段：{String(mapping.mapping.node_id ?? "-")}</small>
          <small>
            边字段：{String(mapping.mapping.source_id ?? "-")} → {String(mapping.mapping.target_id ?? "-")}
          </small>
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
            <span>{dataset.name}</span>
            <small>{dataset.row_count} rows</small>
          </button>
        ))}
      </div>
    </section>
  );
}
