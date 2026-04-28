import { useEffect, useState } from "react";
import { DatasetSummary, listDatasets, uploadDataset } from "../services/api";

type Props = {
  selectedDatasetId: number | null;
  onSelect: (datasetId: number) => void;
};

export function DataUpload({ selectedDatasetId, onSelect }: Props) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [message, setMessage] = useState("上传 CSV 后会写入数据库，并生成演示用人物信息。");

  async function refresh() {
    setDatasets(await listDatasets());
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    const dataset = await uploadDataset(file);
    setMessage(`已入库：${dataset.original_filename}，共 ${dataset.row_count} 行。`);
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
