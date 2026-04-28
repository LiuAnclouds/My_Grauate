import { useState } from "react";
import { AuthPanel } from "./components/AuthPanel";
import { DataUpload } from "./components/DataUpload";
import { GraphWorkspace } from "./components/GraphWorkspace";
import { InferenceResults } from "./components/InferenceResults";
import { PipelinePanel } from "./components/PipelinePanel";

export default function App() {
  const [email, setEmail] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [inferenceVersion, setInferenceVersion] = useState(0);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);

  function handleDatasetSelect(datasetId: number) {
    setSelectedDatasetId(datasetId);
    setHighlightedNodeId(null);
  }

  if (!email) {
    return <AuthPanel onAuthed={setEmail} />;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">DyRIFT-TGAT</p>
          <h1>动态图欺诈节点推理系统</h1>
        </div>
        <div className="topbar-actions">
          <span className="account-pill">{email}</span>
          <button className="ghost-button" onClick={() => setEmail(null)}>
            退出
          </button>
        </div>
      </header>

      <div className="dashboard-layout">
        <aside className="sidebar">
          <DataUpload selectedDatasetId={selectedDatasetId} onSelect={handleDatasetSelect} />
          <PipelinePanel
            datasetId={selectedDatasetId}
            onInferenceComplete={() => setInferenceVersion((value) => value + 1)}
          />
        </aside>

        <section className="workspace">
          <GraphWorkspace
            datasetId={selectedDatasetId}
            refreshKey={inferenceVersion}
            highlightedNodeId={highlightedNodeId}
          />
          <InferenceResults
            datasetId={selectedDatasetId}
            refreshKey={inferenceVersion}
            onNodeFocus={setHighlightedNodeId}
          />
        </section>
      </div>
    </main>
  );
}
