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

  return (
    <main>
      <header className="topbar">
        <div>
          <p className="eyebrow">DyRIFT-TGAT</p>
          <h1>动态图欺诈节点推理系统</h1>
        </div>
        <div className="account-pill">{email ?? "未登录"}</div>
      </header>
      <div className="layout">
        <aside>
          <AuthPanel onAuthed={setEmail} />
          <DataUpload selectedDatasetId={selectedDatasetId} onSelect={handleDatasetSelect} />
        </aside>
        <section className="workspace">
          <GraphWorkspace
            datasetId={selectedDatasetId}
            refreshKey={inferenceVersion}
            highlightedNodeId={highlightedNodeId}
          />
          <div className="two-column">
            <PipelinePanel
              datasetId={selectedDatasetId}
              onInferenceComplete={() => setInferenceVersion((value) => value + 1)}
            />
            <InferenceResults
              datasetId={selectedDatasetId}
              refreshKey={inferenceVersion}
              onNodeFocus={setHighlightedNodeId}
            />
          </div>
        </section>
      </div>
    </main>
  );
}
