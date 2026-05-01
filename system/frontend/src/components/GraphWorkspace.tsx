import cytoscape, { Core } from "cytoscape";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createGraphTask,
  fetchGraph,
  fetchTimeline,
  GraphNode,
  GraphResponse,
  listDatasets,
  ProcessingEventItem
} from "../services/api";

type Props = {
  datasetId: number | null;
  refreshKey: number;
  highlightedNodeId: string | null;
  timelineNodeId: string | null;
  compact?: boolean;
};

const networkNames: Record<string, string> = {
  xinye_dgraph: "星链零售网络",
  elliptic_transactions: "清算支付网络",
  ellipticpp_transactions: "枢纽综合网络"
};

function displayNetworkName(summary: Record<string, unknown> | undefined) {
  const technicalName = String(summary?.technical_name ?? "");
  if (technicalName && networkNames[technicalName]) {
    return networkNames[technicalName];
  }
  return String(summary?.business_name ?? "-");
}

function isGraphReadyStatus(status: string) {
  return ["graph_ready", "feature_ready", "inference_completed"].includes(status);
}

function graphStatusLabel(status: string) {
  if (status === "graph_ready") return "已构建关系图谱";
  if (status === "feature_ready") return "已完成处理";
  if (status === "inference_completed") return "已生成风险名单";
  return "待构建关系图谱";
}

export function GraphWorkspace({ datasetId, refreshKey, highlightedNodeId, timelineNodeId, compact = false }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [message, setMessage] = useState("选择业务网络后，这里将展示对象关系结构。");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [datasetStatus, setDatasetStatus] = useState("");
  const [buildEvents, setBuildEvents] = useState<ProcessingEventItem[]>([]);
  const [viewRequested, setViewRequested] = useState(false);
  const [busy, setBusy] = useState(false);

  const activeNodeId = timelineNodeId ?? highlightedNodeId;
  const graphReady = isGraphReadyStatus(datasetStatus);

  useEffect(() => {
    if (!datasetId) {
      setGraph(null);
      setSelectedNode(null);
      setDatasetStatus("");
      setBuildEvents([]);
      setViewRequested(false);
      setMessage("选择业务网络后，这里将展示对象关系结构。");
      return;
    }
    setGraph(null);
    setSelectedNode(null);
    setBuildEvents([]);
    setViewRequested(false);
    listDatasets()
      .then((items) => {
        const current = items.find((item) => item.id === datasetId);
        const status = current?.status ?? "";
        setDatasetStatus(status);
        if (isGraphReadyStatus(status)) {
          setMessage("关系图谱已构建，可点击查看。");
          if (compact) setViewRequested(true);
        } else {
          setMessage("当前业务网络尚未构建关系图谱。");
        }
      })
      .catch((error) => setMessage(error.message));
  }, [compact, datasetId, refreshKey]);

  useEffect(() => {
    if (!datasetId || !graphReady || !viewRequested) return;
    fetchGraph(datasetId)
      .then((data) => {
        setGraph(data);
        setMessage(`已装载 ${data.nodes.length} 个对象、${data.edges.length} 条关系。`);
      })
      .catch((error) => setMessage(error.message));
  }, [datasetId, graphReady, refreshKey, viewRequested]);

  const nodeMap = useMemo(() => {
    const entries = (graph?.nodes ?? []).map((node) => [node.id, node] as const);
    return new Map(entries);
  }, [graph]);

  useEffect(() => {
    if (activeNodeId) {
      setSelectedNode(nodeMap.get(activeNodeId) ?? null);
    }
  }, [activeNodeId, nodeMap]);

  useEffect(() => {
    if (!containerRef.current || !graph) return;
    cyRef.current?.destroy();
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: [
        ...graph.nodes.map((node) => ({
          data: {
            id: node.id,
            label: node.label,
            size: node.size,
            color: node.color,
            riskScore: node.risk_score,
            riskLabel: node.risk_label,
            sourceType: node.source_type,
            featureCount: node.feature_count,
            focused: node.id === activeNodeId ? "yes" : "no"
          }
        })),
        ...graph.edges.map((edge) => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.edge_type,
            focused: edge.source === activeNodeId || edge.target === activeNodeId ? "yes" : "no",
            highlighted: edge.highlighted ? "yes" : "no"
          }
        }))
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            width: "data(size)",
            height: "data(size)",
            label: "data(label)",
            color: "#122031",
            "font-size": 10,
            "text-outline-width": 3,
            "text-outline-color": "#ffffff"
          }
        },
        {
          selector: 'node[riskLabel = "suspicious"]',
          style: {
            "border-color": "#7f1d1d",
            "border-width": 4
          }
        },
        {
          selector: 'node[riskLabel = "normal"]',
          style: {
            "border-color": "#14532d",
            "border-width": 3
          }
        },
        {
          selector: 'node[focused = "yes"]',
          style: {
            "background-color": "#2563eb",
            "border-color": "#0f3d91",
            "border-width": 6
          }
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "#b7c3d0",
            "target-arrow-color": "#b7c3d0",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            opacity: 0.85
          }
        },
        {
          selector: 'edge[highlighted = "yes"]',
          style: {
            width: 3,
            "line-color": "#3b82f6",
            "target-arrow-color": "#3b82f6"
          }
        },
        {
          selector: 'edge[focused = "yes"]',
          style: {
            width: 3.6,
            "line-color": "#2563eb",
            "target-arrow-color": "#2563eb"
          }
        }
      ],
      layout: {
        name: "cose",
        animate: true,
        fit: true,
        padding: 36
      }
    });

    cyRef.current.on("tap", "node", (event) => {
      const nodeId = String(event.target.data("id"));
      setSelectedNode(nodeMap.get(nodeId) ?? null);
    });

    if (activeNodeId) {
      const target = cyRef.current.getElementById(activeNodeId);
      if (target.nonempty()) {
        cyRef.current.animate({ center: { eles: target }, zoom: 1.12 }, { duration: 400 });
      }
    }

    return () => {
      cyRef.current?.destroy();
      cyRef.current = null;
    };
  }, [graph, activeNodeId, nodeMap]);

  async function handleBuildGraph() {
    if (!datasetId || graphReady || busy) return;
    setBusy(true);
    setGraph(null);
    setViewRequested(false);
    setMessage("正在构建关系图谱。");
    try {
      const task = await createGraphTask(datasetId);
      const timeline = await fetchTimeline(datasetId);
      setDatasetStatus("graph_ready");
      setBuildEvents(timeline.events.filter((event) => event.stage === "graph_construction"));
      setMessage(task.message || "关系图谱构建完成，可以查看。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "关系图谱构建失败。");
    } finally {
      setBusy(false);
    }
  }

  function handleViewGraph() {
    if (!graphReady || busy) return;
    setViewRequested(true);
  }

  const emptyGraphTitle = !datasetId
    ? "尚未选择业务网络"
    : graphReady
      ? "关系图谱已构建"
      : "关系图谱尚未构建";
  const emptyGraphText = !datasetId
    ? "请先进入“业务网络”栏目选择或导入一个网络。"
    : graphReady
      ? "点击“查看关系图谱”载入对象关系。"
      : "请先点击“构建关系图谱”。";

  return (
    <section className={compact ? "panel graph-panel workspace-panel graph-page-panel compact-graph-panel app-panel" : "panel graph-panel workspace-panel graph-page-panel app-panel graph-investigation-panel"}>
      <div className="graph-panel-top app-section-heading">
        <div>
          <p className="eyebrow">Relationship Network</p>
          <h2>关系网络图</h2>
          <p>先构建关系图谱，再查看对象关系。</p>
        </div>
        {compact ? (
          <div className="legend-row enterprise-legend">
            <span><i className="legend-dot blue" /> 当前定位对象</span>
            <span><i className="legend-dot red" /> 高风险对象</span>
            <span><i className="legend-dot green" /> 低风险对象</span>
          </div>
        ) : (
          <div className="graph-action-controls">
            <button
              className={graphReady ? "graph-state-button ready" : "primary graph-action-button"}
              disabled={!datasetId || graphReady || busy}
              onClick={handleBuildGraph}
              type="button"
            >
              {busy ? "构建中..." : graphReady ? "已构建关系图谱" : "构建关系图谱"}
            </button>
            <button
              className={graphReady ? "secondary graph-action-button" : "graph-state-button pending"}
              disabled={!datasetId || !graphReady || busy}
              onClick={handleViewGraph}
              type="button"
            >
              {graphReady ? "查看关系图谱" : "未构建关系图谱"}
            </button>
          </div>
        )}
      </div>

      {!compact ? (
        <div className="graph-build-panel">
          <div>
            <span>图谱状态</span>
            <strong>{graphStatusLabel(datasetStatus)}</strong>
          </div>
          <div className="graph-build-steps">
            {(buildEvents.length ? buildEvents : [
              { id: -1, title: "等待构建", detail: datasetId ? "点击构建后生成图谱过程。" : "请先选择业务网络。", progress: 0 }
            ] as ProcessingEventItem[]).map((event) => (
              <span key={event.id} className={event.progress >= 1 ? "graph-build-step complete" : "graph-build-step"}>
                <i style={{ width: `${Math.round(event.progress * 100)}%` }} />
                <b>{event.title}</b>
                <small>{event.detail}</small>
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className="graph-layout enterprise-graph-layout">
        <div ref={containerRef} className="graph-canvas enterprise-canvas">
          {!graph ? (
            <div className="empty-graph-state">
              <strong>{emptyGraphTitle}</strong>
              <span>{emptyGraphText}</span>
            </div>
          ) : null}
        </div>
        <aside className="graph-sidecard insight-card app-secondary-rail-card">
          <h3>对象画像</h3>
          {selectedNode ? (
            <div className="detail-list">
              <div><span>对象编号</span><strong>{selectedNode.id}</strong></div>
              <div><span>姓名</span><strong>{selectedNode.label}</strong></div>
              <div><span>区域</span><strong>{selectedNode.region}</strong></div>
              <div><span>职业</span><strong>{selectedNode.occupation}</strong></div>
              <div><span>风险标签</span><strong>{selectedNode.risk_label === "suspicious" ? "高风险" : selectedNode.risk_label === "normal" ? "低风险" : "待分析"}</strong></div>
              <div><span>风险分数</span><strong>{selectedNode.risk_score?.toFixed(4) ?? "-"}</strong></div>
            </div>
          ) : (
            <p className="hint">点击图中的对象节点，或在下方风险台账中选择记录，可在此查看对象画像。</p>
          )}
          {graph?.summary ? (
            <div className="graph-summary">
              <h4>网络概览</h4>
              <small>业务网络：{displayNetworkName(graph.summary)}</small>
              <small>对象规模：{String(graph.summary.node_count ?? "-")}</small>
              <small>关系规模：{String(graph.summary.edge_count ?? "-")}</small>
            </div>
          ) : null}
        </aside>
      </div>
      <p className="hint">{message}</p>
    </section>
  );
}
