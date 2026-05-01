import cytoscape, { Core } from "cytoscape";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createGraphTask,
  fetchGraph,
  GraphNode,
  GraphResponse,
  listDatasets
} from "../services/api";
import type { AnalysisFrame } from "./PipelinePanel";

type Props = {
  datasetId: number | null;
  refreshKey: number;
  highlightedNodeId: string | null;
  timelineNodeId: string | null;
  analysisFrame?: AnalysisFrame | null;
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
  if (status === "building") return "正在构建关系图谱";
  if (status === "graph_ready") return "已构建关系图谱";
  if (status === "feature_ready") return "已完成处理";
  if (status === "inference_completed") return "已生成风险名单";
  return "待构建关系图谱";
}

const buildSteps = [
  { ratio: 0.18, title: "装载风险对象", detail: "对象节点开始进入图谱画布。" },
  { ratio: 0.42, title: "组织关系边", detail: "交易方向与关联关系逐步连接。" },
  { ratio: 0.72, title: "计算图谱布局", detail: "系统正在整理节点位置与关系层次。" },
  { ratio: 1, title: "图谱构建完成", detail: "最终关系图谱已生成。" }
];

function sanitizeGraph(data: GraphResponse): GraphResponse {
  const nodeIds = new Set(data.nodes.map((node) => node.id));
  return {
    ...data,
    edges: data.edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
  };
}

function graphSlice(data: GraphResponse, ratio: number): GraphResponse {
  const nodeCount = Math.max(1, Math.ceil(data.nodes.length * ratio));
  const nodes = data.nodes.slice(0, nodeCount);
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = data.edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
  return { ...data, nodes, edges };
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function disposeCytoscape(cy: Core | null) {
  if (!cy) return;
  try {
    cy.elements().stop(true, false);
    cy.destroy();
  } catch {
    // Cytoscape may already be tearing down during fast graph updates.
  }
}

export function GraphWorkspace({ datasetId, refreshKey, highlightedNodeId, timelineNodeId, analysisFrame, compact = false }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const nodeMapRef = useRef<Map<string, GraphNode>>(new Map());
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [message, setMessage] = useState("选择业务网络后，这里将展示对象关系结构。");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [datasetStatus, setDatasetStatus] = useState("");
  const [viewRequested, setViewRequested] = useState(false);
  const [busy, setBusy] = useState(false);
  const [buildStepIndex, setBuildStepIndex] = useState(-1);

  const activeNodeId = timelineNodeId ?? highlightedNodeId;
  const graphReady = isGraphReadyStatus(datasetStatus);
  const buildStep = buildStepIndex >= 0 ? buildSteps[buildStepIndex] : null;
  const buildProgress = buildStep ? Math.round(buildStep.ratio * 100) : 0;
  const showAnalysisAnimator = compact && Boolean(analysisFrame?.started);
  const animationStage = Math.max(0, analysisFrame?.stageIndex ?? 0);

  useEffect(() => {
    if (!datasetId) {
      setGraph(null);
      setSelectedNode(null);
      setDatasetStatus("");
      setViewRequested(false);
      setBuildStepIndex(-1);
      setMessage("选择业务网络后，这里将展示对象关系结构。");
      return;
    }
    setGraph(null);
    setSelectedNode(null);
    setViewRequested(false);
    setBuildStepIndex(-1);
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
        const visibleGraph = sanitizeGraph(data);
        setGraph(visibleGraph);
        const filteredCount = data.edges.length - visibleGraph.edges.length;
        setMessage(
          filteredCount > 0
            ? `已装载 ${visibleGraph.nodes.length} 个对象、${visibleGraph.edges.length} 条关系，已隐藏 ${filteredCount} 条跨页关系。`
            : `已装载 ${visibleGraph.nodes.length} 个对象、${visibleGraph.edges.length} 条关系。`
        );
      })
      .catch((error) => setMessage(error.message));
  }, [datasetId, graphReady, refreshKey, viewRequested]);

  const nodeMap = useMemo(() => {
    const entries = (graph?.nodes ?? []).map((node) => [node.id, node] as const);
    return new Map(entries);
  }, [graph]);

  useEffect(() => {
    nodeMapRef.current = nodeMap;
  }, [nodeMap]);

  useEffect(() => {
    if (activeNodeId) {
      setSelectedNode(nodeMap.get(activeNodeId) ?? null);
    }
  }, [activeNodeId, nodeMap]);

  useEffect(() => {
    if (!containerRef.current || !graph) {
      disposeCytoscape(cyRef.current);
      cyRef.current = null;
      return;
    }
    const nodeIds = new Set(graph.nodes.map((node) => node.id));
    const nodeElements = graph.nodes.map((node) => ({
      data: {
        id: node.id,
        label: node.label,
        size: node.size,
        color: node.color,
        riskScore: node.risk_score,
        riskLabel: node.risk_label,
        sourceType: node.source_type,
        featureCount: node.feature_count,
        focused: "no"
      }
    }));
    const edgeElements = graph.edges
      .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map((edge) => ({
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.edge_type,
          focused: "no",
          highlighted: edge.highlighted ? "yes" : "no"
        }
      }));
    const elements = [...nodeElements, ...edgeElements];
    try {
      let cy = cyRef.current;
      if (!cy) {
        cy = cytoscape({
          container: containerRef.current,
          elements: [],
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
          layout: { name: "preset" }
        });
        cy.on("tap", "node", (event) => {
          const nodeId = String(event.target.data("id"));
          setSelectedNode(nodeMapRef.current.get(nodeId) ?? null);
        });
        cyRef.current = cy;
      }

      const nextElementIds = new Set(elements.map((element) => element.data.id));
      cy.batch(() => {
        cy.elements()
          .filter((element) => !nextElementIds.has(element.id()))
          .remove();
        elements.forEach((element) => {
          const existing = cy.getElementById(element.data.id);
          if (existing.nonempty()) {
            existing.data(element.data);
          } else {
            cy.add(element);
          }
        });
      });

      cy.layout({
        name: "cose",
        animate: false,
        fit: true,
        padding: 36
      }).run();
    } catch (error) {
      setMessage(error instanceof Error ? `关系图谱渲染失败：${error.message}` : "关系图谱渲染失败。");
      return;
    }

  }, [graph]);

  useEffect(() => {
    const currentCy = cyRef.current;
    if (!currentCy) return;

    currentCy.batch(() => {
      currentCy.nodes().forEach((node) => {
        node.data("focused", node.id() === activeNodeId ? "yes" : "no");
      });
      currentCy.edges().forEach((edge) => {
        const sourceId = edge.source().id();
        const targetId = edge.target().id();
        edge.data("focused", sourceId === activeNodeId || targetId === activeNodeId ? "yes" : "no");
      });
    });

    if (activeNodeId) {
      const target = currentCy.getElementById(activeNodeId);
      if (target.nonempty()) {
        currentCy.stop();
        currentCy.animate({ center: { eles: target }, zoom: 1.12 }, { duration: 260 });
      }
    }
  }, [activeNodeId, graph]);

  useEffect(() => {
    return () => {
      disposeCytoscape(cyRef.current);
      cyRef.current = null;
    };
  }, []);

  async function handleBuildGraph() {
    if (!datasetId || graphReady || busy) return;
    setBusy(true);
    setGraph(null);
    setViewRequested(false);
    setSelectedNode(null);
    setDatasetStatus("building");
    setBuildStepIndex(0);
    setMessage("正在构建关系图谱。");
    try {
      const task = await createGraphTask(datasetId);
      const fullGraph = sanitizeGraph(await fetchGraph(datasetId));
      for (let index = 0; index < buildSteps.length; index += 1) {
        setBuildStepIndex(index);
        setMessage(`${buildSteps[index].title}：${buildSteps[index].detail}`);
        setGraph(graphSlice(fullGraph, buildSteps[index].ratio));
        await delay(index === buildSteps.length - 1 ? 700 : 950);
      }
      setGraph(fullGraph);
      setDatasetStatus("graph_ready");
      setBuildStepIndex(-1);
      setMessage(task.message || `关系图谱构建完成：${fullGraph.nodes.length} 个对象、${fullGraph.edges.length} 条关系。`);
    } catch (error) {
      setBuildStepIndex(-1);
      setDatasetStatus("");
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
    : busy
      ? "正在构建关系图谱"
      : graphReady
      ? "关系图谱已构建"
      : "关系图谱尚未构建";
  const emptyGraphText = !datasetId
    ? "请先进入“业务网络”栏目选择或导入一个网络。"
    : busy
      ? "对象和关系会逐步出现在画布中。"
      : graphReady
      ? "点击“查看关系图谱”载入对象关系。"
      : "请先点击“构建关系图谱”。";

  return (
    <section className={compact ? "panel graph-panel workspace-panel graph-page-panel compact-graph-panel app-panel" : "panel graph-panel workspace-panel graph-page-panel app-panel graph-investigation-panel"}>
      <div className="graph-panel-top app-section-heading">
        <div>
          <p className="eyebrow">Relationship Network</p>
          <h2>关系网络图</h2>
          <p>{graphStatusLabel(datasetStatus)}</p>
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

      <div className="graph-layout enterprise-graph-layout">
        <div className="graph-canvas enterprise-canvas">
          <div ref={containerRef} className="graph-cytoscape-layer" />
          {showAnalysisAnimator ? (
            <div className={`analysis-graph-animator stage-${animationStage}`} aria-hidden="true">
              <div className="analysis-graph-scan" />
              <div className="analysis-graph-particles">
                {Array.from({ length: 14 }).map((_, index) => <i key={index} />)}
              </div>
              <div className="analysis-graph-flows">
                {Array.from({ length: 6 }).map((_, index) => <span key={index} />)}
              </div>
              <div className="analysis-graph-orbit">
                <b />
                <b />
                <b />
              </div>
              <div className="analysis-graph-caption">
                <span>{analysisFrame?.stageLabel}</span>
                <strong>{analysisFrame?.title}</strong>
                <small>{analysisFrame?.progress}%</small>
              </div>
            </div>
          ) : null}
          {busy || buildStep ? (
            <div className="graph-build-overlay">
              <span>{buildStep?.title ?? "构建关系图谱"}</span>
              <strong>{buildProgress}%</strong>
              <i><b style={{ width: `${buildProgress}%` }} /></i>
              <small>{buildStep?.detail ?? "正在准备对象和关系。"}</small>
            </div>
          ) : null}
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
