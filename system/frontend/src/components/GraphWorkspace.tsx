import cytoscape, { Core } from "cytoscape";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchGraph, GraphNode, GraphResponse } from "../services/api";

type Props = {
  datasetId: number | null;
  refreshKey: number;
  highlightedNodeId: string | null;
  timelineNodeId: string | null;
};

export function GraphWorkspace({ datasetId, refreshKey, highlightedNodeId, timelineNodeId }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [message, setMessage] = useState("选择分析资产后，这里将展示关系网络主舞台。");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  const activeNodeId = timelineNodeId ?? highlightedNodeId;

  useEffect(() => {
    if (!datasetId) {
      setGraph(null);
      setSelectedNode(null);
      return;
    }
    fetchGraph(datasetId)
      .then((data) => {
        setGraph(data);
        setMessage(`已装载 ${data.nodes.length} 个对象、${data.edges.length} 条关系。`);
      })
      .catch((error) => setMessage(error.message));
  }, [datasetId, refreshKey]);

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

  return (
    <section className="panel graph-panel workspace-panel">
      <div className="panel-heading aligned-start">
        <div>
          <p className="eyebrow">Relationship Network</p>
          <h2>关系网络主舞台</h2>
          <p className="section-copy">集中查看对象关联结构、风险着色结果与任务过程中的节点联动。</p>
        </div>
        <div className="legend-row enterprise-legend">
          <span><i className="legend-dot blue" /> 当前定位对象</span>
          <span><i className="legend-dot red" /> 高风险对象</span>
          <span><i className="legend-dot green" /> 低风险对象</span>
        </div>
      </div>

      <div className="graph-layout enterprise-graph-layout">
        <div ref={containerRef} className="graph-canvas enterprise-canvas" />
        <aside className="graph-sidecard insight-card">
          <h3>对象画像</h3>
          {selectedNode ? (
            <div className="detail-list">
              <div><span>对象编号</span><strong>{selectedNode.id}</strong></div>
              <div><span>姓名</span><strong>{selectedNode.label}</strong></div>
              <div><span>区域</span><strong>{selectedNode.region}</strong></div>
              <div><span>职业</span><strong>{selectedNode.occupation}</strong></div>
              <div><span>来源类型</span><strong>{selectedNode.source_type ?? "dataset"}</strong></div>
              <div><span>特征维度</span><strong>{selectedNode.feature_count}</strong></div>
              <div><span>风险标签</span><strong>{selectedNode.risk_label === "suspicious" ? "高风险" : selectedNode.risk_label === "normal" ? "低风险" : "待分析"}</strong></div>
              <div><span>风险分数</span><strong>{selectedNode.risk_score?.toFixed(4) ?? "-"}</strong></div>
            </div>
          ) : (
            <p className="hint">点击图中的对象节点，或在下方风险台账中选择记录，可在此查看对象画像。</p>
          )}
          {graph?.summary ? (
            <div className="graph-summary">
              <h4>网络概览</h4>
              <small>数据资产：{String(graph.summary.business_name ?? "-")}</small>
              <small>对象规模：{String(graph.summary.node_count ?? "-")}</small>
              <small>关系规模：{String(graph.summary.edge_count ?? "-")}</small>
              <small>特征维度：{String(graph.summary.feature_count ?? "-")}</small>
            </div>
          ) : null}
        </aside>
      </div>
      <p className="hint">{message}</p>
    </section>
  );
}
