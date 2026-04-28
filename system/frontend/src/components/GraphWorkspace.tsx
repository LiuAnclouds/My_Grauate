import cytoscape, { Core } from "cytoscape";
import { useEffect, useRef, useState } from "react";
import { fetchGraph, GraphResponse } from "../services/api";

type Props = {
  datasetId: number | null;
};

export function GraphWorkspace({ datasetId }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [message, setMessage] = useState("选择或上传一个数据集后展示图关系。");

  useEffect(() => {
    if (!datasetId) return;
    fetchGraph(datasetId)
      .then((data) => {
        setGraph(data);
        setMessage(`已加载 ${data.nodes.length} 个节点、${data.edges.length} 条边。`);
      })
      .catch((error) => setMessage(error.message));
  }, [datasetId]);

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
            color: node.color
          }
        })),
        ...graph.edges.map((edge) => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.edge_type
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
            color: "#1d2733",
            "font-size": 9,
            "text-outline-width": 2,
            "text-outline-color": "#ffffff"
          }
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "#9aa8b6",
            "target-arrow-color": "#9aa8b6",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier"
          }
        },
        {
          selector: "node:selected",
          style: {
            "background-color": "#1f7aec",
            "border-color": "#0a4fb3",
            "border-width": 3
          }
        }
      ],
      layout: {
        name: "cose",
        animate: true,
        fit: true,
        padding: 40
      }
    });
    return () => {
      cyRef.current?.destroy();
      cyRef.current = null;
    };
  }, [graph]);

  return (
    <section className="panel graph-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Graph View</p>
          <h2>关系图谱</h2>
        </div>
      </div>
      <div ref={containerRef} className="graph-canvas" />
      <p className="hint">{message}</p>
    </section>
  );
}
