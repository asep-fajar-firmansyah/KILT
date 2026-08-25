"""Render SynMeS candidate-triple retrieval records as self-contained HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def load_record(path: Path, record_index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == record_index:
                return json.loads(line)
    raise IndexError(f"Record index {record_index} does not exist in {path}")


def abbreviated_relation(relation: str) -> str:
    return relation.replace("_", " ").replace(".", " ")


def cluster_positions(record: dict, entities: list[str]) -> tuple[dict[str, tuple[float, float]], set[str], list[tuple[str, float, float]]]:
    topics = [str(topic) for topic in record["meta"].get("topic_entities", [])]
    ownership = {entity: set() for entity in entities}
    for item in record.get("provenance", []):
        owner = item.get("topic_entity")
        if owner not in topics:
            continue
        head, _, tail = item["triple"]
        ownership.setdefault(head, set()).add(owner)
        ownership.setdefault(tail, set()).add(owner)

    centers = []
    for index, topic in enumerate(topics):
        angle = 2 * math.pi * index / max(len(topics), 1) - math.pi / 2
        centers.append((topic, 550 + 320 * math.cos(angle), 410 + 250 * math.sin(angle)))

    positions = {}
    bridge_entities = set()
    for topic, center_x, center_y in centers:
        cluster_entities = [
            entity for entity in entities if ownership.get(entity) == {topic} and entity != topic
        ]
        positions[topic] = (center_x, center_y)
        for index, entity in enumerate(cluster_entities):
            angle = 2 * math.pi * index / max(len(cluster_entities), 1) - math.pi / 2
            positions[entity] = (center_x + 125 * math.cos(angle), center_y + 95 * math.sin(angle))

    shared_entities = [entity for entity in entities if entity not in positions]
    for index, entity in enumerate(shared_entities):
        angle = 2 * math.pi * index / max(len(shared_entities), 1) - math.pi / 2
        positions[entity] = (550 + 120 * math.cos(angle), 410 + 90 * math.sin(angle))
        bridge_entities.add(entity)
    return positions, bridge_entities, centers


def render_html(record: dict) -> str:
    triples = [tuple(triple) for triple in record["candidate_triples"]]
    topics = set(record["meta"].get("topic_entities", []))
    entities = list(dict.fromkeys(entity for triple in triples for entity in (triple[0], triple[2])))
    positions, bridge_entities, centers = cluster_positions(record, entities)
    node_ids = {entity: f"node-{index}" for index, entity in enumerate(entities)}

    marker = '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/></marker></defs>'
    cluster_svg = "".join(
        f'<g class="cluster"><circle cx="{center_x:.1f}" cy="{center_y:.1f}" r="165"/>'
        f'<text x="{center_x:.1f}" y="{max(center_y - 145, 24):.1f}">{html.escape(topic)}</text></g>'
        for topic, center_x, center_y in centers
    )
    edge_svg = []
    for index, (head, relation, tail) in enumerate(triples):
        start_x, start_y = positions[head]
        end_x, end_y = positions[tail]
        direction_x, direction_y = end_x - start_x, end_y - start_y
        distance = max(math.hypot(direction_x, direction_y), 1)
        offset_x, offset_y = direction_x / distance * 62, direction_y / distance * 34
        line_start_x, line_start_y = start_x + offset_x, start_y + offset_y
        line_end_x, line_end_y = end_x - offset_x, end_y - offset_y
        label_x, label_y = (start_x + end_x) / 2, (start_y + end_y) / 2 - 8
        edge_svg.append(
            f'<g class="edge" data-head="{node_ids[head]}" data-tail="{node_ids[tail]}">'
            f'<line x1="{line_start_x:.1f}" y1="{line_start_y:.1f}" x2="{line_end_x:.1f}" y2="{line_end_y:.1f}" marker-end="url(#arrow)"/>'
            f'<text class="edge-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(abbreviated_relation(relation))}</text></g>'
        )

    node_svg = []
    for entity, (x, y) in positions.items():
        node_class = "topic" if entity in topics else "bridge" if entity in bridge_entities else "candidate"
        label = html.escape(entity)
        node_svg.append(
            f'<g id="{node_ids[entity]}" class="node {node_class}" transform="translate({x:.1f},{y:.1f})">'
            '<ellipse rx="62" ry="34"/>'
            f'<text>{label}</text></g>'
        )

    grouped_provenance = {topic: [] for topic in record["meta"].get("topic_entities", [])}
    shared_provenance = []
    provenance = record.get("provenance", [])
    for item in provenance:
        owner = item.get("topic_entity")
        if owner in grouped_provenance:
            grouped_provenance[owner].append(item)
        else:
            shared_provenance.append(item)
    if not provenance:
        for triple in triples:
            owners = [topic for topic in grouped_provenance if topic in {triple[0], triple[2]}]
            item = {"triple": triple}
            if owners:
                grouped_provenance[owners[0]].append(item)
            else:
                shared_provenance.append(item)

    table_groups = []
    for topic, items in grouped_provenance.items():
        if items:
            table_groups.append((topic, items))
    if shared_provenance:
        table_groups.append(("Shared evidence", shared_provenance))

    rows = "".join(
        f'<tr class="group-row"><th colspan="3">{html.escape(topic)} ({len(items)} triples)</th></tr>'
        + "".join(
            f"<tr><td>{html.escape(item['triple'][0])}</td><td>{html.escape(item['triple'][1])}</td><td>{html.escape(item['triple'][2])}</td></tr>"
            for item in items
        )
        for topic, items in table_groups
    )
    question = html.escape(record["question"])
    graph_id = html.escape(str(record["id"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Candidate Triple Retrieval - {graph_id}</title>
<style>
body {{ margin: 0; background: #f8fafc; color: #172033; font-family: Georgia, serif; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 32px; }}
h1 {{ font-size: 24px; margin: 0 0 8px; }}
.question {{ max-width: 900px; line-height: 1.5; margin: 0 0 24px; }}
.graph {{ width: 100%; height: auto; background: #ffffff; border: 1px solid #cbd5e1; }}
.cluster circle {{ fill: #eff6ff; stroke: #93c5fd; stroke-dasharray: 6 5; }}
.cluster text {{ fill: #1e3a8a; font: 13px sans-serif; font-weight: 700; text-anchor: middle; }}
line {{ stroke: #64748b; stroke-width: 1.5; }}
.edge-label {{ fill: #334155; font: 12px sans-serif; text-anchor: middle; paint-order: stroke; stroke: #ffffff; stroke-width: 4px; stroke-linejoin: round; }}
.node {{ cursor: grab; }}
.node.dragging {{ cursor: grabbing; }}
.node ellipse {{ stroke-width: 2; }}
.node text {{ font: 12px sans-serif; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }}
.topic ellipse {{ fill: #bfdbfe; stroke: #1d4ed8; }}
.bridge ellipse {{ fill: #fde68a; stroke: #b45309; }}
.candidate ellipse {{ fill: #f1f5f9; stroke: #64748b; }}
.legend {{ display: flex; gap: 20px; margin: 16px 0 28px; font: 14px sans-serif; }}
.swatch {{ display: inline-block; width: 12px; height: 12px; border: 1px solid #475569; margin-right: 6px; }}
table {{ width: 100%; border-collapse: collapse; background: #ffffff; font: 14px sans-serif; }}
th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #e2e8f0; }}
.group-row th {{ background: #dbeafe; color: #1e3a8a; font-size: 15px; text-align: left; }}
</style>
</head>
<body>
<main>
<h1>Candidate Triple Retrieval: {graph_id}</h1>
<p class="question">{question}</p>
<svg class="graph" viewBox="0 0 1100 820" role="img" aria-label="Candidate triple knowledge graph">{marker}{cluster_svg}{''.join(edge_svg)}{''.join(node_svg)}</svg>
<div class="legend"><span><span class="swatch" style="background:#bfdbfe"></span>Topic entity</span><span><span class="swatch" style="background:#fde68a"></span>Shared bridge</span><span><span class="swatch" style="background:#f1f5f9"></span>Topic-owned retrieval</span><span>Drag any node to rearrange the graph.</span></div>
<h2>Retrieved Triples ({len(triples)})</h2>
<table><thead><tr><th>Head</th><th>Relation</th><th>Tail</th></tr></thead><tbody>{rows}</tbody></table>
</main>
<script>
const graph = document.querySelector('.graph');
let activeNode = null;

function positionOf(node) {{
    const match = node.getAttribute('transform').match(/translate\\(([-\\d.]+),([-.\\d]+)\\)/);
    return {{ x: Number(match[1]), y: Number(match[2]) }};
}}

function updateEdges() {{
    document.querySelectorAll('.edge').forEach((edge) => {{
        const head = positionOf(document.getElementById(edge.dataset.head));
        const tail = positionOf(document.getElementById(edge.dataset.tail));
        const dx = tail.x - head.x;
        const dy = tail.y - head.y;
        const distance = Math.max(Math.hypot(dx, dy), 1);
        const offsetX = dx / distance * 62;
        const offsetY = dy / distance * 34;
        const line = edge.querySelector('line');
        const label = edge.querySelector('text');
        line.setAttribute('x1', head.x + offsetX);
        line.setAttribute('y1', head.y + offsetY);
        line.setAttribute('x2', tail.x - offsetX);
        line.setAttribute('y2', tail.y - offsetY);
        label.setAttribute('x', (head.x + tail.x) / 2);
        label.setAttribute('y', (head.y + tail.y) / 2 - 8);
    }});
}}

function svgPoint(event) {{
    const point = graph.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(graph.getScreenCTM().inverse());
}}

graph.addEventListener('pointerdown', (event) => {{
    const node = event.target.closest('.node');
    if (!node) return;
    activeNode = node;
    activeNode.classList.add('dragging');
    graph.setPointerCapture(event.pointerId);
}});

graph.addEventListener('pointermove', (event) => {{
    if (!activeNode) return;
    const point = svgPoint(event);
    activeNode.setAttribute('transform', `translate(${{point.x.toFixed(1)}},${{point.y.toFixed(1)}})`);
    updateEdges();
}});

function endDrag(event) {{
    if (!activeNode) return;
    activeNode.classList.remove('dragging');
    if (graph.hasPointerCapture(event.pointerId)) graph.releasePointerCapture(event.pointerId);
    activeNode = null;
}}

graph.addEventListener('pointerup', endDrag);
graph.addEventListener('pointercancel', endDrag);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Retrieval JSONL output path.")
    parser.add_argument("--output", required=True, type=Path, help="HTML file to create.")
    parser.add_argument("--record-index", type=int, default=0, help="Zero-based JSONL record index.")
    args = parser.parse_args()
    if args.record_index < 0:
        parser.error("--record-index must be non-negative")
    args.output.write_text(render_html(load_record(args.input, args.record_index)), encoding="utf-8")
    print(f"Wrote visualization to {args.output}")


if __name__ == "__main__":
    main()