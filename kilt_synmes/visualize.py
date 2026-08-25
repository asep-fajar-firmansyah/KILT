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


def render_html(record: dict) -> str:
    triples = [tuple(triple) for triple in record["candidate_triples"]]
    topics = set(record["meta"].get("topic_entities", []))
    entities = list(dict.fromkeys(entity for triple in triples for entity in (triple[0], triple[2])))
    positions = {}
    for index, entity in enumerate(entities):
        angle = 2 * math.pi * index / max(len(entities), 1) - math.pi / 2
        positions[entity] = (450 + 320 * math.cos(angle), 375 + 260 * math.sin(angle))

    marker = '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/></marker></defs>'
    edge_svg = []
    for head, relation, tail in triples:
        start_x, start_y = positions[head]
        end_x, end_y = positions[tail]
        direction_x, direction_y = end_x - start_x, end_y - start_y
        distance = max(math.hypot(direction_x, direction_y), 1)
        offset_x, offset_y = direction_x / distance * 62, direction_y / distance * 34
        line_start_x, line_start_y = start_x + offset_x, start_y + offset_y
        line_end_x, line_end_y = end_x - offset_x, end_y - offset_y
        label_x, label_y = (start_x + end_x) / 2, (start_y + end_y) / 2 - 8
        edge_svg.append(
            f'<line x1="{line_start_x:.1f}" y1="{line_start_y:.1f}" x2="{line_end_x:.1f}" y2="{line_end_y:.1f}" marker-end="url(#arrow)"/>'
            f'<text class="edge-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(abbreviated_relation(relation))}</text>'
        )

    node_svg = []
    for entity, (x, y) in positions.items():
        node_class = "topic" if entity in topics else "candidate"
        label = html.escape(entity)
        node_svg.append(
            f'<g class="node {node_class}" transform="translate({x:.1f},{y:.1f})">'
            '<ellipse rx="62" ry="34"/>'
            f'<text>{label}</text></g>'
        )

    rows = "".join(
        f"<tr><td>{html.escape(head)}</td><td>{html.escape(relation)}</td><td>{html.escape(tail)}</td></tr>"
        for head, relation, tail in triples
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
line {{ stroke: #64748b; stroke-width: 1.5; }}
.edge-label {{ fill: #334155; font: 12px sans-serif; text-anchor: middle; paint-order: stroke; stroke: #ffffff; stroke-width: 4px; stroke-linejoin: round; }}
.node ellipse {{ stroke-width: 2; }}
.node text {{ font: 12px sans-serif; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }}
.topic ellipse {{ fill: #bfdbfe; stroke: #1d4ed8; }}
.candidate ellipse {{ fill: #f1f5f9; stroke: #64748b; }}
.legend {{ display: flex; gap: 20px; margin: 16px 0 28px; font: 14px sans-serif; }}
.swatch {{ display: inline-block; width: 12px; height: 12px; border: 1px solid #475569; margin-right: 6px; }}
table {{ width: 100%; border-collapse: collapse; background: #ffffff; font: 14px sans-serif; }}
th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #e2e8f0; }}
</style>
</head>
<body>
<main>
<h1>Candidate Triple Retrieval: {graph_id}</h1>
<p class="question">{question}</p>
<svg class="graph" viewBox="0 0 900 750" role="img" aria-label="Candidate triple knowledge graph">{marker}{''.join(edge_svg)}{''.join(node_svg)}</svg>
<div class="legend"><span><span class="swatch" style="background:#bfdbfe"></span>Topic entity</span><span><span class="swatch" style="background:#f1f5f9"></span>Retrieved entity</span></div>
<h2>Retrieved Triples ({len(triples)})</h2>
<table><thead><tr><th>Head</th><th>Relation</th><th>Tail</th></tr></thead><tbody>{rows}</tbody></table>
</main>
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