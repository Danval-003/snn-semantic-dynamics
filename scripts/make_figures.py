#!/usr/bin/env python3
"""Genera la figura factorial central como SVG sin dependencias adicionales."""

from __future__ import annotations

import json
from pathlib import Path


WIDTH, HEIGHT = 1200, 390
COLORS = {
    "orth+_semantic+": "#168aad",
    "orth+_semantic-": "#f77f00",
    "orth-_semantic+": "#52b788",
    "orth-_semantic-": "#d62828",
    "semantic": "#2a9d8f",
    "orthographic": "#e76f51",
    "rate": "#5a189a",
    "temporal": "#4361ee",
}


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;")


def panel(svg, x0, title, series, y_min, y_max, zero=False):
    left, top, width, height = x0 + 48, 52, 315, 250
    svg.append(f'<text x="{x0 + 190}" y="24" text-anchor="middle" class="title">{esc(title)}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" class="axis"/>')
    if zero and y_min < 0 < y_max:
        zy = top + height * (y_max / (y_max - y_min))
        svg.append(f'<line x1="{left}" y1="{zy:.1f}" x2="{left+width}" y2="{zy:.1f}" class="zero"/>')
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = top + height - height * tick / 4
        svg.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.1f}</text>')
    xs = [left + 35, left + width / 2, left + width - 35]
    for index, x in enumerate(xs, start=1):
        svg.append(f'<text x="{x:.1f}" y="{top+height+22}" text-anchor="middle" class="tick">L{index}</text>')
    for name, values, color in series:
        points = []
        for x, value in zip(xs, values, strict=True):
            y = top + height - (value - y_min) / (y_max - y_min) * height
            points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    legend_y = top + height + 47
    cursor = left
    for name, _, color in series:
        svg.append(f'<line x1="{cursor}" y1="{legend_y}" x2="{cursor+16}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{cursor+21}" y="{legend_y+4}" class="legend">{esc(name)}</text>')
        cursor += 21 + len(name) * 6.2 + 18


def main():
    summary = json.loads(Path("runs/factorial/summary.json").read_text(encoding="utf-8"))
    aggregate = summary["aggregate"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        """<style>
        .title{font:600 15px sans-serif;fill:#172b4d}.tick,.legend{font:11px sans-serif;fill:#42526e}
        .axis{stroke:#42526e;stroke-width:1}.zero{stroke:#8993a4;stroke-width:1;stroke-dasharray:5 4}
        </style>""",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    cell_labels = {
        "orth+_semantic+": "orth+/sem+",
        "orth+_semantic-": "orth+/sem−",
        "orth-_semantic+": "orth−/sem+",
        "orth-_semantic-": "orth−/sem−",
    }
    cell_series = [
        (
            cell_labels[cell],
            [row["van_rossum_mean"]["mean"] for row in aggregate["cells"][cell]],
            COLORS[cell],
        )
        for cell in cell_labels
    ]
    rate_effects = aggregate["effects"]["rate"]
    effect_series = [
        ("semantic S", [row["semantic_effect_relative"]["mean"] for row in rate_effects], COLORS["semantic"]),
        ("orthographic O", [row["orthographic_effect_relative"]["mean"] for row in rate_effects], COLORS["orthographic"]),
    ]
    abstraction_series = [
        (
            "Van Rossum",
            [row["abstraction_index_relative"]["mean"] for row in aggregate["effects"]["van_rossum"]],
            COLORS["temporal"],
        ),
        (
            "Population/rate",
            [row["abstraction_index_relative"]["mean"] for row in rate_effects],
            COLORS["rate"],
        ),
    ]
    panel(svg, 0, "A  Factorial distances", cell_series, 0.0, 1.8)
    panel(svg, 400, "B  Relative effects (rate)", effect_series, 0.0, 1.4)
    panel(svg, 800, "C  Abstraction A = S − O", abstraction_series, -1.1, 0.4, zero=True)
    svg.append("</svg>")
    output = Path("reports/figures/factorial_abstraction.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
