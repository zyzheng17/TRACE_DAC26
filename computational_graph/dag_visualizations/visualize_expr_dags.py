from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expr_graphs import EXPRESSIONS, NODE_TYPES, build_graph_template, topological_levels  # noqa: E402


ID_TO_NODE_TYPE = {idx: name for name, idx in NODE_TYPES.items()}

EXPRESSION_TITLES = {
    "add": r"$(x + y)\ \mathrm{mod}\ p$",
    "sub": r"$(x - y)\ \mathrm{mod}\ p$",
    "xy": r"$(x \cdot y)\ \mathrm{mod}\ p$",
    "x2_y2": r"$(x^2 + y^2)\ \mathrm{mod}\ p$",
    "x2_xy_y2": r"$(x^2 + xy + y^2)\ \mathrm{mod}\ p$",
    "x2_xy_y2_x": r"$(x^2 + xy + y^2 + x)\ \mathrm{mod}\ p$",
    "x3_xy": r"$(x^3 + xy)\ \mathrm{mod}\ p$",
    "x3_xy2_y": r"$(x^3 + xy^2 + y)\ \mathrm{mod}\ p$",
}

TYPE_COLORS = {
    "IN_X": "#60A5FA",
    "IN_Y": "#A78BFA",
    "ADD": "#34D399",
    "SUB": "#F97316",
    "MUL": "#FBBF24",
    "MOD": "#F87171",
}

TYPE_TEXT = {
    "IN_X": "x",
    "IN_Y": "y",
    "ADD": "+",
    "SUB": "-",
    "MUL": "*",
    "MOD": "mod p",
}


def node_positions(data):
    levels = topological_levels(data.num_nodes, data.edge_index)
    by_level = defaultdict(list)
    for node_idx, level in enumerate(levels.tolist()):
        by_level[level].append(node_idx)

    positions = {}
    max_width = max(len(nodes) for nodes in by_level.values())
    for level, nodes in sorted(by_level.items()):
        start_y = (len(nodes) - 1) / 2.0
        for row, node_idx in enumerate(nodes):
            positions[node_idx] = (level * 2.45, start_y - row)
    return positions, levels, max_width


def edge_radii(edges):
    counts = Counter(edges)
    seen = Counter()
    radii = []
    for edge in edges:
        count = counts[edge]
        seen[edge] += 1
        if count == 1:
            radii.append(0.0)
            continue
        offset = seen[edge] - (count + 1) / 2.0
        radii.append(offset * 0.34)
    return radii


def draw_node(ax, xy, label, color, is_output=False):
    x, y = xy
    width = 1.04
    height = 0.62
    shadow = FancyBboxPatch(
        (x - width / 2 + 0.04, y - height / 2 - 0.04),
        width,
        height,
        boxstyle="round,pad=0.05,rounding_size=0.13",
        linewidth=0,
        facecolor="#CBD5E1",
        alpha=0.35,
        zorder=2,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.05,rounding_size=0.13",
        linewidth=2.3 if is_output else 1.4,
        edgecolor="#7F1D1D" if is_output else "#334155",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
        zorder=4,
    )


def draw_edge(ax, start, end, position_idx, rad):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.65,
        color="#475569",
        shrinkA=26,
        shrinkB=28,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arrow)

    mx = (start[0] + end[0]) / 2.0
    my = (start[1] + end[1]) / 2.0 + rad * 0.75
    ax.text(
        mx,
        my,
        f"position {position_idx}",
        ha="center",
        va="center",
        fontsize=8,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.17", "fc": "#F8FAFC", "ec": "#CBD5E1", "lw": 0.7},
        zorder=5,
    )


def render_expression(expr_name, output_dir):
    data = build_graph_template(expr_name)
    positions, levels, max_width = node_positions(data)
    output_node = int(data.output_node_idx.item())
    edges = [tuple(edge) for edge in data.edge_index.t().tolist()]
    radii = edge_radii(edges)

    max_level = int(levels.max().item())
    fig_width = max(7.4, 2.15 * (max_level + 1))
    fig_height = max(4.2, 1.1 * max_width + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for edge_idx, ((src, dst), rad) in enumerate(zip(edges, radii)):
        draw_edge(ax, positions[src], positions[dst], int(data.edge_pos[edge_idx]), rad)

    for node_idx in range(data.num_nodes):
        node_type = ID_TO_NODE_TYPE[int(data.node_type[node_idx])]
        type_text = TYPE_TEXT[node_type]
        output_suffix = "\nOUT" if node_idx == output_node else ""
        label = f"v{node_idx}\n{type_text}{output_suffix}"
        draw_node(ax, positions[node_idx], label, TYPE_COLORS[node_type], node_idx == output_node)

    for level in range(max_level + 1):
        x = level * 2.45
        ax.text(
            x,
            -max_width / 2.0 - 0.85,
            f"level {level}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#64748B",
        )

    legend_handles = [
        Patch(facecolor=TYPE_COLORS[name], edgecolor="#334155", label=name)
        for name in ["IN_X", "IN_Y", "ADD", "SUB", "MUL", "MOD"]
        if any(ID_TO_NODE_TYPE[int(t)] == name for t in data.node_type.tolist())
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.5,
    )

    ax.set_title(f"{expr_name}: {EXPRESSION_TITLES[expr_name]}", fontsize=15, fontweight="bold", pad=18)
    ax.set_xlim(-0.9, max_level * 2.45 + 0.9)
    ax.set_ylim(-max_width / 2.0 - 1.35, max_width / 2.0 + 1.05)
    ax.axis("off")
    fig.tight_layout()

    png_path = output_dir / f"{expr_name}.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    png_path.chmod(0o644)
    return png_path


def write_gallery(output_dir, rendered):
    lines = [
        "# Computational Graph DAG Visualizations",
        "",
        "Each expression is rendered as an independent directed acyclic graph. Edge labels mark ordered operator inputs.",
        "",
    ]
    for expr_name, png_path in rendered:
        lines.extend(
            [
                f"## `{expr_name}`",
                "",
                f'<img src="./{png_path.name}" alt="{expr_name}" width="900">',
                "",
                f"- PNG: `{png_path.name}`",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    output_dir = Path(__file__).resolve().parent
    rendered = []
    for expr_name in EXPRESSIONS:
        png_path = render_expression(expr_name, output_dir)
        rendered.append((expr_name, png_path))
    write_gallery(output_dir, rendered)
    print(f"Rendered {len(rendered)} DAGs to {output_dir}")


if __name__ == "__main__":
    main()
