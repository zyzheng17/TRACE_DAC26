from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


EXAMPLES = {
    "aig_not_a_and_b": {
        "title": "AIG: Y = NOT(A) AND B",
        "nodes": [
            ("A", "PI"),
            ("B", "PI"),
            ("inv_A", "NOT"),
            ("Y", "AND"),
        ],
        "edges": [(0, 2, 0), (2, 3, 0), (1, 3, 1)],
        "output": 3,
        "positions": {
            0: (0.0, 0.75),
            1: (0.0, -0.75),
            2: (2.55, 0.75),
            3: (5.10, 0.0),
        },
    },
    "pm_inv_nand2": {
        "title": "Post-Mapping Netlist: Y = INV(NAND2(A, B))",
        "nodes": [
            ("A", "PI"),
            ("B", "PI"),
            ("U1", "NAND2"),
            ("Y", "INV"),
        ],
        "edges": [(0, 2, 0), (1, 2, 1), (2, 3, 0)],
        "output": 3,
    },
    "rtl_register_wire_update": {
        "title": "RTL Data Flow: R2 <= R0 + R1",
        "nodes": [
            ("R0", "REG"),
            ("R1", "REG"),
            ("", "ADD"),
            ("R2", "REG"),
        ],
        "edges": [(0, 2, 0), (1, 2, 1), (2, 3, 0)],
        "output": 3,
    },
}

TYPE_COLORS = {
    "PI": "#60A5FA",
    "SIGNAL": "#93C5FD",
    "NOT": "#F97316",
    "AND": "#34D399",
    "NAND2": "#FBBF24",
    "INV": "#F97316",
    "ADD": "#A78BFA",
    "WIRE": "#38BDF8",
    "EQ": "#C084FC",
    "MUX": "#818CF8",
    "REG": "#F87171",
}


def topological_levels(num_nodes, edges):
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    indegree = [0] * num_nodes
    for src, dst, _ in edges:
        incoming[dst].append(src)
        outgoing[src].append(dst)
        indegree[dst] += 1

    queue = deque(node for node, degree in enumerate(indegree) if degree == 0)
    levels = [0] * num_nodes
    while queue:
        node = queue.popleft()
        for dst in outgoing[node]:
            levels[dst] = max(levels[dst], levels[node] + 1)
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    return levels


def node_positions(num_nodes, edges):
    levels = topological_levels(num_nodes, edges)
    by_level = defaultdict(list)
    for node_idx, level in enumerate(levels):
        by_level[level].append(node_idx)

    positions = {}
    max_width = max(len(nodes) for nodes in by_level.values())
    for level, nodes in sorted(by_level.items()):
        start_y = (len(nodes) - 1) / 2.0
        for row, node_idx in enumerate(nodes):
            positions[node_idx] = (level * 2.55, start_y - row)
    return positions, levels, max_width


def edge_radii(edges):
    edge_pairs = [(src, dst) for src, dst, _ in edges]
    counts = Counter(edge_pairs)
    seen = Counter()
    radii = []
    for edge in edge_pairs:
        count = counts[edge]
        seen[edge] += 1
        if count == 1:
            radii.append(0.0)
            continue
        offset = seen[edge] - (count + 1) / 2.0
        radii.append(offset * 0.34)
    return radii


def draw_node(ax, xy, node_name, node_type, is_output=False):
    x, y = xy
    width = 1.12
    height = 0.68
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
        facecolor=TYPE_COLORS[node_type],
        zorder=3,
    )
    ax.add_patch(box)
    output_suffix = "\nOUT" if is_output else ""
    label = node_type if not node_name else f"{node_name}\n{node_type}"
    ax.text(
        x,
        y,
        f"{label}{output_suffix}",
        ha="center",
        va="center",
        fontsize=9.5,
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
        shrinkA=28,
        shrinkB=30,
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


def render_example(name, example, output_dir):
    nodes = example["nodes"]
    edges = example["edges"]
    positions, levels, max_width = node_positions(len(nodes), edges)
    if "positions" in example:
        positions.update(example["positions"])
        max_width = max(max_width, 2)
    radii = edge_radii(edges)
    output_node = example["output"]

    max_level = max(levels)
    fig_width = max(7.0, 2.25 * (max_level + 1))
    fig_height = max(4.2, 1.1 * max_width + 2.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for (src, dst, position_idx), rad in zip(edges, radii):
        draw_edge(ax, positions[src], positions[dst], position_idx, rad)

    for node_idx, (node_name, node_type) in enumerate(nodes):
        draw_node(ax, positions[node_idx], node_name, node_type, node_idx == output_node)

    for level in range(max_level + 1):
        ax.text(
            level * 2.55,
            -max_width / 2.0 - 0.9,
            f"level {level}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#64748B",
        )

    used_types = []
    for _, node_type in nodes:
        if node_type not in used_types:
            used_types.append(node_type)
    legend_handles = [
        Patch(facecolor=TYPE_COLORS[node_type], edgecolor="#334155", label=node_type)
        for node_type in used_types
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.5,
    )

    ax.set_title(example["title"], fontsize=14, fontweight="bold", pad=18)
    ax.set_xlim(-0.9, max_level * 2.55 + 0.9)
    ax.set_ylim(-max_width / 2.0 - 1.35, max_width / 2.0 + 1.05)
    ax.axis("off")
    fig.tight_layout()

    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    png_path.chmod(0o644)
    return png_path


def write_gallery(output_dir, rendered):
    lines = [
        "# Circuit Modality Computational Graph Examples",
        "",
        "Each image shows how one circuit representation can be viewed as an ordered computational graph.",
        "",
    ]
    for name, png_path in rendered:
        lines.extend(
            [
                f"## `{name}`",
                "",
                f'<img src="./{png_path.name}" alt="{name}" width="900">',
                "",
            ]
        )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    readme_path.chmod(0o644)


def main():
    output_dir = Path(__file__).resolve().parent
    rendered = []
    for name, example in EXAMPLES.items():
        rendered.append((name, render_example(name, example, output_dir)))
    write_gallery(output_dir, rendered)
    print(f"Rendered {len(rendered)} circuit graphs to {output_dir}")


if __name__ == "__main__":
    main()
