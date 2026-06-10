# Tutorial: Computational Graphs

This tutorial is a small, self-contained setting for understanding TRACE's core idea on computational graphs.

## What Is A Computational Graph?

A computational graph is a directed graph `G = (V, E)` where:

- Nodes are operands or atomic operators, such as `IN_X`, `IN_Y`, `ADD`, `SUB`, `MUL`, and the final `MOD`.
- Directed edges carry values from producer nodes to consumer operator nodes.
- The graph topology defines the computation itself, not just a relational structure over passive entities.

The paper overview figures are included below.

![Computational graph overview](assets/overview.png)


In this tutorial, a graph might represent:

```text
z = x^2 + x*y + y^2 mod p
```

Each sample keeps the same graph topology but assigns different values to input nodes `x` and `y`. This tutorial runs one forward pass through TRACE to show how the graph is represented and how information flows through the model.

## What Makes A Good Computational Graph Model?

A good computational graph model should reflect the physics of execution.

- It must be position-aware. `x - y` and `y - x` have the same unordered neighbor set but different semantics.
- It must model joint interactions between operands. Multiplication is not the independent sum of two input messages.
- It must respect topological causality. A node should be updated only after its input values have been represented.

Standard message passing GNNs are often a poor fit because they aggregate neighbors as unordered sets. This is natural for many relational graphs, but not for functional operators.

## TRACE Idea In This Tutorial

The minimal TRACE encoder in `model.py` follows the computation order:

1. Embed each node by operation type and current value.
2. Visit nodes level by level in topological order.
3. For each operator node, collect its ordered input operands using edge-slot embeddings.
4. Apply a small Transformer over `[operator token, input slot 0, input slot 1, ...]`.
5. Read out the designated output node.

This mirrors the TRACE idea used in the main circuit tasks: preserve computational order, preserve operand slots, and model local operator interactions with an expressive set encoder.

## Expressions

Available expressions:

- `add`: `(x + y) mod p`
- `sub`: `(x - y) mod p`
- `xy`: `(x * y) mod p`
- `x2_y2`: `(x^2 + y^2) mod p`
- `x2_xy_y2`: `(x^2 + x*y + y^2) mod p`
- `x2_xy_y2_x`: `(x^2 + x*y + y^2 + x) mod p`
- `x3_xy`: `(x^3 + x*y) mod p`
- `x3_xy2_y`: `(x^3 + x*y^2 + y) mod p`

The `x^2` and `x^3` terms are expanded into multiplication nodes, not separate power operators.

## Run The Demo

Run the default demo:

```bash
cd TRACE/tasks/tutorial/computational_graph
python demo.py
```

Try a different expression and input assignment:

```bash
python demo.py --expr x3_xy2_y --x 7 --y 11
```

The demo prints:

- The computational graph nodes, edge slots, and topological levels.
- The TRACE encoder components and embedding sizes.
- The input tensor shapes passed to the model.
- The forward flow, level by level, including which source nodes feed each operator node.
- The untrained model output and the ground-truth arithmetic value.

This is intentionally not a training script. The model uses random weights, so the scalar prediction is only used to illustrate the input/output interface.

CUDA is used by default when available. Add `--use_cpu` to force a CPU-only run.

## Files

- `expr_graphs.py`: expression templates and computational graph construction.
- `data.py`: modular arithmetic ground-truth functions.
- `model.py`: minimal TRACE-style encoder.
- `demo.py`: one-pass tutorial that prints graph construction, model architecture, forward flow, and input/output tensors.
- `assets/`: figures copied from the computational-graph paper draft.
