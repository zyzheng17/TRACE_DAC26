import argparse

import torch
from torch_geometric.loader import DataLoader

from data import compute_expression
from expr_graphs import NODE_TYPES, add_sample_values, build_graph_template
from model import TRACEComputationalGraph


ID_TO_NODE_TYPE = {idx: name for name, idx in NODE_TYPES.items()}


def parse_args():
    parser = argparse.ArgumentParser(description='Tutorial: inspect TRACE forward on a computational graph.')
    parser.add_argument('--expr', default='x2_xy_y2', choices=['add', 'sub', 'xy', 'x2_y2', 'x2_xy_y2', 'x2_xy_y2_x', 'x3_xy', 'x3_xy2_y'])
    parser.add_argument('--x', default=3, type=int)
    parser.add_argument('--y', default=5, type=int)
    parser.add_argument('--p', default=97, type=int)
    parser.add_argument('--hidden', default=64, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    return parser.parse_args()


def print_graph(data):
    print('\n== Computational Graph ==')
    print(f'expr: {data.expr_name}')
    print(f'num_nodes: {data.num_nodes}')
    print(f'output_node_idx: {int(data.output_node_idx.item())}')

    print('\nNodes:')
    for node_idx in range(data.num_nodes):
        node_type = ID_TO_NODE_TYPE[int(data.node_type[node_idx])]
        value = int(data.node_value[node_idx])
        level = int(data.forward_level[node_idx])
        print(f'  v{node_idx}: type={node_type:<5} input_value={value:<3} level={level}')

    print('\nEdges:')
    for edge_idx, (src, dst) in enumerate(data.edge_index.t().tolist()):
        slot = int(data.edge_pos[edge_idx])
        print(f'  e{edge_idx}: v{src} -> v{dst} slot={slot}')


def print_model(model):
    print('\n== TRACE Encoder ==')
    print(f'type_embedding: {tuple(model.type_embedding.weight.shape)}')
    print(f'value_embedding: {tuple(model.value_embedding.weight.shape)}')
    print(f'slot_embedding: {tuple(model.slot_embedding.weight.shape)}')
    print(f'operator_encoder_layers: {len(model.operator_encoder.layers)}')
    print(f'hidden_dim: {model.hidden}')
    print('readout: output-node embedding -> scalar prediction in [0, 1]')


def print_forward_trace(trace):
    print('\n== Forward Flow ==')
    for item in trace:
        print(f"Level {item['level']}: Transformer input shape {item['sequence_shape']}")
        for op in item['operators']:
            sources = ', '.join(f"v{s}" for s in op['sources'])
            slots = ', '.join(str(s) for s in op['slots'])
            print(f"  target v{op['target']} receives [{sources}] at slots [{slots}]")


def main():
    args = parse_args()
    device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else f'cuda:{args.gpu_id}')

    template = build_graph_template(args.expr)
    data = add_sample_values(template, args.x, args.y)
    target = compute_expression(args.expr, args.x, args.y, args.p)
    data.y = torch.tensor([target / (args.p - 1)], dtype=torch.float32)

    loader = DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    model = TRACEComputationalGraph(
        p=args.p,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    model.eval()

    print_graph(data)
    print_model(model)

    print('\n== Model Input Tensors ==')
    print(f'node_type shape: {tuple(batch.node_type.shape)}')
    print(f'node_value shape: {tuple(batch.node_value.shape)}')
    print(f'edge_index shape: {tuple(batch.edge_index.shape)}')
    print(f'edge_pos shape: {tuple(batch.edge_pos.shape)}')
    print(f'forward_level shape: {tuple(batch.forward_level.shape)}')

    with torch.no_grad():
        pred, trace = model(batch, return_trace=True)

    print_forward_trace(trace)

    pred_value = float(pred.item() * (args.p - 1))
    print('\n== Output ==')
    print(f'untrained_model_prediction_normalized: {float(pred.item()):.4f}')
    print(f'untrained_model_prediction_value: {pred_value:.2f}')
    print(f'ground_truth_value: {target}')
    print('\nNote: this tutorial runs one forward pass with random weights. It demonstrates TRACE architecture and data flow, not trained accuracy.')


if __name__ == '__main__':
    main()
