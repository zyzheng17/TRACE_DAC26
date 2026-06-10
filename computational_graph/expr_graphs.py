import torch
from torch_geometric.data import Data


NODE_TYPES = {
    'IN_X': 0,
    'IN_Y': 1,
    'ADD': 2,
    'SUB': 3,
    'MUL': 4,
    'MOD': 5,
}

EXPRESSIONS = ['add', 'sub', 'xy', 'x2_y2', 'x2_xy_y2', 'x2_xy_y2_x', 'x3_xy', 'x3_xy2_y']


def _data(expr_name, node_types, edges, output_node_idx):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_pos = torch.zeros(edge_index.size(1), dtype=torch.long)
    for target in range(len(node_types)):
        target_edges = (edge_index[1] == target).nonzero(as_tuple=True)[0]
        for slot, edge_idx in enumerate(target_edges):
            edge_pos[edge_idx] = slot

    return Data(
        node_type=torch.tensor(node_types, dtype=torch.long),
        edge_index=edge_index,
        edge_pos=edge_pos,
        output_node_idx=torch.tensor([output_node_idx], dtype=torch.long),
        expr_name=expr_name,
        num_nodes=len(node_types),
    )


def build_graph_template(expr_name):
    t = NODE_TYPES
    builders = {
        'add': lambda: _data('add', [t['IN_X'], t['IN_Y'], t['ADD'], t['MOD']], [(0, 2), (1, 2), (2, 3)], 3),
        'sub': lambda: _data('sub', [t['IN_X'], t['IN_Y'], t['SUB'], t['MOD']], [(0, 2), (1, 2), (2, 3)], 3),
        'xy': lambda: _data('xy', [t['IN_X'], t['IN_Y'], t['MUL'], t['MOD']], [(0, 2), (1, 2), (2, 3)], 3),
        'x2_y2': lambda: _data(
            'x2_y2',
            [t['IN_X'], t['IN_Y'], t['MUL'], t['MUL'], t['ADD'], t['MOD']],
            [(0, 2), (0, 2), (1, 3), (1, 3), (2, 4), (3, 4), (4, 5)],
            5,
        ),
        'x2_xy_y2': lambda: _data(
            'x2_xy_y2',
            [t['IN_X'], t['IN_Y'], t['MUL'], t['MUL'], t['MUL'], t['ADD'], t['ADD'], t['MOD']],
            [(0, 2), (0, 2), (0, 3), (1, 3), (1, 4), (1, 4), (2, 5), (3, 5), (5, 6), (4, 6), (6, 7)],
            7,
        ),
        'x2_xy_y2_x': lambda: _data(
            'x2_xy_y2_x',
            [t['IN_X'], t['IN_Y'], t['MUL'], t['MUL'], t['MUL'], t['ADD'], t['ADD'], t['ADD'], t['MOD']],
            [(0, 2), (0, 2), (0, 3), (1, 3), (1, 4), (1, 4), (2, 5), (3, 5), (5, 6), (4, 6), (6, 7), (0, 7), (7, 8)],
            8,
        ),
        'x3_xy': lambda: _data(
            'x3_xy',
            [t['IN_X'], t['IN_Y'], t['MUL'], t['MUL'], t['MUL'], t['ADD'], t['MOD']],
            [(0, 2), (0, 2), (2, 3), (0, 3), (0, 4), (1, 4), (3, 5), (4, 5), (5, 6)],
            6,
        ),
        'x3_xy2_y': lambda: _data(
            'x3_xy2_y',
            [t['IN_X'], t['IN_Y'], t['MUL'], t['MUL'], t['MUL'], t['MUL'], t['ADD'], t['ADD'], t['MOD']],
            [(0, 2), (0, 2), (2, 3), (0, 3), (1, 4), (1, 4), (0, 5), (4, 5), (3, 6), (5, 6), (6, 7), (1, 7), (7, 8)],
            8,
        ),
    }
    if expr_name not in builders:
        raise ValueError(f'Unknown expression {expr_name}. Available: {sorted(builders)}')
    return builders[expr_name]()


def add_sample_values(template, x_val, y_val):
    data = template.clone()
    values = torch.zeros(data.num_nodes, dtype=torch.long)
    values[(data.node_type == NODE_TYPES['IN_X']).nonzero(as_tuple=True)[0]] = int(x_val)
    values[(data.node_type == NODE_TYPES['IN_Y']).nonzero(as_tuple=True)[0]] = int(y_val)
    data.node_value = values
    data.forward_level = topological_levels(data.num_nodes, data.edge_index)
    return data


def topological_levels(num_nodes, edge_index):
    levels = torch.zeros(num_nodes, dtype=torch.long)
    for node in range(num_nodes):
        incoming = edge_index[0, edge_index[1] == node]
        if incoming.numel() > 0:
            levels[node] = levels[incoming].max() + 1
    return levels
