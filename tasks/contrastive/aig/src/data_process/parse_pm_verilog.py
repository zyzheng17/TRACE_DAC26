import argparse
import csv
from pathlib import Path
import re

import deepgate as dg
import networkx as nx
import torch
import torch_geometric


DEFAULT_CELL_CSV = Path(__file__).with_name('sky130.csv')


def parse_verilog_to_graph(file_path):
    graph = nx.DiGraph()
    content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
    module_match = re.search(r'module\s+(\S+)\s*\(', content)
    if module_match:
        graph.graph['module_name'] = module_match.group(1)

    inputs = re.findall(r'input\s+([\w,\s]+);', content)
    outputs = re.findall(r'output\s+([\w,\s]+);', content)
    inputs = [item.strip() for group in inputs for item in group.split(',')]
    outputs = [item.strip() for group in outputs for item in group.split(',')]

    for signal in inputs:
        graph.add_node(signal, gate_type='input')

    net_dict = {}
    gate_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(([^;]+)\);')
    connection_pattern = re.compile(r'\.(\w+)\((\w+)\)')
    for gate_type, gate_name, connections in gate_pattern.findall(content):
        graph.add_node(gate_name, gate_type=gate_type)
        for pin, net in connection_pattern.findall(connections):
            if net in inputs:
                graph.add_edge(net, gate_name)
            elif net in outputs:
                continue
            else:
                entry = net_dict.setdefault(net, {'i': [], 'o': []})
                entry['i' if pin in ['X', 'Y', 'z'] else 'o'].append(gate_name)

    for entry in net_dict.values():
        for dst in entry['i']:
            for src in entry['o']:
                graph.add_edge(src, dst)
    return graph


def bitstring_to_tensor(bitstring):
    if len(bitstring) != 64 or any(char not in '01' for char in bitstring):
        raise ValueError('cell encoding must be a 64-bit 0/1 string')
    return torch.tensor([int(char) for char in bitstring], dtype=torch.float32)


def read_csv(file_path):
    result = {}
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 3:
                value = row[2].strip()
                result[row[0].strip()] = bitstring_to_tensor(value * (64 // len(value)))
    result['input'] = bitstring_to_tensor('0' * 64)
    return result


def verilog2graph(file_path, cell_csv=DEFAULT_CELL_CSV):
    cell_lib = read_csv(cell_csv)
    graph = torch_geometric.utils.from_networkx(parse_verilog_to_graph(file_path))
    graph.x = torch.stack([cell_lib[gate_type] for gate_type in graph.gate_type])
    del graph.module_name
    del graph.gate_type
    forward_level, forward_index, backward_level, backward_index = dg.return_order_info(graph.edge_index, graph.x.shape[0])
    graph.forward_level = forward_level
    graph.forward_index = forward_index
    graph.backward_level = backward_level
    graph.backward_index = backward_index
    return graph


def parse_args():
    parser = argparse.ArgumentParser(description='Parse a mapped PM Verilog netlist into a PyG graph.')
    parser.add_argument('verilog', type=Path)
    parser.add_argument('--cell_csv', type=Path, default=DEFAULT_CELL_CSV)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(verilog2graph(args.verilog, args.cell_csv))
