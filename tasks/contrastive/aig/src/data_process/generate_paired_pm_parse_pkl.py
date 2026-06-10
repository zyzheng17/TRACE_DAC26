import os 
import glob
import deepgate as dg
from torch_geometric.data import Data, InMemoryDataset, Batch
import torch
import pickle
import numpy as np 
import random
import copy
import time
import argparse
import torch.nn.functional as F

import networkx as nx
import utils.aiger_utils as aiger_utils
import utils.circuit_utils as circuit_utils
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from parse_pm_verilog import verilog2graph

gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}


def get_parse_args():
    parser = argparse.ArgumentParser()

    # Range
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=200000, type=int)
    
    # Input
    parser.add_argument('--pm_dir', default='../../../../data/pm_netlist/raw', type=str)
    
    # Output
    parser.add_argument('--save_path', default='../../../../data/pm_netlist/processed/pm_pair_pkl', type=str)
    
    args = parser.parse_args()
    
    return args

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'forward_index'] :
            return self.forward_index.shape[0]
        elif key in ['rd_edge_index', 'rd_forward_index'] :
            return self.rd_forward_index.shape[0]
        elif key in ['syn_edge_index', 'syn_forward_index'] :
            return self.syn_forward_index.shape[0]
        elif key in ['pm_edge_index', 'pm_forward_index'] :
            return self.pm_forward_index.shape[0]
        elif 'batch' in key:
            return 1
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if  "edge_index" in key :
            return 1
        else:
            return 0



def parse_aig(aig_file, cir_name):
    x_data, edge_index = aiger_utils.aig_to_xdata(aig_file)
    fanin_list, fanout_list = circuit_utils.get_fanin_fanout(x_data, edge_index)
    
    # Replace DFF as PPI and PPO
    no_ff = 0
    for idx in range(len(x_data)):
        if x_data[idx][1] == gate_to_index['DFF']:
            no_ff += 1
            x_data[idx][1] = gate_to_index['PI']
            for fanin_idx in fanin_list[idx]:
                fanout_list[fanin_idx].remove(idx)
            fanin_list[idx] = []
    
    # Get x_data and edge_index
    edge_index = []
    for idx in range(len(x_data)):
        for fanin_idx in fanin_list[idx]:
            edge_index.append([fanin_idx, idx])

    x_data, edge_index = circuit_utils.remove_unconnected(x_data, edge_index)

    if len(edge_index) == 0:
        return None
    

    x_one_hot = dg.construct_node_feature(x_data, 3)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))
    
    # print(f'node: {x_data.shape},max level: {forward_level.max()}')
    # break

    graph = OrderedData()
    graph.x = x_one_hot
    graph.edge_index = edge_index
    graph.name = cir_name
    graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long).unsqueeze(1)
    graph.forward_index = forward_index
    # graph.backward_index = backward_index
    graph.forward_level = forward_level
    graph.backward_level = backward_level
    return graph
    
def check_not(graphs):
    for key in graphs.keys():
        graph = graphs[key]
        edge_index = graph.edge_index
        x = graph.x
        gate = graph.gate
        edge_index = edge_index.t().contiguous()
        edge_index = edge_index.numpy()
        x = x.numpy()
        gate = gate.numpy()
        for edge in edge_index:
            if gate[edge[0]] == 2 and gate[edge[1]] == 2:
                print(f'{key} 2 not gates')

def add_virtual_node(graph):    
    graph.x = torch.cat([graph.x, torch.zeros(graph.x.shape[0],1)], dim=-1)
    graph.x = torch.cat([graph.x, torch.tensor([[0,0,0,1]])], dim=0)

    virtual_node_index = graph.forward_index[-1] + 1
    virtual_edge_index = torch.stack([graph.forward_index, virtual_node_index*torch.ones_like(graph.forward_index)])

    # graph.edge_index = torch.cat([graph.edge_index,virtual_edge], dim=0)

    graph.gate = torch.cat([graph.gate, torch.tensor([[3]])], dim=0)
    graph.forward_index = torch.cat([graph.forward_index, torch.tensor([virtual_node_index])], dim=0)
    graph.forward_level = torch.cat([graph.forward_level, torch.tensor([-1])], dim=0)
    graph.backward_level = torch.cat([graph.backward_level, torch.tensor([-1])], dim=0)

    graph.virtual_node_index = torch.tensor([virtual_node_index])
    graph.virtual_edge_index = virtual_edge_index

    return graph

def combine_aig(data):
    args, aig_idx, cir_name = data
    syn_pm_file = os.path.join(args.pm_dir, cir_name + '_syn.v')
    pm_file = os.path.join(args.pm_dir, cir_name + '.v')

    start_time = time.time()

    try:
        syn_pm = verilog2graph(syn_pm_file)
        map_aig = verilog2graph(pm_file)
    except Exception as e:
        print(f"Error parsing AIG files for {cir_name}: {e}")
        return

    if  syn_pm is None:
        return  
    
    if aig_idx%100 == 0:
        print('Parse: {} ({:})'.format(
        cir_name, aig_idx))

    
    graphs = OrderedData()

    graphs.syn_x = syn_pm.x
    graphs.syn_edge_index = syn_pm.edge_index
    graphs.syn_forward_index = syn_pm.forward_index
    graphs.syn_forward_level = syn_pm.forward_level
    graphs.syn_backward_level = syn_pm.backward_level
    graphs.syn_batch = torch.zeros(graphs.syn_x.shape[0], dtype=torch.long)

    graphs.pm_x = map_aig.x
    graphs.pm_edge_index = map_aig.edge_index
    graphs.pm_forward_index = map_aig.forward_index
    graphs.pm_forward_level = map_aig.forward_level
    graphs.pm_backward_level = map_aig.backward_level
    graphs.pm_batch = torch.zeros(graphs.pm_x.shape[0], dtype=torch.long)

    
    if aig_idx%100 == 0:
        print(f'{cir_name} finished')

    output_file = os.path.join(args.save_path, f'{cir_name}.pt')
    # output_file = f"<output_dir>/{cir_name}.pkl"
    torch.save(graphs, output_file)
    # with open(output_file, 'wb') as f:
    #     pickle.dump(graphs, f)


    # return graphs


if __name__ == '__main__':    
    args = get_parse_args()
    
    aig_files = glob.glob('{}/*.v'.format(args.pm_dir))
    aig_namelist = []
    for aig_file in aig_files:
        aig_name = os.path.basename(aig_file).replace('.v', '')
        if aig_name.split('_')[-1] == 'rd' or aig_name.split('_')[-1] == 'syn':
            continue
        aig_namelist.append(aig_name)
    
    no_circuits = len(aig_namelist)
    tot_time = 0

    ori_num = []
    rd_num = []
    syn_num = []
    ori_num_and = []
    rd_num_and = []
    syn_num_and = []


    data = [(args,aig_idx, cir_name) for aig_idx, cir_name in enumerate(aig_namelist)]

    combine_aig(data[0])

    num_workers = 8
    with Pool(num_workers) as pool:
        # graph_list = pool.map(combine_aig, data)
        pool.map(combine_aig, data)

    # output_file = "<output_dir>/combined_graphs_pm.pkl"
    # with open(output_file, 'wb') as f:
    #     pickle.dump(graph_list, f)
    # print(f"save {len(graph_list)} graphs to {output_file}")
    print('finish all')
