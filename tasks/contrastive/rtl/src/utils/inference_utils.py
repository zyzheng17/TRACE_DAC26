import os 
import glob
import deepgate as dg
from torch_geometric.data import Data, InMemoryDataset, Batch
import torch
import numpy as np 
import random
import copy
import time
import argparse
import torch.nn.functional as F
import pandas as pd
import src.utils.circuit_utils as circuit_utils
import multiprocessing
from datetime import datetime 
from torch.utils.data import DataLoader, Dataset 

gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}
NODE_CONNECT_SAMPLE_RATIO = 0.1

K_HOP = 4

NO_NODES = [2, 10000]

import sys
sys.setrecursionlimit(1000000)

"""
input: .aig
output: .aig
func: read aig file, and prepare training label for DG model

"""


# def get_parse_args():
#     parser = argparse.ArgumentParser()

#     # Range
#     parser.add_argument('--start', default=0, type=int)
#     parser.add_argument('--end', default=100000, type=int)
    
#     # Input
#     parser.add_argument('--aig_dir', default='./data/aig/subgraph', type=str)
#     parser.add_argument('--csv_dir', default='./data/adder_meandata.csv', type=str)
    
#     # Output
#     parser.add_argument('--npz_path', default='./data/train_data/graphs.npz', type=str)
    
#     args = parser.parse_args()
    
#     return args

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):

        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'edge' in key:
            return 1
        elif key == 'forward_index' or key == 'backward_index':
            return 0
        else:
            return 0
class MyDataset(Dataset):  
    def __init__(self, data_list):  
        self.data_list = data_list  
  
    def __len__(self):  
        return len(self.data_list)  
  
    def __getitem__(self, idx):  
        return self.data_list[idx] 

class MyInformation:
    def __init__(self, cir_name,x_data, edge_index,forward_level,forward_index,backward_level,backward_index,x_one_hot):
        self.cir_name = cir_name
        self.x_data = x_data
        self.edge_index = edge_index
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
        self.x_one_hot = x_one_hot

def myaig_to_xdata(aig_filename, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2}):
    block_size = 8
    with open(aig_filename, 'rb') as file:   
        first_line = file.readline()
        first_line = first_line.decode('ascii')
        header = first_line.strip().split(" ")  
        n_variables = eval(header[1])
        n_inputs = eval(header[2])
        n_outputs = eval(header[4])
        n_and = eval(header[5])
        no_latch = eval(header[3])
        assert no_latch == 0, 'The AIG has latches.'
        x_data = []
        edge_index = []
        # PI 
        for i in range(n_inputs):
            x_data.append([len(x_data), gate_to_index['PI']])
        # AND 
        for i in range(n_and):
            x_data.append([len(x_data), gate_to_index['AND']])
        has_not = [-1] * (n_inputs+n_and)
        for i in range(n_outputs):
            line = file.readline()
        for i in range(n_and):
            t = 0
            child1 = 0
            child2 = 0
            while(True):
                block = file.read(1)
                unsigned_int = int.from_bytes(block, byteorder='little')
                child1 |= ( ( unsigned_int & 0x7f ) << ( 7 * t ) )
                if unsigned_int & 0x80 == 0:
                    break
                t += 1
            t = 0
            while(True):
                block = file.read(1)
                unsigned_int = int.from_bytes(block, byteorder='little')
                child2 |= ( ( unsigned_int & 0x7f ) << ( 7 * t ) )
                if unsigned_int & 0x80 == 0:
                    break
                t += 1
            fanin_1_index = int(int(2*(i + 1 + n_inputs) - child1)/2) - 1
            fanin_2_index = int(int(2*(i + 1 + n_inputs) - child1 - child2)/2) - 1
            fanin_1_not = int(2*(i + 1 + n_inputs) - child1) % 2
            fanin_2_not = int(2*(i + 1 + n_inputs) - child1 - child2) % 2
            if fanin_1_not == 1:
                if has_not[fanin_1_index] == -1:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not_index = len(x_data)-1
                    edge_index.append([fanin_1_index, not_index])
                    has_not[fanin_1_index] = not_index
                fanin_1_index = has_not[fanin_1_index]
            if fanin_2_not == 1:
                if has_not[fanin_2_index] == -1:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not_index = len(x_data)-1
                    edge_index.append([fanin_2_index, not_index])
                    has_not[fanin_2_index] = not_index
                fanin_2_index = has_not[fanin_2_index]
            #x_data.append([len(x_data), gate_to_index['AND']])
            edge_index.append([fanin_1_index , i + n_inputs])
            edge_index.append([fanin_2_index , i + n_inputs])
    with open(aig_filename, 'rb') as file:
        first_line = file.readline()  
        for i in range(n_outputs):
            line = file.readline()
            line = line.decode('ascii')
            arr = line.replace('\n', '').split(' ')
            if len(arr) != 1:
                continue
            po_index = int(int(arr[0]) / 2) - 1
            if po_index < 0:
                continue
            po_not = int(arr[0]) % 2
            if po_not == 1:
                if has_not[po_index] == -1:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not_index = len(x_data)-1
                    edge_index.append([po_index, not_index])
                    has_not[po_index] = not_index
    return x_data, edge_index


def mulprocessworker(aig_dir, aig_idx, cir_name):
    aig_file = os.path.join(aig_dir, cir_name + '.aig')
    x_data, edge_index = myaig_to_xdata(aig_file)
    #read csv
    
    # print('Parse: {} ({:} / {:}), Size: {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
    #     cir_name, aig_idx, no_circuits, len(x_data), 
    #     tot_time, tot_time / ((aig_idx + 1) / no_circuits) - tot_time, 
    #     len(graphs)
    # ))


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
    # xremove_unconnected_data, edgeremove_unconnected_index = circuit_utils.remove_unconnected(x_data, edge_index)
    if len(edge_index) == 0 or len(x_data) < NO_NODES[0] or len(x_data) > NO_NODES[1]:
        print(cir_name)
        return
    x_data = np.array(x_data)
    x_one_hot = dg.construct_node_feature(x_data, 3)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, len(x_data))
    re = MyInformation(cir_name,x_data, edge_index,forward_level,forward_index,backward_level,backward_index,x_one_hot)
    return re

def read_aigs(aig_dir, csv_path=None, batch_size = 1024):
    
    #1: get all aig paths in aig_dir
    aig_namelist_path = os.path.join(aig_dir, 'aig_namelist.txt')
    if not os.path.exists(aig_namelist_path):
        aig_files = glob.glob('{}/*.aig'.format(aig_dir))
        aig_namelist = []
        for aig_file in aig_files:
            with open(aig_file, 'rb') as f:
                firstline = f.readline().decode('ascii')
                firstline = firstline.strip()
                words = firstline.split(" ")
                if int(words[1]) < 512 and int(words[1]) > 2:
                    aig_name = os.path.basename(aig_file).replace('.aig', '')
                    aig_namelist.append(aig_name)     
        with open(aig_namelist_path, 'w') as f:
            for aig_name in aig_namelist:
                f.write(aig_name + '\n')
    else:
        with open(aig_namelist_path, 'r') as f:
            aig_namelist = f.readlines()
            aig_namelist = [x.strip() for x in aig_namelist]
    
    no_circuits = len(aig_namelist)
    tot_time = 0

    # mapping partitioned aigs to global node
    node_list = aig_namelist[:]

    global_node_list = [int(n.split('/')[-1].split('_')[1]) for n in node_list]
    global_node_list = torch.tensor(global_node_list)

    center_list = [int(n.split('/')[-1].split('_')[2]) for n in node_list]
    center_list = torch.tensor(center_list)

    AIGs = []
    graphs = {}
    tot_num_nodes = 0
    # if csv_path is not None, prepare classification label
    c = torch.zeros([global_node_list.shape[0],4]).float()
    if csv_path!=None:
        label = pd.read_csv(csv_path)
        c1 = label['c1'].to_numpy()
        c2 = label['c2'].to_numpy()
        c3 = label['c3'].to_numpy()
        c4 = label['c4'].to_numpy()

        c[global_node_list,0] = torch.tensor(c1).float()
        c[global_node_list,1] = torch.tensor(c2).float()
        c[global_node_list,2] = torch.tensor(c3).float()
        c[global_node_list,3] = torch.tensor(c4).float()

    stime = datetime.now()
    # with multiprocessing.Pool(processes=2) as pool:
    #     async_results = [] 
    #     for aig_idx, cir_name in enumerate(aig_namelist):
    #         async_results.append(pool.apply_async(mulprocessworker, (aig_dir, aig_idx, cir_name)))
    #     # pool = multiprocessing.Pool(processes=16)
    #     for ar in async_results:
    #         result = ar.get()
    #         graphs[result.cir_name] = result
    for aig_idx, cir_name in enumerate(aig_namelist):
        if aig_idx % batch_size == 0:
            tot_num_nodes = 0
            naig_idx = 0
        aig_file = os.path.join(aig_dir, cir_name + '.aig')
        # csv_file = os.path.join(csv_path, cir_name + '.csv')
        
        start_time = time.time()
        #read aig
        x_data, edge_index = myaig_to_xdata(aig_file)
        #read csv
        


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
        # xremove_unconnected_data, edgeremove_unconnected_index = circuit_utils.remove_unconnected(x_data, edge_index)
        if len(edge_index) == 0 or len(x_data) < NO_NODES[0] or len(x_data) > NO_NODES[1]:
            print(cir_name)
            continue
        x_data = np.array(x_data)
        x_one_hot = dg.construct_node_feature(x_data, 3)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, len(x_data))
        forward_level = torch.tensor(forward_level)
        forward_index = torch.tensor(forward_index)
        backward_level = torch.tensor(backward_level)
        backward_index = torch.tensor(backward_index)
        graph = OrderedData()
        graph.x = x_one_hot
        graph.edge_index = edge_index + tot_num_nodes
        # graph.name = cir_name
        graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long).unsqueeze(1)
        graph.forward_index = forward_index + tot_num_nodes
        graph.backward_index = backward_index + tot_num_nodes
        graph.forward_level = forward_level
        graph.backward_level = backward_level
        graph.batch = torch.ones_like(forward_index) * naig_idx

        # graph = OrderedData()
        # graph.x = graphs[cir_name].x_one_hot
        # graph.edge_index = graphs[cir_name].edge_index + tot_num_nodes
        # # graph.name = cir_name
        # graph.gate = torch.tensor(graphs[cir_name].x_data[:, 1], dtype=torch.long).unsqueeze(1)
        # graph.forward_index = graphs[cir_name].forward_index + tot_num_nodes
        # graph.backward_index = graphs[cir_name].backward_index + tot_num_nodes
        # graph.forward_level = graphs[cir_name].forward_level
        # graph.backward_level = graphs[cir_name].backward_level
        # graph.batch = torch.ones_like(graphs[cir_name].forward_index) * naig_idx
        
        global_node = cir_name
        global_node = int(cir_name.split('/')[-1].split('_')[1]) 
        global_node = torch.tensor(global_node)

        graph.global_node = global_node.long().unsqueeze(0)
        graph.center_node = torch.tensor(int(cir_name.split('/')[-1].split('_')[2])).long().unsqueeze(0) + tot_num_nodes
        global_virtual_edge = [[x ,graph.center_node] for x,_ in x_data]
        global_virtual_edge = torch.tensor(global_virtual_edge, dtype=torch.long).t().contiguous()
        graph.edge_index = torch.cat((global_virtual_edge, graph.edge_index), dim=1)
        if csv_path is not None:
            graph.c_label = c[global_node].unsqueeze(0)
        else:
            graph.c_label = torch.zeros(1,4)
        # for key in graph.keys():
        #     if key == 'connect_pair_index' and len(graph[key]) == 0:
        #         succ = False
        #         break
        #     if 'prob' in key or 'sim' in key or 'ratio' in key or 'ged' in key or 'label' in key:
        #         graph[key] = torch.tensor(graph[key], dtype=torch.float)

        #     else:
        #         graph[key] = torch.tensor(graph[key], dtype=torch.long)

        end_time = time.time()
        # tot_time += end_time - start_time
        # tot_num_nodes += graphs[cir_name].forward_index.shape[0]
        tot_num_nodes += forward_index.shape[0]
        naig_idx += 1
        
        # Save graph 
        AIGs.append(graph)
        # graphs[cir_name] = graph
    # for res in async_results:  
    #     result = res.get()  # 获取每个任务的返回值    
    #     AIGs.append(result)
    # dtime = datetime.now()
    # print(f"mulprocess {dtime - stime}")
    # batch_data,_ = InMemoryDataset().collate(AIGs)
    # batch = Batch.from_data_list(AIGs)
    # dataset = MyDataset(AIGs)  
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # batch_data,_ = InMemoryDataset().collate(AIGs)
    return AIGs

def newread_aigs(aig_dir, csv_path=None):
    aig_namelist_path = os.path.join(aig_dir, 'aig_namelist.txt')
    if not os.path.exists(aig_namelist_path):
        aig_files = glob.glob('{}/*.aig'.format(aig_dir))
        aig_namelist = []
        for aig_file in aig_files:
            with open(aig_file, 'rb') as f:
                firstline = f.readline().decode('ascii')
                firstline = firstline.strip()
                words = firstline.split(" ")
                if int(words[1]) < 256 and int(words[1]) > 2:
                    aig_name = os.path.basename(aig_file).replace('.aig', '')
                    aig_namelist.append(aig_name)     
        with open(aig_namelist_path, 'w') as f:
            for aig_name in aig_namelist:
                f.write(aig_name + '\n')
    else:
        with open(aig_namelist_path, 'r') as f:
            aig_namelist = f.readlines()
            aig_namelist = [x.strip() for x in aig_namelist]
    
    
    no_circuits = len(aig_namelist)
    tot_time = 0
    graphs = []
    node_list = aig_namelist[:]

    global_node_list = [int(n.split('/')[-1].split('_')[1]) for n in node_list]
    global_node_list = torch.tensor(global_node_list)

    center_list = [int(n.split('/')[-1].split('_')[2]) for n in node_list]
    center_list = torch.tensor(center_list)
    c = torch.zeros([global_node_list.shape[0],4]).float()
    if csv_path != None:
        label = pd.read_csv(csv_path)
        c1 = label['c1'].to_numpy()
        c2 = label['c2'].to_numpy()
        c3 = label['c3'].to_numpy()
        c4 = label['c4'].to_numpy()

        c[global_node_list,0] = torch.tensor(c1).float()
        c[global_node_list,1] = torch.tensor(c2).float()
        c[global_node_list,2] = torch.tensor(c3).float()
        c[global_node_list,3] = torch.tensor(c4).float()

    for aig_idx, cir_name in enumerate(aig_namelist):
        aig_file = os.path.join(aig_dir, cir_name + '.aig')
        # csv_file = os.path.join(args.csv_dir, cir_name + '.csv')
        
        start_time = time.time()
        with open(aig_file) as file:
            header = file.readline
        #read aig
        if header == '':
            continue
        x_data, edge_index = myaig_to_xdata(aig_file)
        #read csv
        
        print('Parse: {} ({:} / {:}), Size: {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
            cir_name, aig_idx, no_circuits, len(x_data), 
            tot_time, tot_time / ((aig_idx + 1) / no_circuits) - tot_time, 
            len(graphs)
        ))


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
        if len(edge_index) == 0 or len(x_data) < NO_NODES[0] or len(x_data) > NO_NODES[1]:
            print(cir_name)
            continue
        x_one_hot = dg.construct_node_feature(x_data, 3)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))
        
        graph = OrderedData()
        graph.x = x_one_hot
        graph.edge_index = edge_index
        graph.name = cir_name
        graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long).unsqueeze(1)
        graph.forward_index = forward_index
        graph.backward_index = backward_index
        graph.forward_level = forward_level
        graph.backward_level = backward_level
        
        global_node = cir_name
        global_node = int(cir_name.split('/')[-1].split('_')[1]) 
        global_node = torch.tensor(global_node)

        graph.global_node = global_node.long().unsqueeze(0)
        graph.c_label = c[global_node].unsqueeze(0)
        graph.center_node = torch.tensor(int(cir_name.split('/')[-1].split('_')[2])).long().unsqueeze(0)
        for key in graph.keys():
            if (key == 'connect_pair_index' and len(graph[key]) == 0) or key == 'name':
                succ = False
                break
            if 'prob' in key or 'sim' in key or 'ratio' in key or 'ged' in key or 'label' in key:
                graph[key] = torch.tensor(graph[key], dtype=torch.float)

            else:
                graph[key] = torch.tensor(graph[key], dtype=torch.long)


        #DG3 node level task
        # prob, tt_pair_index, tt_sim, con_index, con_label = circuit_utils.prepare_dg2_labels_cpp2(graph, 15000, fast=False, simulator='./simulator/simulator')
        # graph.connect_pair_index = torch.tensor(con_index).T
        # graph.connect_label = con_label
        
        # assert max(prob).item() <= 1.0 and min(prob).item() >= 0.0
        # if len(tt_pair_index) == 0:
        #     tt_pair_index = torch.zeros((2, 0), dtype=torch.long)
        # else:
        #     tt_pair_index = tt_pair_index.t().contiguous()
        # graph.prob = prob
        # graph.tt_pair_index = tt_pair_index
        # graph.tt_sim = tt_sim



        end_time = time.time()
        tot_time += end_time - start_time
        
        # Save graph 
        g = {}
        for key in graph.keys():
            if key == 'name' or key == 'batch' or key == 'ptr':
                continue
            if torch.is_tensor(graph[key]):
                g[key] = graph[key].cpu().numpy()
            else:
                g[key] = graph[key]
        graphs.append(graph)
    batch_data,_ = InMemoryDataset().collate(graphs)
    return batch_data


    