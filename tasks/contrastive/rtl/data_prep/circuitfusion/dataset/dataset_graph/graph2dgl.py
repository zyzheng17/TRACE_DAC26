import torch
import numpy as np
import pickle, json, time, re, sys
from DG import Graph, Node
import networkx as nx
from multiprocessing import Pool
import dgl
from dgl import from_networkx
import dgl



def node_feat_extra_word(node_name, node_class:Node, g_nx:nx.DiGraph, node_dict):
    ## 1. fanin
    fanin_iter = g_nx.predecessors(node_name)
    fanin = sum(1 for _ in fanin_iter)

    ## 2. fanout
    fanout_iter = g_nx.successors(node_name)
    fanout = sum(1 for _ in fanout_iter)
    ## 3. node type
    node_tpe_ori = node_class.type

    total_num = 23

    node_type_cp = [0 for i in range(total_num)]

    if node_tpe_ori in ['Reg']:
        node_type = node_type_cp.copy()
        node_type[0] = 1
    elif node_tpe_ori in ['Input', 'Inout']:
        node_type = node_type_cp.copy()
        node_type[1] = 1
    elif node_tpe_ori in ['Output']:
        node_type = node_type_cp.copy()
        node_type[2] = 1
    elif node_tpe_ori in ['Operator', 'UnaryOperator', 'Concat', 'Repeat']:
        op_temp = re.findall(r'([A-Z][a-z]*)(\d+)', node_name)
        op = op_temp[0][0]
        if op == 'Plus':
            node_type = node_type_cp.copy()
            node_type[5] = 1
        elif op == 'Times':
            node_type = node_type_cp.copy()
            node_type[6] = 1
        elif op in ['Minus', 'Uminus']:
            node_type = node_type_cp.copy()
            node_type[7] = 1
        elif op == 'Divide':
            node_type = node_type_cp.copy()
            node_type[8] = 1
        elif op in ['Mux', 'Cond', 'Case']:
            node_type = node_type_cp.copy()
            node_type[9] = 1
        elif op == 'Concat':
            node_type = node_type_cp.copy()
            node_type[10] = 1
        elif op in ['And', 'Land','Uand']:
            node_type = node_type_cp.copy()
            node_type[11] = 1
        elif op in ['Or', 'Lor','Uor']:
            node_type = node_type_cp.copy()
            node_type[12] = 1
        elif op in ['Unot', 'Ulnot']:
            node_type = node_type_cp.copy()
            node_type[13] = 1
        elif op in ['Xor', 'Uxor']:
            node_type = node_type_cp.copy()
            node_type[14] = 1
        elif op == 'Eq':
            node_type = node_type_cp.copy()
            node_type[15] = 1
        elif op == 'GreaterEq':
            node_type = node_type_cp.copy()
            node_type[16] = 1
        elif op == 'LessEq':
            node_type = node_type_cp.copy()
            node_type[17] = 1
        elif op in ['GreaterThan', 'Than']:
            node_type = node_type_cp.copy()
            node_type[18] = 1
        elif op == 'LessThan':
            node_type = node_type_cp.copy()
            node_type[19] = 1
        elif op in ['Sra', 'Srl']:
            node_type = node_type_cp.copy()
            node_type[20] = 1
        elif op in ['Sla', 'Sll']:
            node_type = node_type_cp.copy()
            node_type[21] = 1
        else:
            print(op)
            assert False
    elif node_tpe_ori in ['Concat', 'Repeat']:
        node_type = node_type_cp.copy()
        node_type[10] = 1
    elif node_tpe_ori in ['Constant']:
        node_type = node_type_cp.copy()
        node_type[4] = 1
    elif node_tpe_ori in ['Wire']:
        node_type = node_type_cp.copy()
        node_type[3] = 1
    elif node_tpe_ori in ['Partselect', 'Pointer']:
        if not node_class.father in node_dict:
            node_type = node_type_cp.copy()
            node_type[0] = 1
        elif node_dict[node_class.father].type in ['Reg']:
            node_type = node_type_cp.copy()
            node_type[0] = 1
        elif node_dict[node_class.father].type in ['Input', 'Inout']:
            node_type = node_type_cp.copy()
            node_type[1] = 1
        elif node_dict[node_class.father].type in ['Output']:
            node_type = node_type_cp.copy()
            node_type[2] = 1
        elif node_dict[node_class.father].type in ['Constant']:
            node_type = node_type_cp.copy()
            node_type[6] = 1
        elif node_dict[node_class.father].type in ['Wire']:
            node_type = node_type_cp.copy()
            node_type[5] = 1
        else:
            node_type = node_type_cp.copy()
            node_type[0] = 1

    # 4. node width
    node_width = node_dict[node_name].width
    
    if not node_width:
        node_width = float(1)

    feat_vec = [fanin, fanout, fanin+fanout, node_width]
    feat_vec.extend(node_type)

    
    # assert len(feat_vec) == 7
    # feat_vec = torch.FloatTensor(feat_vec)
    feat_vec = np.array(feat_vec)
    return feat_vec

def get_edge_node_tpe(node_name, node_dict):
    tpe_ = 0
    if not node_name in node_dict:
        tpe_ = 0
        return tpe_
    
    node_class = node_dict[node_name]
    node_tpe_ori = node_class.type
    if node_tpe_ori in ['Reg']:
        tpe_ = 1
    elif node_tpe_ori in ['Input', 'Inout']:
        tpe_ = 2
    elif node_tpe_ori in ['Output']:
        tpe_ = 3
    elif node_tpe_ori in ['Wire']:
        tpe_ = 4
    elif node_tpe_ori in ['Partselect', 'Pointer']:
        if not node_class.father in node_dict:
            tpe_ = 0
        elif node_dict[node_class.father].type in ['Reg']:
            tpe_ = 1
        elif node_dict[node_class.father].type in ['Input', 'Inout']:
            tpe_ = 2
        elif node_dict[node_class.father].type in ['Output']:
            tpe_ = 3
        elif node_dict[node_class.father].type in ['Constant']:
            tpe_ = 11
        elif node_dict[node_class.father].type in ['Wire']:
            tpe_ = 4
        else:
            tpe_ = 0
    elif node_tpe_ori in ['Operator', 'UnaryOperator', 'Concat', 'Repeat']:
        op_temp = re.findall(r'([A-Z][a-z]*)(\d+)', node_name)
        op = op_temp[0][0]
        if op in ['Plus', 'Minus', 'Uminus', 'Times', 'Divide']:
            tpe_ = 5
        elif op in ['Mux', 'Cond', 'Case']:
            tpe_ = 6
        elif op == 'Concat':
            tpe_ = 7
        elif op in ['And', 'Land','Uand', 'Or', 'Lor','Uor', 'Xor', 'Uxor', 'Unot', 'Ulnot']:
            tpe_ = 8
        elif op in ['Eq', 'GreaterEq', 'LessEq', 'GreaterThan', 'Than', 'LessThan']:
            tpe_ = 9
        elif op in ['Sra', 'Srl', 'Sla', 'Sll']:
            tpe_ = 10
        else:
            print(op)
            assert False
    elif node_tpe_ori in ['Concat', 'Repeat']:
        tpe_ = 7
    elif node_tpe_ori in ['Constant']:
        tpe_ = 11
    else:
        print(node_tpe_ori)
        assert False
    
    return tpe_

def edge_feat_extra_word(edge_pair, g_nx:nx.DiGraph, node_dict):
    edge_feat = [0 for i in range(12)]

    scr, dst = edge_pair
    if (scr not in node_dict) or (dst not in node_dict):
        return edge_feat
    scr_node = node_dict[scr]
    dst_node = node_dict[dst]
    
    scr_tpe = get_edge_node_tpe(scr, node_dict)
    dst_tpe = get_edge_node_tpe(dst, node_dict)
    ### one-hot encoding of edge feature ###
    
    edge_feat[scr_tpe] = 1
    edge_feat[dst_tpe] = 1
    edge_feat = edge_feat
    return edge_feat
    
