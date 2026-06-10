import re
import networkx as nx
import torch_geometric
import csv
import torch
import deepgate as dg
from torch_geometric.data import Data
import numpy as np
import copy
import random
# from cut_cone_ziyang import random_bfs_sample_node
from collections import deque, defaultdict
import glob
import os
from torch_geometric.utils import to_networkx

def count_predecessors_dag(graph):
    """
    统计有向无环图中每个节点的前驱节点个数
    :param graph: 以邻接表形式表示的有向无环图，graph[u] 是节点 u 的所有直接后继节点的列表
    :return: 一个字典，键是节点，值是该节点的前驱节点个数
    """
    # 初始化每个节点的前驱节点个数为0
    predecessors_count = {}
    in_degree = {}
    for u in graph:
        predecessors_count[u] = 0
        in_degree[u] = 0
        for v in graph[u]:
            predecessors_count[v] = 0
            in_degree[v] = 0

    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # 使用队列进行拓扑排序
    from collections import deque
    queue = deque([node for node in graph if in_degree[node] == 0])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            predecessors_count[v] += predecessors_count[u] + 1
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return predecessors_count

def has_cycle(edge_index, num_nodes):
    """
    使用拓扑排序检测图中是否存在环。
    
    参数:
    - edge_index: torch.Tensor，形状为 [2, num_edges]，表示图的边索引。
    - num_nodes: int，图中节点的数量。

    返回:
    - bool: 如果存在环，返回 True；否则返回 False。
    """
    # 构建入度表和邻接表
    in_degree = torch.zeros(num_nodes, dtype=torch.int)
    adjacency_list = defaultdict(list)

    for src, dst in edge_index.tolist():
        in_degree[dst] += 1
        adjacency_list[src].append(dst)

    # 将入度为 0 的节点加入队列
    queue = deque([node for node in range(num_nodes) if in_degree[node] == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1
        for neighbor in adjacency_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 如果访问的节点数小于总节点数，说明存在环
    return visited_count < num_nodes


def random_bfs_sample_node(sub_x_data, sub_edge, root, visit_ratio=0.8):
    # 构建图
    graph = defaultdict(list)
    ori_graph = defaultdict(list)
    for u, v in sub_edge:
        ori_graph[u].append(v)
        graph[v].append(u)

    node_cnt = count_predecessors_dag(ori_graph)

    # with 50% probability, we start from the root node
    # Otherwise we start from a random predecessor node 
    if random.random() < 0.5:
        pre = graph[root]
        cnt = [node_cnt[p] for p in pre]
        new_root = torch.argmax(torch.tensor(cnt)).item()
        root = pre[new_root]

    # total_nodes = sub_x_data.shape[0]
    total_nodes = len(sub_x_data)
    target_visit_count = int(total_nodes * visit_ratio)

    visited = set()
    queue = deque([root])
    result_nodes = set()
    result_edges = set()

    while queue and len(result_nodes) < target_visit_count:
        node = queue.popleft()

        if node not in visited:
            visited.add(node)
            result_nodes.add(node)

            neighbors = graph[node]
            random.shuffle(neighbors)  # 随机打乱邻居节点顺序
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
                    result_edges.add((node, neighbor))
                    result_nodes.add(neighbor)

    return result_nodes, result_edges, root


def top_sort(edge_index, graph_size):

    cell_ids = torch.arange(graph_size, dtype=int)

    node_order = torch.zeros(graph_size, dtype=int)
    unevaluated_nodes = torch.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~torch.isin(cell_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return node_order.long()

def return_order_info(edge_index, num_nodes):
    ns = torch.LongTensor([i for i in range(num_nodes)])
    forward_level = top_sort(edge_index, num_nodes)
    ei2 = torch.LongTensor([list(edge_index[1]), list(edge_index[0])])
    backward_level = top_sort(ei2, num_nodes)
    forward_index = ns
    backward_index = torch.LongTensor([i for i in range(num_nodes)])
    
    return forward_level, forward_index, backward_level, backward_index

def parse_cell_to_aig(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 提取模块
    modules = re.findall(r'module\s+(\w+)\s*\((.*?)\);(.*?)endmodule', content, re.S)

    graphs = {}

    for module_name, ports, body in modules:
        # 提取输入、输出和中间信号
        inputs = re.findall(r'input\s+(\w+);', body)
        outputs = re.findall(r'output\s+(\w+);', body)
        wires = re.findall(r'wire\s+(\w+);', body)

        nodes = wires
        node_indices = {node: i for i, node in enumerate(nodes)}

        # 边列表
        edge_index = []

        # 解析 assign 语句
        gate_type = {}
        assigns = re.findall(r'assign\s+(\w+)\s*=\s*(.*?);', body)
        for target, expr in assigns:
            expr = expr.strip()

            if '&' in expr:  # AND 操作
                gate_type[target] = 'AND'
                sources = [s.strip() for s in expr.split('&')]
                for source in sources:
                    edge_index.append([node_indices[source], node_indices[target]])
            elif '~' in expr:  # NOT 操作

                gate_type[target] = 'NOT'
                source = expr.replace('~', '').strip()
                edge_index.append([node_indices[source], node_indices[target]])

        # 转换为 PyG 的张量格式
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
       
        gate_type_list = []
        
        type2id = {
            'AND': 1,
            'NOT': 2,
        }

        for node in nodes:
            if node in gate_type:
                gate_type_list.append(type2id[gate_type[node]])
            elif node in inputs:
                gate_type_list.append(node)

        # edge_index = remove_assign_gates(edge_index, gate_type_list)

        # 创建 PyG 图对象
        graph = Data(
            forward_index=torch.arange(0,len(nodes)),  # 节点特征
            edge_index=edge_index,  # 边索引
            gate_type = gate_type_list,
        )
        if edge_index.shape[0] == 0:
            return None
        forward_level, forward_index, backward_level, backward_index = return_order_info(graph.edge_index, graph.forward_index.shape[0])
        graph.forward_level = forward_level
        graph.forward_index = forward_index
        graph.backward_level = backward_level
        graph.backward_index = backward_index
        graphs[module_name] = graph

    return graphs

def remove_pin_node(graph):

    pin_mask = ~np.isin(graph.gate_type, np.array(['input','1','2']))

    assign_nodes = graph.forward_index[pin_mask]

    # 转置 edge_index 方便操作
    edge_index = graph.edge_index.t()  # 转换为 [num_edges, 2]

    # 新的边列表
    new_edges = []

    # 遍历所有 assign 节点
    for node in assign_nodes:
        # 找到所有入边和出边
        in_edges = edge_index[edge_index[:, 1] == node]  # 入边：目标是该节点
        out_edges = edge_index[edge_index[:, 0] == node]  # 出边：源是该节点

        # 在入点和出点之间添加新边
        for in_edge in in_edges:
            for out_edge in out_edges:
                new_edges.append([in_edge[0].item(), out_edge[1].item()])

    # 移除 assign 节点的边
    edge_index = edge_index[
        (edge_index[:, 0] != assign_nodes.unsqueeze(1)).all(dim=0) &
        (edge_index[:, 1] != assign_nodes.unsqueeze(1)).all(dim=0)
    ]

    # 添加新边
    new_edges = torch.tensor(new_edges, dtype=torch.long)
    edge_index = torch.cat([edge_index, new_edges], dim=0).t()

    # 更新 gate_type，移除 assign 节点
    graph.edge_index = edge_index
    graph.gate_type = graph.gate_type[~pin_mask]
    graph.forward_index = graph.forward_index[~pin_mask]
    graph.aig_to_cell = graph.aig_to_cell[~pin_mask]
    
    return graph

def replace_node_with_subgraph_pyg(graph, cell_id, subgraph, subgraph_name):
    """
    将一个节点替换为一个子图。

    参数:
    - graph: 原始 PyG 图 (Data 对象)
    - cell_id: 要替换的节点 ID
    - subgraph: 要插入的子图 (Data 对象)
    - entry_nodes: 子图的入口节点索引列表
    - exit_nodes: 子图的出口节点索引列表

    返回:
    - graph: 更新后的 PyG 图
    """
    entry_nodes = subgraph.forward_index[torch.logical_and(subgraph.forward_level==0 , subgraph.backward_level!=0)]
    exit_nodes = subgraph.forward_index[torch.logical_and(subgraph.forward_level!=0 , subgraph.backward_level==0)]
    # 获取目标节点的入边和出边
    edge_index = graph.edge_index
    in_edges = edge_index[:, edge_index[1] == cell_id]  # 入边
    out_edges = edge_index[:, edge_index[0] == cell_id]  # 出边

    out_edges_pin = np.array(graph.edge_pin_o)[graph.edge_index[0] == cell_id]
    pins_i = np.array(graph.edge_pin_o)[graph.edge_index[1] == cell_id]
    pins_o = np.array(subgraph.gate_type)[(subgraph.forward_level==0) & (subgraph.backward_level!=0)]
    assert len(pins_i) == len(pins_o)


    # 删除目标节点及其相关边
    mask = (edge_index[0] != cell_id) & (edge_index[1] != cell_id)
    edge_index = edge_index[:, mask]
    graph.edge_pin_i = np.array(graph.edge_pin_i)[mask]
    graph.edge_pin_o = np.array(graph.edge_pin_o)[mask]

    # 添加子图的节点和边
    num_original_nodes = graph.forward_index.shape[0]
    subgraph_edge_index = subgraph.edge_index + num_original_nodes
    edge_index = torch.cat([edge_index, subgraph_edge_index], dim=1)
    graph.edge_pin_i = np.concatenate([graph.edge_pin_i, np.array(len(subgraph_edge_index[1]) * ['None'])])
    graph.edge_pin_o = np.concatenate([graph.edge_pin_o, np.array(len(subgraph_edge_index[1]) * ['None'])])



    # 将入边连接到子图的入口节点, 并且根据pin的类型连接
    for pin_i,u in zip(pins_i,in_edges[0]):
        for pin_o,entry_node in zip(pins_o,entry_nodes):
            if pin_i == pin_o: 
                edge_index = torch.cat([edge_index, torch.tensor([[u, entry_node + num_original_nodes]], dtype=torch.long).t()], dim=1)
                graph.edge_pin_i = np.concatenate([graph.edge_pin_i, np.array(['None'])])
                graph.edge_pin_o = np.concatenate([graph.edge_pin_o, np.array(['None'])])

    # 将子图的出口节点连接到出边的目标节点
    for pin, v in zip(out_edges_pin,out_edges[1]):
        for exit_node in exit_nodes:
            edge_index = torch.cat([edge_index, torch.tensor([[exit_node + num_original_nodes, v]], dtype=torch.long).t()], dim=1)
            graph.edge_pin_i = np.concatenate([graph.edge_pin_i, np.array(['None'])])
            graph.edge_pin_o = np.concatenate([graph.edge_pin_o, np.array([pin])])
            

    # 更新节点特征
    # graph.x = torch.cat([graph.x, subgraph.x], dim=0)
    graph.gate_type = graph.gate_type + subgraph.gate_type
    graph.aig_to_cell = torch.cat([graph.aig_to_cell, cell_id * torch.ones_like(subgraph.forward_index,dtype=int)], dim=0)
    graph.forward_index = torch.cat([graph.forward_index, subgraph.forward_index + num_original_nodes], dim=0)
    graph.edge_index = edge_index

    return graph

def parse_verilog_to_graph(file_path):

    G = nx.DiGraph()
    AIG = Data()

    with open(file_path, 'r') as file:
        content = file.read()

        # 提取模块名
        module_match = re.search(r"module\s+(\S+)\s*\(", content)
        if module_match:
            module_name = module_match.group(1)
            G.graph['module_name'] = module_name

        # 提取输入和输出
        inputs_list = re.findall(r"input\s+([\w,\s]+);", content)
        outputs_list = re.findall(r"output\s+([\w,\s]+);", content)

        inputs_list = [item.strip() for sublist in inputs_list for item in sublist.split(',')]
        outputs_list = [item.strip() for sublist in outputs_list for item in sublist.split(',')]

        # AIG.forward_index = torch.arange(0, len(inputs_list))
        # AIG.gate_name = inputs_list
        
        for input_signal in inputs_list:
            G.add_node(input_signal, gate_type='input')

        net_dict = {}

        # 提取逻辑门及其连接
        gate_pattern = re.compile(r"(\w+)\s+(\w+)\s*\(([^;]+)\);")
        for match in gate_pattern.finditer(content):

            gate_type, gate_name, connections = match.groups()

            G.add_node(gate_name, gate_type=gate_type)
            # 提取连接关系
            connection_pattern = re.compile(r"\.(\w+)\((\w+)\)")

            # cell_aig = cell_lib_aig[gate_type]

            for conn_match in connection_pattern.finditer(connections):
                pin, net = conn_match.groups()
                if net in inputs_list:
                    G.add_edge(net, gate_name,edge_pin_i=net, edge_pin_o=pin)
                elif net in outputs_list:
                    # print(net)
                    continue
                else:
                    if net not in net_dict:
                        net_dict[net] = {'i':[],'o':[]}
                    if pin in ['X','Y','z']:
                        net_dict[net]['i'].append([gate_name,pin])
                    else:
                        net_dict[net]['o'].append([gate_name,pin])
            
            for net in net_dict:
                for i,pin_i in net_dict[net]['i']:
                    for o,pin_o in net_dict[net]['o']:
                        G.add_edge(i, o, edge_pin_i=pin_i, edge_pin_o=pin_o)

    return G

def bitstring_to_tensor(bitstring):
    """
    将一个 64 位的 '01' 字符串转换为一个 64 维的 PyTorch 张量。

    参数:
    - bitstring: str, 长度为 64 的 '01' 字符串。

    返回:
    - tensor: torch.Tensor, 形状为 (64,) 的张量，元素为 0 或 1。
    """
    # 确保输入是长度为 64 的字符串
    assert len(bitstring) == 64, "输入字符串长度必须为 64 位"
    assert all(c in '01' for c in bitstring), "输入字符串只能包含字符 '0' 和 '1'"

    # 将字符串转换为整数列表
    bit_list = [int(c) for c in bitstring]

    # 转换为 PyTorch 张量
    tensor = torch.tensor(bit_list, dtype=torch.float32)
    return tensor

def read_csv(file_path):

    result_dict = {}

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # 确保行有足够的列
            if len(row) >= 3:
                key = row[0].strip()  # 第一列作为键
                value = row[2].strip()  # 第三列作为值
                result_dict[key] = value * (64// len(value))  # 扩展值到64位
    result_dict['input'] = '0' * 64

    for k in result_dict:
        result_dict[k] = bitstring_to_tensor(result_dict[k])
    return result_dict

def verilog2graph(file_path, csv_path=None):
    csv_path = csv_path or os.path.join(os.path.dirname(__file__), 'sky130.csv')
    cell_lib = read_csv(csv_path)
    graph = parse_verilog_to_graph(file_path)
    graph = torch_geometric.utils.from_networkx(graph)
    graph.x = torch.stack([cell_lib[i] for i in graph.gate_type])
    del graph.module_name
    # del graph.gate_type
    forward_level, forward_index, backward_level, backward_index = return_order_info(graph.edge_index, graph.x.shape[0])
    graph.forward_level = forward_level
    graph.forward_index = forward_index
    graph.backward_level = backward_level
    graph.backward_index = backward_index
    return graph


class BoundaryData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['pm_edge_index', 'pm_forward_index', 'aig_to_cell', 'sub_aig_to_cell'] :
            return self.pm_forward_index.shape[0]
        elif key in ['aig_edge_index', 'aig_forward_index'] :
            return self.aig_forward_index.shape[0]
        elif key in ['sub_aig_edge_index', 'sub_aig_forward_index'] :
            return self.sub_aig_forward_index.shape[0]
        elif 'batch' in key:
            return 1
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if  "edge_index" in key :
            return 1
        else:
            return 0
