import os, time, json
from multiprocessing import Pool
import torch
import networkx as nx
import numpy as np
import pickle


# def convert_one_design(design, ep, output_dir):
#     design_dir = f"../../rtl2code/cone_code/{design}/{ep}.v"
#     print('Current Design: ', design)
#     os.system(f'python3 analyze.py {design_dir} -D {design} -N {ep} -C {cmd2} -O {output_dir}')


def extract_dg5_topology_from_graph(graph_file, node_dict_file):
    """
    从原始的图文件中提取DG5所需的电路拓扑信息
    """
    # 读取原始图数据
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    with open(node_dict_file, 'rb') as f:
        node_dict = pickle.load(f)
    
    # 创建NetworkX图
    G = nx.DiGraph()
    
    # 添加节点和边
    for src, dst_list in graph_data.items():
        # 跳过DG5数据字段
        if isinstance(src, str) and src.startswith('dg5_'):
            continue
        for dst in dst_list:
            G.add_edge(src, dst)
    
    # 为所有节点添加属性
    for node_name in G.nodes():
        if node_name in node_dict:
            node = node_dict[node_name]
            G.nodes[node_name]['type'] = node.type
            G.nodes[node_name]['width'] = node.width
        else:
            # 如果节点不在node_dict中，推断类型
            G.nodes[node_name]['type'] = infer_node_type_from_name(node_name)
            G.nodes[node_name]['width'] = 1
    
    # 计算拓扑排序和层级信息
    forward_level, forward_index = compute_topological_levels(G)
    
    # 提取fanin信息
    fanin_info = extract_fanin_info(G)
    
    # 确保有PI节点（forward_level=0）
    node_list = list(G.nodes())
    input_nodes = [node for node in node_list if G.nodes[node]['type'] in ['Input', 'PI', 'Reg']]
    if input_nodes:
        for node in input_nodes:
            node_idx = node_list.index(node)
            forward_level[node_idx] = 0
    else:
        # 如果没有输入节点，将第一个节点设为PI
        forward_level[0] = 0
    
    print(f"Topology extracted:")
    print(f"  - Total nodes: {len(node_list)}")
    print(f"  - Input nodes: {len(input_nodes)}")
    print(f"  - Forward levels: {forward_level}")
    print(f"  - PI nodes (level 0): {forward_level.count(0)}")
    
    return {
        'graph_data': graph_data,
        'node_dict': node_dict,
        'gate_types': [G.nodes[i]['type'] for i in G.nodes()],
        'edge_index': list(G.edges()),
        'forward_level': forward_level,
        'forward_index': forward_index,
        'fanin_info': fanin_info,
        'node_names': list(G.nodes())
    }


def infer_node_type_from_name(node_name):
    """
    从节点名称推断节点类型
    """
    # 简单的启发式规则
    if 'input' in node_name.lower() or 'in' in node_name.lower():
        return 'Input'
    elif 'output' in node_name.lower() or 'out' in node_name.lower():
        return 'Output'
    elif 'reg' in node_name.lower():
        return 'Reg'
    elif 'wire' in node_name.lower():
        return 'Wire'
    else:
        return 'Wire'  # 默认为Wire


def compute_topological_levels(G):
    """
    计算拓扑排序和层级信息，处理包含环的图
    """
    try:
        # 首先尝试检测并移除环
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"Warning: Graph contains {len(cycles)} cycles, attempting to break them")
            # 尝试移除环中的一些边来打破环
            G_acyclic = G.copy()
            for cycle in cycles:
                if len(cycle) > 1:
                    # 移除环中的最后一条边
                    G_acyclic.remove_edge(cycle[-1], cycle[0])
            
            # 再次检查是否还有环
            try:
                topo_order = list(nx.topological_sort(G_acyclic))
            except nx.NetworkXError:
                print("Warning: Still contains cycles after edge removal, using simple level assignment")
                return simple_level_assignment(G)
        else:
            # 没有环，直接进行拓扑排序
            topo_order = list(nx.topological_sort(G))
        
        # 计算每个节点的层级
        levels = {}
        for node in topo_order:
            # 找到所有前驱节点的最大层级
            pred_levels = [levels.get(pred, 0) for pred in G.predecessors(node)]
            levels[node] = max(pred_levels) + 1 if pred_levels else 0
        
        # 转换为列表格式
        node_list = list(G.nodes())
        forward_level = [levels.get(node, 0) for node in node_list]
        forward_index = [node_list.index(node) for node in topo_order]
        
        return forward_level, forward_index
    
    except Exception as e:
        print(f"Warning: Error in topological sort: {e}, using simple level assignment")
        return simple_level_assignment(G)


def simple_level_assignment(G):
    """
    简单的层级分配，用于处理无法进行拓扑排序的图
    """
    node_list = list(G.nodes())
    forward_level = [0] * len(node_list)
    forward_index = list(range(len(node_list)))
    
    # 使用简单的BFS来分配层级
    visited = set()
    queue = []
    
    # 找到入度为0的节点作为起始点
    in_degrees = dict(G.in_degree())
    for node in node_list:
        if in_degrees.get(node, 0) == 0:
            queue.append((node, 0))
            visited.add(node)
    
    # 如果没有入度为0的节点，从第一个节点开始
    if not queue:
        queue.append((node_list[0], 0))
        visited.add(node_list[0])
    
    # BFS分配层级
    while queue:
        current_node, level = queue.pop(0)
        node_idx = node_list.index(current_node)
        forward_level[node_idx] = level
        
        # 处理后继节点
        for successor in G.successors(current_node):
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, level + 1))
    
    # 处理剩余的未访问节点
    for i, node in enumerate(node_list):
        if node not in visited:
            forward_level[i] = 0  # 默认层级为0
    
    return forward_level, forward_index


def extract_fanin_info(G):
    """
    提取每个节点的fanin信息
    """
    fanin_info = {}
    for node in G.nodes():
        fanin_nodes = list(G.predecessors(node))
        fanin_info[node] = {
            'count': len(fanin_nodes),
            'nodes': fanin_nodes
        }
    return fanin_info


def create_dg5_format_data(topology_data):
    """
    将电路拓扑数据转换为DG5格式，使用与00_prep_labels_dg5.py相同的逻辑
    """
    graph_data = topology_data['graph_data']
    node_dict = topology_data['node_dict']
    gate_types = topology_data['gate_types']
    edge_index = topology_data['edge_index']
    forward_level = topology_data['forward_level']
    forward_index = topology_data['forward_index']
    node_names = topology_data['node_names']
    
    # 创建节点名称到索引的映射
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    
    # 转换边索引为数字索引 - 注意边的方向：[src, dst]
    edge_index_numeric = []
    for src, dst in edge_index:
        if src in name_to_idx and dst in name_to_idx:
            edge_index_numeric.append([name_to_idx[src], name_to_idx[dst]])
    
    # 从node_dict中提取正确的门类型信息
    # 保持RTL电路的原始节点类型
    gate_types_numeric = []
    for node_name in node_names:
        if node_name in node_dict:
            node = node_dict[node_name]
            node_type = node.type
            
            # 根据node_type分配门类型编号
            if node_type in ['Reg', 'Input', 'Inout']:
                gate_types_numeric.append(0)  # PI类型
            elif node_type in ['Output', 'Wire']:
                gate_types_numeric.append(1)  # 输出/连线类型
            elif node_type in ['Constant']:
                gate_types_numeric.append(2)  # 常量类型
            elif node_type in ['Operator', 'UnaryOperator']:
                # 操作符类型，需要从节点名称中提取具体操作
                import re
                op_temp = re.findall(r'([A-Z][a-z]*)(\d+)', node_name)
                if op_temp:
                    op = op_temp[0][0]
                    if op in ['And', 'Land', 'Uand']:
                        gate_types_numeric.append(3)  # AND
                    elif op in ['Or', 'Lor', 'Uor']:
                        gate_types_numeric.append(4)  # OR
                    elif op in ['Unot', 'Ulnot']:
                        gate_types_numeric.append(5)  # NOT
                    elif op in ['Xor', 'Uxor']:
                        gate_types_numeric.append(6)  # XOR
                    elif op in ['Eq']:
                        gate_types_numeric.append(7)  # 等于比较
                    elif op in ['Cond', 'Mux']:
                        gate_types_numeric.append(8)  # 条件/多路选择
                    elif op in ['Plus', 'Times', 'Minus', 'Uminus', 'Divide']:
                        gate_types_numeric.append(9)  # 算术运算
                    elif op in ['GreaterEq', 'LessEq', 'GreaterThan', 'Than', 'LessThan']:
                        gate_types_numeric.append(10)  # 比较运算
                    elif op in ['Sra', 'Srl', 'Sla', 'Sll']:
                        gate_types_numeric.append(11)  # 移位运算
                    else:
                        gate_types_numeric.append(12)  # 其他操作符
                else:
                    gate_types_numeric.append(12)  # 默认其他操作符
            elif node_type in ['Concat', 'Repeat']:
                gate_types_numeric.append(13)  # 连接/重复
            elif node_type in ['Partselect', 'Pointer']:
                gate_types_numeric.append(14)  # 部分选择/指针
            else:
                gate_types_numeric.append(15)  # 其他类型
        else:
            gate_types_numeric.append(0)  # 默认为PI
    
    # 创建x_data格式: [node_id, gate_type] - 与00_prep_labels_dg5.py相同
    num_nodes = len(node_names)
    x_data = []
    for i in range(num_nodes):
        x_data.append([i, gate_types_numeric[i]])  # [node_id, gate_type]
    
    # 转换为numpy数组
    x_data = np.array(x_data, dtype=np.int64)
    
    # 转换为PyTorch张量
    gate_tensor = torch.tensor(gate_types_numeric, dtype=torch.long)
    
    # 创建边索引张量 - 注意格式：[2, num_edges]
    if edge_index_numeric:
        edge_tensor = torch.tensor(edge_index_numeric, dtype=torch.long).t().contiguous()
    else:
        edge_tensor = torch.zeros((2, 0), dtype=torch.long)
    
    # 使用deepgate库生成正确的节点特征和拓扑信息
    try:
        import deepgate as dg
        # 现在有16种门类型（0-15），所以num_gate_types=16
        x_one_hot = dg.construct_node_feature(x_data, 16)
        
        # 处理edge_index有环的情况 - 删除造成环的边
        edge_tensor = remove_cycles_from_edges(edge_tensor)
        
        # 重新计算forward_level和forward_index
        forward_level_tensor, forward_index_tensor, backward_level, backward_index = dg.return_order_info(edge_tensor, x_one_hot.size(0))
        x_tensor = x_one_hot
    except ImportError:
        print("Warning: deepgate not available, using fallback method")
        # 创建16维的one-hot编码
        x_tensor = torch.zeros(num_nodes, 16, dtype=torch.float)
        for i, gate_type in enumerate(gate_types_numeric):
            if 0 <= gate_type < 16:
                x_tensor[i, gate_type] = 1.0
        
        # 使用原始的forward_level和forward_index
        forward_level_tensor = torch.tensor(forward_level, dtype=torch.long)
        forward_index_tensor = torch.tensor(forward_index, dtype=torch.long)
        
        # 确保有PI节点（forward_level=0）
        if forward_level_tensor.min() > 0:
            # 找到输入节点
            input_nodes = (gate_tensor == 0).nonzero(as_tuple=True)[0]
            if len(input_nodes) > 0:
                forward_level_tensor[input_nodes] = 0
            else:
                forward_level_tensor[0] = 0  # 将第一个节点设为PI
    
    # 创建概率张量（用于PI节点初始化）
    prob_tensor = torch.rand(num_nodes, dtype=torch.float)
    
    print(f"DG5 data created:")
    print(f"  - Gate types: {gate_tensor}")
    print(f"  - Forward levels: {forward_level_tensor}")
    print(f"  - PI nodes (level 0): {(forward_level_tensor == 0).sum().item()}")
    print(f"  - Max level: {forward_level_tensor.max().item()}")
    print(f"  - Edge count: {edge_tensor.shape[1]}")
    print(f"  - X tensor shape: {x_tensor.shape}")
    
    return {
        'gate': gate_tensor,
        'edge_index': edge_tensor,
        'forward_level': forward_level_tensor,
        'forward_index': forward_index_tensor,
        'x': x_tensor,
        'prob': prob_tensor,
        'num_nodes': num_nodes,
        'node_names': node_names
    }


def remove_cycles_from_edges(edge_tensor):
    """
    检测并删除造成环的边
    """
    if edge_tensor.shape[1] == 0:
        return edge_tensor
    
    # 转换为NetworkX图来检测环
    import networkx as nx
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加所有边
    for i in range(edge_tensor.shape[1]):
        src = edge_tensor[0, i].item()
        dst = edge_tensor[1, i].item()
        G.add_edge(src, dst)
    
    # 检测环
    try:
        cycles = list(nx.simple_cycles(G))
    except:
        # 如果检测环失败，返回原始边
        return edge_tensor
    
    if not cycles:
        # 没有环，返回原始边
        return edge_tensor
    
    print(f"检测到 {len(cycles)} 个环，正在删除造成环的边...")
    
    # 删除造成环的边
    edges_to_remove = set()
    for cycle in cycles:
        if len(cycle) > 1:
            # 删除环中的最后一条边
            src = cycle[-1]
            dst = cycle[0]
            edges_to_remove.add((src, dst))
    
    # 创建新的边张量，排除要删除的边
    new_edges = []
    for i in range(edge_tensor.shape[1]):
        src = edge_tensor[0, i].item()
        dst = edge_tensor[1, i].item()
        if (src, dst) not in edges_to_remove:
            new_edges.append([src, dst])
    
    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        print(f"删除了 {len(edges_to_remove)} 条边，剩余 {new_edge_tensor.shape[1]} 条边")
        return new_edge_tensor
    else:
        print("警告：删除了所有边，返回空边张量")
        return torch.zeros((2, 0), dtype=torch.long)


def convert_one_design(design, ep, output_dir):
    design_dir = f"../../rtl2code/cone_code/{design}/{ep}.v"
    print('Current Design: ', design)
    
    # 运行原始的Graphormer数据处理
    os.system(f'python3 analyze.py {design_dir} -D {design} -N {ep} -C {cmd2} -O {output_dir}')
    
    # 添加DG5数据处理
    graph_file = f"{output_dir}/{ep}_{cmd2}.pkl"
    node_dict_file = f"{output_dir}/{ep}_{cmd2}_node_dict.pkl"
    
    if os.path.exists(graph_file) and os.path.exists(node_dict_file):
        
        # ######### debug
        # if graph_file != "../cone_graph/ori/fpu//div_by_zero_ori.pkl":
        #     return
        # #########
        
        # 从原始图数据中提取DG5拓扑信息
        topology_data = extract_dg5_topology_from_graph(graph_file, node_dict_file)
        
        # 创建DG5格式数据
        dg5_data = create_dg5_format_data(topology_data)
        
        # 将DG5数据添加到Graphormer数据中
        # 读取现有的Graphormer数据
        with open(graph_file, 'rb') as f:
            graphormer_data = pickle.load(f)
        
        # 添加DG5数据字段
        graphormer_data['dg5_gate'] = dg5_data['gate']
        graphormer_data['dg5_edge_index'] = dg5_data['edge_index']
        graphormer_data['dg5_forward_level'] = dg5_data['forward_level']
        graphormer_data['dg5_forward_index'] = dg5_data['forward_index']
        graphormer_data['dg5_x'] = dg5_data['x']
        graphormer_data['dg5_prob'] = dg5_data['prob']
        graphormer_data['dg5_num_nodes'] = dg5_data['num_nodes']
        graphormer_data['dg5_node_names'] = dg5_data['node_names']
        
        # 保存合并后的数据
        with open(graph_file, 'wb') as f:
            pickle.dump(graphormer_data, f)
        
        print(f"Added DG5 data to Graphormer file: {graph_file}")
        print(f"  - Nodes: {dg5_data['num_nodes']}")
        print(f"  - Edges: {dg5_data['edge_index'].shape[1]}")
        print(f"  - Max level: {dg5_data['forward_level'].max().item()}")
        print(f"  - PI nodes (level 0): {(dg5_data['forward_level'] == 0).sum().item()}")
            
       
    

if __name__ == '__main__':
    cmd = 'ast' ## for word-level
    # cmd = 'sog' ## for bit-level
    global cmd2
    # cmd2 = "ori"
    cmd2 = "pos"

    
    # design_lst = ['spi']
    design_lst = [
            "spi", 
              "b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b09", "b10",
              "b11", "b12", "b13", "b14", "b15", "b17", "b18", "b19", "b20",
              "b21", "b22",
              "fpu", 
              "i2c_master_top", "mc_top", "pcm_slv_top" , "sasc_top", "simple_spi_top", "tv80s", "usb_phy", "usbf_top", "wb_dma_top"
              ]


    for design in design_lst:
        with open (f"../../dataset/reg_lst/{design}.json", 'r') as f:
            reg_lst = json.load(f)

        output_dir = f'../cone_graph/{cmd2}/{design}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for ep in reg_lst:
            # ######### debug
            # if design != "fpu" or ep != "div_by_zero":
            #     continue
            # #########
            
            print(design + ' ' + ep)
            convert_one_design(design, ep, output_dir)


    
    

    



