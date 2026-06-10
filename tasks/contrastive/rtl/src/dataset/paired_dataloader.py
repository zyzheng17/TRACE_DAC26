import torch
from torch.utils.data import DataLoader
from itertools import zip_longest
import dgl
import os
import pickle

class PairedDataLoader:
    """
    自定义数据加载器，用于将原始数据和正样本数据融合到一个batch中
    """
    def __init__(self, orig_loader, pos_loader, encoder_type='legacy_gru'):
        self.orig_loader = orig_loader
        self.pos_loader = pos_loader
        self.encoder_type = encoder_type
        self.orig_iter = None
        self.pos_iter = None
        
    def __iter__(self):
        self.orig_iter = iter(self.orig_loader)
        self.pos_iter = iter(self.pos_loader)
        return self
    
    def __next__(self):
        try:
            orig_batch = next(self.orig_iter)
            pos_batch = next(self.pos_iter)
            
            # 根据编码器类型创建不同的数据格式
            if self.encoder_type == 'legacy_gru':
                return self._create_deepgate2_batch(orig_batch, pos_batch)
            elif self.encoder_type in ['TRACE']:
                result = self._create_dg5_batch(orig_batch, pos_batch)
                # 如果返回None，说明没有TRACE数据，继续获取下一个batch
                if result is None:
                    return self.__next__()
                return result
            elif self.encoder_type == 'Graphormer':
                return self._create_graphormer_batch(orig_batch, pos_batch)
            else:
                # 默认使用legacy_gru格式
                return self._create_deepgate2_batch(orig_batch, pos_batch)
                
        except StopIteration:
            raise StopIteration
    
    def _create_deepgate2_batch(self, orig_batch, pos_batch):
        """
        创建legacy_gru格式的batch对象
        """
        # 解包原始数据
        orig_attn_mask, orig_node_feat, orig_in_degree, orig_out_degree, orig_path_data, orig_dist = orig_batch
        pos_attn_mask, pos_node_feat, pos_in_degree, pos_out_degree, pos_path_data, pos_dist = pos_batch
        
        # 创建一个新的batch对象，模拟原始的属性结构
        class CombinedBatch:
            def __init__(self):
                # 基本属性
                self.batch_size = orig_node_feat.shape[0]  # 第一维是batch_size
                self.x = orig_node_feat
                device = orig_node_feat.device  # 获取设备信息
                
                # 原始数据属性 - 需要展平为一维，并确保都在相同设备上
                # 假设第一个batch中的数据，展平所有节点
                self.gate = orig_node_feat.view(-1, orig_node_feat.shape[-1]).argmax(dim=-1).to(device)  # 转为类别索引
                self.edge_index = self._create_simple_edge_index(orig_node_feat.shape[1], device)  # 基于节点数创建边
                self.forward_level = orig_in_degree.view(-1).to(device)  # 展平为一维并移到正确设备
                self.forward_index = torch.arange(orig_node_feat.shape[1], device=device)  # 节点索引
                self.backward_level = torch.zeros_like(self.forward_level, device=device)  # 创建默认的backward_level
                
                # 正样本数据属性（加syn_前缀）- 同样展平并确保在相同设备上
                self.syn_gate = pos_node_feat.view(-1, pos_node_feat.shape[-1]).argmax(dim=-1).to(device)
                self.syn_edge_index = self._create_simple_edge_index(pos_node_feat.shape[1], device)
                self.syn_forward_level = pos_in_degree.view(-1).to(device)
                self.syn_forward_index = torch.arange(pos_node_feat.shape[1], device=device)
                self.syn_backward_level = torch.zeros_like(self.syn_forward_level, device=device)
                
            def _create_simple_edge_index(self, num_nodes, device):
                """创建简单的边索引（链式连接）"""
                if num_nodes <= 1:
                    return torch.zeros((2, 0), dtype=torch.long, device=device)
                
                # 创建一个简单的链式连接图
                edges = []
                for i in range(num_nodes - 1):
                    edges.append([i, i + 1])
                
                return torch.tensor(edges, dtype=torch.long, device=device).t()
                
        return CombinedBatch()
    
    def _create_dg5_batch(self, orig_batch, pos_batch):
        """
        创建TRACE格式的batch对象
        处理多个图对，每个batch包含batch_size个图对
        """
        # 解包原始数据
        orig_attn_mask, orig_node_feat, orig_in_degree, orig_out_degree, orig_path_data, orig_dist = orig_batch
        pos_attn_mask, pos_node_feat, pos_in_degree, pos_out_degree, pos_path_data, pos_dist = pos_batch
        
        device = orig_node_feat.device
        
        # 获取batch size
        batch_size = orig_node_feat.shape[0]
        
        # 检查是否有预生成的TRACE数据
        if hasattr(orig_batch, 'dg5_gate') and hasattr(pos_batch, 'dg5_gate'):
            # 如果有预生成的TRACE数据，我们需要处理多个图
            orig_gates = []
            orig_edge_indices = []
            orig_forward_levels = []
            orig_forward_indices = []
            orig_xs = []
            orig_num_nodes = []
            
            syn_gates = []
            syn_edge_indices = []
            syn_forward_levels = []
            syn_forward_indices = []
            syn_xs = []
            syn_num_nodes = []
            
            # 获取TRACE数据的实际batch size
            dg5_batch_size = 1
            if hasattr(orig_batch.dg5_gate, '__getitem__') and hasattr(orig_batch.dg5_gate, '__len__'):
                dg5_batch_size = len(orig_batch.dg5_gate)
            elif hasattr(orig_batch.dg5_gate, 'shape') and len(orig_batch.dg5_gate.shape) > 0:
                dg5_batch_size = orig_batch.dg5_gate.shape[0]
            
            # 检查TRACE数据的实际结构
            # 如果TRACE数据是batch，我们应该使用TRACE的batch size
            # 如果TRACE数据是单个图，我们应该使用Graphormer的batch size
            if hasattr(orig_batch.dg5_gate, 'shape') and len(orig_batch.dg5_gate.shape) > 1:
                # TRACE数据是batch，使用TRACE的batch size
                actual_batch_size = orig_batch.dg5_gate.shape[0]
            else:
                # TRACE数据是单个图，使用Graphormer的batch size
                actual_batch_size = batch_size
            
            # 处理每个图对
            for i in range(actual_batch_size):
                # 检查TRACE数据是否为batch
                if hasattr(orig_batch.dg5_gate, 'shape') and len(orig_batch.dg5_gate.shape) > 1:
                    # TRACE数据是batch，使用索引i
                    orig_gates.append(orig_batch.dg5_gate[i])
                    orig_edge_indices.append(orig_batch.dg5_edge_index[i])
                    orig_forward_levels.append(orig_batch.dg5_forward_level[i])
                    orig_forward_indices.append(orig_batch.dg5_forward_index[i])
                    orig_xs.append(orig_batch.dg5_x[i])
                    orig_num_nodes.append(orig_batch.dg5_num_nodes[i])
                    
                    syn_gates.append(pos_batch.dg5_gate[i])
                    syn_edge_indices.append(pos_batch.dg5_edge_index[i])
                    syn_forward_levels.append(pos_batch.dg5_forward_level[i])
                    syn_forward_indices.append(pos_batch.dg5_forward_index[i])
                    syn_xs.append(pos_batch.dg5_x[i])
                    syn_num_nodes.append(pos_batch.dg5_num_nodes[i])
                else:
                    # TRACE数据是单个图，重复使用
                    orig_gates.append(orig_batch.dg5_gate)
                    orig_edge_indices.append(orig_batch.dg5_edge_index)
                    orig_forward_levels.append(orig_batch.dg5_forward_level)
                    orig_forward_indices.append(orig_batch.dg5_forward_index)
                    orig_xs.append(orig_batch.dg5_x)
                    orig_num_nodes.append(orig_batch.dg5_num_nodes)
                    
                    syn_gates.append(pos_batch.dg5_gate)
                    syn_edge_indices.append(pos_batch.dg5_edge_index)
                    syn_forward_levels.append(pos_batch.dg5_forward_level)
                    syn_forward_indices.append(pos_batch.dg5_forward_index)
                    syn_xs.append(pos_batch.dg5_x)
                    syn_num_nodes.append(pos_batch.dg5_num_nodes)
        else:
            print("No TRACE data found, skipping this graph")
            return None
        
        # 创建TRACE图对象列表
        class TRACEGraph:
            def __init__(self, data):
                self.gate = data['gate']
                self.edge_index = data['edge_index']
                self.forward_level = data['forward_level']
                self.forward_index = data['forward_index']
                self.x = data['x']
                self.num_nodes = data['num_nodes']
                
                # 确保所有张量都在同一设备上
                device = self.gate.device
                self.gate = self.gate.to(device)
                self.edge_index = self.edge_index.to(device)
                self.forward_level = self.forward_level.to(device)
                self.forward_index = self.forward_index.to(device)
                self.x = self.x.to(device)
                
                # 添加edge_weight属性，AddRandomWalkPE需要这个属性
                # 如果没有明确的边权重，设置为None
                self.edge_weight = None
                
                # 添加num_edges属性，AddRandomWalkPE需要这个属性
                # num_edges等于edge_index的列数
                self.num_edges = self.edge_index.shape[1] if self.edge_index.numel() > 0 else 0
                
                # 确保num_nodes是正确的值（从x的维度计算）
                # 如果data['num_nodes']不正确，使用x的第一维大小
                if self.num_nodes != self.x.shape[0]:
                    self.num_nodes = self.x.shape[0] if self.x.numel() > 0 else 0
                
                # 添加batch属性，用于标识每个节点属于哪个图
                # 对于单个图，所有节点的batch_id都是0
                self.batch = torch.zeros(self.num_nodes, dtype=torch.long, device=device)
            
            def __setitem__(self, key, value):
                """支持字典式赋值，如 graph['pe'] = value"""
                setattr(self, key, value)
            
            def __getitem__(self, key):
                """支持字典式访问，如 graph['pe']"""
                return getattr(self, key)
        
        # 创建多个TRACE图对象
        orig_graphs = []
        syn_graphs = []
        
        for i in range(actual_batch_size):
            orig_data = {
                'gate': orig_gates[i],
                'edge_index': orig_edge_indices[i],
                'forward_level': orig_forward_levels[i],
                'forward_index': orig_forward_indices[i],
                'x': orig_xs[i],
                'num_nodes': orig_num_nodes[i]
            }
            syn_data = {
                'gate': syn_gates[i],
                'edge_index': syn_edge_indices[i],
                'forward_level': syn_forward_levels[i],
                'forward_index': syn_forward_indices[i],
                'x': syn_xs[i],
                'num_nodes': syn_num_nodes[i]
            }
            
            orig_graphs.append(TRACEGraph(orig_data))
            syn_graphs.append(TRACEGraph(syn_data))
        
        # 创建batch对象
        class Batch:
            def __init__(self):
                self.gate = orig_graphs
                self.syn_gate = syn_graphs
                self.batch_size = actual_batch_size
        
        batch = Batch()
        
        return batch
    
    def _create_graphormer_batch(self, orig_batch, pos_batch):
        """
        创建Graphormer格式的batch对象
        """
        # Graphormer期望的是原始的6元组格式
        # 我们需要将原始数据和正样本数据合并到一个batch中
        orig_attn_mask, orig_node_feat, orig_in_degree, orig_out_degree, orig_path_data, orig_dist = orig_batch
        pos_attn_mask, pos_node_feat, pos_in_degree, pos_out_degree, pos_path_data, pos_dist = pos_batch
        
        # 确保两个batch的维度匹配
        if orig_node_feat.shape[1] != pos_node_feat.shape[1]:
            # 如果节点数量不匹配，使用较小的那个
            min_nodes = min(orig_node_feat.shape[1], pos_node_feat.shape[1])
            orig_node_feat = orig_node_feat[:, :min_nodes, :]
            pos_node_feat = pos_node_feat[:, :min_nodes, :]
            orig_attn_mask = orig_attn_mask[:, :min_nodes, :min_nodes]
            pos_attn_mask = pos_attn_mask[:, :min_nodes, :min_nodes]
            orig_in_degree = orig_in_degree[:, :min_nodes]
            pos_in_degree = pos_in_degree[:, :min_nodes]
            orig_out_degree = orig_out_degree[:, :min_nodes]
            pos_out_degree = pos_out_degree[:, :min_nodes]
            orig_path_data = orig_path_data[:, :min_nodes, :min_nodes, :]
            pos_path_data = pos_path_data[:, :min_nodes, :min_nodes, :]
            orig_dist = orig_dist[:, :min_nodes, :min_nodes]
            pos_dist = pos_dist[:, :min_nodes, :min_nodes]
        
        # 合并两个batch的数据
        combined_attn_mask = torch.cat([orig_attn_mask, pos_attn_mask], dim=0)
        combined_node_feat = torch.cat([orig_node_feat, pos_node_feat], dim=0)
        combined_in_degree = torch.cat([orig_in_degree, pos_in_degree], dim=0)
        combined_out_degree = torch.cat([orig_out_degree, pos_out_degree], dim=0)
        combined_path_data = torch.cat([orig_path_data, pos_path_data], dim=0)
        combined_dist = torch.cat([orig_dist, pos_dist], dim=0)
        
        # 修复attn_mask的形状问题
        # Graphormer会添加一个虚拟节点，所以attn_mask需要相应调整
        num_graphs, max_num_nodes, _ = combined_node_feat.shape
        device = combined_attn_mask.device
        
        # 创建新的attn_mask，考虑虚拟节点
        # attn_bias的形状是 [num_graphs, max_num_nodes + 1, max_num_nodes + 1, num_heads]
        # 所以attn_mask应该是 [num_graphs, max_num_nodes + 1, max_num_nodes + 1]
        new_attn_mask = torch.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1, device=device)
        
        # 将原始attn_mask复制到新位置（跳过第一个位置，因为那是虚拟节点）
        # 确保原始attn_mask的形状与目标位置匹配
        if combined_attn_mask.shape[1] == max_num_nodes:
            new_attn_mask[:, 1:, 1:] = combined_attn_mask
        else:
            # 如果形状不匹配，需要调整
            min_size = min(combined_attn_mask.shape[1], max_num_nodes)
            new_attn_mask[:, 1:min_size+1, 1:min_size+1] = combined_attn_mask[:, :min_size, :min_size]
        
        # 虚拟节点（第0个位置）应该能够看到所有其他节点，所以不设置mask
        # 其他节点也应该能够看到虚拟节点，所以也不设置mask
        
        # 返回Graphormer期望的格式
        return (new_attn_mask, combined_node_feat, combined_in_degree, 
                combined_out_degree, combined_path_data, combined_dist)
    
    def __len__(self):
        return min(len(self.orig_loader), len(self.pos_loader))
    
    @property
    def batch_size(self):
        return self.orig_loader.batch_size
    
    @property
    def num_workers(self):
        return self.orig_loader.num_workers
    
    @property
    def pin_memory(self):
        return self.orig_loader.pin_memory
    
    @property
    def drop_last(self):
        return self.orig_loader.drop_last 