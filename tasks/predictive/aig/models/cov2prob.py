import torch


def cov2prob(graph, cov):
    pred_prob = torch.zeros_like(graph.prob)
    pred_prob[graph.forward_level == 0] = graph.prob[graph.forward_level == 0]

    max_num_layers = torch.max(graph.forward_level).item() + 1
    for level in range(max_num_layers):
        and_nodes = graph.forward_index[
            torch.logical_and(graph.gate.squeeze() == 1, graph.forward_level == level)
        ]
        if and_nodes.size(0) > 0:
            and_mask = torch.logical_and(
                graph.gate[graph.edge_index[1]].squeeze() == 1,
                graph.forward_level[graph.edge_index[1]] == level,
            )
            and_edge_index = graph.edge_index[:, and_mask]
            and_tgt_node_idx = and_edge_index[1].reshape(-1, 2)[:, 0]
            and_src_node_idx = and_edge_index[0].reshape(-1, 2)
            pred_prob[and_tgt_node_idx] = (
                pred_prob[and_src_node_idx[:, 0]] * pred_prob[and_src_node_idx[:, 1]]
                + cov[and_tgt_node_idx]
            )

        not_nodes = graph.forward_index[
            torch.logical_and(graph.gate.squeeze() == 2, graph.forward_level == level)
        ]
        if not_nodes.size(0) > 0:
            not_mask = torch.logical_and(
                graph.gate[graph.edge_index[1]].squeeze() == 2,
                graph.forward_level[graph.edge_index[1]] == level,
            )
            not_edge_index = graph.edge_index[:, not_mask]
            pred_prob[not_edge_index[1]] = 1 - pred_prob[not_edge_index[0]]

    return pred_prob
