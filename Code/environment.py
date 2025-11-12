import torch
import torch.nn.functional as F
import networkx as nx
import random
from graph_utils import pairwise_connectivity
import numpy as np

# === 通用图压缩轨迹类 ===
class GraphCompressionEpisode:
    def __init__(self, graph, tensor_type, device, stop_ratio):
        self.graph = graph.copy()
        self.device = device
        self.tensor_type = tensor_type

        self.node_list = list(self.graph.nodes())
        assert all(isinstance(n, int) for n in self.node_list), f"[FATAL] 图中存在非法节点: {self.node_list}"

        self.total_nodes = len(self.node_list)
        self.removal_limit = int(stop_ratio * self.total_nodes)

        self.compress_graph = graph.copy()
        self.node_list = list(self.compress_graph.nodes())
        self.node_list_map = {nid: idx for idx, nid in enumerate(self.node_list)}

        self.initial_connectivity = pairwise_connectivity(self.compress_graph)

        self.state_sequence = []
        self.action_sequence = []
        self.reward_sequence = []

    def get_current_state(self):
        adj = nx.to_scipy_sparse_array(self.compress_graph, format='coo')
        indices = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long, device=self.device)
        values = torch.tensor(adj.data, dtype=self.tensor_type, device=self.device)
        shape = adj.shape
        adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))

        num_nodes = self.compress_graph.number_of_nodes()
        features = torch.ones((num_nodes, 1), dtype=self.tensor_type, device=self.device)
        return adj_tensor, features

    def step(self, action, reward):
        node_id = self.node_list[action]

        if node_id not in self.compress_graph:
            raise RuntimeError(f"[FATAL] node_id {node_id} 不在 compress_graph 中")
        self.compress_graph.remove_node(node_id)

        # 记录轨迹
        self.state_sequence.append(self.get_current_state())
        self.action_sequence.append(action)
        self.reward_sequence.append(reward)

        # 记录已删除节点
        self.removed_node_ids.append(node_id)

        # 保持 node_list 与 node_list_map 始终一致
        self.node_list = list(self.compress_graph.nodes())
        self.node_list_map = {nid: idx for idx, nid in enumerate(self.node_list)}

        next_adj, _ = self.get_current_state()
        return next_adj, reward, self.is_terminal(), node_id

    def is_terminal(self):
        return len(self.action_sequence) >= self.removal_limit

    def get_trajectory(self):
        return self.state_sequence, self.action_sequence, self.reward_sequence

    def sample_random_action(self):
        valid_node_ids = set(self.compress_graph.nodes())
        available_actions = [
            idx for idx, nid in enumerate(self.node_list)
            if nid in valid_node_ids
        ]
        if not available_actions:
            raise RuntimeError("[FATAL] No valid nodes left to sample.")
        return random.choice(available_actions)

# === Important Node Episode ===
class INAEpisode(GraphCompressionEpisode):
    def __init__(self, graph, tensor_type, device, stop_ratio, reward_net, agent):
        super().__init__(graph, tensor_type, device, stop_ratio)
        self.reward_net = reward_net
        self.agent = agent
        self.removed_node_ids = []

    def step(self, action, reward):
        return super().step(action, reward)

    def select_action(self):
        adj, features = self.get_current_state()
        batch = torch.ones((1, features.size(0)), dtype=self.tensor_type, device=self.device).to_sparse()
        with torch.no_grad():
            node_emb, graph_emb = self.agent.encode(adj, batch, features)
            q_values = self.agent.decode(node_emb, graph_emb, q_for_all=True)
            return torch.argmax(q_values).item()

    def compute_step_reward(self, node_emb, graph_emb, action):
        # 验证输入形状
        if node_emb.dim() != 2 or node_emb.shape[1] != 128:
            raise ValueError(f"[ERROR] node_emb shape invalid: {node_emb.shape}")
        if graph_emb.dim() != 2 or graph_emb.shape[1] != 128:
            raise ValueError(f"[ERROR] graph_emb shape invalid: {graph_emb.shape}")

        with torch.no_grad():
            r = self.reward_net(
                node_emb.unsqueeze(0),
                graph_emb,
                torch.tensor([action], device=self.device)
            ).squeeze()
        return r.item()

    def true_return(self):
        current_connectivity = pairwise_connectivity(self.compress_graph)
        drop_ratio = max((self.initial_connectivity - current_connectivity) / self.initial_connectivity, 0.0)
        return drop_ratio

# === Redundant Node Episode ===
class RNAEpisode(GraphCompressionEpisode):
    def __init__(self, graph, tensor_type, device, stop_ratio, reward_net, agent, important_node_ids=None):
        super().__init__(graph, tensor_type, device, stop_ratio)
        self.reward_net = reward_net
        self.agent = agent
        self.important_node_ids = important_node_ids if important_node_ids else []
        self.initial_connectivity = pairwise_connectivity(self.compress_graph)
        self.node_list_map = {n: i for i, n in enumerate(self.node_list)}
        self.original_embedding = None
        self.removed_node_ids = []

    def step(self, action, reward):
        return super().step(action, reward)

    def select_action(self):
        adj, features = self.get_current_state()
        batch = torch.ones((1, features.size(0)), dtype=self.tensor_type, device=self.device).to_sparse()
        with torch.no_grad():
            node_emb, graph_emb = self.agent.encode(adj, batch, features)
            q_values = self.agent.decode(node_emb, graph_emb, q_for_all=True)
            return torch.argmax(q_values).item()

    def compute_step_reward(self, node_emb, graph_emb, action):
        # 验证输入形状
        if node_emb.dim() != 2 or node_emb.shape[1] != 128:
            raise ValueError(f"[ERROR] node_emb shape invalid: {node_emb.shape}")
        if graph_emb.dim() != 2 or graph_emb.shape[1] != 128:
            raise ValueError(f"[ERROR] graph_emb shape invalid: {graph_emb.shape}")

        with torch.no_grad():
            reward_tensor = self.reward_net(
                node_emb.unsqueeze(0),  # [1, N, 128]
                graph_emb,  # [1, 128]
                torch.tensor([action], device=self.device)  # [1]
            ).squeeze()

        return reward_tensor.item()

    def store_original_embedding(self):
        adj, features = self.get_current_state()
        batch = torch.ones((1, features.size(0)), dtype=self.tensor_type,
                           device=self.device).to_sparse()

        with torch.no_grad():
            node_emb, _ = self.agent.encode(adj, batch, features)

        original_embeddings = []
        self.original_node_ids = []
        for nid in self.important_node_ids:
            if nid in self.node_list_map:
                idx = self.node_list_map[nid]
                self.original_node_ids.append(nid)
                original_embeddings.append(node_emb[idx].detach().cpu())
        if original_embeddings:
            self.original_embedding = torch.stack(original_embeddings, dim=0)  # CPU tensor
        else:
            self.original_embedding = None

    def true_return(self):
        reward = 1.0  # 初始满分

        # === 当前连接性下降 ===
        if self.compress_graph.number_of_nodes() == 0:
            conn_drop = 1.0  # 图为空，连接性完全丢失
        else:
            current_conn = pairwise_connectivity(self.compress_graph)
            conn_drop = (self.initial_connectivity - current_conn) / self.initial_connectivity

        # === 是否误删重要节点 ===
        removed_ids = set(self.removed_node_ids)
        overlap = set(self.important_node_ids).intersection(removed_ids)
        num_deleted_important = len(overlap)
        if len(self.important_node_ids) > 0:
            delete_penalty = num_deleted_important / len(self.important_node_ids)
        else:
            delete_penalty = 0.0

        # === 嵌入漂移（余弦相似度）===
        embed_change = 0.0
        if self.original_embedding is not None:
            adj, features = self.get_current_state()
            batch = torch.ones((1, features.size(0)), dtype=self.tensor_type, device=self.device).to_sparse()
            with torch.no_grad():
                node_emb, _ = self.agent.encode(adj, batch, features)

            current_embeds = []
            original_embeds = []
            for i, nid in enumerate(self.original_node_ids):
                if nid in self.node_list_map:
                    idx = self.node_list_map[nid]
                    current_embeds.append(node_emb[idx].detach().cpu())
                    original_embeds.append(self.original_embedding[i])

            if current_embeds:
                current_emb = torch.stack(current_embeds)
                original_emb = torch.stack(original_embeds)
                embed_change = 1 - F.cosine_similarity(current_emb, original_emb, dim=-1).mean().item()

        # === 超参数阈值调整 ===
        conn_thresh = 0.1
        embed_thresh = 0.5
        delete_thresh = 0

        # === 惩罚项计算（线性映射） ===
        conn_penalty = min(max((conn_drop - conn_thresh) / (1.0 - conn_thresh), 0.0), 1.0)
        embed_penalty = min(max((embed_change - embed_thresh) / (1.0 - embed_thresh), 0.0), 1.0)
        delete_penalty = min(max((delete_penalty - delete_thresh) / (1.0 - delete_thresh), 0.0), 1.0)

        # === 惩罚项权重 ===
        total_penalty = (0.5 * conn_penalty +
                         0.3 * embed_penalty +
                         0.2 * delete_penalty)

        # === 计算最终回报 ===
        reward -= total_penalty
        return max(reward, 0.0)


# === 控制器 ===
class DualAgentController:
    def __init__(self, tensor_type, device):
        self.tensor_type = tensor_type
        self.device = device

    def load_graph(self, graph, agent_ina, agent_rna, reward_net_ina, reward_net_rna):
        self.graph = graph
        self.agent_ina = agent_ina
        self.agent_rna = agent_rna
        self.reward_net_ina = reward_net_ina
        self.reward_net_rna = reward_net_rna

        graph_for_ina = graph.copy()
        graph_for_rna = graph.copy()

        self.INEpisode = INAEpisode(graph_for_ina, self.tensor_type, self.device, stop_ratio=0.9,
                                    reward_net=self.reward_net_ina, agent=self.agent_ina)

        self.RNEpisode = RNAEpisode(graph_for_rna, self.tensor_type, self.device, stop_ratio=0.3,
                                    reward_net=self.reward_net_rna, agent=self.agent_rna)


