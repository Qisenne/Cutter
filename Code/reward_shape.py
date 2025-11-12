import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardNetworkGraphAware(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

    def forward(self, *args):
        if len(args) == 3:
            node_embs, graph_embs, actions = args
            # actions: (batch,), node_embs: (batch, N, D), graph_embs: (batch, D)
            batch_size, max_nodes, embed_dim = node_embs.shape
            batch_indices = torch.arange(batch_size, device=node_embs.device)
            action_node_embs = node_embs[batch_indices, actions]
            combined = torch.cat([action_node_embs, graph_embs], dim=-1)
            rewards = self.fc(combined).squeeze(-1)
            return torch.tanh(rewards)

        elif len(args) == 2:
            node_emb, graph_emb = args
            # node_emb: (N, D), graph_emb: (D,) or (1, D)
            if graph_emb.dim() == 2 and graph_emb.size(0) == 1:
                graph_emb = graph_emb.squeeze(0)
            num_nodes = node_emb.shape[0]
            graph_emb_expanded = graph_emb.unsqueeze(0).expand(num_nodes, -1)
            combined = torch.cat([node_emb, graph_emb_expanded], dim=-1)
            rewards = self.fc(combined).squeeze(-1)
            return torch.tanh(rewards)

    def compute_single(self, node_emb, graph_emb, action_idx):
        action_node_emb = node_emb[action_idx]
        if graph_emb.dim() == 2 and graph_emb.size(0) == 1:
            graph_emb = graph_emb.squeeze(0)
        combined = torch.cat([action_node_emb, graph_emb], dim=-1)
        return torch.tanh(self.fc(combined).squeeze(-1))

    def compute_embedding(self, node_emb, graph_emb, action_idx):
        action_node_emb = node_emb[action_idx]
        if graph_emb.dim() == 2 and graph_emb.size(0) == 1:
            graph_emb = graph_emb.squeeze(0)
        return torch.cat([action_node_emb, graph_emb], dim=-1)

class AffineTransformation(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        return self.scale * x + self.bias

@torch.no_grad()
def compute_state_action_embedding_by_decoder(
    shared_encoder, task_encoder, decoder,
    tensor_type, device, replay_buffer
):
    @torch.no_grad()
    def embedding_fn(state_idx, node_id):
        state, feat, node_id_list = replay_buffer.get_state(state_idx, device)
        # state: sparse adjacency, feat: node features
        node_matrix = state
        if not node_matrix.is_sparse:
            node_matrix = node_matrix.to_sparse()
        batch_node = torch.ones((1, node_matrix.size(0)),
                                dtype=tensor_type, device=device).to_sparse()

        # 编码
        h_n, h_g = shared_encoder(node_matrix, batch_node, input_node_features=feat.to(device))
        node_emb, graph_emb = task_encoder(h_n, h_g, node_matrix, batch_node)

        if node_id not in node_id_list:
            print(f"[Proto WARN] node_id {node_id} not in current graph (state {state_idx})")
            return None

        index = node_id_list.index(node_id)
        emb = decoder.compute_single_embedding(node_emb, graph_emb, index)
        return emb.detach().cpu()

    return embedding_fn

@torch.no_grad()
def encode_sequence(s_seq, a_seq, subseq_encoder):
    """编码一个子序列 (s_seq, a_seq) 为向量"""
    s_input = s_seq.unsqueeze(0)  # (1, L, D)
    a_input = a_seq.unsqueeze(0)  # (1, L)
    emb = subseq_encoder(s_input, a_input)  # (1, hidden_dim)
    return emb.squeeze(0)

@torch.no_grad()
def compute_reward_supervised_by_prototypes(
    s_seq, a_seq, subseq_encoder, avg_pos, avg_neg
):
    # 在 no_grad 环境，防止梯度跟踪
    curr_emb = encode_sequence(s_seq, a_seq, subseq_encoder)  # (hidden_dim,)
    # 确保传入的 avg_pos, avg_neg 已 detach 并在 CPU
    sim_pos = F.cosine_similarity(curr_emb, avg_pos, dim=0)
    sim_neg = F.cosine_similarity(curr_emb, avg_neg, dim=0)
    target_reward = sim_pos - sim_neg
    return target_reward.clamp(-1.0, 1.0)
