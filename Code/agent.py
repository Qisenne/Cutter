import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

class SharedEncoderGraph(nn.Module):
    def __init__(self, input_size, embedding_size, tensor_type, device):
        super().__init__()
        self.device = device
        self.tensor_type = tensor_type
        self.embedding_size = embedding_size

        self.fc_input = nn.Linear(input_size, embedding_size)
        self.W_gcn = nn.Parameter(torch.empty(embedding_size, embedding_size, dtype=tensor_type, device=device))
        nn.init.xavier_uniform_(self.W_gcn)
        self.graph_fc = nn.Linear(embedding_size, embedding_size)

    def forward(self, node_matrix, node_batch, input_node_features=None):
        node_matrix = node_matrix.to(self.device)
        node_batch = node_batch.to(self.device)

        N = node_matrix.shape[0]
        if input_node_features is None:
            input_node_features = torch.ones((N, 1), dtype=self.tensor_type, device=self.device)
        else:
            input_node_features = input_node_features.to(self.device)

        h = F.relu(self.fc_input(input_node_features), inplace=False)
        h_struct = sparse.mm(node_matrix, h) @ self.W_gcn
        h_struct = F.normalize(F.relu(h_struct, inplace=False), p=2, dim=-1)

        h_graph = sparse.mm(node_batch, h_struct)
        h_graph = F.normalize(F.relu(self.graph_fc(h_graph), inplace=False), p=2, dim=-1)

        return h_struct, h_graph

class TaskEncoder(nn.Module):
    def __init__(self, embedding_size, depth, tensor_type, device):
        super().__init__()
        self.depth = depth
        self.device = device
        self.W_2 = nn.Parameter(torch.empty(embedding_size, embedding_size, dtype=tensor_type, device=device))
        self.W_3 = nn.Parameter(torch.empty(embedding_size, embedding_size, dtype=tensor_type, device=device))
        nn.init.xavier_uniform_(self.W_2)
        nn.init.xavier_uniform_(self.W_3)
        self.fc_n = nn.Linear(2 * embedding_size, embedding_size, device=device)
        self.fc_g = nn.Linear(2 * embedding_size, embedding_size, device=device)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, h_n_l, h_g_l, node_matrix, node_batch):
        h_n_l = h_n_l.to(self.device)
        h_g_l = h_g_l.to(self.device)
        node_matrix = node_matrix.to(self.device)
        node_batch = node_batch.to(self.device)

        for _ in range(self.depth):
            h_n_old = h_n_l.clone()
            h_g_old = h_g_l.clone()

            h_nn = sparse.mm(node_matrix, h_n_old)
            z1 = torch.matmul(h_g_old.expand(h_n_old.size(0), -1), self.W_2)
            z2 = torch.matmul(h_nn, self.W_3)

            h_n_l = self.fc_n(torch.cat([z1, z2], dim=1))
            h_n_l = F.normalize(self.relu(h_n_l), p=2, dim=-1)

            h_ng = sparse.mm(node_batch, h_n_l)
            z3 = torch.matmul(h_g_old, self.W_2)
            z4 = torch.matmul(h_ng, self.W_3)

            h_g_l = self.fc_g(torch.cat([z3, z4], dim=1))
            h_g_l = F.normalize(self.relu(h_g_l), p=2, dim=-1)

        return h_n_l, h_g_l

class QDecoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.fc1 = nn.Linear(2 * embedding_size, 2 * embedding_size)
        self.fc2 = nn.Linear(2 * embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, 1)
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(2 * embedding_size)

        # === 初始化，避免陷入负域 or 零梯度 ===
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, node_embedding, graph_embedding, actions=None, q_for_all=False):
        if node_embedding.dim() == 2:
            num_nodes = node_embedding.size(0)

            if not q_for_all:
                raise NotImplementedError("2D输入仅支持 q_for_all")

            if graph_embedding.dim() == 3:
                graph_embedding = graph_embedding.squeeze(0)
            if graph_embedding.dim() == 2 and graph_embedding.size(0) == 1:
                graph_embedding = graph_embedding.squeeze(0)
            assert graph_embedding.dim() == 1

            Z_s_expand = graph_embedding.unsqueeze(0).expand(num_nodes, -1)
            x = torch.cat([node_embedding, Z_s_expand], dim=-1)

            x = self.fc1(x)
            x = self.activation(x)
            x = self.layernorm(x)
            x = self.fc2(x)
            x = self.activation(x)
            x = self.fc3(x)

            return x.squeeze(-1)

        elif node_embedding.dim() == 3:
            B, N, D = node_embedding.shape

            if q_for_all:
                Z_s_expand = graph_embedding.unsqueeze(1).expand(-1, N, -1)
                x = torch.cat([node_embedding, Z_s_expand], dim=-1)
                x = self.fc1(x)
                x = self.activation(x)
                x = self.layernorm(x)
                x = self.fc2(x)
                x = self.activation(x)
                x = self.fc3(x)
                return x.squeeze(-1)
            else:
                idx = torch.arange(B, device=node_embedding.device)
                node_sel = node_embedding[idx, actions]
                x = torch.cat([node_sel, graph_embedding], dim=-1)
                x = self.fc1(x)
                x = self.activation(x)
                x = self.layernorm(x)
                x = self.fc2(x)
                x = self.activation(x)
                x = self.fc3(x)
                return x.squeeze(-1)

    def compute_single_embedding(self, node_emb, graph_emb, action_idx):
        action_node_emb = node_emb[action_idx]
        if graph_emb.dim() == 2 and graph_emb.size(0) == 1:
            graph_emb = graph_emb.squeeze(0)
        return torch.cat([action_node_emb, graph_emb], dim=-1)

class INAAgent:
    def __init__(self, encoder_shared, encoder_ina, decoder_ina):
        self.encoder_shared = encoder_shared
        self.encoder_ina = encoder_ina
        self.decoder_ina = decoder_ina

    def encode(self, adj, batch, features=None):
        node_emb, graph_emb = self.encoder_shared(adj, batch, features)
        node_emb, graph_emb = self.encoder_ina(node_emb, graph_emb, adj, batch)
        return node_emb, graph_emb

    def decode(self, node_emb, graph_emb, actions=None, q_for_all=False):
        return self.decoder_ina(node_emb, graph_emb, actions, q_for_all)

class RNAAgent:
    def __init__(self, encoder_shared, encoder_rna, decoder_rna):
        self.encoder_shared = encoder_shared
        self.encoder_rna = encoder_rna
        self.decoder_rna = decoder_rna

    def encode(self, adj, batch, features=None):
        node_emb, graph_emb = self.encoder_shared(adj, batch, features)
        node_emb, graph_emb = self.encoder_rna(node_emb, graph_emb, adj, batch)
        return node_emb, graph_emb

    def decode(self, node_emb, graph_emb, actions=None, q_for_all=False):
        return self.decoder_rna(node_emb, graph_emb, actions, q_for_all)
