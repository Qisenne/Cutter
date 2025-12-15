import torch
import torch.nn as nn

# Subsequence Encoder: Encodes (s_seq, a_seq) into a vector
class SubsequenceEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action_num=500):
        super(SubsequenceEncoder, self).__init__()
        self.action_embedding = nn.Embedding(max_action_num, action_dim)
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, state_embeds, actions):
        actions = actions.to(state_embeds.device)
        act_embeds = self.action_embedding(actions)
        x = torch.cat([state_embeds, act_embeds], dim=-1)
        x = self.input_proj(x)
        output, h_n = self.gru(x)
        return h_n.squeeze(0)


# Extract the (s_seq, a_seq) corresponding to the maximum step of the true return from the trajectory.
def extract_prototype_from_traj(traj, encoder_shared, encoder_task, replay_buffer, k, device):

    returns = [step[4]["delta_true_return"] for step in traj]
    if not returns:
        raise RuntimeError("Empty trajectory, unable to extract prototype.")
    max_idx = max(range(len(returns)), key=lambda i: returns[i])

    start = max(0, max_idx - k)
    indices = list(range(start, max_idx + 1))

    s_embeds, actions = [], []
    for i in indices:
        state_id, action, *_ = traj[i]
        try:
            adj, feat, _ = replay_buffer.get_state(state_id, device)
        except KeyError:
            raise RuntimeError(f"[Prototype extraction] state_id={state_id} not exist in buffer")

        sparse_adj = adj.to_sparse()
        batch_node = torch.ones((1, sparse_adj.shape[0]), device=device).to_sparse()
        h_n, h_g = encoder_shared(sparse_adj, batch_node, feat.to(device))
        node_emb, _ = encoder_task(h_n, h_g, sparse_adj, batch_node)
        s_embeds.append(node_emb.mean(dim=0))
        actions.append(action)

    s_seq = torch.stack(s_embeds, dim=0)
    a_seq = torch.tensor(actions, dtype=torch.long, device=device)
    return s_seq, a_seq


def extract_avg_prototypes_from_buffer(
    replay_buffer, role, encoder_shared, encoder_task,
    subseq_encoder, k, device
):
    top_list = replay_buffer.top_k_trajs[role]
    bot_list = replay_buffer.bottom_k_trajs[role]

    def encode_one(traj):
        s_seq, a_seq = extract_prototype_from_traj(
            traj, encoder_shared, encoder_task, replay_buffer, k, device
        )
        return subseq_encoder(s_seq.unsqueeze(0), a_seq.unsqueeze(0)).squeeze(0)

    bottom_embs = []
    for _, traj in bot_list:
        emb = encode_one(traj)
        if emb is not None:
            bottom_embs.append(emb)

    top_embs = []
    for _, traj in reversed(top_list):
        emb = encode_one(traj)
        if emb is not None:
            top_embs.append(emb)

    avg_neg = torch.stack(bottom_embs, dim=0).mean(dim=0) if bottom_embs else None
    avg_pos = torch.stack(top_embs,    dim=0).mean(dim=0) if top_embs    else None
    return avg_pos, avg_neg
