import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from graph_utils import gen_graph,pairwise_connectivity
from reward_shape import compute_reward_supervised_by_prototypes
import gc

def prepare_train_data(n_train, num_min, num_max):
    print('\n[AgentI/RNA] Generating training graphs...')
    return [gen_graph(num_min, num_max) for _ in tqdm(range(n_train))]

def run_episode(shared_encoder, task_encoder, decoder, epsilon, env, role='ina',
                follow=False, follow_traj=None, is_warmup=False):
    ######For privacy, temporarily hide.#######
    pass

def soft_update(target_net, source_net, tau=0.01):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        with torch.no_grad():
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def train_reward_net_full(
        reward_net, subseq_encoder, avg_pos, avg_neg,
        encoder_shared, encoder_task, optimizer,
        replay_buffer, affine, role, device,
        k=3, batch_size=64, proto_weight=1.0
):
    ######For privacy, temporarily hide.#######
    pass

def train_dqn(shared_encoder, task_encoder, decoder, decoder_target, optimizer,
              replay_buffer, device, batch_size=64, gamma=0.99):
    half = batch_size // 2
    batch_self   = replay_buffer.sample(half,   condition='self')
    batch_follow = replay_buffer.sample(half,   condition='follow')
    batch = batch_self + batch_follow
    if len(batch) < batch_size:
        print("[DQN Debug] Incomplete batch due to insufficient self/follow data.")
        return 0.0
    random.shuffle(batch)

    # 拆包
    states_idx, actions, next_states_idx, dones, infos, *_ = zip(*batch)
    # 恢复 raw 数据
    states_raw      = [replay_buffer.get_state(i, device) for i in states_idx]
    next_states_raw = [replay_buffer.get_state(i, device) for i in next_states_idx]
    adjs,   feats,   _    = zip(*states_raw)
    next_adjs, next_feats, _ = zip(*next_states_raw)

    # —— 当前 states 编码 ——
    node_emb_list, graph_emb_list = [], []
    for adj, feat in zip(adjs, feats):
        # 一次性 to(device) 并转稀疏
        if not adj.is_sparse:
            sparse_state = adj.to(device).to_sparse()
        else:
            sparse_state = adj.to(device)
        feat_state = feat.to(device)

        n_nodes   = sparse_state.shape[0]
        batch_node= torch.ones((1, n_nodes), dtype=torch.float32, device=device).to_sparse()

        h_n_s, h_g_s   = shared_encoder(sparse_state, batch_node, feat_state)
        h_n_r, h_g_r   = task_encoder(h_n_s, h_g_s, sparse_state, batch_node)

        node_emb_list.append(h_n_r)           # (n_nodes, hidden_dim)
        graph_emb_list.append(h_g_r.squeeze(0))  # (hidden_dim,)

    # pad node embeddings 到同一长度
    padded_node_embs  = torch.nn.utils.rnn.pad_sequence(node_emb_list, batch_first=True)
    padded_graph_embs = torch.stack(graph_emb_list)  # (batch, hidden_dim)

    # 取 Q(s, a)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    q_values = decoder(padded_node_embs, padded_graph_embs, actions_tensor)

    # —— 下一个 states 编码 ——
    next_node_emb_list, next_graph_emb_list, node_counts = [], [], []
    for adj, feat in zip(next_adjs, next_feats):
        if not adj.is_sparse:
            sparse_next = adj.to(device).to_sparse()
        else:
            sparse_next = adj.to(device)
        feat_next = feat.to(device)

        n_nodes    = sparse_next.shape[0]
        node_counts.append(n_nodes)
        batch_node = torch.ones((1, n_nodes), dtype=torch.float32, device=device).to_sparse()

        h_n_s, h_g_s = shared_encoder(sparse_next, batch_node, feat_next)
        h_n_r, h_g_r = task_encoder(h_n_s, h_g_s, sparse_next, batch_node)

        next_node_emb_list.append(h_n_r)
        next_graph_emb_list.append(h_g_r.squeeze(0))

    padded_next_node_embs  = torch.nn.utils.rnn.pad_sequence(next_node_emb_list, batch_first=True)
    padded_next_graph_embs = torch.stack(next_graph_emb_list)

    # —— 构建下一个 state mask ——
    max_nodes = max(node_counts)
    mask_next = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    for i, cnt in enumerate(node_counts):
        mask_next[i, :cnt] = True

    # —— 计算 target Q ——
    rewards_tensor = torch.tensor([b[-1] for b in batch], dtype=torch.float32, device=device)
    dones_tensor   = torch.tensor(dones, dtype=torch.float32, device=device)

    with torch.no_grad():
        all_next_q = decoder_target(
            padded_next_node_embs,
            padded_next_graph_embs,
            q_for_all=True
        )  # (batch, seq_len)
        # 只保留真实节点部分
        L = min(all_next_q.size(1), mask_next.size(1))
        all_next_q = all_next_q[:, :L]
        mask_trim  = mask_next[:, :L]
        all_next_q[~mask_trim] = float('-inf')
        max_next_q = all_next_q.max(dim=1).values

    target_q = rewards_tensor + gamma * max_next_q * (1 - dones_tensor)

    # —— 优化 step ——
    loss = F.mse_loss(q_values.squeeze(), target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    # else:
    #     gc.collect()

    return loss.item()

def pad_and_stack_adjacency(adj_list, device):
    """
    仅返回 mask，用于标记每个样本前 n_nodes 为真实节点，后面为 padding。
    """
    node_counts = [adj.shape[0] for adj in adj_list]
    max_nodes   = max(node_counts)
    mask = torch.zeros((len(adj_list), max_nodes), dtype=torch.bool, device=device)
    for i, cnt in enumerate(node_counts):
        mask[i, :cnt] = True
    return mask

