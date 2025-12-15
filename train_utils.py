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
    episode = env.INEpisode if role == 'ina' else env.RNEpisode
    replay_buffer = env.replay_ina if role == 'ina' else env.replay_rna
    traj = []

    state, feature = episode.get_current_state()
    feature = feature.to(env.device)

    while True:
        if follow:
            if not follow_traj or len(follow_traj) == 0:
                break
            if len(traj) >= len(follow_traj):
                break
            step = follow_traj[len(traj)]
            node_id = step[5]
            action = episode.node_list_map[node_id]
            is_exploit = True
        else:
            if episode.is_terminal():
                break
            if random.random() < epsilon:
                action = episode.sample_random_action()
                is_exploit = False
            else:
                action = episode.select_action()
                is_exploit = True
            action = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
            node_id = episode.node_list[action]

        sparse_state = state.to(env.device)
        if not sparse_state.is_sparse:
            sparse_state = sparse_state.to_sparse()
        batch_node = torch.ones((1, sparse_state.shape[0]),
                                 dtype=torch.float32,
                                 device=env.device).to_sparse()
        h_n, h_g = shared_encoder(sparse_state, batch_node, input_node_features=feature)
        node_emb, graph_emb = task_encoder(h_n, h_g, sparse_state, batch_node)
        reward = episode.compute_step_reward(node_emb, graph_emb, action)
        node_id_list = episode.node_list.copy()
        state_idx = replay_buffer.store_state(state, feature, node_id_list)
        next_state, _, done, _ = episode.step(action, reward=reward)
        next_feature = torch.ones((next_state.shape[0], 1),
                                  dtype=torch.float32,
                                  device=env.device)
        next_node_id_list = episode.node_list.copy()
        next_idx = replay_buffer.store_state(next_state, next_feature, next_node_id_list)
        delta_true_return = episode.true_return()
        info = {"delta_true_return": delta_true_return}
        traj.append((state_idx, action, next_idx, done,
                     info, node_id, is_exploit, reward))
        state, feature = next_state, next_feature
        if done:
            break
    return traj, episode.true_return()

def soft_update(target_net, source_net, tau=0.01):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        with torch.no_grad():
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def train_reward_net_full(
        reward_net, subseq_encoder, avg_pos, avg_neg,
        encoder_shared, encoder_task, optimizer,
        replay_buffer, affine, role, device,
        k=3, batch_size=16, proto_weight=1.0
):

    trajs, returns = replay_buffer.get_recent_trajectories(
        role=role, num_trajectories=batch_size
    )
    if len(trajs) == 0:
        return 0.0, 0.0, 0.0

    mse_loss_fn = nn.MSELoss()
    total_reward_loss = 0.0
    total_affine_loss = 0.0
    total_proto_loss = 0.0
    count_steps = 0

    # 对每条轨迹单独进行更新
    for traj, R in zip(trajs, returns):
        optimizer.zero_grad()

        traj_step_losses = []
        for i in range(1, len(traj)):
            state_idx, action, *rest = traj[i]
            state, feat, _ = replay_buffer.get_state(state_idx, device)
            sparse_adj = state.to_sparse()
            batch_node = torch.ones((1, sparse_adj.size(0)), device=device).to_sparse()
            with torch.no_grad():
                h_n, h_g = encoder_shared(sparse_adj, batch_node, feat.to(device))
                node_emb, graph_emb = encoder_task(h_n, h_g, sparse_adj, batch_node)
            pooled = node_emb.mean(dim=0)
            seq_start = max(0, i - k)
            s_embeds = []
            actions = []
            with torch.no_grad():
                for j in range(seq_start, i + 1):
                    st_idx, act, *rest_j = traj[j]
                    adj_j, feat_j, _ = replay_buffer.get_state(st_idx, device)
                    sparse_adj_j = adj_j.to_sparse()
                    batch_node_j = torch.ones((1, sparse_adj_j.size(0)), device=device).to_sparse()
                    h_n_j, h_g_j = encoder_shared(sparse_adj_j, batch_node_j, feat_j.to(device))
                    node_emb_j, _ = encoder_task(h_n_j, h_g_j, sparse_adj_j, batch_node_j)
                    s_embeds.append(node_emb_j.mean(dim=0))
                    actions.append(act)
            s_seq = torch.stack(s_embeds, dim=0)
            a_seq = torch.tensor(actions, dtype=torch.long, device=device)
            with torch.no_grad():
                target_r = compute_reward_supervised_by_prototypes(
                    s_seq, a_seq, subseq_encoder,
                    avg_pos.to(device), avg_neg.to(device)
                ).clamp(-1, 1)
            pred_r = reward_net(pooled.unsqueeze(0), graph_emb).view(-1)
            loss_r = mse_loss_fn(pred_r, target_r.view(-1))
            traj_step_losses.append(loss_r)
            total_reward_loss += loss_r.item()
            count_steps += 1
        if traj_step_losses:
            loss_proto = proto_weight * sum(traj_step_losses) / len(traj_step_losses)
            total_proto_loss += loss_proto.item()
            R_pred_sum = sum(l.item() for l in traj_step_losses)
            R_affine = affine(torch.tensor([R_pred_sum], device=device))
            loss_affine = mse_loss_fn(R_affine.view(-1), torch.tensor([R], device=device))
            total_affine_loss += loss_affine.item()
            (loss_proto + loss_affine).backward()
            optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
    avg_reward_loss = total_reward_loss / max(count_steps, 1)
    avg_affine_loss = total_affine_loss / len(trajs)
    avg_proto_loss = total_proto_loss / max(count_steps, 1)
    return avg_reward_loss, avg_affine_loss, avg_proto_loss

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

    states_idx, actions, next_states_idx, dones, infos, *_ = zip(*batch)
    states_raw      = [replay_buffer.get_state(i, device) for i in states_idx]
    next_states_raw = [replay_buffer.get_state(i, device) for i in next_states_idx]
    adjs,   feats,   _    = zip(*states_raw)
    next_adjs, next_feats, _ = zip(*next_states_raw)

    node_emb_list, graph_emb_list = [], []
    for adj, feat in zip(adjs, feats):
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

    # pad node embeddings
    padded_node_embs  = torch.nn.utils.rnn.pad_sequence(node_emb_list, batch_first=True)
    padded_graph_embs = torch.stack(graph_emb_list)  # (batch, hidden_dim)

    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    q_values = decoder(padded_node_embs, padded_graph_embs, actions_tensor)

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

    max_nodes = max(node_counts)
    mask_next = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    for i, cnt in enumerate(node_counts):
        mask_next[i, :cnt] = True

    rewards_tensor = torch.tensor([b[-1] for b in batch], dtype=torch.float32, device=device)
    dones_tensor   = torch.tensor(dones, dtype=torch.float32, device=device)

    with torch.no_grad():
        all_next_q = decoder_target(
            padded_next_node_embs,
            padded_next_graph_embs,
            q_for_all=True
        )  # (batch, seq_len)
        L = min(all_next_q.size(1), mask_next.size(1))
        all_next_q = all_next_q[:, :L]
        mask_trim  = mask_next[:, :L]
        all_next_q[~mask_trim] = float('-inf')
        max_next_q = all_next_q.max(dim=1).values

    target_q = rewards_tensor + gamma * max_next_q * (1 - dones_tensor)

    loss = F.mse_loss(q_values.squeeze(), target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def pad_and_stack_adjacency(adj_list, device):

    node_counts = [adj.shape[0] for adj in adj_list]
    max_nodes   = max(node_counts)
    mask = torch.zeros((len(adj_list), max_nodes), dtype=torch.bool, device=device)
    for i, cnt in enumerate(node_counts):
        mask[i, :cnt] = True
    return mask

