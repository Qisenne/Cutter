import torch
import random
import numpy as np
from tqdm import tqdm
from environment import DualAgentController
from agent import SharedEncoderGraph, TaskEncoder, QDecoder,INAAgent,RNAAgent
from experience_replay_buffer import ExperienceReplayBuffer
from reward_shape import RewardNetworkGraphAware, AffineTransformation
from prototype_extract_updater import extract_avg_prototypes_from_buffer,SubsequenceEncoder
import networkx as nx
import os
import gc
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from train_utils import (
    prepare_train_data, run_episode,
    train_dqn, train_reward_net_full,
    soft_update
)
# === 超参数 ===
GAMMA = 0.99
LR = 1e-4
REPLAY_SIZE = 20000
TARGET_UPDATE_FREQ = 100
PRINT_FREQ = 10
EVAL_FREQ = 50
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000
EPISODES = 4000
TOTAL_EXPLORATIONS = 500
N_TRAIN_GRAPH = 3000
NODE_NUM_MIN = 100
NODE_NUM_MAX = 150
PROTOTYPE_TOP_K = 10
UPDATE_PROTOTYPE_FREQ = 10

# === 设备 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
tensor_type = torch.float32
# === 训练图 ===
train_graphs = prepare_train_data(N_TRAIN_GRAPH, num_min=NODE_NUM_MIN, num_max=NODE_NUM_MAX)
# === 模型初始化 ===
encoder_shared = SharedEncoderGraph(input_size=1, embedding_size=128, tensor_type=tensor_type, device=device).to(device)
encoder_ina = TaskEncoder(embedding_size=128, depth=3, tensor_type=tensor_type, device=device).to(device)
encoder_rna = TaskEncoder(embedding_size=128, depth=3, tensor_type=tensor_type, device=device).to(device)

decoder_ina = QDecoder(embedding_size=128).to(device)
decoder_ina_target = QDecoder(embedding_size=128).to(device)
decoder_rna = QDecoder(embedding_size=128).to(device)
decoder_rna_target = QDecoder(embedding_size=128).to(device)

reward_net_ina = RewardNetworkGraphAware(128).to(device)
reward_net_rna = RewardNetworkGraphAware(128).to(device)
affine_ina = AffineTransformation().to(device)
affine_rna = AffineTransformation().to(device)
# === 优化器 ===
optimizer_ina = torch.optim.Adam(list(encoder_shared.parameters()) + list(encoder_ina.parameters()) + list(decoder_ina.parameters()), lr=LR)
optimizer_rna = torch.optim.Adam(list(encoder_shared.parameters()) + list(encoder_rna.parameters()) + list(decoder_rna.parameters()), lr=LR)
optimizer_reward_ina = torch.optim.Adam(list(reward_net_ina.parameters()) + list(affine_ina.parameters()), lr=LR)
optimizer_reward_rna = torch.optim.Adam(list(reward_net_rna.parameters()) + list(affine_rna.parameters()), lr=LR)
# === Agent ===
agent_ina = INAAgent(encoder_shared, encoder_ina, decoder_ina)
agent_rna = RNAAgent(encoder_shared, encoder_rna, decoder_rna)
# === 经验池 ===
replay_ina = ExperienceReplayBuffer(REPLAY_SIZE)
replay_rna = ExperienceReplayBuffer(REPLAY_SIZE)
# === 环境控制器 ===
env = DualAgentController(tensor_type=tensor_type, device=device)
env.replay_ina = replay_ina
env.replay_rna = replay_rna
# === 原型管理器 ===
subseq_encoder = SubsequenceEncoder(state_dim=128, action_dim=16, hidden_dim=128, max_action_num=500).to(device)
avg_pos_ina = torch.zeros(128, device=device)
avg_neg_ina = torch.zeros(128, device=device)
avg_pos_rna = torch.zeros(128, device=device)
avg_neg_rna = torch.zeros(128, device=device)
# ====== Warmup阶段，填充经验池 ======
print("[Warmup] Start filling replay buffers...")
exploration_count = 0
with tqdm(total=TOTAL_EXPLORATIONS, desc="[Warmup]", leave=True, ncols=100, unit="episode") as pbar:
    while exploration_count < TOTAL_EXPLORATIONS:
        # === Step 1: 随机选图 ===
        graph_original = random.choice(train_graphs)

        # === Step 2: INA 主动探索 ===
        env.load_graph(graph_original.copy(), agent_ina, agent_rna, reward_net_ina, reward_net_rna)
        traj_ina, ret_ina = run_episode(
            encoder_shared, encoder_ina, decoder_ina,
            epsilon=1.0, env=env, role='ina', follow=False
        )
        follow_traj_ina = traj_ina
        # important_node_ids = [step[5] for step in traj_ina]

        sorted_traj = sorted(traj_ina, key=lambda x: x[-1], reverse=True)
        top_k = max(1, int(0.1 * len(traj_ina)))
        important_node_ids = [step[5] for step in traj_ina[:top_k]]

        # === Step 3: RNA 跟随 INA ===
        env.load_graph(graph_original.copy(), agent_ina, agent_rna, reward_net_ina, reward_net_rna)
        env.RNEpisode.important_node_ids = important_node_ids
        traj_rna_follow, ret_rna_follow = run_episode(
            encoder_shared, encoder_rna, decoder_rna,
            epsilon=1.0, env=env, role='rna', follow=True, follow_traj=follow_traj_ina
        )

        # === Step 4: RNA 主动探索 ===
        env.load_graph(graph_original.copy(), agent_ina, agent_rna, reward_net_ina, reward_net_rna)
        env.RNEpisode.important_node_ids = important_node_ids
        env.RNEpisode.store_original_embedding()
        traj_rna_self, ret_rna_self = run_episode(
            encoder_shared, encoder_rna, decoder_rna,
            epsilon=1.0, env=env, role='rna', follow=False
        )
        follow_traj_rna = traj_rna_self

        # === Step 5: INA 跟随 RNA ===
        env.load_graph(graph_original.copy(), agent_ina, agent_rna, reward_net_ina, reward_net_rna)
        env.RNEpisode.important_node_ids = important_node_ids
        traj_ina_follow, ret_ina_follow = run_episode(
            encoder_shared, encoder_ina, decoder_ina,
            epsilon=1.0, env=env, role='ina', follow=True, follow_traj=follow_traj_rna
        )

        # === 单步经验填充 ===
        for step in traj_ina:
            replay_ina.add(step, trajectory_type='self')
        for step in traj_ina_follow:
            replay_ina.add(step, trajectory_type='follow')
        for step in traj_rna_self:
            replay_rna.add(step, trajectory_type='self')
        for step in traj_rna_follow:
            replay_rna.add(step, trajectory_type='follow')

        # === 轨迹整体保存 ===
        replay_ina.add_trajectory(traj_ina, ret_ina, role='ina')
        replay_ina.add_trajectory(traj_ina_follow, ret_ina_follow, role='ina')
        replay_rna.add_trajectory(traj_rna_self, ret_rna_self, role='rna')
        replay_rna.add_trajectory(traj_rna_follow, ret_rna_follow, role='rna')

        replay_ina.update_prototypes('ina', traj_ina, ret_ina)
        replay_ina.update_prototypes('ina', traj_ina_follow, ret_ina_follow)
        replay_rna.update_prototypes('rna', traj_rna_self, ret_rna_self)
        replay_rna.update_prototypes('rna', traj_rna_follow, ret_rna_follow)

        exploration_count += 1

        if exploration_count % UPDATE_PROTOTYPE_FREQ == 0:
            avg_pos_ina, avg_neg_ina = extract_avg_prototypes_from_buffer(
                replay_buffer=replay_ina,
                role='ina',
                encoder_shared=encoder_shared,
                encoder_task=encoder_ina,
                subseq_encoder=subseq_encoder,
                k=3,
                device=device,
            )
            avg_pos_rna, avg_neg_rna = extract_avg_prototypes_from_buffer(
                replay_buffer=replay_rna,
                role='rna',
                encoder_shared=encoder_shared,
                encoder_task=encoder_rna,
                subseq_encoder=subseq_encoder,
                k=3,
                device=device,
            )
        pbar.update(1)

print("[Warmup] Forcing initial prototype evaluation...")

# ====== 主训练循环 ======
total_steps = 0
with tqdm(total=EPISODES, desc="Training Progress", dynamic_ncols=True) as pbar_train:
    for episode in range(EPISODES):
        # epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * total_steps / EPSILON_DECAY)
        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY)
        # === Step 1: 随机选图 ===
        graph_original = random.choice(train_graphs)

        # === Step 2: INA 主动探索 ===
        env.load_graph(graph_original.copy(), agent_ina, agent_rna, reward_net_ina, reward_net_rna)
        traj_ina, ret_ina = run_episode(
            encoder_shared, encoder_ina, decoder_ina,
            epsilon=epsilon, env=env, role='ina', follow=False
        )
        follow_traj_ina = traj_ina
        # important_node_ids = [step[5] for step in traj_ina]

        sorted_traj = sorted(traj_ina, key=lambda x: x[-1], reverse=True)
        top_k = max(1, int(0.1 * len(traj_ina)))
        important_node_ids = [step[5] for step in traj_ina[:top_k]]

        # === Step 3: RNA 跟随 INA ===
        env.load_graph(graph_original.copy(), agent_ina, agent_rna, reward_net_ina, reward_net_rna)
        env.RNEpisode.important_node_ids = important_node_ids
        traj_rna_follow, ret_rna_follow = run_episode(
            encoder_shared, encoder_rna, decoder_rna,
            epsilon=epsilon, env=env, role='rna', follow=True, follow_traj=follow_traj_ina
        )

        # === Step 4: RNA 主动探索 ===
        env.load_graph(graph_original.copy(), agent_ina, agent_rna, reward_net_ina, reward_net_rna)
        env.RNEpisode.important_node_ids = important_node_ids
        env.RNEpisode.store_original_embedding()
        traj_rna_self, ret_rna_self = run_episode(
            encoder_shared, encoder_rna, decoder_rna,
            epsilon=epsilon, env=env, role='rna', follow=False
        )
        follow_traj_rna = traj_rna_self

        # === Step 5: INA 跟随 RNA ===
        env.load_graph(graph_original.copy(), agent_ina, agent_rna, reward_net_ina, reward_net_rna)
        env.RNEpisode.important_node_ids = important_node_ids
        traj_ina_follow, ret_ina_follow = run_episode(
            encoder_shared, encoder_ina, decoder_ina,
            epsilon=epsilon, env=env, role='ina', follow=True, follow_traj=follow_traj_rna
        )

        # === 单步经验填充 ===
        for step in traj_ina:
            state_idx, action, next_idx, done, info, node_id, is_exploit, reward = step
            replay_ina.add((state_idx, action, next_idx, done,info, node_id, is_exploit, reward), trajectory_type='self')

        for step in traj_ina_follow:
            state_idx, action, next_idx, done, info, node_id, is_exploit, reward = step
            replay_ina.add((state_idx, action, next_idx, done,info, node_id, is_exploit, reward), trajectory_type='follow')

        for step in traj_rna_self:
            state_idx, action, next_idx, done, info, node_id, is_exploit, reward = step
            replay_rna.add((state_idx, action, next_idx, done,info, node_id, is_exploit, reward), trajectory_type='self')

        for step in traj_rna_follow:
            state_idx, action, next_idx, done, info, node_id, is_exploit, reward = step
            replay_rna.add((state_idx, action, next_idx, done,info, node_id, is_exploit, reward), trajectory_type='follow')

        # === 轨迹整体保存 ===
        replay_ina.add_trajectory(traj_ina, ret_ina, role='ina')
        replay_rna.add_trajectory(traj_rna_follow, ret_rna_follow, role='rna')
        replay_rna.add_trajectory(traj_rna_self, ret_rna_self, role='rna')
        replay_ina.add_trajectory(traj_ina_follow, ret_ina_follow, role='ina')

        replay_ina.update_prototypes('ina', traj_ina, ret_ina)
        replay_ina.update_prototypes('ina', traj_ina_follow, ret_ina_follow)
        replay_rna.update_prototypes('rna', traj_rna_self, ret_rna_self)
        replay_rna.update_prototypes('rna', traj_rna_follow, ret_rna_follow)

        loss_reward_ina, mse_loss_ina, proto_loss_ina = train_reward_net_full(
            reward_net=reward_net_ina,
            subseq_encoder=subseq_encoder,
            avg_pos=avg_pos_ina,
            avg_neg=avg_neg_ina,
            encoder_shared=encoder_shared,
            encoder_task=encoder_ina,
            optimizer=optimizer_reward_ina,
            replay_buffer=replay_ina,
            affine=affine_ina,
            role='ina',
            device=device
        )
        loss_reward_rna, mse_loss_rna, proto_loss_rna = train_reward_net_full(
            reward_net=reward_net_rna,
            subseq_encoder=subseq_encoder,
            avg_pos=avg_pos_rna,
            avg_neg=avg_neg_rna,
            encoder_shared=encoder_shared,
            encoder_task=encoder_rna,
            optimizer=optimizer_reward_rna,
            replay_buffer=replay_rna,
            affine=affine_rna,
            role='rna',
            device=device
        )

        loss_ina = train_dqn(encoder_shared, encoder_ina, decoder_ina, decoder_ina_target, optimizer_ina, replay_ina, device)
        loss_rna = train_dqn(encoder_shared, encoder_rna, decoder_rna, decoder_rna_target, optimizer_rna, replay_rna, device)

        if episode % TARGET_UPDATE_FREQ == 0:
            soft_update(decoder_ina_target, decoder_ina, tau=0.1)
            soft_update(decoder_rna_target, decoder_rna, tau=0.1)

        if episode % UPDATE_PROTOTYPE_FREQ == 0:
            avg_pos_ina, avg_neg_ina = extract_avg_prototypes_from_buffer(
                replay_buffer=replay_ina,
                role='ina',
                encoder_shared=encoder_shared,
                encoder_task=encoder_ina,
                subseq_encoder=subseq_encoder,
                k=3,
                device=device,
            )
            avg_pos_rna, avg_neg_rna = extract_avg_prototypes_from_buffer(
                replay_buffer=replay_rna,
                role='rna',
                encoder_shared=encoder_shared,
                encoder_task=encoder_rna,
                subseq_encoder=subseq_encoder,
                k=3,
                device=device,
            )

        total_steps += 1

        if episode % PRINT_FREQ == 0:
            print(f"[Ep {episode}] ε={epsilon:.4f}, Loss INA: {loss_ina:.4f}, Loss RNA: {loss_rna:.4f}")
            print(f"[Ep {episode}] Reward Loss INA: {loss_reward_ina:.4f}, Reward Loss RNA: {loss_reward_rna:.4f}")
            print(f"[Ep {episode}] AffineLoss INA: {mse_loss_ina:.4f}, ProtoLoss INA: {proto_loss_ina:.4f}")
            print(f"[Ep {episode}] AffineLoss RNA: {mse_loss_rna:.4f}, ProtoLoss RNA: {proto_loss_rna:.4f}")
        if episode % EVAL_FREQ == 0:
            test_graph = nx.barabasi_albert_graph(n=1000, m=3)
           # test_ina_rna(test_graph, encoder_shared, encoder_ina, decoder_ina, encoder_rna, decoder_rna, device, episode)
        pbar_train.update(1)
os.makedirs("results", exist_ok=True)
np.save("results/ret_rna_with_reward_net.npy", np.array(returns_with_reward_net))
