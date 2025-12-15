import random
from collections import OrderedDict

class ExperienceReplayBuffer:
    def __init__(self, capacity,proto_k=5):
        self.capacity = capacity
        self.self_exploration = []
        self.follow_exploration = []
        self.trajectories = {'ina': [], 'rna': []}

        self.state_pool = OrderedDict()
        self.proto_k = proto_k
        self.top_k_trajs = {'ina': [], 'rna': []}
        self.bottom_k_trajs = {'ina': [], 'rna': []}
        self._next_state_id = 0

    def store_state(self, adj, feature, node_id_list):
        # detach from any graph and move to CPU
        adj_cpu = adj.detach().cpu()
        feat_cpu = feature.detach().cpu()
        state_id = self._next_state_id
        self._next_state_id += 1
        self.state_pool[state_id] = (adj_cpu, feat_cpu, node_id_list)
        return state_id

    def get_state(self, state_id, device):
        try:
            adj_cpu, feat_cpu, node_id_list = self.state_pool[state_id]
        except KeyError:
            raise KeyError(f"[ReplayBuffer] state_id {state_id} not in pool (size={len(self.state_pool)}).")
        adj = adj_cpu.to(device)
        feat = feat_cpu.to(device)
        return adj, feat, node_id_list

    def add(self, experience, trajectory_type='self'):
        if trajectory_type == 'self':
            self.self_exploration.append(experience)
            if len(self.self_exploration) > self.capacity:
                self.self_exploration.pop(0)
        elif trajectory_type == 'follow':
            self.follow_exploration.append(experience)
            if len(self.follow_exploration) > self.capacity:
                self.follow_exploration.pop(0)
        else:
            raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

    def add_trajectory(self, trajectory, ret, role='ina'):
        self.trajectories[role].append((trajectory, ret))
        if len(self.trajectories[role]) > self.capacity:
            self.trajectories[role].pop(0)

    def sample(self, batch_size, condition='mixed'):
        if condition == 'self':
            return random.sample(self.self_exploration, min(batch_size, len(self.self_exploration)))
        elif condition == 'follow':
            return random.sample(self.follow_exploration, min(batch_size, len(self.follow_exploration)))
        else:
            total = self.self_exploration + self.follow_exploration
            return random.sample(total, min(batch_size, len(total)))

    def get_recent_trajectories(self, role='ina', num_trajectories=10):
        trajs = self.trajectories[role]
        selected = trajs[-num_trajectories:] if len(trajs) >= num_trajectories else trajs
        trajectories, returns = zip(*selected)
        return list(trajectories), list(returns)

    def get_size(self):
        return {
            'self': len(self.self_exploration),
            'follow': len(self.follow_exploration),
            'trajectories_ina': len(self.trajectories['ina']),
            'trajectories_rna': len(self.trajectories['rna']),
            'total': len(self.self_exploration) + len(self.follow_exploration)
        }

    def update_prototypes(self, role: str, traj, traj_return: float):
        top_list = self.top_k_trajs[role]
        if len(top_list) < self.proto_k:
            top_list.append((traj_return, traj))
            top_list.sort(key=lambda x: x[0])
        else:
            if traj_return > top_list[0][0]:
                top_list[0] = (traj_return, traj)
                top_list.sort(key=lambda x: x[0])

        bot_list = self.bottom_k_trajs[role]
        if len(bot_list) < self.proto_k:
            bot_list.append((traj_return, traj))
            bot_list.sort(key=lambda x: x[0])
        else:
            if traj_return < bot_list[-1][0]:
                bot_list[-1] = (traj_return, traj)
                bot_list.sort(key=lambda x: x[0])
