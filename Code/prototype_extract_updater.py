import torch
import torch.nn as nn

# 子序列编码器：将 (s_seq, a_seq) 编码为向量
class SubsequenceEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action_num=500):
        super(SubsequenceEncoder, self).__init__()
        self.action_embedding = nn.Embedding(max_action_num, action_dim)  # 整数动作映射成向量
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)   # (s a)拼接后，投影到hd
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)       # 使用单层GRU编码

    def forward(self, state_embeds, actions):
        actions = actions.to(state_embeds.device)
        act_embeds = self.action_embedding(actions)                       # 动作转换成向量
        x = torch.cat([state_embeds, act_embeds], dim=-1)          # 拼接(s a)
        x = self.input_proj(x)                                            # 维度投影
        output, h_n = self.gru(x)                                         # 输出序列聚合
        return h_n.squeeze(0)


# 从轨迹中提取 Ture return 最大 step 的 (s_seq, a_seq)
def extract_prototype_from_traj(traj, encoder_shared, encoder_task, replay_buffer, k, device):
    ######For privacy, temporarily hide.#######
    pass


# 从历史轨迹中提取 Top-K / Bottom-K 原型，平均得到 avg_pos / avg_neg
def extract_avg_prototypes_from_buffer():
    ######For privacy, temporarily hide.#######
    pass