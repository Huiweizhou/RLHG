"""
这段代码实现了一个基于深度学习的智能体（Agent），用于处理关系和实体的动态嵌入。它使用 PyTorch 框架构建了多个模块，
包括历史编码器（HistoryEncoder）、策略多层感知机（PolicyMLP）、动态嵌入（DynamicEmbedding）、静态嵌入（StaticEmbedding）和主要的智能体类（Agent）。
该智能体能够根据历史动作和当前状态预测下一个动作。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# HistoryEncoder: 处理历史动作的编码，使用 LSTM 进行时间序列建模。
class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super(HistoryEncoder, self).__init__()  # 初始化父类
        self.config = config  # 保存配置
        # 创建 LSTM 单元，输入维度为动作维度，隐藏状态维度为状态维度
        self.lstm_cell = torch.nn.LSTMCell(input_size=config['action_dim'],
                                           hidden_size=config['state_dim'])

    # 设置隐藏层参数，初始化为0
    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')  # 隐藏状态初始化
            self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')  # 单元状态初始化
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])  # 隐藏状态初始化
            self.cx = torch.zeros(batch_size, self.config['state_dim'])  # 单元状态初始化

    # 前向传播函数，mask 表示是否为NO_OP
    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_, self.cx_ = self.lstm_cell(prev_action, (self.hx, self.cx))  # 更新 LSTM 状态
        # 依据 mask 更新隐藏状态和单元状态
        self.hx = torch.where(mask, self.hx, self.hx_)
        self.cx = torch.where(mask, self.cx, self.cx_)
        return self.hx  # 返回更新后的隐藏状态

# PolicyMLP: 多层感知机，用于根据智能体状态生成动作的概率分布。
class PolicyMLP(nn.Module):
    def __init__(self, config):
        super(PolicyMLP, self).__init__()
        self.mlp_l1= nn.Linear(config['mlp_input_dim'], config['mlp_hidden_dim'], bias=True)
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['action_dim'], bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = self.mlp_l2(hidden).unsqueeze(1)
        return output

class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t):
        super(DynamicEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent - dim_t)
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float())
        self.b = torch.nn.Parameter(torch.zeros(dim_t).float())

    def forward(self, entities, dt):
        dt = dt.unsqueeze(-1)
        batch_size = dt.size(0)
        seq_len = dt.size(1)

        dt = dt.view(batch_size, seq_len, 1)
        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))
        t = t.squeeze(1)  # [batch_size, time_dim]

        e = self.ent_embs(entities)
        return torch.cat((e, t), -1)

class StaticEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent):
        super(StaticEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent)

    def forward(self, entities, timestamps=None):
        return self.ent_embs(entities)

# 主要功能包括根据历史信息进行动作预测、计算得分、更新实体嵌入以及恢复嵌入。
# 通过这些功能，智能体可以在复杂的环境中进行决策。
class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()  # 初始化父类 nn.Module
        self.num_rel = config['num_rel'] + 2  # 计算关系数量
        self.config = config  # 保存配置

        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        # 定义操作常量
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] + 1  # Padding relation
        self.tPAD = 0  # Padding time

        # 根据配置选择动态或静态嵌入方法
        if self.config['entities_embeds_method'] == 'dynamic':
            self.ent_embs = DynamicEmbedding(config['num_ent']+1, config['ent_dim'], config['time_dim'])
        else:
            self.ent_embs = StaticEmbedding(config['num_ent']+1, config['ent_dim'])

        # 创建关系嵌入层
        self.rel_embs = nn.Embedding(self.num_rel, config['rel_dim'])

        # 创建历史编码器和策略MLP
        self.policy_step = HistoryEncoder(config)
        self.policy_mlp = PolicyMLP(config)

    # 前向传播函数，处理输入并预测动作。
    def forward(self, prev_relation, current_entities, current_timestamps,
                query_relation, query_entity, query_timestamps, action_space):
        """
        Args:
            prev_relation: [batch_size]
            current_entities: [batch_size]
            current_timestamps: [batch_size]
            query_relation: embeddings of query relation，[batch_size, rel_dim]
            query_entity: embeddings of query entity, [batch_size, ent_dim]
            query_timestamps: [batch_size]
            action_space: [batch_size, max_actions_num, 3] (relations, entities, timestamps)
        """
        # embeddings
        # 计算当前时间差
        current_delta_time = query_timestamps - current_timestamps
        # 获取当前实体的嵌入
        current_embds = self.ent_embs(current_entities, current_delta_time)  # [batch_size, ent_dim]
        # 获取前一关系的嵌入
        prev_relation_embds = self.rel_embs(prev_relation)  # [batch_size, rel_dim]

        # Pad Mask
        pad_mask = torch.ones_like(action_space[:, :, 0]) * self.rPAD  # [batch_size, action_number]
        pad_mask = torch.eq(action_space[:, :, 0], pad_mask)  # [batch_size, action_number]

        # History Encode
        # 处理历史编码
        NO_OP_mask = torch.eq(prev_relation, torch.ones_like(prev_relation) * self.NO_OP)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]
        # 拼接前一关系和当前实体的嵌入
        prev_action_embedding = torch.cat([prev_relation_embds, current_embds], dim=-1)  # [batch_size, rel_dim + ent_dim]
        # 通过历史编码器
        lstm_output = self.policy_step(prev_action_embedding, NO_OP_mask)  # [batch_size, state_dim]

        # Neighbor/condidate_actions embeddings
        # 获取邻居/候选动作的嵌入
        action_num = action_space.size(1)  # 动作数量
        # 计算邻居时间差
        neighbors_delta_time = query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 2]
        # 获取邻居实体的嵌入
        neighbors_entities = self.ent_embs(action_space[:, :, 1], neighbors_delta_time)  # [batch_size, action_num, ent_dim]
        # 获取邻居关系的嵌入
        neighbors_relations = self.rel_embs(action_space[:, :, 0])  # [batch_size, action_num, rel_dim]

        # agent state representation
        # 生成智能体状态表示
        agent_state = torch.cat([lstm_output, query_entity, query_relation], dim=-1)  # [batch_size, state_dim + ent_dim + rel_dim]

        # 通过策略 MLP 计算输出
        output = self.policy_mlp(agent_state)  # [batch_size, 1, action_dim] action_dim == rel_dim + ent_dim

        # 计算根据MLP生成的动作分布和当前实体的动作空间中动作的相似度
        scores = torch.sum(torch.mul(output, torch.cat([neighbors_relations, neighbors_entities], dim=-1)), dim=2)

        # Padding mask
        scores = scores.masked_fill(pad_mask, -1e10)  # [batch_size ,action_number]

        # 计算动作概率
        action_prob = torch.softmax(scores, dim=1)

        # 随机选择一个动作
        action_id = torch.multinomial(action_prob, 1)  # Randomly select an action. [batch_size, 1]

        # # 计算 logits
        logits = torch.nn.functional.log_softmax(scores, dim=1)  # [batch_size, action_number]
        # 生成 one-hot 编码
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        # 计算损失
        loss = - torch.sum(torch.mul(logits, one_hot), dim=1)
        # 返回损失、logits 和选择的动作 ID
        return loss, logits, action_id
