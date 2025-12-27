"""
Episode 类实现了整个学习过程中的一个回合，即与环境的交互逻辑，包括前向传播和束搜索（Beam Search）策略。
forward 方法用于训练过程中执行一系列动作并返回损失和相关信息，而 beam_search 方法则用于在测试时通过束搜索来选择最佳动作。
此类利用了智能体的政策来决定在特定状态下采取的动作，并通过与环境的交互来优化智能体的决策过程。
"""

import torch
import torch.nn as nn

class Episode(nn.Module):
    def __init__(self, env, agent, config):
        super(Episode, self).__init__()
        self.config = config
        self.env = env
        self.agent = agent
        self.path_length = config['path_length']
        self.num_rel = config['num_rel']
        # This is now the default/new value, but dynamic values will be passed during test/valid
        self.max_action_num = config['max_action_num_new']


    def forward(self, query_entities, query_timestamps, query_relations):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            all_loss: list
            all_logits: list
            all_actions_idx: list
            current_entities: torch.tensor, [batch_size]
            current_timestamps: torch.tensor, [batch_size]
        """
        # 获取查询实体和关系的嵌入
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)

        # 初始化当前实体和时间戳
        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP

        all_loss = []  # 存储所有损失
        all_logits = []  # 存储所有 logits
        all_actions_idx = []  # 存储所有动作索引

        # 创建一个满足函数签名的占位符，因为训练时不需要它
        batch_size = query_entities.shape[0]
        paths_history_placeholder = [[] for _ in range(batch_size)]

        # 设置初始隐藏状态
        self.agent.policy_step.set_hiddenx(query_relations.shape[0])
        for t in range(self.path_length):
            if t == 0:  # 检查是否为第一步
                first_step = True
            else:
                first_step = False

            # 获取下一个动作空间
            # MODIFIED: Use the fixed training max_action_num from config
            action_space = self.env.next_actions(
                current_entites,
                current_timestamps,
                (query_entities, query_relations),  # 修正：传递元组
                query_timestamps,
                paths_history_placeholder,  # <--- 传入占位符
                self.config['max_action_num_training'],
                first_step
            )

            # 通过智能体计算损失、logits 和动作索引
            loss, logits, action_id = self.agent(
                prev_relations,
                current_entites,
                current_timestamps,
                query_relations_embeds,
                query_entities_embeds,
                query_timestamps,
                action_space,
            )

            # 从动作空间中选择动作
            chosen_relation = torch.gather(action_space[:, :, 0], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity = torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=action_id).reshape(action_space.shape[0])

            # 保存当前步骤的结果
            all_loss.append(loss)
            all_logits.append(logits)
            all_actions_idx.append(action_id)

            # 更新当前实体、时间戳和前一关系
            current_entites = chosen_entity
            current_timestamps = chosen_entity_timestamps
            prev_relations = chosen_relation

        return all_loss, all_logits, all_actions_idx, current_entites, current_timestamps

    def beam_search(self, query_entities, query_timestamps, query_relations, max_action_nums):
        batch_size = query_entities.shape[0]
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)

        self.agent.policy_step.set_hiddenx(batch_size)

        # 初始化路径历史，每个batch项一个空列表
        paths_history = [[] for _ in range(batch_size)]

        # --- 第一步 ---
        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel

        action_space = self.env.next_actions(
            current_entites, current_timestamps,
            (query_entities, query_relations),
            query_timestamps,
            paths_history, max_action_nums,
            True
        )

        _, logits, _ = self.agent(
            prev_relations, current_entites, current_timestamps,
            query_relations_embeds, query_entities_embeds,
            query_timestamps, action_space
        )

        beam_size = min(self.config['beam_size'], action_space.shape[1])
        beam_log_prob, top_k_action_id = torch.topk(logits, beam_size, dim=1)
        beam_log_prob = beam_log_prob.reshape(-1)

        # 更新状态和路径历史
        chosen_relations = torch.gather(action_space[:, :, 0], 1, top_k_action_id).reshape(-1)
        chosen_entities = torch.gather(action_space[:, :, 1], 1, top_k_action_id).reshape(-1)

        # 扩展路径历史以匹配beam大小
        new_paths_history = []
        for i in range(batch_size):
            for j in range(beam_size):
                action_idx = top_k_action_id[i, j]
                rel_id = action_space[i, action_idx, 0].item()
                ent_id = action_space[i, action_idx, 1].item()
                # 复制父路径并添加新步骤
                new_paths_history.append(paths_history[i] + [(rel_id, ent_id)])
        paths_history = new_paths_history

        current_entites = chosen_entities
        current_timestamps = torch.gather(action_space[:, :, 2], 1, top_k_action_id).reshape(-1)
        prev_relations = chosen_relations

        # 扩展其他状态以匹配beam大小
        self.agent.policy_step.hx = self.agent.policy_step.hx.repeat_interleave(beam_size, dim=0)
        self.agent.policy_step.cx = self.agent.policy_step.cx.repeat_interleave(beam_size, dim=0)
        max_action_nums_roll = max_action_nums.repeat_interleave(beam_size)
        query_entities_roll = query_entities.repeat_interleave(beam_size)
        query_relations_roll = query_relations.repeat_interleave(beam_size)

        # --- 后续步骤循环 ---
        for t in range(1, self.path_length):
            query_timestamps_roll = query_timestamps.repeat_interleave(beam_size)
            query_entities_embeds_roll = query_entities_embeds.repeat_interleave(beam_size, dim=0)
            query_relations_embeds_roll = query_relations_embeds.repeat_interleave(beam_size, dim=0)

            action_space = self.env.next_actions(
                current_entites, current_timestamps,
                (query_entities_roll, query_relations_roll), query_timestamps_roll,
                paths_history, max_action_nums_roll, False
            )

            _, logits, _ = self.agent(
                prev_relations, current_entites, current_timestamps,
                query_relations_embeds_roll, query_entities_embeds_roll,
                query_timestamps_roll, action_space
            )

            hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
            cx_tmp = self.agent.policy_step.cx.reshape(batch_size, beam_size, -1)
            paths_history_tmp = [paths_history[i:i + beam_size] for i in range(0, len(paths_history), beam_size)]

            action_space_size = action_space.shape[1]
            beam_tmp = beam_log_prob.reshape(-1, 1) + logits
            beam_tmp = beam_tmp.reshape(batch_size, -1)

            next_beam_size = min(self.config['beam_size'], beam_tmp.shape[1])
            top_k_log_prob, top_k_action_id = torch.topk(beam_tmp, next_beam_size, dim=1)

            offset = top_k_action_id // action_space_size

            # 更新隐藏状态和路径历史
            new_paths_history = []
            for i in range(batch_size):
                for j in range(next_beam_size):
                    parent_beam_idx = offset[i, j].item()
                    parent_path = paths_history_tmp[i][parent_beam_idx]

                    action_idx_in_space = top_k_action_id[i, j] % action_space_size
                    action_space_start = i * beam_size * action_space_size + parent_beam_idx * action_space_size

                    rel_id = action_space.reshape(batch_size, beam_size, action_space_size, 3)[
                        i, parent_beam_idx, action_idx_in_space, 0].item()
                    ent_id = action_space.reshape(batch_size, beam_size, action_space_size, 3)[
                        i, parent_beam_idx, action_idx_in_space, 1].item()

                    new_paths_history.append(parent_path + [(rel_id, ent_id)])
            paths_history = new_paths_history

            offset_hx = offset.unsqueeze(-1).expand(-1, -1, self.config['state_dim'])
            offset_cx = offset.unsqueeze(-1).expand(-1, -1, self.config['state_dim'])  # 修复：定义 offset_cx
            self.agent.policy_step.hx = torch.gather(hx_tmp, 1, offset_hx).reshape(-1, self.config['state_dim'])
            self.agent.policy_step.cx = torch.gather(cx_tmp, 1, offset_cx).reshape(-1, self.config['state_dim'])

            action_space_reshaped = action_space.reshape(batch_size, -1, 3)
            current_entites = torch.gather(action_space_reshaped[:, :, 1], 1, top_k_action_id).reshape(-1)
            current_timestamps = torch.gather(action_space_reshaped[:, :, 2], 1, top_k_action_id).reshape(-1)
            prev_relations = torch.gather(action_space_reshaped[:, :, 0], 1, top_k_action_id).reshape(-1)

            beam_log_prob = top_k_log_prob.reshape(-1)
            beam_size = next_beam_size  # 更新当前的beam size

            # 更新用于下一轮的扩展张量
            max_action_nums_roll = torch.gather(max_action_nums_roll.reshape(batch_size, -1), 1, offset).reshape(-1)
            query_entities_roll = torch.gather(query_entities_roll.reshape(batch_size, -1), 1, offset).reshape(-1)
            query_relations_roll = torch.gather(query_relations_roll.reshape(batch_size, -1), 1, offset).reshape(-1)

        return action_space.reshape(batch_size, -1, 3)[:, :, 1], beam_tmp
