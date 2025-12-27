"""
PG 类实现了一个策略梯度（Policy Gradient）算法的基础结构，主要用于强化学习。
它包含奖励计算、累计折扣奖励计算、熵正则化损失以及策略损失的计算。
"""

import torch
import numpy as np
import math
from src.baseline import ReactiveBaseline
import torch.nn.functional as F

class PG(object):
    def __init__(self, config, agent):
        self.config = config
        self.agent = agent  # <--- NEW: 保存 agent 实例
        self.positive_reward = 1.0  # 正奖励
        self.negative_reward = 0.0  # 负奖励
        self.baseline = ReactiveBaseline(config, config['lambda'])  # 初始化基线
        self.now_epoch = 0  # 当前训练的 epoch

    # 计算当前状态下的奖励
    # Args:
    #   current_entities: 当前实体的张量；
    #   answers: 正确答案的张量。
    # Return:
    #   奖励值的张量。
    # def get_reward(self, current_entites, answers):
    #      positive = torch.ones_like(current_entites, dtype=torch.float32) * self.positive_reward
    #      negative = torch.ones_like(current_entites, dtype=torch.float32) * self.negative_reward
    #      reward = torch.where(current_entites == answers, positive, negative)
    #      return reward

    def get_reward(self, current_entities, answers):
        """
        计算一个结合了二元成功和语义相似度的塑形奖励。
        这为模型提供了更密集的奖励信号，以增强泛化能力。
        Args:
            current_entities: torch.tensor, [batch_size], 智能体最终到达的实体。
            answers: torch.tensor, [batch_size], 真实的答案实体。
        Return:
            塑形后的奖励张量。
        """
        # 1. 二元成功奖励 (如果完全正确，则为1.0)
        binary_success_reward = (current_entities == answers).float()

        # 2. 语义相似度奖励 (即使没答对，但如果答案在语义上接近，也给予部分奖励)
        with torch.no_grad():  # 在计算奖励时不需要计算梯度
            # 获取当前实体和答案实体的嵌入。
            # 我们使用一个零张量作为时间戳，以获取实体的通用静态嵌入。
            current_embs = self.agent.ent_embs(current_entities, torch.zeros_like(current_entities, device=current_entities.device))
            answer_embs = self.agent.ent_embs(answers, torch.zeros_like(answers, device=answers.device))

            # 使用余弦相似度计算嵌入之间的相似度。
            # F.cosine_similarity 返回值在 [-1, 1]，我们将其缩放到 [0, 1] 区间。
            semantic_similarity = (F.cosine_similarity(current_embs, answer_embs, dim=-1) + 1.0) / 2.0

        # 3. 结合两种奖励
        # 使用一个超参数 alpha 来平衡两种奖励。
        # alpha 决定了“完全正确”的额外奖励有多大。
        alpha = self.config['reward_alpha']

        # 最终奖励 = (1 - alpha) * 相似度奖励 + alpha * 成功奖励
        # - 如果成功: 相似度为1, 最终奖励为 (1-alpha)*1 + alpha*1 = 1
        # - 如果失败: 最终奖励为 (1-alpha)*相似度 + alpha*0 = (1-alpha)*相似度
        # 这样既保证了成功的奖励最高，也为“接近成功”提供了平滑的部分奖励。
        # shaped_reward = (1 - alpha) * binary_success_reward + alpha * semantic_similarity
        shaped_reward = (1 - alpha) * semantic_similarity + alpha * binary_success_reward

        # 确保奖励不会因为浮点误差等原因超过最大值
        shaped_reward = torch.clamp(shaped_reward, max=self.positive_reward)

        return shaped_reward

    # calc_cum_discounted_reward
    # Args:
    #   rewards: 当前奖励的张量。
    # Return:
    #   累计折扣奖励的张量。
    def calc_cum_discounted_reward(self, rewards):
        running_add = torch.zeros([rewards.shape[0]])
        cum_disc_reward = torch.zeros([rewards.shape[0], self.config['path_length']])
        if self.config['cuda']:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.config['path_length'] - 1] = rewards
        for t in reversed(range(self.config['path_length'])):
            running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    # 计算熵正则化损失。
    # Args:
    #   all_logits: 所有logits的列表。
    # Return:
    #   熵损失的标量值。
    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)
        entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))
        return entropy_loss

    # 计算 REINFORCE 损失。
    # Args:
    #   all_loss: 所有损失的列表；
    #   all_logits: 所有logits的列表；
    #   cum_discounted_reward: 累计折扣奖励的张量。
    # Return:
    #   总损失的标量值。
    def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward):
        loss = torch.stack(all_loss, dim=1)
        base_value = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - base_value

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)
        entropy_loss = self.config['ita'] * math.pow(self.config['zita'], self.now_epoch) * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss
        return total_loss
