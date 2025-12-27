"""
ReactiveBaseline 类用于维护一个基线值，该值根据目标值进行更新。
它可以用于强化学习等场景中，以便动态调整基线，帮助代理评估当前状态的价值。
"""

import torch

class ReactiveBaseline(object):
    def __init__(self, config, update_rate):
        self.update_rate = update_rate  # 保存更新速率
        self.value = torch.zeros(1)  # 初始化基线值为零张量
        if config['cuda']:
            self.value = self.value.cuda()

    # 获取当前的基线值
    def get_baseline_value(self):
        return self.value

    # 更新基线值
    def update(self, target):
        """
        接收一个目标值，并根据指定的更新速率更新基线值。更新公式采用加权平均的方式，使得基线值逐渐接近目标值。
        具体公式为：new_value = (1 - update_rate) * old_value + update_rate * target。
        """
        self.value = torch.add((1 - self.update_rate) * self.value, self.update_rate * target)
