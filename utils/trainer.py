import torch
import json
import os
import tqdm

class Trainer(object):
    def __init__(self, model, pg, optimizer, args, distribution=None):
        self.model = model  # 模型实例
        self.pg = pg  # 策略梯度实例
        self.optimizer = optimizer  # 优化器
        self.args = args  # 参数配置
        self.distribution = distribution  # 可选的分布函数

    # 在一个 epoch 中训练模型
    def train_epoch(self, dataloader, ntriple):
        self.model.train()  # 设置模型为训练模式
        total_loss = 0.0  # 初始化总损失
        total_reward = 0.0  # 初始化总奖励
        counter = 0  # 计数器，用于计算平均值
        # 使用 tqdm 创建进度条
        with tqdm.tqdm(total=ntriple, unit='ex') as bar:
            bar.set_description('Train')  # 设置进度条描述
            for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                # 将数据移动到 GPU（如果可用）
                if self.args.cuda:
                    src_batch = src_batch.cuda()
                    rel_batch = rel_batch.cuda()
                    dst_batch = dst_batch.cuda()
                    time_batch = time_batch.cuda()

                # 执行前向传播
                all_loss, all_logits, _, current_entities, current_time = self.model(src_batch, time_batch, rel_batch)

                # 计算奖励
                reward = self.pg.get_reward(current_entities, dst_batch)
                # 计算奖励
                cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward)
                # 计算强化学习损失
                reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)
                # 更新基线
                self.pg.baseline.update(torch.mean(cum_discounted_reward))
                # 增加当前 epoch 计数
                self.pg.now_epoch += 1

                # 优化步骤
                self.optimizer.zero_grad()  # 清零梯度
                reinfore_loss.backward()  # 反向传播
                if self.args.clip_gradient:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)  # 梯度裁剪
                self.optimizer.step()  # 更新参数

                # 累加损失和奖励
                total_loss += reinfore_loss
                total_reward += torch.mean(reward)
                counter += 1  # 更新计数器
                bar.update(self.args.batch_size)  # 更新进度条
                bar.set_postfix(loss='%.4f' % reinfore_loss, reward='%.4f' % torch.mean(reward).item())  # 显示损失和奖励
        return total_loss / counter, total_reward / counter  # 返回平均损失和奖励

    # 保存模型和优化器的参数
    def save_model(self, checkpoint_path='checkpoint.pth'):
        """Save the parameters of the model and the optimizer,"""
        argparse_dict = vars(self.args)  # 将参数转换为字典
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)  # 保存配置为 JSON 文件

        # 保存模型和优化器的状态字典
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            os.path.join(self.args.save_path, checkpoint_path)
        )
