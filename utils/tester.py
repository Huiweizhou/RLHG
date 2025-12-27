import torch
import tqdm
import numpy as np
import random


# 实现了一个用于测试模型性能的类，计算与时间相关的指标（如 MRR 和 Hits@K）。
class Tester(object):
    # MODIFIED: __init__ now accepts sets of seen triplets
    def __init__(self, model, args, train_entities, train_triplets, train_valid_triplets):
        self.model = model  # 模型实例
        self.args = args  # 参数配置
        self.train_entities = train_entities  # 训练实体
        self.train_triplets = train_triplets  # Set of (h,r,t) from training data
        self.train_valid_triplets = train_valid_triplets  # Set of (h,r,t) from train+valid data

    # 获取答案的位置，如果答案不在数组中，排名将是实体总数。
    # Args:
    #   score: list, 实体得分
    #   answer: int, 正确的实体
    #   entities_space: 与得分对应的实体
    # num_ent: 实体总数
    #   Return: 正确答案的排名。
    def get_rank(self, score, answer, entities_space, num_ent):
        """Get the location of the answer, if the answer is not in the array,
        the ranking will be the total number of entities.
        Args:
            score: list, entity score
            answer: int, the ground truth entity
            entities_space: corresponding entity with the score
            num_ent: the total number of entities
        Return: the rank of the ground truth.
        """
        if answer not in entities_space:
            rank = num_ent  # 如果答案不在集合中，返回总数
        else:
            answer_prob = score[entities_space.index(answer)]  # 获取正确答案的得分
            score.sort(reverse=True)  # 降序排序得分
            rank = score.index(answer_prob) + 1  # 获取排名
        return rank

    # 获取时间感知的过滤指标(MRR, Hits@1/3/10)。
    # Args:
    #   ntriple: 测试样本数量。
    #   skip_dict: 时间感知过滤，从基数据集获取
    #   num_ent: 实体数量。
    #   mode: 'valid' or 'test', to select the correct set of seen triplets
    # Return:
    #   字典(key -> MRR / HITS @ 1 / HITS @ 3 / HITS @ 10, values -> float)
    # MODIFIED: Added 'mode' parameter to handle validation vs. testing logic
    def test(self, dataloader, ntriple, skip_dict, num_ent, mode='valid'):
        """Get time-aware filtered metrics(MRR, Hits@1/3/10).
        Args:
            ntriple: number of the test examples.
            skip_dict: time-aware filter. Get from baseDataset
            num_ent: number of the entities.
            mode: 'valid' or 'test'
        Return: a dict (key -> MRR/HITS@1/HITS@3/HITS@10, values -> float)
        """
        self.model.eval()
        logs = []

        # MODIFIED: Select the appropriate set of seen triplets based on the mode
        if mode == 'valid':
            seen_triplets = self.train_triplets
            print("Tester running in VALIDATION mode.")
        else:  # 'test' mode
            seen_triplets = self.train_valid_triplets
            print("Tester running in TEST mode.")

        with torch.no_grad():
            with tqdm.tqdm(total=ntriple, unit='ex') as bar:
                for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                    batch_size = dst_batch.size(0)

                    if self.args.cuda:
                        src_batch = src_batch.cuda()
                        rel_batch = rel_batch.cuda()
                        dst_batch = dst_batch.cuda()
                        time_batch = time_batch.cuda()

                    # MODIFIED: Dynamically create a batch of max_action_num values
                    max_action_nums_batch = []
                    for i in range(batch_size):
                        src = src_batch[i].item()
                        rel = rel_batch[i].item()
                        dst = dst_batch[i].item()
                        # Check if the (h,r,t) triplet (ignoring timestamp) has been seen
                        if (src, rel, dst) in seen_triplets:
                            max_action_nums_batch.append(self.args.max_action_num_seen)
                        else:
                            max_action_nums_batch.append(self.args.max_action_num_new)

                    max_action_nums_batch = torch.tensor(max_action_nums_batch, device=src_batch.device)

                    # MODIFIED: Pass the dynamic max_action_nums to beam_search
                    current_entities, beam_prob = \
                        self.model.beam_search(src_batch, time_batch, rel_batch, max_action_nums_batch)

                    if self.args.cuda:
                        current_entities = current_entities.cpu()
                        beam_prob = beam_prob.cpu()

                    current_entities = current_entities.numpy()
                    beam_prob = beam_prob.numpy()

                    MRR = 0
                    for i in range(batch_size):
                        candidate_answers = current_entities[i]
                        candidate_score = beam_prob[i]

                        # sort by score from largest to smallest
                        idx = np.argsort(-candidate_score)
                        candidate_answers = candidate_answers[idx]
                        candidate_score = candidate_score[idx]

                        # remove duplicate entities
                        candidate_answers, idx = np.unique(candidate_answers, return_index=True)
                        candidate_answers = list(candidate_answers)
                        candidate_score = list(candidate_score[idx])

                        src = src_batch[i].item()
                        rel = rel_batch[i].item()
                        dst = dst_batch[i].item()
                        time = time_batch[i].item()

                        filter = skip_dict[(src, rel, time)]  # a set of ground truth entities
                        tmp_entities = candidate_answers.copy()
                        tmp_prob = candidate_score.copy()
                        # time-aware filter
                        for j in range(len(tmp_entities)):
                            if tmp_entities[j] in filter and tmp_entities[j] != dst:
                                candidate_answers.remove(tmp_entities[j])
                                candidate_score.remove(tmp_prob[j])

                        ranking_raw = self.get_rank(candidate_score, dst, candidate_answers, num_ent)

                        # with open('ranking_raw.txt', 'a') as results_file:
                        #     results_file.write(f"{i}\t{src}\t{rel}\t{dst}\t{time}\t{ranking_raw}\n")

                        logs.append({
                            'MRR': 1.0 / ranking_raw,
                            'HITS@1': 1.0 if ranking_raw <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking_raw <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking_raw <= 10 else 0.0,
                        })
                        MRR = MRR + 1.0 / ranking_raw

                    bar.update(batch_size)
                    bar.set_postfix(MRR='{}'.format(MRR / batch_size))
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        return metrics
