import networkx as nx
from collections import defaultdict
import numpy as np
import torch
import random
import logging


class Env(object):
    def __init__(self, examples, config, state_action_space=None, id2relation=None, id2entity=None, llm_enhancer=None):
        self.config = config
        self.num_rel = config['num_rel']
        self.graph, self.label2nodes = self.build_graph(examples)
        self.NO_OP = self.num_rel
        self.ePAD = config['num_ent']
        self.rPAD = config['num_rel'] + 1
        self.tPAD = 0
        self.state_action_space = state_action_space
        if state_action_space:
            self.state_action_space_key = self.state_action_space.keys()

        self.max_time = max([ex[3] for ex in examples]) if examples else 0

        self.id2relation = id2relation
        self.id2entity = id2entity
        self.llm_enhancer = llm_enhancer
        self.llm_top_k = self.config['llm_top_k']
        self.mode = 'train'

    def set_mode(self, mode):
        if mode in ['train', 'valid', 'test']:
            self.mode = mode
            logging.info(f"Environment mode set to: {self.mode}")
        else:
            raise ValueError("Mode must be one of 'train', 'valid', or 'test'")

    def build_graph(self, examples):
        graph = nx.MultiDiGraph()
        label2nodes = defaultdict(set)
        examples.sort(key=lambda x: x[3], reverse=True)
        for example in examples:
            src = example[0]
            rel = example[1]
            dst = example[2]
            time = example[3]

            src_node = (src, time)
            dst_node = (dst, time)
            if src_node not in label2nodes[src]:
                graph.add_node(src_node, label=src)
            if dst_node not in label2nodes[dst]:
                graph.add_node(dst_node, label=dst)

            graph.add_edge(src_node, dst_node, relation=rel)
            label2nodes[src].add(src_node)
            label2nodes[dst].add(dst_node)
        return graph, label2nodes

    def get_state_actions_space_complete(self, entity, time, current_=False, max_action_num=None):
        """Get the action space of the current state.
        Args:
            entity: The entity of the current state;
            time: Maximum timestamp for candidate actions;
            current_: Can the current time of the event be used;
            max_action_num: Maximum number of events stored;
        Return:
            numpy array，shape: [number of events，3], (relation, dst, time)
        """
        if self.state_action_space:
            if (entity, time, current_) in self.state_action_space_key:
                return np.array(list(self.state_action_space[(entity, time, current_)])[:max_action_num], dtype=np.dtype('int32'))
        nodes = self.label2nodes[entity].copy()
        if current_:
            # Delete future events, you can see current events, before query time
            nodes = list(filter((lambda x: x[1] <= time), nodes))
        else:
            # No future events, no current events
            nodes = list(filter((lambda x: x[1] < time), nodes))
        nodes.sort(key=lambda x: x[1], reverse=True)
        actions_space = []
        i = 0
        for node in nodes:
            for src, dst, rel in self.graph.out_edges(node, data=True):
                actions_space.append((rel['relation'], dst[0], dst[1]))
                i += 1
                if max_action_num and i >= max_action_num:
                    break
            if max_action_num and i >= max_action_num:
                break

        # 添加负采样
        if max_action_num and len(actions_space) < max_action_num:
            neg_samples = max_action_num - len(actions_space)
            # print(f"neg_samples: {neg_samples}")
            neg_entities = np.random.randint(0, self.config['num_ent'], neg_samples)
            neg_relations = np.random.randint(0, self.config['num_rel'], neg_samples)
            neg_times = np.random.randint(1, self.max_time, neg_samples)
            neg_actions = np.stack([neg_relations, neg_entities, neg_times], axis=1)
            actions_space.extend(neg_actions)

        return np.array(list(actions_space), dtype=np.dtype('int32'))

    def next_actions(self, entites, times, query_info, query_times, path_histories, max_action_nums, first_step=False):
        if self.config['cuda']:
            entites = entites.cpu()
            times = times.cpu()
            query_times = times.cpu()
            if torch.is_tensor(max_action_nums):
                max_action_nums = max_action_nums.cpu()

        entites = entites.numpy()
        times = times.numpy()
        query_times = query_times.numpy()
        query_heads, query_relations = query_info[0].cpu().numpy(), query_info[1].cpu().numpy()

        if torch.is_tensor(max_action_nums):
            max_action_nums = max_action_nums.numpy()

        actions = self.get_padd_actions(entites, times, (query_heads, query_relations), query_times, path_histories,
                                        max_action_nums, first_step)

        if self.config['cuda']:
            actions = torch.tensor(actions, dtype=torch.long, device='cuda')
        else:
            actions = torch.tensor(actions, dtype=torch.long)
        return actions

    # 这是优化的核心，实现了“收集-批处理调用-分发”的逻辑
    def get_padd_actions(self, entites, times, query_info, query_times, path_histories, max_action_nums, first_step=False):
        if isinstance(max_action_nums, (list, np.ndarray)):
            max_padd_size = int(np.max(max_action_nums))
        else:
            max_padd_size = int(max_action_nums)

        actions = np.ones((entites.shape[0], max_padd_size, 3), dtype=np.int32)
        actions[:, :, 0] *= self.rPAD
        actions[:, :, 1] *= self.ePAD
        actions[:, :, 2] *= self.tPAD

        query_heads, query_relations = query_info

        # ==================== 优化部分：阶段1 - 收集 ====================
        llm_batch_info = []
        llm_processing_indices = []  # 记录哪些样本索引需要LLM处理
        original_action_arrays = {}  # 存储所有样本的原始动作空间

        for i in range(entites.shape[0]):
            current_max_action_num = max_action_nums[i] if isinstance(max_action_nums, (list, np.ndarray)) else max_action_nums
            can_use_current_time = times[i] != query_times[i]

            action_array = self.get_state_actions_space_complete(
                entites[i], times[i], can_use_current_time, current_max_action_num
            )
            original_action_arrays[i] = action_array

            # 检查是否需要LLM增强
            if self.llm_enhancer and self.mode != 'train' and action_array.shape[0] > 1 and first_step == True:
                llm_processing_indices.append(i)

                # 准备LLM输入
                query_head_id, query_rel_id = query_heads[i], query_relations[i]
                head_text = self.id2entity.get(query_head_id, str(query_head_id))
                query_rel_text = self.id2relation.get(query_rel_id, str(query_rel_id))

                path_history_text = [(self.id2relation.get(r, str(r)), self.id2entity.get(e, str(e))) for r, e in
                                     path_histories[i]]
                candidate_texts = [(self.id2relation.get(r, str(r)), self.id2entity.get(e, str(e))) for r, e, _ in
                                   action_array]

                llm_batch_info.append({
                    'query': (head_text, query_rel_text),
                    'candidates': candidate_texts,
                    'path_history': path_history_text
                })

        # ==================== 阶段2 - 分块批处理调用 ====================
        processed_action_arrays = {}
        if llm_batch_info and first_step == True:
            # 定义一个可控的、用于LLM推理的批处理大小，以避免OOM
            llm_process_batch_size = self.config.get('llm_batch_size')  # 您可以根据显存大小调整此值
            all_scores = []

            # logging.info(f"========需要LLM增强的样本总数: {len(llm_batch_info)}，将以大小为 {llm_process_batch_size} 的批次进行处理========")

            # 将收集到的所有样本分块处理
            for i in range(0, len(llm_batch_info), llm_process_batch_size):
                # 获取当前的小批次数据
                mini_batch_info = llm_batch_info[i:i + llm_process_batch_size]

                # logging.info(f"--------> 正在处理LLM批次 {i // llm_process_batch_size + 1}，大小: {len(mini_batch_info)}...")

                # 使用小批次调用LLM增强器
                mini_batch_scores = self.llm_enhancer.score_candidates_batch(mini_batch_info)

                # 将小批次得到的分数追加到总分列表
                all_scores.extend(mini_batch_scores)

            # ==================== 优化部分：阶段3 - 分发 ====================
            # 此处的逻辑保持不变，它现在使用从所有小批次中收集到的完整all_scores列表
            for idx, original_index in enumerate(llm_processing_indices):
                action_space = original_action_arrays[original_index]
                llm_scores = all_scores[idx]

                # 如果只有一个或没有动作，则无需重新排序
                if len(llm_scores) <= 1:
                    processed_action_arrays[original_index] = action_space
                    continue

                # --- 代码修改：引入时间间隔约束 ---
                # 1. 计算基于时间差的转移概率
                lambda_val = self.config['llm_lambda']
                alpha_val = self.config['llm_time_alpha']
                current_time = float(times[original_index])
                edge_times = action_space[:, 2].astype(np.float32)

                # 计算时间差 (T-t)，由于t<=T，结果非负。这符合“时间越近权重越大”的描述。
                time_diffs = current_time - edge_times
                # 根据公式 w(t) = exp(-λ * (T-t)) 计算权重
                time_weights = np.exp(-lambda_val * time_diffs)
                
                # 归一化为概率分布
                sum_weights = np.sum(time_weights)
                if sum_weights > 0:
                    time_probs = time_weights / sum_weights
                else:
                    # 避免除以零的边缘情况
                    time_probs = np.zeros_like(time_weights)

                # 2. 归一化LLM分数
                llm_scores_np = np.array(llm_scores)
                # 使用softmax将LLM分数转换为概率分布，以保证尺度一致
                # 减去最大值是为了数值稳定性，防止溢出
                exp_scores = np.exp(llm_scores_np - np.max(llm_scores_np))
                sum_exp_scores = np.sum(exp_scores)
                if sum_exp_scores > 0:
                    llm_probs = exp_scores / sum_exp_scores
                else:
                    llm_probs = np.zeros_like(llm_scores_np)

                # 3. 加权组合两种分数
                combined_scores = alpha_val * llm_probs + (1 - alpha_val) * time_probs
                # print(f"llm_probs: {llm_probs}")
                # print(f"time_probs: {time_probs}")
                # --- 代码修改结束 ---

                # 使用组合后的分数对动作进行排序
                scored_actions = sorted(zip(action_space, combined_scores), key=lambda x: x[1], reverse=True)

                current_max_action_num = max_action_nums[original_index] if isinstance(max_action_nums, (
                    list, np.ndarray)) else max_action_nums
                top_k = self.llm_top_k if current_max_action_num is None else min(self.llm_top_k, current_max_action_num)

                # 存储经过LLM处理和时间约束排序后的动作数组
                processed_action_arrays[original_index] = np.array([action for action, score in scored_actions[:top_k]], dtype=np.int32)

        # ==================== 最终组装 ====================
        for i in range(entites.shape[0]):
            actions[i, 0, 0] = self.NO_OP
            actions[i, 0, 1] = entites[i]
            actions[i, 0, 2] = times[i]

            # 如果样本经过了LLM处理，则使用处理后的结果，否则使用原始结果
            if i in processed_action_arrays:
                action_array = processed_action_arrays[i]
            else:
                action_array = original_action_arrays[i]

            if action_array.shape[0] == 0:
                continue

            start_idx = 1 if not first_step else 0
            num_actions_to_copy = min(action_array.shape[0], max_padd_size - start_idx)

            if num_actions_to_copy > 0:
                actions[i, start_idx:num_actions_to_copy + start_idx, ] = action_array[:num_actions_to_copy]

        return actions
