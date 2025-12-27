import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging
from typing import List, Tuple

import warnings

# 忽略所有类型的警告
warnings.filterwarnings("ignore")


class LLMEnhancer:
    def __init__(self, model_path, use_cuda=True):
        """
        初始化LLM增强器。
        Args:
            model_path (str): 本地LLM模型的路径。
            use_cuda (bool): 是否使用GPU。
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        logging.info(f"LLM Enhancer is using device: {self.device}")

        # --- 开始优化/添加的代码 (1/2) ---

        # # 存储Prompt日志文件路径
        # self.prompt_log_file = "./prompts.txt"

        # 如果指定了日志文件路径，则在程序开始时清空该文件，确保一个干净的日志环境
        if self.prompt_log_file:
            try:
                with open(self.prompt_log_file, 'w', encoding='utf-8') as f:
                    # 'w'模式打开文件会清空其内容
                    pass
                logging.info(f"Prompts will be logged to '{self.prompt_log_file}'. The file has been cleared for this session.")
            except IOError as e:
                logging.error(f"Failed to open or clear prompt log file '{self.prompt_log_file}': {e}. Prompt logging will be disabled.")
                self.prompt_log_file = None # 如果文件无法写入，则禁用日志功能

        # --- 结束优化/添加的代码 (1/2) ---

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left"
        )
        # --- 开始修改/添加的代码 ---

        # 定义并设置聊天模板，以解决 apply_chat_template 的报错问题
        # 注意：请确认这个模板格式是否与您的 'meditron-7b' 模型完全匹配。
        # 不同的模型（如 Llama, Mistral, Zephyr）使用不同的特殊标记和结构。
        chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '</s>\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '</s>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '</s>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"""
        self.tokenizer.chat_template = chat_template

        # --- 结束修改/添加的代码 ---
        # 为批处理添加填充标记
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            load_in_4bit=True  # 4bit量化减少显存
        )
        self.model.eval()
        logging.info(f"LLM Enhancer loaded model from {self.model_path}")

        self.system_prompt = (
            "You are an expert in knowledge graph path evaluation. "
            "Your task is to assess the semantic relevance of each candidate path based on the given initial query and historical reasoning paths. "
            "For each candidate path, provide a score between 0.0 (completely irrelevant) and 1.0 (highly relevant). "
            "The final output should be a comma-separated list of floating-point numbers (e.g., '0.9, 0.2, 0.75'), "
            "where each score strictly corresponds to its respective candidate path."
        )

    def _create_single_prompt(self, query: Tuple[str, str], candidates: List[Tuple[str, str]],
                              path_history: List[Tuple[str, str]]) -> str:
        """
        根据查询、候选列表和路径历史生成单个LLM输入提示词。
        """
        head_entity, query_relation = query

        if not path_history:
            path_taken_str = "[Start]"
            current_position_str = head_entity
        else:
            path_parts = [f'(Subject: "{head_entity}")']
            for rel, ent in path_history:
                path_parts.append(f'--{rel}--> (Object: "{ent}")')
            path_taken_str = " ".join(path_parts)
            current_position_str = path_history[-1][1]

        candidate_lines = ["Evaluate the following candidate next steps:"]
        for i, (cand_relation, cand_entity) in enumerate(candidates, 1):
            candidate_lines.append(f'{i}. (Relation: "{cand_relation}", Object: "{cand_entity}")')
        candidates_str = "\n".join(candidate_lines)

        user_prompt = (
            f'Original Query: (Subject: "{head_entity}", Relation: "{query_relation}", Object: ?)\n'
            f'Path Taken: {path_taken_str}\n'
            f'Current Position: "{current_position_str}"\n\n'
            f'{candidates_str}'
        )

        full_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        return full_prompt

    def _parse_score_list(self, text: str, expected_count: int) -> List[float]:
        """
        从LLM的输出文本中解析出逗号分隔的分数列表。
        """
        score_strings = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        scores = [float(s) for s in score_strings]

        if len(scores) != expected_count:
            scores = (scores + [0.0] * expected_count)[:expected_count]
        # print(f"scores: {scores}")
        return scores

    @torch.no_grad()
    def score_candidates_batch(self, batch_info: List[dict]) -> List[List[float]]:
        """
        对一批候选动作进行评分，只调用一次LLM。
        Args:
            batch_info (List[dict]): 一个列表，每个元素是一个字典，包含 'query', 'candidates', 'path_history'。
        Returns:
            List[List[float]]: 一个分数列表的列表，每个子列表对应输入批次中的一个样本。
        """
        if not batch_info:
            return []

        # 1. 为批次中的每个样本创建Prompt
        prompts = [self._create_single_prompt(info['query'], info['candidates'], info['path_history']) for info in batch_info]

        # --- 开始优化/添加的代码 (2/2) ---
        # 如果在初始化时设置了日志文件路径，则将本批次的prompt写入文件
        if self.prompt_log_file:
            try:
                # 使用 'a' (append) 模式来追加内容，而不是覆盖
                with open(self.prompt_log_file, 'a', encoding='utf-8') as f:
                    for prompt in prompts:
                        # 将prompt内的换行符替换为'\\n'，以确保每个prompt在文件中只占一行
                        sanitized_prompt = prompt.replace('\n', '\\n')
                        f.write(sanitized_prompt + '\n')
            except IOError as e:
                logging.error(f"Could not write prompts to log file '{self.prompt_log_file}': {e}")
        # --- 结束优化/添加的代码 (2/2) ---

        # 2. 批量调用LLM
        # 使用 padding=True 来处理不同长度的prompt
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # 估算一个合理的max_new_tokens，以批次中最长的候选列表为基准
        max_candidates_len = max(len(info['candidates']) for info in batch_info)
        max_new_tokens = max_candidates_len * 6 + 30  # 为每个分数留出更宽松的空间

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 3. 批量解析输出
        # 只解码生成的部分，跳过输入的prompt
        decoded_outputs = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        all_scores = []
        for i, decoded_text in enumerate(decoded_outputs):
            expected_count = len(batch_info[i]['candidates'])
            scores = self._parse_score_list(decoded_text, expected_count)
            all_scores.append(scores)

        return all_scores

    # 保留旧的单次调用接口以备后用，但内部可以重构为使用批处理方法
    def score_candidates(self, query: Tuple[str, str], candidates: List[Tuple[str, str]],
                         path_history: List[Tuple[str, str]]) -> List[float]:
        """
        对单个样本的候选动作进行评分的便捷接口。
        内部调用批处理方法以保持代码统一。
        """
        if not candidates:
            return []

        batch_info = [{'query': query, 'candidates': candidates, 'path_history': path_history}]
        # 调用批处理方法，并返回第一个（也是唯一一个）结果
        return self.score_candidates_batch(batch_info)[0]
