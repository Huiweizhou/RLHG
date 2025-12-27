from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import warnings

# 忽略所有类型的警告
warnings.filterwarnings("ignore")  # 移除了 category 参数

def load_llama_model():
    # 参数设置
    model_name = r"/media/lsw/TKG-RL-Demo/BioMistral-7B"  # 替换为实际模型路径/名称
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    return model, tokenizer


def generate_response(model, tokenizer, prompt):
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成参数设置（可根据需要调整）
    generate_kwargs = {
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    # 生成回答
    outputs = model.generate(**inputs, **generate_kwargs)

    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]  # 返回生成的回答部分


if __name__ == "__main__":
    # 初始化模型
    print("Loading BioMistral-7B...")
    model, tokenizer = load_llama_model()

    user_input = """
    Original Query: (Subject: "ganglioside", Relation: "inferred", Object: ?) 
    Path Taken: [Start] 
    Current Position: "ganglioside" 
    
    Evaluate the following candidate next steps: 
    1. (Relation: "inferred", Object: "neurotoxic") 
    2. (Relation: "inferred", Object: "gasogenic infections") 
    3. (Relation: "inferred", Object: "neuroblastoma") 
    4. (Relation: "inferred", Object: "gliomas") 
    5. (Relation: "inferred", Object: "myeloma") 
    6. (Relation: "inferred", Object: "cell carcinoma") 
    7. (Relation: "inferred", Object: "toxic") 
    8. (Relation: "inferred", Object: "pulmonary tubercolosis") 
    9. (Relation: "inferred", Object: "Chronic Hyperphenylalanemic") 
    10. (Relation: "inferred", Object: "anaplastic carcinoma") 
    11. (Relation: "inferred", Object: "AUTOIMMUNIZATION") 
    12. (Relation: "inferred", Object: "neurotoxic") 
    13. (Relation: "inferred", Object: "gasogenic infections") 
    14. (Relation: "inferred", Object: "neuroblastoma") 
    15. (Relation: "inferred", Object: "gliomas") 
    16. (Relation: "inferred", Object: "myeloma") 
    17. (Relation: "inferred", Object: "cell carcinoma") 
    18. (Relation: "inferred", Object: "toxic") 
    19. (Relation: "inferred", Object: "pulmonary tubercolosis") 
    20. (Relation: "inferred", Object: "Chronic Hyperphenylalanemic") 
    """

    # 构建提示模板
    prompt = f"""<|system|>
            "You are an expert in knowledge graph reasoning. Your task is to evaluate a list of candidate next steps in a reasoning path, "
            "given a starting query and the path taken so far. For each candidate, provide a logical plausibility score.\n\n"
            "Respond ONLY with a comma-separated list of floating-point numbers (e.g., \"0.9, 0.2, 0.75\"), "
            "with each number corresponding to the candidate in the same order. The list should contain exactly as many scores as there are candidates."
            <|user|>
            {user_input}
            <|assistant|>
            """

    # 生成回答
    response = generate_response(model, tokenizer, prompt)
    print("\nAssistant:", response.strip(), "\n")
