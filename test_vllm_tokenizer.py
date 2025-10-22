from transformers import AutoTokenizer

model_path = "/home/aistudio/config_folder"
prompt = "who are you？"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 打印 tokenizer 配置信息
print("--- vLLM/Transformers Tokenizer Config ---")
print(tokenizer)
print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
print("\n")


# 测试1：默认调用
print("--- Test 1: Default call ---")
output_default = tokenizer(prompt)
print(f"Prompt: {prompt!r}")
print(f"Input IDs: {output_default.input_ids}")
print(f"Decoded: {tokenizer.decode(output_default.input_ids)}")
print("\n")

# 测试2：不加特殊token
print("--- Test 2: add_special_tokens=False ---")
output_no_special = tokenizer(prompt, add_special_tokens=False)
print(f"Prompt: {prompt!r}")
print(f"Input IDs: {output_no_special.input_ids}")
print(f"Decoded: {tokenizer.decode(output_no_special.input_ids)}")
print("\n")

# 测试3: 检查你的日志输出
vllm_ids = [23246, 457, 390, 1219]
print("--- Check vLLM Log IDs ---")
print(f"IDs from log: {vllm_ids}")
print(f"Decoded: {tokenizer.decode(vllm_ids)}")