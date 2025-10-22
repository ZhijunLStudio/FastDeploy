from paddleformers.transformers import AutoTokenizer

model_path = "/home/aistudio/config_folder"
prompt = "who are you？"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 打印 tokenizer 配置信息
print("--- FD/PaddleFormers Tokenizer Config ---")
print(tokenizer)
print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
print("\n")


# 测试1：默认调用
print("--- Test 1: Default call ---")
output_default = tokenizer(prompt)
print(f"Prompt: {prompt!r}")
print(f"Input IDs: {output_default['input_ids']}")
print(f"Decoded: {tokenizer.decode(output_default['input_ids'])}")
print("\n")

# 测试2：不加特殊token
print("--- Test 2: add_special_tokens=False ---")
output_no_special = tokenizer(prompt, add_special_tokens=False)
print(f"Prompt: {prompt!r}")
print(f"Input IDs: {output_no_special['input_ids']}")
print(f"Decoded: {tokenizer.decode(output_no_special['input_ids'])}")
print("\n")

# 测试3: 检查你的日志输出
fd_ids = [1171, 94106, 94106]
print("--- Check FD Log IDs ---")
print(f"IDs from log: {fd_ids}")
print(f"Decoded: {tokenizer.decode(fd_ids)}")