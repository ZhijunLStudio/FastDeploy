import json
import re

# --- 配置 ---
# 包含您提供的完整80层权重映射的JSON文件
original_index_path = "/home/aistudio/data/models/36280/MiniMax-M1-80k/model.safetensors.index.json"
# 将要生成的新5层权重映射文件的路径
new_index_path = "model.safetensors.index.json"
# 您想要保留的层数
num_layers_to_keep = 10

# --- 脚本 ---
print(f"正在从 '{original_index_path}' 加载原始权重索引...")
with open(original_index_path, 'r') as f:
    original_index = json.load(f)

new_weight_map = {}
original_weight_map = original_index['weight_map']

print(f"正在筛选权重，保留前 {num_layers_to_keep} 层以及头尾部分...")

for key, value in original_weight_map.items():
    # 使用正则表达式匹配层权重
    match = re.match(r"model\.layers\.(\d+)\.", key)
    
    if match:
        # 如果是层权重，检查层号
        layer_idx = int(match.group(1))
        if layer_idx < num_layers_to_keep:
            new_weight_map[key] = value
    else:
        # 如果不是层权重（例如 embed_tokens, norm, lm_head），则保留
        # 确保不会意外丢弃不以 "model.layers." 开头的其他权重
        if not key.startswith("model.layers."):
            new_weight_map[key] = value

new_index = {
    "metadata": original_index['metadata'],
    "weight_map": new_weight_map
}

print(f"正在将新的截断后索引保存到 '{new_index_path}'...")
with open(new_index_path, 'w') as f:
    json.dump(new_index, f, indent=4)

print("完成！新的索引文件已成功创建。")
print(f"原始权重数量: {len(original_weight_map)}")
print(f"新权重数量: {len(new_weight_map)}")