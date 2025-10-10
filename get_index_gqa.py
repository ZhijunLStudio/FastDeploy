import json
import re

original_index_path = "/home/aistudio/data/models/36280/MiniMax-M1-80k/model.safetensors.index.json"
new_index_path = "/home/aistudio/config_folder/model.safetensors.index.json" # 输出文件名

# 我们想要保留的 GQA 层的原始索引
layers_to_keep = [7]

print(f"Loading original index from '{original_index_path}'...")
with open(original_index_path, 'r') as f:
    original_index = json.load(f)

new_weight_map = {}
original_weight_map = original_index['weight_map']

print(f"Filtering weights to keep layers {layers_to_keep} and non-layer weights...")

for key, value in original_weight_map.items():
    match = re.match(r"model\.layers\.(\d+)\.", key)
    
    if match:
        layer_idx = int(match.group(1))
        if layer_idx in layers_to_keep:
            # 这是我们想要的层，保留它
            new_weight_map[key] = value
    else:
        # 非层权重 (embed_tokens, norm, lm_head)，总是保留
        new_weight_map[key] = value

new_index = {
    "metadata": original_index['metadata'], # total_size 会不准，但不影响加载
    "weight_map": new_weight_map
}

print(f"Saving new filtered index to '{new_index_path}'...")
with open(new_index_path, 'w') as f:
    json.dump(new_index, f, indent=4)

print("Done!")