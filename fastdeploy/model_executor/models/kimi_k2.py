# /home/aistudio/work/FastDeploy/fastdeploy/model_executor/models/kimi_k2.py
from __future__ import annotations
import re
import paddle
import numpy as np

from .deepseek_v3 import DeepseekV3ForCausalLM, DeepSeekV3PretrainedModel
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.models.model_base import ModelCategory, ModelRegistry
from fastdeploy.model_executor.utils import get_tensor


from paddleformers.utils.log import logger
import pprint

def print_tensor_stats(tensor, name):
    """打印Paddle张量的统计信息 (强制 float32)"""
    if tensor is None:
        logger.info(f"DEBUG_FD: {name} is None")
        return
    with paddle.no_grad():
        stats = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        num_elements = tensor.numel()
        if num_elements.item() > 0:
            tensor_float = tensor.astype('float32')
            tensor_cpu = tensor_float.cpu()
            stats["max"] = f"{tensor_cpu.max().item():.6f}"
            stats["min"] = f"{tensor_cpu.min().item():.6f}"
            stats["mean"] = f"{tensor_cpu.mean().item():.6f}"
            
            if num_elements.item() > 1:
                stats["std"] = f"{tensor_cpu.std().item():.6f}"
            else:
                stats["std"] = "0.000000"

            flat_data = tensor_cpu.flatten().numpy()[:5]
            stats["first_5_values"] = flat_data
        logger.info(f"\n--- [FD DEBUG] {name} ---\n{pprint.pformat(stats, indent=2)}\n--------------------------\n")


@ModelRegistry.register_model_class(
    architecture="KimiK2ForCausalLM",
    module_name="kimi_k2",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION
)
class KimiK2ForCausalLM(DeepseekV3ForCausalLM):
    """
    KimiK2 model, architecturally identical to DeepseekV3.
    This class implements a self-contained, robust weight loading method
    based on the exact checkpoint structure.
    """

    @classmethod
    def name(cls):
        return "KimiK2ForCausalLM"

    def _load_moe_experts(self, model_param_name, weights_map):
        """
        Helper function to find, merge, and stack all expert weights from the checkpoint.
        """
        print(f"--- [KIMI DEBUG] Entering _load_moe_experts for '{model_param_name}'")
        match = re.search(r"layers\.(\d+)\.mlp\.experts\.(.+)", model_param_name)
        if not match:
            print(f"--- [KIMI DEBUG]   - Regex failed to match. Exiting.")
            return None, set()

        layer_idx = match.group(1)
        param_full_suffix = match.group(2)
        
        num_experts = self.fd_config.model_config.n_routed_experts
        expert_tensors = []
        processed_names = set()

        print(f"  > Strategy: MoE Expert Stacking for '{param_full_suffix}' in layer {layer_idx}")

        is_up_gate = param_full_suffix.startswith("up_gate_proj")
        is_down = param_full_suffix.startswith("down_proj")
        
        suffix = "weight" if "weight_scale_inv" not in param_full_suffix else "weight_scale_inv"
        print(f"--- [KIMI DEBUG]   - Determined suffix: '{suffix}'")
        
        for i in range(num_experts):
            if is_up_gate:
                gate_name = f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.{suffix}"
                up_name = f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.{suffix}"
                
                if gate_name not in weights_map or up_name not in weights_map:
                    print(f"--- [KIMI DEBUG]   - ERROR: Missing weights for expert {i}. Looking for '{gate_name}' and '{up_name}'.")
                    return None, set()
                
                gate_tensor = weights_map[gate_name]
                up_tensor = weights_map[up_name]
                expert_tensor = paddle.concat([gate_tensor, up_tensor], axis=0)
                processed_names.add(gate_name)
                processed_names.add(up_name)
            elif is_down:
                weight_name = f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.{suffix}"
                if weight_name not in weights_map:
                    print(f"--- [KIMI DEBUG]   - ERROR: Missing weight for expert {i}. Looking for '{weight_name}'.")
                    return None, set()
                expert_tensor = weights_map[weight_name]
                processed_names.add(weight_name)
            else:
                return None, set()

            expert_tensors.append(expert_tensor)
        
        if not expert_tensors:
            print(f"--- [KIMI DEBUG]   - No expert tensors collected. Exiting.")
            return None, set()

        stacked_tensor = paddle.stack(expert_tensors, axis=0)
        print(f"--- [KIMI DEBUG]   - Successfully stacked {len(expert_tensors)} expert tensors. Final shape: {stacked_tensor.shape}")
        return stacked_tensor, processed_names


    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        """
        Final, definitive, and fully-logged weight loading logic for KimiK2.
        This version reverse-maps model parameters to checkpoint weights with
        explicit transformation rules based on confirmed naming patterns.
        """
        
        print("\n--- [KIMI LOADER V6 DEBUG] STARTING WEIGHT LOAD (Manual Aggregation with Super Logging) ---")

        params_dict = dict(self.named_parameters())
        tp_size = self.fd_config.parallel_config.tensor_parallel_size
        tp_rank = self.fd_config.parallel_config.tensor_parallel_rank

        print("[KIMI V6 DEBUG] Buffering all weights from iterator into memory...")
        weights_map = {name: get_tensor(weight) for name, weight in weights_iterator}
        print(f"[KIMI V6 DEBUG] Buffering complete. Total unique weights in map: {len(weights_map)}")

        processed_ckpt_names = set()

        for model_param_name, param in params_dict.items():
            print(f"\n[KIMI V6 DEBUG] >>> Attempting to load parameter: '{model_param_name}' with shape {param.shape}")
            
            loaded_tensor = None
            
            # --- Determine loading strategy based on parameter name ---
            # Strategy 0: MoE Expert Weights
            if "mlp.experts." in model_param_name:
                loaded_tensor, names = self._load_moe_experts(model_param_name, weights_map)
                processed_ckpt_names.update(names)

            # Strategy 1: Merged ColumnParallel Weights
            elif model_param_name.endswith(".up_gate_proj.weight"):
                gate_name = model_param_name.replace(".up_gate_proj.", ".gate_proj.")
                up_name = model_param_name.replace(".up_gate_proj.", ".up_proj.")
                if gate_name in weights_map and up_name in weights_map:
                    print(f"  > Strategy: Merged ColumnParallel. Shards: '{gate_name}', '{up_name}'")
                    gate_w = weights_map[gate_name]
                    up_w = weights_map[up_name]
                    loaded_tensor = paddle.concat([gate_w, up_w], axis=0)
                    processed_ckpt_names.update([gate_name, up_name])

            # Strategy 1.1: Merged ColumnParallel Weight Scales
            elif model_param_name.endswith(".up_gate_proj.weight_scale_inv"):
                gate_scale_name = model_param_name.replace(".up_gate_proj.", ".gate_proj.")
                up_scale_name = model_param_name.replace(".up_gate_proj.", ".up_proj.")
                if gate_scale_name in weights_map and up_scale_name in weights_map:
                    print(f"  > Strategy: Merged ColumnParallel Scales. Shards: '{gate_scale_name}', '{up_scale_name}'")
                    gate_scale_w = weights_map[gate_scale_name]
                    up_scale_w = weights_map[up_scale_name]
                    loaded_tensor = paddle.concat([gate_scale_w, up_scale_w], axis=0)
                    processed_ckpt_names.update([gate_scale_name, up_scale_name])

            # Strategy 2: Merged Replicated Weights
            elif model_param_name.endswith(".qkv_a_proj_with_mqa.weight"):
                q_a_name = model_param_name.replace(".qkv_a_proj_with_mqa.", ".q_a_proj.")
                kv_a_name = model_param_name.replace(".qkv_a_proj_with_mqa.", ".kv_a_proj_with_mqa.")
                if q_a_name in weights_map and kv_a_name in weights_map:
                    print(f"  > Strategy: Merged Replicated. Shards: '{q_a_name}', '{kv_a_name}'")
                    q_a_w = weights_map[q_a_name]
                    kv_a_w = weights_map[kv_a_name]
                    loaded_tensor = paddle.concat([q_a_w, kv_a_w], axis=0)
                    processed_ckpt_names.update([q_a_name, kv_a_name])

            # Strategy 2.1: Merged Replicated Weight Scales
            elif model_param_name.endswith(".qkv_a_proj_with_mqa.weight_scale_inv"):
                q_a_scale_name = model_param_name.replace(".qkv_a_proj_with_mqa.", ".q_a_proj.")
                kv_a_scale_name = model_param_name.replace(".qkv_a_proj_with_mqa.", ".kv_a_proj_with_mqa.")
                if q_a_scale_name in weights_map and kv_a_scale_name in weights_map:
                    print(f"  > Strategy: Merged Replicated Scales. Shards: '{q_a_scale_name}', '{kv_a_scale_name}'")
                    q_a_scale_w = weights_map[q_a_scale_name]
                    kv_a_scale_w = weights_map[kv_a_scale_name]
                    loaded_tensor = paddle.concat([q_a_scale_w, kv_a_scale_w], axis=0)
                    processed_ckpt_names.update([q_a_scale_name, kv_a_scale_name])

            # Strategy 3: Special Names
            elif model_param_name == "model.embed_tokens.embeddings.weight":
                ckpt_name = "model.embed_tokens.weight"
                if ckpt_name in weights_map:
                    print(f"  > Strategy: Special Name Mapping. Source: '{ckpt_name}'")
                    loaded_tensor = weights_map[ckpt_name]
                    processed_ckpt_names.add(ckpt_name)
            
            elif model_param_name == "lm_head.linear.weight":
                ckpt_name = "lm_head.weight"
                if ckpt_name in weights_map:
                    print(f"  > Strategy: Special Name Mapping. Source: '{ckpt_name}'")
                    loaded_tensor = weights_map[ckpt_name]
                    processed_ckpt_names.add(ckpt_name)

            # Strategy 4: Direct Match
            elif model_param_name in weights_map:
                print(f"  > Strategy: Direct Match. Source: '{model_param_name}'")
                loaded_tensor = weights_map[model_param_name]
                processed_ckpt_names.add(model_param_name)

            if loaded_tensor is None:
                print(f"  > WARNING: No corresponding weight found for param '{model_param_name}'. Skipping.")
                continue

            print(f"    - Loaded tensor shape (raw): {loaded_tensor.shape}")

            # --- Apply Transformations ---
            is_column_parallel = any(s in model_param_name for s in [".q_b_proj.", ".kv_b_proj.", ".up_gate_proj.", "lm_head."])
            is_row_parallel = any(s in model_param_name for s in [".o_proj.", ".down_proj."])
            
            if tp_size > 1:
                # ... (TP sharding logic from your V2 code, which is correct)
                if "mlp.experts" in model_param_name and loaded_tensor.ndim >= 3:
                    if "up_gate_proj" in model_param_name:
                        dim_to_shard = 1
                    elif "down_proj" in model_param_name:
                         dim_to_shard = 2
                    else:
                        dim_to_shard = -1 
                    if dim_to_shard != -1:
                        print(f"    - Applying TP shard on MoE dim={dim_to_shard}")
                        total_size = loaded_tensor.shape[dim_to_shard]
                        block_size = total_size // tp_size
                        start, end = tp_rank * block_size, (tp_rank + 1) * block_size
                        if dim_to_shard == 1: loaded_tensor = loaded_tensor[:, start:end, ...]
                        else: loaded_tensor = loaded_tensor[:, :, start:end, ...]
                        print(f"    - Sharded tensor shape: {loaded_tensor.shape}")
                
                elif "weight_scale_inv" in model_param_name:
                    dim_to_shard = -1
                    if is_column_parallel: dim_to_shard = 0
                    elif is_row_parallel: dim_to_shard = 1
                    if dim_to_shard != -1:
                        print(f"    - Applying TP shard on scale dim={dim_to_shard}")
                        total_size = loaded_tensor.shape[dim_to_shard]
                        block_size = total_size // tp_size
                        start, end = tp_rank * block_size, (tp_rank + 1) * block_size
                        if dim_to_shard == 0: loaded_tensor = loaded_tensor[start:end, :]
                        else: loaded_tensor = loaded_tensor[:, start:end]
                        print(f"    - Sharded tensor shape: {loaded_tensor.shape}")
                
                elif is_column_parallel or "embed_tokens" in model_param_name:
                    dim_to_shard = 0
                    print(f"    - Applying TP shard on dim={dim_to_shard} (ColumnParallel/Vocab)")
                    total_size = loaded_tensor.shape[dim_to_shard]
                    block_size = total_size // tp_size
                    start, end = tp_rank * block_size, (tp_rank + 1) * block_size
                    loaded_tensor = loaded_tensor[start:end, :]
                    print(f"    - Sharded tensor shape: {loaded_tensor.shape}")

                elif is_row_parallel:
                    dim_to_shard = 1
                    print(f"    - Applying TP shard on dim={dim_to_shard} (RowParallel)")
                    total_size = loaded_tensor.shape[dim_to_shard]
                    block_size = total_size // tp_size
                    start, end = tp_rank * block_size, (tp_rank + 1) * block_size
                    loaded_tensor = loaded_tensor[:, start:end]
                    print(f"    - Sharded tensor shape: {loaded_tensor.shape}")
            
            if is_row_parallel and "weight" in model_param_name and loaded_tensor.ndim == 2:
                loaded_tensor = loaded_tensor.transpose([1, 0])
                print(f"    - Transposed for RowParallel -> {loaded_tensor.shape}")

            if param.shape != loaded_tensor.shape and np.prod(param.shape) == np.prod(loaded_tensor.shape):
                print(f"    - Reshaping loaded tensor from {loaded_tensor.shape} to {param.shape}")
                loaded_tensor = loaded_tensor.reshape(param.shape)
            
            print(f"    - Final check before copy: param.shape={param.shape}, loaded_tensor.shape={loaded_tensor.shape}")
            assert param.shape == loaded_tensor.shape, f"Final shape mismatch for {model_param_name}: param {param.shape} vs loaded {loaded_tensor.shape}"
            param.copy_(loaded_tensor.cast(param.dtype), False)
            print(f"  > SUCCESS: Loaded into '{model_param_name}'")

        unprocessed_weights = set(weights_map.keys()) - processed_ckpt_names
        if unprocessed_weights:
            unprocessed_weights = {w for w in unprocessed_weights if "inv_freq" not in w}
            unprocessed_weights = {w for w in unprocessed_weights if not any(x in w for x in [
                'q_a_proj.weight_scale_inv', 'kv_a_proj_with_mqa.weight_scale_inv', 
                'gate_proj.weight_scale_inv', 'up_proj.weight_scale_inv',
            ])}
            unprocessed_weights = {w for w in unprocessed_weights if 'mlp.experts' not in w}

            if unprocessed_weights:
                print(f"\n[KIMI LOADER] WARNING: The following weights were not used: {unprocessed_weights}")

        print("\n--- [KIMI LOADER V6 DEBUG] WEIGHT LOAD COMPLETE ---")


class KimiK2PretrainedModel(DeepSeekV3PretrainedModel):
    @classmethod
    def arch_name(cls):
        return "KimiK2ForCausalLM"