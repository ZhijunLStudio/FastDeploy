# /home/aistudio/work/FastDeploy/fastdeploy/model_executor/models/kimi_k2.py
from __future__ import annotations
import re
import paddle

# --- Imports ---
from .deepseek_v3 import DeepseekV3ForCausalLM, DeepSeekV3PretrainedModel
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.models.model_base import ModelCategory, ModelRegistry
from fastdeploy.model_executor.utils import default_weight_loader, process_weights_after_loading

@ModelRegistry.register_model_class(
    architecture="KimiK2ForCausalLM",
    module_name="kimi_k2",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION
)
class KimiK2ForCausalLM(DeepseekV3ForCausalLM):
    """
    KimiK2 model, which is architecturally identical to DeepseekV3.
    This class inherits directly from DeepseekV3ForCausalLM and only overrides
    the weight loading method to handle the difference in weight naming prefixes.
    """

    @classmethod
    def name(cls):
        return "KimiK2ForCausalLM"

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        """
        Loads weights for the KimiK2 model.
        
        The internal model structure uses the 'deepseek_v3.' prefix due to inheritance.
        However, Kimi's weight files use the 'model.' prefix.
        This method adapts the loading process by replacing the prefix before matching weights.
        """
        
        # This mapping is copied directly from `deepseek_v3.py` and is essential for
        # correctly routing split weights (like gate/up_proj) to merged parameters.
        stacked_params_mapping = [
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            ("experts.gate_correction_bias", "gate.e_score_correction_bias", None),
            ("qkv_a_proj_with_mqa", "q_a_proj", "q_a"),
            ("qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", "kv_a"),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()))
        
        for loaded_weight_name, loaded_weight in weights_iterator:
            
            # --- 这是唯一的、根本性的修复 ---
            # 将从文件中读到的 'model.' 前缀替换为模型内部使用的 'deepseek_v3.' 前缀
            # 模仿 Ernie 和 Qwen2 的成功做法
            if loaded_weight_name.startswith("model."):
                 loaded_weight_name = loaded_weight_name.replace("model.", "deepseek_v3.", 1)
            # --- 修复结束 ---

            # The rest of the logic is an exact copy of the battle-tested logic from `DeepseekV3ForCausalLM`.
            model_param_name = None
            param = None
            found = False
            
            # --- Start of copied block from DeepseekV3ForCausalLM.load_weights ---
            for p_name, w_name, s_id in stacked_params_mapping:
                if w_name not in loaded_weight_name:
                    continue
                if "mlp.experts." in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(w_name, p_name)

                if model_param_name not in params_dict:
                    continue

                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, s_id)
                found = True
                break
            
            if not found:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in loaded_weight_name:
                        continue
                    model_param_name = loaded_weight_name.replace(weight_name, param_name)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                    found = True
                    break

            if not found:
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight)

            if model_param_name and param is not None:
                model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight|embeddings|linear)$", "", model_param_name)
                if "kv_b_proj" in model_sublayer_name:
                    kv_model_sublayer_name = model_sublayer_name.replace("kv_b_proj", "kv_b_proj_bmm")
                    process_weights_after_loading_fn(kv_model_sublayer_name)
                process_weights_after_loading_fn(model_sublayer_name, param)
            # --- End of copied block ---

class KimiK2PretrainedModel(DeepSeekV3PretrainedModel):
    @classmethod
    def arch_name(cls):
        return "KimiK2ForCausalLM"