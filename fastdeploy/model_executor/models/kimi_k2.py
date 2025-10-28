# /home/aistudio/work/FastDeploy/fastdeploy/model_executor/models/kimi_k2.py

"""
Implementation for KimiK2 model, which shares the DeepSeekV3 architecture
but has different weight loading requirements.
"""

from __future__ import annotations
import re
import paddle

# --- 1. Inherit from DeepSeekV3 classes for maximum code reuse ---
from .deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepSeekV3PretrainedModel,
)

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelRegistry,
)
from fastdeploy.model_executor.utils import (
    default_weight_loader,
    process_weights_after_loading,
)

# --- 2. Create the KimiK2 model class and register it with its own unique name ---
# The key here is that the 'architecture' name is now specific to KimiK2.
# We will later modify the loader to look for this name first.
@ModelRegistry.register_model_class(
    architecture="KimiK2ForCausalLM", 
    module_name="kimi_k2",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class KimiK2ForCausalLM(DeepseekV3ForCausalLM):
    """
    KimiK2 Causal LM model for FastDeploy.
    This class inherits the entire model structure from DeepseekV3ForCausalLM
    but overrides the weight loading logic.
    """
    def __init__(self, fd_config: FDConfig):
        # 不能直接调用 super().__init__(fd_config)，因为它会用 deepseek_v3 前缀创建模型
        # 我们需要手动调用更基类的 __init__，然后自己创建模型
        from fastdeploy.model_executor.models.model_base import ModelForCasualLM
        from .deepseek_v3 import DeepSeekV3Model # 导入基类
        from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
        
        ModelForCasualLM.__init__(self, fd_config)

        # --- 核心修改：在创建模型前，修改配置中的前缀 ---
        # 这一步至关重要，它告诉所有子模块使用 'model' 作为参数名前缀
        fd_config.model_config.pretrained_config.prefix_name = "model"

        # 现在用修改后的配置来创建模型
        self.model = DeepSeekV3Model(fd_config)
        
        # 重新创建 lm_head 等其他部分
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        self.position_ids_buffer = paddle.empty([fd_config.scheduler_config.max_num_batched_tokens], dtype=paddle.int32)
        self.mask_encoder_batch_buffer = paddle.empty(
            [fd_config.scheduler_config.max_num_batched_tokens, 1], dtype=paddle.int32
        )
    
    @classmethod
    def name(cls):
        """Returns the unique name for this model class."""
        return "KimiK2ForCausalLM"

    # --- 3. Override the weight loading method ---
    # This is the most critical change for KimiK2.
    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Loads weights for the KimiK2 model. The primary difference from the
        DeepSeekV3 implementation is the removal of the hardcoded name replacement,
        as KimiK2 weights already use the 'model.' prefix.
        """
        # The mapping logic from DeepSeekV3 is perfectly reusable.
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
            # This is the key difference: We DO NOT replace any prefixes.
            # We use the loaded_weight_name directly.
            
            # The rest of the logic is identical to DeepseekV3ForCausalLM.load_weights
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                if "mlp.experts." in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)

                if model_param_name not in params_dict:
                    continue

                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
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
                    break
                else:
                    model_param_name = loaded_weight_name
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            if "kv_b_proj" in model_sublayer_name:
                kv_model_sublayer_name = model_sublayer_name.replace("kv_b_proj", "kv_b_proj_bmm")
                process_weights_after_loading_fn(kv_model_sublayer_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

# --- 4. Create the corresponding PretrainedModel class for tensor parallelism ---
class KimiK2PretrainedModel(DeepSeekV3PretrainedModel):
    """
    KimiK2 PretrainedModel for tensor parallelism. Inherits all logic
    from DeepSeekV3PretrainedModel as the internal parameter names are identical.
    """
    @classmethod
    def arch_name(self):
        # This must match the name of your main model class
        return "KimiK2ForCausalLM"