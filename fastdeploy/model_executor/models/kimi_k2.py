# /home/aistudio/work/FastDeploy/fastdeploy/model_executor/models/kimi_k2.py

from __future__ import annotations
import re
import paddle

# --- 1. 导入所有我们需要继承或替换的官方类 ---
from .deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepSeekV3PretrainedModel,
    DeepSeekV3Model,
    DeepSeekV3DecoderLayer,
    DeepseekV3MLAAttention,
)
from fastdeploy.model_executor.layers.linear import MergedReplicatedLinear
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.models.model_base import (
    ModelForCasualLM,
    ModelCategory,
    ModelRegistry,
)
from fastdeploy.model_executor.utils import (
    default_weight_loader,
    process_weights_after_loading,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead


# --- 2. 创建一个修正版的 MergedReplicatedLinear ---
# 我们在自己的文件里定义一个新类，它继承自官方类，但修正了 __init__ 方法
class _KimiFixedMergedReplicatedLinear(MergedReplicatedLinear):
    def __init__(self, *args, **kwargs):
        # 正常调用父类的 __init__
        super().__init__(*args, **kwargs)

        # 这是我们之前讨论过的核心修正：重新创建权重，并传入 'output_dim': True
        extra_attrs = {
            "output_dim": True,
            "weight_loader": self.weight_loader,
            "model_format": self.fd_config.model_config.model_format,
        }
        self.quant_method.create_weights(self, **extra_attrs)


# --- 3. 创建一个继承自官方 Attention 的新类，并替换掉有问题的层 ---
class KimiK2MLAAttention(DeepseekV3MLAAttention):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        # 调用父类的 __init__，让它完成大部分初始化工作
        super().__init__(fd_config, layer_id, prefix)

        # --- 核心操作：猴子补丁 ---
        # 用我们修正过的版本，替换掉父类创建的那个有问题的 qkv_a_proj_with_mqa 实例
        self.qkv_a_proj_with_mqa = _KimiFixedMergedReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_a_proj_with_mqa",
            input_size=self.hidden_size,
            output_sizes=[self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            with_bias=False,
        )


# --- 4. 创建继承自官方 DecoderLayer 的新类，确保它使用我们修正后的 Attention ---
class KimiK2DecoderLayer(DeepSeekV3DecoderLayer):
    def __init__(self, fd_config: FDConfig, prefix: str = "") -> None:
        # 调用父类的 __init__
        super().__init__(fd_config, prefix)
        
        # 替换掉 self_attn
        layer_id = int(prefix.split(sep=".")[-1])
        self.self_attn = KimiK2MLAAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )


# --- 5. 创建继承自官方 Model 的新类，确保它使用我们修正后的 DecoderLayer ---
class KimiK2Model(DeepSeekV3Model):
    def __init__(self, fd_config: FDConfig = None):
        # 不要调用父类的 __init__，因为它会创建错误的层
        super(DeepSeekV3Model, self).__init__() # 只调用 nn.Layer 的 __init__

        self.num_layers = fd_config.model_config.num_hidden_layers
        
        # 确保前缀正确
        fd_config.model_config.pretrained_config.prefix_name = "model"

        # 正常创建 embed_tokens 和 norm
        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype(),
            prefix="model.embed_tokens",
        )
        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix="model.norm",
        )

        # 使用我们自己的 KimiK2DecoderLayer 来创建层
        self.layers = nn.LayerList(
            [
                KimiK2DecoderLayer(
                    fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )


# --- 6. 最终的模型主类 ---
@ModelRegistry.register_model_class(
    architecture="KimiK2ForCausalLM",
    module_name="kimi_k2",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class KimiK2ForCausalLM(DeepseekV3ForCausalLM):
    def __init__(self, fd_config: FDConfig):
        # 调用最顶层基类的 __init__
        ModelForCasualLM.__init__(self, fd_config)

        # 使用我们完全自定义的 KimiK2Model
        self.model = KimiK2Model(fd_config)

        # 复制其他必要的初始化代码
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        self.position_ids_buffer = paddle.empty(
            [fd_config.scheduler_config.max_num_batched_tokens], dtype=paddle.int32
        )
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