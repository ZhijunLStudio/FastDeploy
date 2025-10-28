"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import paddle
from paddle import nn
from typing_extensions import assert_never

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (
    get_weight_iterator,
    is_weight_cache_enabled,
    load_weights_from_cache,
    measure_time,
    save_model,
)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.adapters import as_embedding_model
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.platforms import current_platform


def _resolve_architecture(model_config: ModelConfig) -> str:
    """
    Resolves the correct model architecture name to load.
    ...
    """
    from paddleformers.utils.log import logger
    from fastdeploy.model_executor.models.model_base import ModelRegistry

    model_type_to_arch = {
        "kimi_k2": "KimiK2ForCausalLM",
    }

    model_type = getattr(model_config, 'model_type', None)

    if model_type and model_type in model_type_to_arch:
        resolved_arch = model_type_to_arch[model_type]
        # --- 核心修正：使用 try-except 来检查是否存在 ---
        try:
            # 尝试获取模型类，如果成功，说明它存在
            ModelRegistry.get_class(resolved_arch)
            logger.info(
                f"Resolved architecture to '{resolved_arch}' based on model_type '{model_type}'."
            )
            return resolved_arch
        except KeyError:
            # 如果 get_class 抛出 KeyError，说明该架构未注册
            logger.warning(
                f"Architecture '{resolved_arch}' for model_type '{model_type}' is defined in map but not registered. Falling back to default."
            )
            pass  # 继续执行下面的 fallback 逻辑
        
    # 如果没有找到或者模型类不存在，则回退到原始逻辑
    return model_config.architectures[0]


class DefaultModelLoaderV1(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def clean_memory_fragments(self) -> None:
        """clean_memory_fragments"""
        if current_platform.is_cuda() or current_platform.is_maca():
            paddle.device.empty_cache()
            paddle.device.synchronize()

    @save_model()
    @measure_time()
    def load_weights(self, model, fd_config: FDConfig, enable_cache: bool = False) -> None:
        weights_iterator = get_weight_iterator(fd_config.model_config.model)
        if enable_cache:
            load_weights_from_cache(model, weights_iterator)
        else:
            model.load_weights(weights_iterator)

        self.clean_memory_fragments()

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        # architectures = fd_config.model_config.architectures[0]
        architectures = _resolve_architecture(fd_config.model_config)
        print(f"Loading model type: {architectures}")
        context = paddle.LazyGuard()
        if fd_config.load_config.dynamic_load_weight:
            # register rl model
            import fastdeploy.rl  # noqa

            architectures = architectures + "RL"

        enable_cache, _, weight_cache_context = is_weight_cache_enabled(fd_config)
        with weight_cache_context:
            with context:
                model_cls = ModelRegistry.get_class(architectures)
                convert_type = fd_config.model_config.convert_type
                if convert_type == "none":
                    pass
                elif convert_type == "embed":
                    model_cls = as_embedding_model(model_cls)
                else:
                    assert_never(convert_type)

                model = model_cls(fd_config)
                
                # --- 在这里插入 Debug 代码 ---
                print("\n" + "="*50)
                print("--- [DEBUG] All Model Parameters and Shapes ---")
                found_problematic_param = False
                for name, param in model.named_parameters():
                    if 0 in param.shape:
                        print(f"  - [!!! PROBLEM !!!] Name: {name}, Shape: {param.shape}")
                        found_problematic_param = True
                    # 为了减少日志量，可以只打印有问题的参数
                    # else:
                    #     print(f"  - Name: {name}, Shape: {param.shape}")
                if not found_problematic_param:
                    print("  - All parameter shapes seem OK (no zero dimensions).")
                print("="*50 + "\n")
                # --- Debug 代码结束 ---

        model.eval()
        # RL model not need set_state_dict
        if fd_config.load_config.dynamic_load_weight:
            return model
        self.load_weights(model, fd_config, enable_cache)
        return model
