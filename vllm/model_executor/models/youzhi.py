# Copyright (c) 2026 PSBC. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2ForCausalLM,
    DeepseekV2MLAAttention,
)
from vllm.model_executor.models.utils import PPMissingLayer

class YouZhiRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        skip: bool = False,
    ) -> None:
        super().__init__(hidden_size, eps=eps)
        self.skip = skip

    def forward(self, *args, **kwargs):
        if 'x' in kwargs:
            x = kwargs['x']
        elif len(args) > 0:
            x = args[0]
        else:
            raise ValueError("Need input x")
        return x

class YouZhiForCausalLM(DeepseekV2ForCausalLM):

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        qk_latent_layernorm = getattr(config, "qk_latent_layernorm", True)

        if not qk_latent_layernorm:
            self._skip_layernorm_names = set()
            for layer in self.model.layers:
                if isinstance(layer, PPMissingLayer):
                    continue
                attn = layer.self_attn
                if isinstance(attn, DeepseekV2MLAAttention):
                    impl = attn.mla_attn.mla_attn.impl
                    if attn.q_lora_rank is not None:
                        new_q_a_ln = YouZhiRMSNorm(
                            attn.q_lora_rank,
                            eps=config.rms_norm_eps,
                            skip=True,
                        )
                        attn.q_a_layernorm = new_q_a_ln
                        if hasattr(impl, 'q_a_layernorm'):
                            impl.q_a_layernorm = new_q_a_ln
                        self._skip_layernorm_names.add(
                            f"model.layers.{layer.layer_idx}.self_attn.q_a_layernorm.weight"
                        )
                    new_kv_a_ln = YouZhiRMSNorm(
                        attn.kv_lora_rank,
                        eps=config.rms_norm_eps,
                        skip=True,
                    )
                    attn.kv_a_layernorm = new_kv_a_ln
                    if hasattr(impl, 'kv_a_layernorm'):
                        impl.kv_a_layernorm = new_kv_a_ln
                    self._skip_layernorm_names.add(
                        f"model.layers.{layer.layer_idx}.self_attn.kv_a_layernorm.weight"
                    )
        else:
            self._skip_layernorm_names = set()

    def load_weights(self, weights):
        loaded_params = super().load_weights(weights)
        loaded_params.update(self._skip_layernorm_names)
        return loaded_params