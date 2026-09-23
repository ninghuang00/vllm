# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
    LayerNorm,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
    _is_shared_expert_fse_compatible,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = self._make_full_attn(config, vllm_config, prefix)
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    def _make_full_attn(self, config, vllm_config, prefix):
        return Qwen3NextAttention(
            config,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
            reduce_results=not self.use_attn_reduce_scatter_for_moe,
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    # Qwen3.5 ships the GDN in_proj checkpoints separately (qwen3-next
    # pre-fuses them); fuse them on top of the qwen3-next QKV/gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return self._get_decoder_layer_cls()(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    @classmethod
    def _get_decoder_layer_cls(cls):
        return Qwen3_5DecoderLayer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # FSE must match construction (Qwen3NextSparseMoeBlock): reroute the
        # shared expert into the extra fused slot only when AITER FSE is both
        # requested and compatible with the quant spec.
        if "moe" in self.config.model_type:
            weights = maybe_fuse_shared_experts(
                weights,
                enabled=rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
                and _is_shared_expert_fse_compatible(self.quant_config),
                n_routed_experts=self.config.num_experts,
                n_shared_experts=1,
                ckpt_prefix="mlp.shared_expert",
            )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = self._get_model_cls()(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    @classmethod
    def _get_model_cls(cls):
        return Qwen3_5Model

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    # Qwen3.5 does not support multimodal pruning (EVS).
    supports_multimodal_pruning = False

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def recompute_mrope_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3.5 does not support multimodal pruning (EVS). "
            "recompute_mrope_positions should never be called."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()


########################################################
# Qwen3.5-9B MLA / DSA attention variants
# Architecture (device-agnostic); NPU impl injected by vllm_ascend patch.
########################################################


class Qwen3_5KVProjWithMQA(nn.Module):
    """Wrap ``k_proj / k_norm / v_proj / kv_a_proj`` into a single module.

    Externally this module looks like DeepSeek's ``kv_a_proj_with_mqa``:
    ``hidden(2048) -> [latent(512), rope(64)]``.

    Internally it runs the original Qwen3.6 compressed-KV pipeline:

        k = k_proj(hidden)            # [N, 512]  = [2*256]
        k = k_norm(k.view(N,2,256))   # RMSNorm per-head
        k_rot  = k[..., :64]          # [N, 2, 64]
        k_pass = k[..., 64:]          # [N, 2, 192]
        v = v_proj(hidden)           # [N, 512]  = [2*256]
        kv_concat = cat(k_rot_flat, k_pass_flat, v_flat)  # [N, 1024]
        out = kv_a_proj(kv_concat)   # [N, 576]  = [latent(512), rope(64)]
    """

    def __init__(
        self,
        hidden_size: int,
        num_kv_heads: int,
        head_dim: int,
        rope_dim: int,
        kv_lora_rank: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.kv_lora_rank = kv_lora_rank

        self.nope_dim = head_dim - rope_dim  # 192
        self.k_proj = ReplicatedLinear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            prefix="k_proj",
        )
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self.v_proj = ReplicatedLinear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            prefix="v_proj",
        )

        kv_input_dim = (
            num_kv_heads * rope_dim       # k_rot  = 128
            + num_kv_heads * self.nope_dim  # k_pass = 384
            + num_kv_heads * head_dim       # v      = 512
        )                                   # total  = 1024
        self.kv_a_proj = nn.Linear(kv_input_dim, kv_lora_rank + rope_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        bs_seq = hidden_states.shape[:-1]

        # --- K path ---
        k = self.k_proj(hidden_states)[0]                       # [N, 512]
        k = k.view(*bs_seq, self.num_kv_heads, self.head_dim)   # [N, 2, 256]
        k = self.k_norm(k)                                       # RMSNorm on 256
        k_rot = k[..., : self.rope_dim]                          # [N, 2, 64]
        k_pass = k[..., self.rope_dim :]                         # [N, 2, 192]

        # --- V path ---
        v = self.v_proj(hidden_states)[0]                       # [N, 512]
        v = v.view(*bs_seq, self.num_kv_heads, self.head_dim)   # [N, 2, 256]

        # --- flatten & concat ---
        k_rot_flat = k_rot.reshape(*bs_seq, -1)                 # [N, 128]
        k_pass_flat = k_pass.reshape(*bs_seq, -1)               # [N, 384]
        v_flat = v.reshape(*bs_seq, -1)                         # [N, 512]

        kv_concat = torch.cat([k_rot_flat, k_pass_flat, v_flat], dim=-1)  # [N, 1024]
        out = self.kv_a_proj(kv_concat)                          # [N, 576]
        return (out,)


class Qwen3_5QProj(nn.Module):
    """Q projection that also carries an output gate.

    The checkpoint stores a single ``q_proj.weight`` of shape
    ``[num_heads * head_dim * 2, hidden_size]`` which we split into
    query (first half, qk_head_dim per head) and gate (second half,
    head_dim per head).

    Uses a single ``ColumnParallelLinear`` for the merged q+gate weight.
    The TP sharding correctly splits by head because the weight layout
    is [head0_q, head0_g, head1_q, head1_g, ...] interleaved per head.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        qk_head_dim: int,
        rms_norm_eps: float,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_head_dim = qk_head_dim  # 256 = nope(192) + rope(64)

        # Interleaved layout: [head0_q(256), head0_g(256), head1_q(256), ...]
        # This ensures TP sharding by dim 0 gives each rank the correct
        # set of heads with both query and gate parts.
        # Total output = num_heads * (qk_head_dim + head_dim) = 16 * 512 = 8192
        self.qg_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * (qk_head_dim + head_dim),  # 8192
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qg_proj",
        )
        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self._qk_head_dim = qk_head_dim
        self._head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        q_and_gate = self.qg_proj(hidden_states)[0]
        # Output is interleaved: [h0_q(256), h0_g(256), h1_q(256), h1_g(256), ...]
        # After TP sharding: [h0_q, h0_g, h1_q, h1_g, ...] for local heads
        # Each head's block = qk_head_dim + head_dim = 512
        block = self._qk_head_dim + self._head_dim  # 512
        local_heads = q_and_gate.shape[-1] // block
        # Reshape to [tokens, local_heads, block] then split q/g
        qg = q_and_gate.view(-1, local_heads, block)
        query = qg[..., : self._qk_head_dim]  # [tokens, heads, qk_head_dim]
        # gate = qg[..., self._qk_head_dim:]  # not needed here
        query = query.view(-1, local_heads, self.head_dim)
        query = self.q_norm(query)
        return (
            query.reshape(-1, local_heads * self.qk_head_dim),
        )

    def get_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the gate tensor [N, local_heads * head_dim]."""
        q_and_gate = self.qg_proj(hidden_states)[0]
        block = self._qk_head_dim + self._head_dim  # 512
        local_heads = q_and_gate.shape[-1] // block
        qg = q_and_gate.view(-1, local_heads, block)
        gate = qg[..., self._qk_head_dim:]  # [tokens, heads, head_dim]
        return gate.reshape(-1, local_heads * self._head_dim)



class Qwen3_5MLAAttention(nn.Module):
    """Full-attention layer using MLA for compressed-KV Qwen3.6."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        hidden_size = config.hidden_size           # 4096
        num_heads = config.num_attention_heads      # 16
        num_kv_heads = config.num_key_value_heads   # 2
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)  # 256

        # TP: local num_heads (for weight absorption and o_proj)
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
        tp_size = get_tensor_model_parallel_world_size()
        num_heads_local = num_heads // tp_size  # 8 when TP=2

        rope_params = config.rope_parameters or {}
        partial_rotary = rope_params.get("partial_rotary_factor", 1.0)
        rope_dim = int(head_dim * partial_rotary)  # 64
        nope_dim = head_dim - rope_dim             # 192

        kv_comp = getattr(config, "kv_compression", {}) or {}
        kv_lora_rank = kv_comp.get("rank", 512)

        # MLA dimension constants
        qk_rope_head_dim = rope_dim     # 64
        qk_nope_head_dim = nope_dim     # 192
        v_head_dim = head_dim            # 256
        q_lora_rank = None

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 256
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads_local
        self.num_kv_heads_original = num_kv_heads  # 4 (for weight broadcast)

        # ── Create modules ──
        self.kv_a_proj_with_mqa = Qwen3_5KVProjWithMQA(
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            kv_lora_rank=kv_lora_rank,
            rms_norm_eps=config.rms_norm_eps,
        )

        self.q_proj = Qwen3_5QProj(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_head_dim=self.qk_head_dim,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # No latent norm — identity placeholder for MLAModules
        self.kv_a_layernorm = nn.Identity()

        # kv_b_proj: output dim is per-kv-head (896); expanded to per-query-head
        # (7168) in process_weights_after_loading for prefill path compatibility.
        self.kv_b_proj = ReplicatedLinear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + v_head_dim),  # 896
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,  # FULL heads × v_head_dim (RowParallelLinear
            # shards input dim by TP internally, so pass the full count)
            hidden_size,              # 4096
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Rotary embedding — we use GPT-NeoX style (rotate_half) to match
        # the original Qwen3.6 model's RoPE implementation.
        # cos/sin are computed via the rotary_emb's cos_sin_cache, NOT via
        # npu_interleave_rope's _cos_cache (which is for interleaved format).
        max_pos = getattr(config, "max_position_embeddings", 32768)
        rope_params_clean = {
            k: v for k, v in rope_params.items()
            if k not in ("mrope_section", "mrope_interleaved", "partial_rotary_factor")
        }
        rope_params_clean["rope_type"] = rope_params_clean.get("rope_type", "default")
        rope_params_clean["partial_rotary_factor"] = 1.0
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_pos,
            rope_parameters=rope_params_clean,
            is_neox_style=True,
        )

        scale = head_dim ** -0.5

        # ── Build MLAModules ──
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=None,
            q_b_proj=None,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )

        # ── Create MultiHeadLatentAttentionWrapper ──
        # This will register as "multi_head_latent_attention" PluggableLayer
        self.attn = MultiHeadLatentAttentionWrapper(
            hidden_size=hidden_size,
            num_heads=num_heads_local,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(positions, hidden_states)



class Qwen3_5MLADecoderLayer(Qwen3_5DecoderLayer):
    """Decoder layer: linear_attn layers unchanged, full_attn layers use MLA."""

    def _make_full_attn(self, config, vllm_config, prefix):
        return Qwen3_5MLAAttention(
            config=config,
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )



# ════════════════════════════════════════════════════════════════════
#  5.  Model
# ════════════════════════════════════════════════════════════════════


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5MLAModel(Qwen3_5Model):
    """Qwen3.5-9B model with MLA-adapted full-attention layers."""

    hf_to_vllm_mapper = (
        WeightsMapper(
            orig_to_new_stacked={
                ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
                ".in_proj_z": (".in_proj_qkvz", 3),
                ".in_proj_b": (".in_proj_ba", 0),
                ".in_proj_a": (".in_proj_ba", 1),
                ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
                ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            }
        )
        | WeightsMapper(
            orig_to_new_substr={
                ".self_attn.k_proj.": ".self_attn.kv_a_proj_with_mqa.k_proj.",
                ".self_attn.k_norm.": ".self_attn.kv_a_proj_with_mqa.k_norm.",
                ".self_attn.v_proj.": ".self_attn.kv_a_proj_with_mqa.v_proj.",
                ".self_attn.kv_a_proj.": ".self_attn.kv_a_proj_with_mqa.kv_a_proj.",
                ".self_attn.q_norm.": ".self_attn.q_proj.q_norm.",
            }
        )
    )

    @classmethod
    def _get_decoder_layer_cls(cls):
        return Qwen3_5MLADecoderLayer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The mapper is applied once at the ForCausalLM level; do NOT
        # re-apply it here to avoid double-mapping (e.g. in_proj_ba→in_proj_baa).
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)



# ════════════════════════════════════════════════════════════════════
#  6.  ForCausalLM
# ════════════════════════════════════════════════════════════════════


class Qwen3_5MLAForCausalLM(Qwen3_5ForCausalLMBase, IsHybrid, SupportsMRoPE):
    """Language model wrapper for dense Qwen3.5-9B with MLA attention.

    Inherits __init__/forward/compute_logits/embed_input_ids from
    Qwen3_5ForCausalLMBase.  Overrides the model class and weight loading
    (interleaved q+gate q_proj layout from the compressed checkpoint).
    """

    @classmethod
    def _get_model_cls(cls):
        return Qwen3_5MLAModel

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features,
    ) -> tuple[torch.Tensor, int]:
        """Return M-RoPE positions for text-only input (3, seq_len)."""
        import numpy as np
        positions = np.arange(len(input_tokens))
        mrope_positions = np.broadcast_to(positions, (3, len(input_tokens)))
        return torch.tensor(mrope_positions), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint's q_proj.weight is laid out as interleaved per-head
        # [h0_q(256), h0_g(256), h1_q(256), h1_g(256), ...] because the original
        # Qwen3.5 model chunks q_proj.view(seq, heads, head_dim*2) into q/gate.
        # ColumnParallelLinear qg_proj expects this interleaved layout for TP
        # sharding by head, so just rename the weight.
        # Also strip model.language_model. prefix and skip visual/mtp weights.
        skip_prefixes = ("model.visual.", "visual.", "mtp.")
        renamed_weights = []
        for name, data in weights:
            if name.startswith(skip_prefixes):
                continue
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]
            if name.endswith(".self_attn.q_proj.weight"):
                new_name = name.replace(".self_attn.q_proj.weight",
                                        ".self_attn.q_proj.qg_proj.weight")
                renamed_weights.append((new_name, data))
            else:
                renamed_weights.append((name, data))

        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(
            renamed_weights, mapper=self.model.hf_to_vllm_mapper
        )





class Qwen3_5Indexer(nn.Module):
    """Lightweight DSA indexer for Qwen3.5-MLA-DSA.

    Structurally identical to DeepSeek-V3.2's ``Indexer``
    (``vllm.model_executor.models.deepseek_v2.Indexer``): learned ``wq_b`` /
    fused ``wk_weights_proj`` / ``k_norm`` + a K-only paged cache, producing
    top-k key indices consumed by the SFA sparse-attention kernel.

    The single difference from DeepSeek-V3.2: ``wq_b`` takes ``hidden_states``
    directly (Qwen3.5 has no Q-lora residual), so its input dim is
    ``hidden_size`` instead of ``q_lora_rank``.  The indexer forward itself is
    driven by ``AscendSFAImpl.indexer_select_pre_process`` /
    ``indexer_select_post_process`` (which call ``wq_b`` / ``wk_weights_proj``
    / ``k_norm`` directly), so this class only owns the weights + cache.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        quant_config,
        cache_config,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        # Qwen3.5 has no Q-lora; wq_b consumes hidden_states directly.
        self.q_lora_rank = hidden_size
        self.softmax_scale = self.head_dim ** -0.5

        # No tensor parallel — replicated.
        self.wq_b = ReplicatedLinear(
            hidden_size,                         # <-- hidden, not q_lora
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # The checkpoint stores them separately (indexer.wk, indexer.weights_proj);
        # load_weights concatenates them into this fused weight.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.topk_indices_buffer = topk_indices_buffer

        # K-only paged cache.  The model creates it here; the runner overrides
        # the allocation with an AscendSFAIndexerCacheSpec (bf16,
        # head_size=index_head_dim on 910B3 non-C8).  The prefix must be unique
        # and registered so model_runner detects it via isinstance(...,
        # DeepseekV32IndexerCache).
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.prefix = prefix

    def forward(self):
        # The indexer compute is driven by AscendSFAImpl's
        # indexer_select_pre_process / indexer_select_post_process; this stub
        # exists only so nn.Module treats the weights as submodules.
        return




class Qwen3_5DSAAttention(nn.Module):
    """Full-attention layer using MLA + DSA indexer for compressed-KV Qwen3.5."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        hidden_size = config.hidden_size           # 4096
        num_heads = config.num_attention_heads      # 16
        num_kv_heads = config.num_key_value_heads   # 2
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)  # 256

        # TP: local num_heads (for weight absorption and o_proj)
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
        tp_size = get_tensor_model_parallel_world_size()
        num_heads_local = num_heads // tp_size  # 8 when TP=2

        rope_params = config.rope_parameters or {}
        partial_rotary = rope_params.get("partial_rotary_factor", 1.0)
        rope_dim = int(head_dim * partial_rotary)  # 64
        nope_dim = head_dim - rope_dim             # 192

        kv_comp = getattr(config, "kv_compression", {}) or {}
        kv_lora_rank = kv_comp.get("rank", 512)

        # MLA dimension constants
        qk_rope_head_dim = rope_dim     # 64
        qk_nope_head_dim = nope_dim     # 192
        v_head_dim = head_dim            # 256
        q_lora_rank = None

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 256
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads_local
        self.num_kv_heads_original = num_kv_heads  # 4 (for weight broadcast)

        # ── Create modules ──
        self.kv_a_proj_with_mqa = Qwen3_5KVProjWithMQA(
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            kv_lora_rank=kv_lora_rank,
            rms_norm_eps=config.rms_norm_eps,
        )

        self.q_proj = Qwen3_5QProj(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_head_dim=self.qk_head_dim,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # No latent norm — identity placeholder for MLAModules
        self.kv_a_layernorm = nn.Identity()

        # kv_b_proj: output dim is per-kv-head (896); expanded to per-query-head
        # (7168) in process_weights_after_loading for prefill path compatibility.
        self.kv_b_proj = ReplicatedLinear(
            kv_lora_rank,
            num_kv_heads * (qk_nope_head_dim + v_head_dim),  # 896
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,  # FULL heads × v_head_dim (RowParallelLinear
            # shards input dim by TP internally, so pass the full count)
            hidden_size,              # 4096
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Rotary embedding — we use GPT-NeoX style (rotate_half) to match
        # the original Qwen3.6 model's RoPE implementation.
        # cos/sin are computed via the rotary_emb's cos_sin_cache, NOT via
        # npu_interleave_rope's _cos_cache (which is for interleaved format).
        max_pos = getattr(config, "max_position_embeddings", 32768)
        rope_params_clean = {
            k: v for k, v in rope_params.items()
            if k not in ("mrope_section", "mrope_interleaved", "partial_rotary_factor")
        }
        rope_params_clean["rope_type"] = rope_params_clean.get("rope_type", "default")
        rope_params_clean["partial_rotary_factor"] = 1.0
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_pos,
            rope_parameters=rope_params_clean,
            is_neox_style=True,
        )

        scale = head_dim ** -0.5

        # ── Build the DSA indexer (only on full-attention layers) ──
        # Structurally identical to DeepSeek-V3.2's Indexer; wq_b takes
        # hidden_size (Qwen3.5 has no Q-lora residual).
        self.indexer = Qwen3_5Indexer(
            vllm_config=vllm_config,
            config=config,
            hidden_size=hidden_size,
            quant_config=quant_config,
            cache_config=cache_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=f"{prefix}.indexer",
        )

        # ── Build MLAModules (is_sparse=True selects the SFA backend) ──
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=None,
            q_b_proj=None,
            q_proj=self.q_proj,
            indexer=self.indexer,
            is_sparse=True,
            topk_indices_buffer=topk_indices_buffer,
        )

        # ── Create MultiHeadLatentAttentionWrapper ──
        # is_sparse=True → MLAAttention base selects AscendSFABackend and
        # constructs an AscendSFAImpl with the indexer wired in.  We then
        # replace that impl with Qwen3_5DSAImpl (Qwen MLA preprocess + SFA
        # indexer/sparse-attn helpers).
        self.attn = MultiHeadLatentAttentionWrapper(
            hidden_size=hidden_size,
            num_heads=num_heads_local,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(positions, hidden_states)


class Qwen3_5DSADecoderLayer(Qwen3_5DecoderLayer):
    """Decoder layer: linear_attn unchanged, full_attn uses MLA+DSA indexer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        *,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        self._topk_indices_buffer = topk_indices_buffer
        super().__init__(vllm_config, layer_type, prefix)

    def _make_full_attn(self, config, vllm_config, prefix):
        return Qwen3_5DSAAttention(
            config=config,
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=self._topk_indices_buffer,
        )





# ════════════════════════════════════════════════════════════════════
#  5.  Model
# ════════════════════════════════════════════════════════════════════


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5DSAModel(Qwen3_5Model):
    """Qwen3.5-9B model with MLA+DSA-adapted full-attention layers."""

    hf_to_vllm_mapper = (
        WeightsMapper(
            orig_to_new_stacked={
                ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
                ".in_proj_z": (".in_proj_qkvz", 3),
                ".in_proj_b": (".in_proj_ba", 0),
                ".in_proj_a": (".in_proj_ba", 1),
                ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
                ".mlp.up_proj": (".mlp.gate_up_proj", 1),
                ".indexer.wk": (".indexer.wk_weights_proj", 0),
                ".indexer.weights_proj": (".indexer.wk_weights_proj", 1),
            }
        )
        | WeightsMapper(
            orig_to_new_substr={
                ".self_attn.k_proj.": ".self_attn.kv_a_proj_with_mqa.k_proj.",
                ".self_attn.k_norm.": ".self_attn.kv_a_proj_with_mqa.k_norm.",
                ".self_attn.v_proj.": ".self_attn.kv_a_proj_with_mqa.v_proj.",
                ".self_attn.kv_a_proj.": ".self_attn.kv_a_proj_with_mqa.kv_a_proj.",
                ".self_attn.q_norm.": ".self_attn.q_proj.q_norm.",
            }
        )
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Skip Qwen3_5Model.__init__: DSA allocates a shared top-k indices
        # buffer and threads it through every DSA decoder layer.
        nn.Module.__init__(self)

        config: Qwen3_5TextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size

        topk_indices_buffer = None
        if getattr(config, "index_topk", None) is not None:
            from vllm.platforms import current_platform
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=current_platform.device_type,
            )

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, config.hidden_size
        )

        def get_layer(prefix: str):
            return self._get_decoder_layer_cls()(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    @classmethod
    def _get_decoder_layer_cls(cls):
        return Qwen3_5DSADecoderLayer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The mapper is applied once at the ForCausalLM level; do NOT
        # re-apply it here to avoid double-mapping.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)



# ════════════════════════════════════════════════════════════════════
#  6.  ForCausalLM
# ════════════════════════════════════════════════════════════════════


class Qwen3_5DSAForCausalLM(Qwen3_5ForCausalLMBase, IsHybrid, SupportsMRoPE):
    """Language model wrapper for dense Qwen3.5-9B with MLA+DSA attention.

    Inherits __init__/forward/compute_logits/embed_input_ids from
    Qwen3_5ForCausalLMBase.  Overrides the model class and weight loading
    (interleaved q+gate q_proj layout from the compressed checkpoint).
    """

    @classmethod
    def _get_model_cls(cls):
        return Qwen3_5DSAModel

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features,
    ) -> tuple[torch.Tensor, int]:
        """Return M-RoPE positions for text-only input (3, seq_len)."""
        import numpy as np
        positions = np.arange(len(input_tokens))
        mrope_positions = np.broadcast_to(positions, (3, len(input_tokens)))
        return torch.tensor(mrope_positions), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint's q_proj.weight is laid out as interleaved per-head
        # [h0_q(256), h0_g(256), h1_q(256), h1_g(256), ...] because the original
        # Qwen3.5 model chunks q_proj.view(seq, heads, head_dim*2) into q/gate.
        # ColumnParallelLinear qg_proj expects this interleaved layout for TP
        # sharding by head, so just rename the weight.
        # Also strip model.language_model. prefix and skip visual/mtp weights.
        skip_prefixes = ("model.visual.", "visual.", "mtp.")
        renamed_weights = []
        for name, data in weights:
            if name.startswith(skip_prefixes):
                continue
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]
            if name.endswith(".self_attn.q_proj.weight"):
                new_name = name.replace(".self_attn.q_proj.weight",
                                        ".self_attn.q_proj.qg_proj.weight")
                renamed_weights.append((new_name, data))
            else:
                renamed_weights.append((name, data))

        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(
            renamed_weights, mapper=self.model.hf_to_vllm_mapper
        )



# ════════════════════════════════════════════════════════════════════
