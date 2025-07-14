# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.model_executor.models.constant_size_cache import ConstantSizeCache


@dataclass
class MambaCacheParams:
    conv_state: torch.Tensor = torch.Tensor()
    ssm_state: torch.Tensor = torch.Tensor()
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return MambaCacheParams(self.conv_state[layer_idx],
                                self.ssm_state[layer_idx],
                                self.state_indices_tensor)


class MambaCacheManager(ConstantSizeCache):

    def __init__(self, vllm_config: VllmConfig, dtype: torch.dtype,
                 num_mamba_layers: int, conv_state_shape: tuple[int, int],
                 temporal_state_shape: tuple[int, int]):

        # Determine max batch size to set size of MambaCache
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        if not vllm_config.model_config.enforce_eager:
            max_batch_size = vllm_config.pad_for_cudagraph(max_batch_size)

        # Initialize parent class
        super().__init__(max_batch_size)

        conv_state = torch.empty(size=(num_mamba_layers, max_batch_size) +
                                 conv_state_shape,
                                 dtype=dtype,
                                 device="cuda")
        temporal_state = torch.empty(size=(num_mamba_layers, max_batch_size) +
                                     temporal_state_shape,
                                     dtype=torch.float32,
                                     device="cuda")

        self._mamba_cache = (conv_state, temporal_state)

    @property
    def cache(self):
        return self._mamba_cache

    def sleep(self):
        print(f"MambaCache, moving tensors to CPU")
        # Move tensors to CPU and free GPU memory
        conv_state_cpu = self._mamba_cache[0].cpu().clone()
        temporal_state_cpu = self._mamba_cache[1].cpu().clone()

        # Free GPU memory by deleting the original tensors
        # Store references to GPU tensors before replacing the tuple
        gpu_conv_state = self._mamba_cache[0]
        gpu_temporal_state = self._mamba_cache[1]

        # Replace the tuple with CPU tensors
        self._mamba_cache = (conv_state_cpu, temporal_state_cpu)

        # Now delete the GPU tensors
        del gpu_conv_state
        del gpu_temporal_state
        torch.cuda.empty_cache()

    def wake_up(self):
        print(f"MambaCache, moving tensors to GPU")
        # Move tensors back to GPU
        self._mamba_cache = (self._mamba_cache[0].to(torch.device("cuda")), self._mamba_cache[1].to(torch.device("cuda")))

    def _copy_cache(self, from_index: int, to_index: int):
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)

    def current_run_tensors(self, **kwargs) -> MambaCacheParams:
        """
        Return the tensors for the current run's conv and ssm state.
        """
        cache_tensors, state_indices_tensor = super().current_run_tensors(
            **kwargs)
        return MambaCacheParams(cache_tensors[0], cache_tensors[1],
                                state_indices_tensor)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the Mamba Cache during the CUDA graph
        replay runs.
        """
        return self._mamba_cache, torch.as_tensor([PAD_SLOT_ID] * batch_size,
                                                  dtype=torch.int32,
                                                  device="cuda")
