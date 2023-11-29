"""Multi-head attention."""
from typing import List, Optional

import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        slot_mapping = input_metadata.slot_mapping.flatten()

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
            )

        if input_metadata.is_prompt:
            # Prompt run.
            if self.num_kv_heads != self.num_heads:
                # For MQA/GQA, project the key and value tensors to the desired
                # number of heads.
                query = query.view(query.shape[0], self.num_kv_heads,
                                   self.num_queries_per_kv, query.shape[-1])
                key = key[:, :,
                          None, :].expand(key.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          key.shape[-1])
                value = value[:, :, None, :].expand(value.shape[0],
                                                    self.num_kv_heads,
                                                    self.num_queries_per_kv,
                                                    value.shape[-1])

            # Set attention bias if not provided. FIXME: This is a hack.
            if input_metadata.attn_bias is None:
                if self.alibi_slopes is None:
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        [seq_len] * batch_size)
                    if self.sliding_window is not None:
                        attn_bias = attn_bias.make_local_attention(
                            self.sliding_window)
                    input_metadata.attn_bias = attn_bias
                else:
                    input_metadata.attn_bias = _make_alibi_bias(
                        self.alibi_slopes, batch_size, seq_len, query.dtype)

            # TODO(woosuk): Too much views. Can we do better?
            if self.alibi_slopes is None:
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
            else:
                query = query.unflatten(0, (batch_size, seq_len))
                key = key.unflatten(0, (batch_size, seq_len))
                value = value.unflatten(0, (batch_size, seq_len))

            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=input_metadata.attn_bias,
                p=0.0,
                scale=self.scale,
            )
            output = out.view_as(query)
        else:
            # Decoding run.
            block_size = value_cache.shape[3]
            output = torch.empty_like(query)
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                self.head_mapping,
                self.scale,
                input_metadata.block_tables,
                input_metadata.context_lens,
                block_size,
                8192,  # FIXME
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


# FIXME: Temporary hack to avoid import errors.
PagedAttentionWithRoPE = PagedAttention
PagedAttentionWithALiBi = PagedAttention


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]
    bias = bias.to(alibi_slopes.device)

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    bias = torch.empty(
        batch_size,
        alibi_slopes.shape[0],
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias
