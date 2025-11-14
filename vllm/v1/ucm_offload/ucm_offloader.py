from __future__ import annotations

import math
import time
from typing import Optional, Dict, List, Union, TYPE_CHECKING

import numpy as np
import torch
import ucmdevice as uc

from vllm.v1.request import Request, RequestStatus
from .state import INVALID_SLOT, UcmSparseRole

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ReqType = Union[str, int]

class UcmOffloader:
    """
    统一 Host Pinned Slab + 分层块偏移 + 层内顺序映射（global_block_id <-> 层内位置）的工具类。

    1) 进程启动后初始化（建议在首次进入 prefill 的第0层前）：
        - offloader.prepare_unified_slab(per_layer_num_actual_tokens, out_token_reserve)
    2) prefill 每层：offloader.offload_blocks_prefill(lid, kv_cache_u8, block_table)
    3) decode：
        - B) 直接写入“新分配的 device blocks”：reload_tokens_decode_into_blocks(...)
        - C) 将新产生 tokens 回写 Host：dump_new_tokens_decode(...)
    """

    def __init__(self, vllm_config: "VllmConfig", role: UcmSparseRole):
        self.num_layers = vllm_config.model_config.hf_config.num_hidden_layers
        self.block_size = vllm_config.cache_config.block_size
        self.element_size = vllm_config.model_config.dtype.itemsize
        self.head_size = vllm_config.model_config.get_head_size()
        self.max_model_len = vllm_config.model_config.max_model_len
        self.token_dim_bytes = int(self.head_size) * self.element_size
        self.block_bytes = self.block_size * self.token_dim_bytes

        # UCM device/ops
        self.dev = uc.MakeDevice(int(0))
        _ = self.dev.Setup()

        # 统一 slab 及布局
        self._slab_host: Optional[torch.Tensor] = None          # pinned host 大块
        self._base_host_ptr: Optional[int] = None               # uintp 地址
        self._layer_offset_blocks: Optional[List[int]] = None   # 每层块偏移（单位：block）
        self._layer_capacity_blocks: Optional[List[int]] = None # 每层预留容量（单位：block）

        # 每层映射
        self._layer_block_ids: Dict[int, torch.Tensor] = {}     # 层内顺序位置 -> global_block_id（保持 offload 时的顺序）
        self._layer_id2pos: Dict[int, Dict[int, int]] = {}      # global_block_id -> 层内顺序位置

        self.preempt_req_output_tokens: Dict[ReqType, int] = {}
        
    def _check_ready(self):
        if (self._slab_host is None or self._base_host_ptr is None or
            self._layer_offset_blocks is None or self._layer_capacity_blocks is None):
            raise RuntimeError("[UcmOffloader] prepare_unified_slab(...) must be called first.")

    def get_layer_base_ptr(self, lid: int) -> int:
        self._check_ready()
        base = self._base_host_ptr
        ofs_blocks = self._layer_offset_blocks[lid]
        return base + ofs_blocks * self.block_bytes

    # --------------------- 一次性分配统一 slab ---------------------
    @torch.inference_mode()
    def prepare_unified_slab(self, per_layer_num_actual_tokens, out_token_reserve: int = 100):
        """
        per_layer_num_actual_tokens: int（对所有层广播）。
        out_token_reserve: 未来 decode 新增 token 的预留容量。
        """
        per_layer_num_actual_tokens = [int(per_layer_num_actual_tokens)] * self.num_layers
        per_layer_blocks = [
            math.ceil((na + int(out_token_reserve)) / self.block_size)
            for na in per_layer_num_actual_tokens
        ]
        self._layer_capacity_blocks = per_layer_blocks

        # 前缀和（块偏移）
        offsets = [0]
        for b in per_layer_blocks[:-1]:
            offsets.append(offsets[-1] + b)
        self._layer_offset_blocks = offsets

        total_blocks = sum(per_layer_blocks)
        total_bytes = total_blocks * self.block_bytes

        self._slab_host = torch.empty(total_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        self._base_host_ptr = int(self._slab_host.data_ptr())

        print(f"[UcmOffloader] Unified slab allocated: {total_blocks} blocks "
              f"({total_bytes/1e6:.2f} MB) for {self.num_layers} layers.")

    # --------------------- prefill：block 粒度 D2H ---------------------
    @torch.inference_mode()
    def offload_blocks_prefill(
        self,
        lid: int,
        kv_cache_u8: torch.Tensor,        # [N_blk, block_size, token_dim_bytes] (CUDA uint8)
        block_table_cuda: torch.Tensor,   # CUDA [num_reqs, max_blk] or [N]
    ):
        self._check_ready()

        # 提取 >0 的全局 block id（保持顺序，避免 unique）
        if block_table_cuda.dim() == 2:
            blk_table = block_table_cuda.reshape(-1)
        else:
            blk_table = block_table_cuda
        blk_table = blk_table.to(torch.int32, non_blocking=True).contiguous()
        blk_table = blk_table[blk_table > 0]
        nblk = int(blk_table.numel())

        cap = self._layer_capacity_blocks[lid]
        if nblk > cap:
            raise RuntimeError(
                f"[UcmOffloader] L{lid}: need {nblk} blocks > reserved {cap} blocks. "
                f"Increase out_token_reserve or re-plan capacity."
            )

        # GPU 源 block 指针
        dev_ptrs = torch.tensor(
            [kv_cache_u8[int(g)].data_ptr() for g in blk_table.tolist()],
            dtype=torch.uint64,
            device="cuda",
        )

        # Host 目的指针：层基址 + 顺序 [0..nblk-1]
        base = self.get_layer_base_ptr(lid)
        slab_idx = torch.arange(nblk, dtype=torch.int64, device="cuda")
        host_ptrs = (base + slab_idx * self.block_bytes).to(torch.uint64)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        self.dev.D2HBatchSync(
            int(dev_ptrs.data_ptr()),
            int(host_ptrs.data_ptr()),
            int(nblk),
            int(self.block_bytes),
        )
        torch.cuda.synchronize(); t1 = time.perf_counter()
        bw = (nblk * self.block_bytes) / (t1 - t0) / (1024**3)
        print(f"[D2H][prefill][L{lid}] nblk={nblk}, {1000*(t1-t0):.3f} ms, {bw:.2f} GiB/s")

        # 映射
        self._layer_block_ids[lid] = blk_table
        self._layer_id2pos[lid] = {int(g): i for i, g in enumerate(blk_table.tolist())}

    # ---- decode：Host -> 直接写入“新分配的 device blocks”（H2D 目的地）----
    @torch.inference_mode()
    def reload_tokens_decode_into_blocks(
        self,
        lid: int,
        topk_indices_global: torch.Tensor,   # CUDA int32/64 [N] or [1,N]
        kv_cache_u8: torch.Tensor,           # [N_blk, block_size, token_dim_bytes] (CUDA uint8)
        dst_blocks_1d: torch.Tensor,         # CUDA int32/64 [B] 新分配 block id
    ):
        if topk_indices_global.dim() == 2:
            topk_indices_global = topk_indices_global.squeeze(0)
        tok_g = topk_indices_global.to(torch.long, non_blocking=True)

        id2pos = self._layer_id2pos.get(lid, None)
        if id2pos is None:
            raise RuntimeError("[UcmOffloader] Prefill offload not done for this layer.")

        blk_size = self.block_size
        tok_blk = (tok_g // blk_size).to("cpu", non_blocking=True).numpy()
        tok_ofs = (tok_g %  blk_size).to("cpu", non_blocking=True).numpy()
        slab_idx = np.fromiter((id2pos.get(int(g), -1) for g in tok_blk),
                               dtype=np.int64, count=tok_blk.size)
        if (slab_idx < 0).any():
            bad = int((slab_idx < 0).sum())
            raise RuntimeError(f"[UcmOffloader] L{lid}: {bad} tokens refer to non-offloaded blocks.")

        base = self.get_layer_base_ptr(lid)
        host_token_addrs = base + slab_idx * self.block_bytes + tok_ofs * self.token_dim_bytes
        host_dev = torch.from_numpy(host_token_addrs.view(np.uint64)).to("cuda", non_blocking=True)

        num_tokens = int(tok_g.numel())
        nblk_needed = (num_tokens + blk_size - 1) // blk_size
        assert int(dst_blocks_1d.numel()) >= nblk_needed, \
            f"dst_blocks({int(dst_blocks_1d.numel())}) < needed({nblk_needed})"

        dst_blocks = dst_blocks_1d.to(torch.long, non_blocking=True)[:nblk_needed]
        dst_base_ptrs = np.array([kv_cache_u8[int(b)].data_ptr() for b in dst_blocks.tolist()],
                                 dtype=np.uintp)

        dst_blk = np.repeat(np.arange(nblk_needed, dtype=np.int64), blk_size)[:num_tokens]
        dst_ofs = np.tile(np.arange(blk_size, dtype=np.int64), nblk_needed)[:num_tokens]
        dst_addrs = (dst_base_ptrs[dst_blk] + dst_ofs * self.token_dim_bytes).astype(np.uintp)
        dst_dev = torch.from_numpy(dst_addrs.view(np.uint64)).to("cuda", non_blocking=True)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        self.dev.H2DBatchSync(int(dst_dev.data_ptr()), int(host_dev.data_ptr()),
                              int(num_tokens), int(self.token_dim_bytes))
        torch.cuda.synchronize(); t1 = time.perf_counter()
        bw = (num_tokens * self.token_dim_bytes) / (t1 - t0) / (1024**3)
        print(f"[H2D][decode-into-newblocks][L{lid}] num_tokens={num_tokens}, {1000*(t1-t0):.3f} ms, {bw:.2f} GiB/s")
        return dst_blocks.to(dtype=torch.int32, device="cuda")

    # -------- decode：新产生 token 回写 Host（token 粒度 D2H）--------
    @torch.inference_mode()
    def dump_new_tokens_decode(
        self,
        lid: int,
        kv_cache_u8: torch.Tensor,            # [N_blk, block_size, token_dim_bytes] (CUDA uint8)
        new_token_global_ids_1d: torch.Tensor # CUDA int32/64 [N_new]
    ) -> None:
        if new_token_global_ids_1d is None or int(new_token_global_ids_1d.numel()) == 0:
            return
        if lid not in self._layer_id2pos:
            # 该层尚未 offload，按你的流程通常是在 prefill 后才会 decode；这里直接跳过。
            return

        self._check_ready()
        id2pos = self._layer_id2pos[lid]
        blk_size = self.block_size

        tok_g = new_token_global_ids_1d.to(torch.long, non_blocking=True)
        tok_blk = (tok_g // blk_size).to("cpu", non_blocking=True).numpy()
        tok_ofs = (tok_g %  blk_size).to("cpu", non_blocking=True).numpy()

        slab_idx = np.fromiter((id2pos.get(int(g), -1) for g in tok_blk),
                               dtype=np.int64, count=tok_blk.size)
        mask = slab_idx >= 0
        if not mask.any():
            return

        tok_blk = tok_blk[mask]
        tok_ofs = tok_ofs[mask]
        slab_idx = slab_idx[mask]

        dev_addrs = np.empty(tok_blk.shape[0], dtype=np.uintp)
        for i, (b, ofs) in enumerate(zip(tok_blk, tok_ofs)):
            dev_addrs[i] = kv_cache_u8[int(b), int(ofs)].data_ptr()
        dev_dev = torch.from_numpy(dev_addrs.view(np.uint64)).to("cuda", non_blocking=True)

        base = self.get_layer_base_ptr(lid)
        host_addrs = base + slab_idx * self.block_bytes + tok_ofs * self.token_dim_bytes
        host_dev = torch.from_numpy(host_addrs.view(np.uint64)).to("cuda", non_blocking=True)

        num_tokens = int(host_addrs.size)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        self.dev.D2HBatchSync(int(dev_dev.data_ptr()),
                              int(host_dev.data_ptr()),
                              int(num_tokens),
                              int(self.token_dim_bytes))
        torch.cuda.synchronize(); t1 = time.perf_counter()
        bw = (num_tokens * self.token_dim_bytes) / (t1 - t0) / (1024**3)
        print(f"[D2H][decode-new][L{lid}] num_tokens={num_tokens}, {1000*(t1-t0):.3f} ms, {bw:.2f} GiB/s")

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        # self.preempt_req_output_tokens = {}
        if request.status == RequestStatus.PREEMPTED:
            self.preempt_req_output_tokens[request.request_id] = (
                request.num_output_tokens
            )

        if request.request_id in self.preempt_req_output_tokens:
            num_output_tokens = (
                request.num_output_tokens
                - self.preempt_req_output_tokens[request.request_id]
            )
        else:
            num_output_tokens = request.num_output_tokens

        if (
            request.num_computed_tokens == 0
            or num_output_tokens == 0
        ):  
            return INVALID_SLOT

        slots_need = 2048
        return slots_need

    def allocate_slots(self, kv_cache_manager, request, num_encoder_tokens, num_slots_sparsed):
        coordinator = kv_cache_manager.coordinator
        block_pool = kv_cache_manager.block_pool
        kv_cache_groups = kv_cache_manager.kv_cache_config.kv_cache_groups

        if request.request_id in self.preempt_req_output_tokens:
            # handle preempt: get the TRUE output_len
            num_output_tokens = (
                request.num_output_tokens
                - self.preempt_req_output_tokens[request.request_id]
            )
        else:
            num_output_tokens = request.num_output_tokens

        if num_output_tokens == 1:
            kv_cache_manager.free(request)

        new_computed_block_list = tuple([] for _ in range(len(kv_cache_groups)))
        num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_slots_sparsed,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )
        manual_preempt = False
        # manual_preempt = (request.num_output_tokens % 10) == 0
        if manual_preempt or num_blocks_to_allocate > block_pool.get_num_free_blocks():
            return None
        coordinator.allocate_new_blocks(request.request_id, num_slots_sparsed)
        blocks = coordinator.single_type_managers[0].req_to_blocks[request.request_id]
        print("====ucm blocks====", blocks)
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks
        return KVCacheBlocks(tuple([blocks]))