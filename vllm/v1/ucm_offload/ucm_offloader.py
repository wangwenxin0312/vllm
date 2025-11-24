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
    统一 Host Pinned Slab + 分层块偏移 + 层内顺序映射（global_block_id <-> 层内位置）。

    1) 进程启动后初始化（prefill 的第0层）：
        - offloader.prepare_unified_slab(max_model_len)
    2) prefill 每层：offloader.offload_blocks_prefill(lid, kv_cache_u8, block_table)
    3) decode：
        - B) 直接写入“新分配的 device blocks”：reload_tokens_decode_into_blocks(...)
        - C) 将新产生 tokens 回写 Host：dump_new_tokens_decode(...)
    """

    def __init__(self, vllm_config: "VllmConfig", role: UcmSparseRole, device_id: int | None = None):
        self.num_layers = vllm_config.model_config.hf_config.num_hidden_layers
        self.block_size = vllm_config.cache_config.block_size
        self.element_size = 1
        self.head_size = 656
        self.max_model_len = vllm_config.model_config.max_model_len
        self.token_dim_bytes = int(self.head_size) * self.element_size
        self.block_bytes = self.block_size * self.token_dim_bytes
        self.role = role
        # UCM device/ops
        if self.role == UcmSparseRole.WORKER:
            self.device_id = int(device_id)
            self.dev = uc.MakeDevice(self.device_id)
            _ = self.dev.Setup()
        else:
            # SCHEDULER 不做 D2H/H2D，不需要 ucdevice
            self.device_id = None
            self.dev = None

        # 统一 slab 及布局
        self._slab_host: Optional[torch.Tensor] = None          # pinned host 大块
        self._base_host_ptr: Optional[int] = None               # uintp 地址
        self._layer_offset_blocks: Optional[List[int]] = None   # 每层块偏移（单位：block）
        self._layer_capacity_blocks: Optional[List[int]] = None # 每层预留容量（单位：block）

        # 每层映射
        self._layer_id2pos: Dict[int, Dict[int, int]] = {}      # global_block_id -> 层内顺序位置
        
        # === token 粒度信息 ===
        self._layer_capacity_tokens: List[int] = []             # 每层 token 容量 = capacity_blocks * block_size
        self._layer_next_token_pos: List[int] = []              # 每层“下一要写入的 token 索引”
        self._layer_token_pos: Dict[int, Dict[int, int]] = {}   # global_token_id -> host 内 token 索引
        self._prev_topk_global: Dict[int, torch.Tensor] = {}

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
    def prepare_unified_slab(self, layer_id, num_actual_tokens):
        per_layer_num_total_tokens = [int(self.max_model_len)] * self.num_layers
        per_layer_blocks = [
            math.ceil((na) / self.block_size) 
            for na in per_layer_num_total_tokens
        ]
        
        self._layer_capacity_blocks = per_layer_blocks
        self._layer_capacity_tokens = [
            b * self.block_size for b in per_layer_blocks
        ]

        self._layer_next_token_pos = [num_actual_tokens for _ in range(self.num_layers)]
        # global_token_id -> host_token_pos 的映射，先置空，decode 时填
        self._layer_token_pos = {
            lid: {} for lid in range(self.num_layers)
        }

        # 前缀和（块偏移）
        offsets = [0]
        for b in per_layer_blocks[:-1]:
            offsets.append(offsets[-1] + b)
        self._layer_offset_blocks = offsets

        total_blocks = sum(per_layer_blocks)
        total_bytes = total_blocks * self.block_bytes

        if self._slab_host is None:
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
        if self.dev is None:
            raise RuntimeError("offload_blocks_prefill should only be called on WORKER role.")
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
            dtype=torch.int64,
            device="cuda",
        )

        # Host 目的指针：层基址 + 顺序 [0..nblk-1]
        base = self.get_layer_base_ptr(lid)
        host_ptrs = torch.arange(
            base,
            base + nblk * self.block_bytes,
            step=self.block_bytes,
            dtype=torch.int64,
            device="cuda",
        )

        # torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.dev.D2HBatchSync(
            int(dev_ptrs.data_ptr()),
            int(host_ptrs.data_ptr()),
            int(nblk),
            int(self.block_bytes),
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bw = (nblk * self.block_bytes) / (t1 - t0) / (1024**3)
        print(f"[D2H][prefill][L{lid}] nblk={nblk}, {1000*(t1-t0):.3f} ms, {bw:.2f} GiB/s")

        # ======= 校验 =========
        ref_tokens = kv_cache_u8[blk_table]
        ref_cpu = ref_tokens.cpu().contiguous().view(-1)
        slab_cpu = self._slab_host.cpu().contiguous().view(-1)   # uint8

        # 计算 base 在整个 slab 里的起始元素 index
        slab_base_ptr = self._slab_host.data_ptr()               # int64
        elem_size = self._slab_host.element_size()               # uint8 一般是 1
        offset_elems = (base - slab_base_ptr) // elem_size       # 起始 index
        num_bytes = ref_cpu.numel()                              # nblk * block_size * token_dim_bytes

        slab_slice = slab_cpu[offset_elems : offset_elems + num_bytes]
        diff_d2h = (ref_cpu != slab_slice)
        # num_diff_d2h = int(diff_d2h.sum().item())
        
        if torch.equal(ref_cpu, slab_slice):
            print(f"[UcmOffloader][OK] L{lid} D2H offload content matches kv_cache.")
        else:
            diff = (ref_cpu != slab_slice)
            num_err = diff.sum().item()
            first_idx = diff.nonzero()[0].item()
            print(
                f"[UcmOffloader][ERR] L{lid} content mismatch: {num_err} bytes differ. "
                f"first idx={first_idx}, ref={int(ref_cpu[first_idx])}, "
                f"slab={int(slab_slice[first_idx])}"
            )
        # ======= 校验 =========

        # 映射
        blk_table_cpu = blk_table.to("cpu", non_blocking=True)
        self._layer_id2pos[lid] = {int(g): i for i, g in enumerate(blk_table_cpu.tolist())}

    # ---- decode：Host -> 直接写入“新分配的 device blocks”（H2D 目的地）----
    @torch.inference_mode()
    def reload_tokens_decode_into_blocks(
        self,
        lid: int,
        topk_indices_global: torch.Tensor,   # CUDA int32/64 [N] or [1,N]
        kv_cache_u8: torch.Tensor,           # [N_blk, block_size, token_dim_bytes] (CUDA uint8)
        dst_blocks_cuda: torch.Tensor,         # CUDA int32/64 [B] 新分配 block id
    ):
        if self.dev is None:
            raise RuntimeError("reload_tokens_decode_into_blocks should only be called on WORKER role.")
        
        if topk_indices_global.dim() == 2:
            topk_indices_global = topk_indices_global.squeeze(0)
        tok_g = topk_indices_global.to(torch.long, non_blocking=True)
        
        if dst_blocks_cuda.dim() == 2:
            dst_blocks_cuda = dst_blocks_cuda.reshape(-1)
        else:
            dst_blocks_cuda = dst_blocks_cuda
        dst_blocks_cuda = dst_blocks_cuda.to(torch.int32, non_blocking=True).contiguous()
        dst_blocks_cuda = dst_blocks_cuda[dst_blocks_cuda > 0]

        # 确保这一层已经在 prefill 阶段 offload 过
        if lid not in self._layer_id2pos:
            raise RuntimeError("[UcmOffloader] Prefill offload not done for this layer.")
        blk_size = self.block_size
        base = self.get_layer_base_ptr(lid)

        # global_block_id -> host slab_idx（block 粒度的布局）
        id2pos_block = self._layer_id2pos[lid]
        # global_token_id -> host token_pos（decode 阶段 append 的 token）
        token_pos_map = self._layer_token_pos[lid]

        tok_g_cpu = tok_g.to("cpu", non_blocking=True).numpy().astype(np.int64)
        num_tokens = tok_g_cpu.size
       
        # 先统一算出每个 token 在 host 内的 token_pos
        token_pos_np = np.empty(num_tokens, dtype=np.int64)
        for i, g in enumerate(tok_g_cpu):
            g_int = int(g)
            pos = token_pos_map.get(g_int, None)
            if pos is not None:
                # 情况 1：decode 新 token，直接用之前 append 时记录的 token_pos
                token_pos_np[i] = pos
            else:
                # 情况 2：prefill token，用 block 映射 + block 内 offset 反推 token_pos
                blk = g_int // blk_size
                ofs = g_int % blk_size
                slab_idx = id2pos_block.get(int(blk), -1)
                if slab_idx < 0:
                    # raise RuntimeError(
                    #     f"[UcmOffloader] L{lid}: token g={g_int} (blk={blk}) "
                    #     f"refers to non-offloaded block in prefill."
                    # )
                    continue
                token_pos_np[i] = slab_idx * blk_size + ofs
        
        token_pos = torch.from_numpy(token_pos_np).to("cuda", non_blocking=True)  # int64
        base = self.get_layer_base_ptr(lid)
        host_dev = (token_pos * self.token_dim_bytes + base).to(torch.int64)

        # 目的端：按 block 粒度，把这些 token 写到新分配的 device blocks 中
        nblk_needed = (num_tokens + blk_size - 1) // blk_size
        assert int(dst_blocks_cuda.numel()) >= nblk_needed, \
            f"dst_blocks({int(dst_blocks_cuda.numel())}) < needed({nblk_needed})"

        dst_blocks = dst_blocks_cuda.to(torch.long, non_blocking=True)[:nblk_needed]
        dst_base_ptrs = np.array(
            [kv_cache_u8[int(b)].data_ptr() for b in dst_blocks.tolist()],
            dtype=np.uintp,
        )

        dst_blk = np.repeat(np.arange(nblk_needed, dtype=np.int64), blk_size)[:num_tokens]
        dst_ofs = np.tile(np.arange(blk_size, dtype=np.int64), nblk_needed)[:num_tokens]
        dst_addrs = (dst_base_ptrs[dst_blk] + dst_ofs * self.token_dim_bytes).astype(np.uintp)
        dst_dev = torch.from_numpy(dst_addrs.view(np.int64)).to("cuda", non_blocking=True)

        t0 = time.perf_counter()
        self.dev.H2DBatchSync(
            int(dst_dev.data_ptr()),
            int(host_dev.data_ptr()),
            int(num_tokens),
            int(self.token_dim_bytes),
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bw = (num_tokens * self.token_dim_bytes) / (t1 - t0) / (1024**3)
        print(f"[H2D][decode-into-newblocks][L{lid}] num_tokens={num_tokens}, "
              f"{1000*(t1-t0):.3f} ms, {bw:.2f} GiB/s")

        # ======= 校验 =========
        try:
            # ---  Host slab ---
            slab = self._slab_host
            slab_cpu = slab.cpu().contiguous()
            slab_base_ptr = slab.data_ptr()
            elem_size = slab.element_size()

            # base 是当前层在 slab 中的起始地址
            offset_elems = (base - slab_base_ptr) // elem_size
            layer_tokens = self._layer_capacity_tokens[lid]
            layer_bytes = layer_tokens * self.token_dim_bytes

            # view 成 [layer_tokens, token_dim_bytes]，按 token 粒度索引
            layer_view = slab_cpu.view(-1)[
                offset_elems : offset_elems + layer_bytes
            ].view(layer_tokens, self.token_dim_bytes)

            token_pos_t = torch.from_numpy(token_pos_np).long()
            host_tokens = layer_view[token_pos_t].contiguous().view(-1)  # 1D uint8

            # --- 从 device kv_cache 侧重建“被写入的数据” ---
            # 这里 dst_blk/dst_ofs 和上面计算 dst_addrs 的逻辑保持一致
            dst_blk_t = torch.from_numpy(dst_blk).long()
            dst_ofs_t = torch.from_numpy(dst_ofs).long()
            dst_blocks_cpu = dst_blocks.to("cpu", non_blocking=True).view(-1)
            dev_tokens = kv_cache_u8[
                dst_blocks_cpu[dst_blk_t], dst_ofs_t
            ].cpu().contiguous().view(-1)  # 1D uint8

            if host_tokens.numel() != dev_tokens.numel():
                print(f"[UcmOffloader][WARN] L{lid} verify size mismatch: "
                      f"host={host_tokens.numel()}, dev={dev_tokens.numel()}")

            diff = (host_tokens != dev_tokens)
            num_diff = int(diff.sum().item())
            if num_diff == 0:
                print(f"[UcmOffloader][OK] L{lid} H2D reload content matches host slab.")
            else:
                first_idx = int(diff.nonzero()[0])
                print(
                    f"[UcmOffloader][ERR] L{lid} H2D content mismatch: "
                    f"{num_diff} bytes differ. first idx={first_idx}, "
                    f"host={int(host_tokens[first_idx])}, "
                    f"dev={int(dev_tokens[first_idx])}"
                )
        except Exception as e:
            print(f"[UcmOffloader][WARN] L{lid} verify failed: {e}")
        # ======= 校验 =========

        return dst_blocks.to(dtype=torch.int32, device="cuda")

    @torch.inference_mode()
    def incre_reload_tokens_decode_into_blocks(
        self,
        lid: int,
        topk_indices_global: torch.Tensor,   # CUDA int32/64 [N] or [1,N]
        kv_cache_u8: torch.Tensor,           # [N_blk, block_size, token_dim_bytes] (CUDA uint8)
        dst_blocks_cuda: torch.Tensor,         # CUDA int32/64 [B] 固定的 block id（整个 decode 过程中不变）
    ):
        if self.dev is None:
            raise RuntimeError("incre_reload_tokens_decode_into_blocks should only be called on WORKER role.")

        # 归一化成 1D
        if topk_indices_global.dim() == 2:
            topk_indices_global = topk_indices_global.squeeze(0)
        tok_g = topk_indices_global.to(torch.long, non_blocking=True)  # [N]

        if lid not in self._layer_id2pos:
            raise RuntimeError("[UcmOffloader] Prefill offload not done for this layer.")

        blk_size = self.block_size
        base = self.get_layer_base_ptr(lid)

        # ---- 取出 block 粒度 & token 粒度的映射 ----
        id2pos_block = self._layer_id2pos[lid]   # global_block_id -> host slab_idx
        token_pos_map = self._layer_token_pos[lid]  # global_token_id -> host token_pos（decode append token）

        # ---- 准备增量 reload 的条件 ----
        tok_g_cpu = tok_g.to("cpu", non_blocking=True)
        dst_blocks_cpu = dst_blocks_cuda.to("cpu", non_blocking=True)

        prev_topk = self._prev_topk_global.get(lid, None)
        prev_dst = self._prev_dst_blocks.get(lid, None)

        full_reload = False
        if prev_topk is None or prev_dst is None:
            full_reload = True
        elif prev_topk.numel() != tok_g_cpu.numel():
            full_reload = True
        elif not torch.equal(prev_dst, dst_blocks_cpu):
            # block 布局变了，保守起见全量重载
            full_reload = True

        # =========================
        # 1) 先算出所有 token 的 host token_pos
        # =========================
        # 注意：这个是 CPU 上的 int64 数组，用于后面全量 / 增量两种路径
        num_tokens = int(tok_g_cpu.numel())
        token_pos_np = np.empty(num_tokens, dtype=np.int64)

        tok_g_np = tok_g_cpu.numpy().astype(np.int64)

        for i, g in enumerate(tok_g_np):
            g_int = int(g)
            pos = token_pos_map.get(g_int, None)
            if pos is not None:
                # 情况 1：decode 新 token，在 dump_new_tokens_decode 时已经 append 并记录了 token_pos
                token_pos_np[i] = pos
            else:
                # 情况 2：prefill token，用 block 映射 + block 内 offset 反推 token_pos
                blk = g_int // blk_size
                ofs = g_int % blk_size
                slab_idx = id2pos_block.get(int(blk), -1)
                if slab_idx < 0:
                    raise RuntimeError(
                        f"[UcmOffloader] L{lid}: token g={g_int} (blk={blk}) "
                        f"refers to non-offloaded block in prefill."
                    )
                token_pos_np[i] = slab_idx * blk_size + ofs

        # =========================
        # 2) 决定哪些槽位需要 H2D
        # =========================
        if full_reload:
            # 全量路径：所有槽位都需要 reload
            changed_idx_np = np.arange(num_tokens, dtype=np.int64)
        else:
            # 增量路径：只 reload topk 发生变化的位置
            prev_topk_np = prev_topk.numpy().astype(np.int64)
            changed_mask = (tok_g_np != prev_topk_np)
            if not changed_mask.any():
                # 没有任何变化，直接返回，复用上一轮 GPU 中的 KV
                print(f"[H2D][decode-into-newblocks][L{lid}] num_tokens=0 (no change)")
                return dst_blocks_cuda.to(dtype=torch.int32, device="cuda")

            changed_idx_np = np.nonzero(changed_mask)[0].astype(np.int64)

        num_changed = int(changed_idx_np.size)

        # =========================
        # 3) 准备 host 端地址（只为 changed 的 token）
        # =========================
        changed_token_pos = token_pos_np[changed_idx_np]  # [num_changed]
        token_pos_cuda = torch.from_numpy(changed_token_pos).to("cuda", non_blocking=True)

        # addr = base + token_pos * token_dim_bytes
        host_dev = (token_pos_cuda * self.token_dim_bytes + base).to(torch.int64)

        # =========================
        # 4) 准备 device 端地址（只为 changed 的槽位）
        # =========================
        # 逻辑窗口位置 i -> block index / offset：
        #   blk_idx = i // blk_size
        #   ofs_idx = i % blk_size
        nblk_needed = (num_tokens + blk_size - 1) // blk_size
        # 当前窗口应该完全落在前 nblk_needed 个 dst_blocks_cuda 里
        dst_blocks_used = dst_blocks_cpu[:nblk_needed]  # CPU tensor

        # 用 numpy 计算 changed 槽位对应的 block index / offset
        changed_blk_idx = (changed_idx_np // blk_size).astype(np.int64)
        changed_ofs_idx = (changed_idx_np % blk_size).astype(np.int64)

        dst_base_ptrs = np.array(
            [kv_cache_u8[int(b)].data_ptr() for b in dst_blocks_used.tolist()],
            dtype=np.uintp,
        )

        dst_addrs = (dst_base_ptrs[changed_blk_idx]
                     + changed_ofs_idx * self.token_dim_bytes).astype(np.uintp)
        dst_dev = torch.from_numpy(dst_addrs.view(np.int64)).to("cuda", non_blocking=True)

        # =========================
        # 5) 执行 H2D 批量拷贝
        # =========================
        t0 = time.perf_counter()
        self.dev.H2DBatchSync(
            int(dst_dev.data_ptr()),
            int(host_dev.data_ptr()),
            int(num_changed),
            int(self.token_dim_bytes),
        )
        t1 = time.perf_counter()
        bw = (num_changed * self.token_dim_bytes) / (t1 - t0) / (1024**3)
        print(f"[H2D][decode-into-newblocks][L{lid}] "
              f"num_tokens={num_changed}/{num_tokens}, {1000*(t1-t0):.3f} ms, {bw:.2f} GiB/s")

        # =========================
        # 6) 更新缓存（为下一轮增量 reload 做准备）
        # =========================
        self._prev_topk_global[lid] = tok_g_cpu.clone()
        self._prev_dst_blocks[lid] = dst_blocks_cpu.clone()

        return dst_blocks_cuda.to(dtype=torch.int32, device="cuda")
    
    # -------- decode：新产生 token 回写 Host（token 粒度 D2H）--------
    @torch.inference_mode()
    def dump_new_tokens_decode(
        self,
        lid: int,
        kv_cache_u8: torch.Tensor,            # [N_blk, block_size, token_dim_bytes] (CUDA uint8)
        new_token_global_ids_1d: torch.Tensor # CUDA int32/64 [N_new]
    ) -> None:
        if self.dev is None:
            raise RuntimeError("dump_new_tokens_decode should only be called on WORKER role.")
        
        if new_token_global_ids_1d is None or int(new_token_global_ids_1d.numel()) == 0:
            return
        # 没有做过 prefill offload 的层，直接跳过
        if lid not in self._layer_id2pos:
            return

        self._check_ready()
        blk_size = self.block_size

        tok_g = new_token_global_ids_1d.to(torch.long, non_blocking=True)
        tok_blk = (tok_g // blk_size).to("cpu", non_blocking=True).numpy()
        tok_ofs_in_gpu_blk = (tok_g % blk_size).to("cpu", non_blocking=True).numpy()

        num_new = int(tok_g.numel())
        cap_tokens = self._layer_capacity_tokens[lid]
        start_pos = self._layer_next_token_pos[lid]
        end_pos = start_pos + num_new
        if end_pos > cap_tokens:
            raise RuntimeError(
                f"[UcmOffloader] L{lid}: host token capacity exceeded "
                f"({end_pos} > {cap_tokens}). Increase out_token_reserve."
            )

        token_pos = torch.arange(
            start_pos,
            end_pos,
            dtype=torch.int64,
            device="cuda",
        )
        base = self.get_layer_base_ptr(lid)
        host_dev = (token_pos * self.token_dim_bytes + base).to(torch.int64)

        # device 端地址：还是用 kv_cache_u8[block, ofs] 拿
        dev_addrs = np.empty(num_new, dtype=np.uintp)
        for i, (b, ofs) in enumerate(zip(tok_blk, tok_ofs_in_gpu_blk)):
            dev_addrs[i] = kv_cache_u8[int(b), int(ofs)].data_ptr()
        dev_dev = torch.from_numpy(dev_addrs.view(np.int64)).to("cuda", non_blocking=True)

        # 记录 global_token_id -> host_token_pos 的映射
        token_pos_map = self._layer_token_pos[lid]
        for i, g in enumerate(tok_g.tolist()):
            token_pos_map[int(g)] = int(token_pos[i])

        # 更新“下一写入 token 索引”
        self._layer_next_token_pos[lid] = end_pos

        num_tokens = num_new
        # torch.cuda.synchronize(); 
        t0 = time.perf_counter()
        self.dev.D2HBatchSync(
            int(dev_dev.data_ptr()),
            int(host_dev.data_ptr()),
            int(num_tokens),
            int(self.token_dim_bytes),
        )
        torch.cuda.synchronize(); 
        t1 = time.perf_counter()
        bw = (num_tokens * self.token_dim_bytes) / (t1 - t0) / (1024**3)
        print(f"[D2H][decode-new][L{lid}] num_tokens={num_tokens}, "
              f"{1000*(t1-t0):.3f} ms, {bw:.2f} GiB/s")

    def estimate_num_slots_sparsed(self, request: Request) -> int:
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
        output_len = request.num_output_tokens
        slots_need = 2048
        return slots_need + output_len

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