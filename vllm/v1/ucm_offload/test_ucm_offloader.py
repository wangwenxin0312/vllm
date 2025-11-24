import pytest
import torch
import numpy as np

import vllm.attention.ops.flashmla as fm

from vllm.v1.ucm_offload.state import (
    init_ucm_offloader,
    UcmSparseRole,
)
from vllm.v1.ucm_offload.ucm_offloader import UcmOffloader


# --------- 构造一个最小 VllmConfig，专门给 UcmOffloader 用 ---------
class DummyHFConfig:
    def __init__(self, num_layers: int):
        self.num_hidden_layers = num_layers

class DummyModelConfig:
    def __init__(self):
        # DeepSeek v3.2: 61 层
        self.hf_config = DummyHFConfig(num_layers=61)
        # 这里我们只需要 dtype.itemsize=1，使 token_dim_bytes = 656
        # 所以用 uint8 充当“模型 dtype”
        self.dtype = torch.uint8

        # 让 get_head_size() * itemsize == 656
        # dtype.itemsize = 1，所以 head_size 直接设 656
        self._head_size = 656

        # 随便设一个最大长度
        self.max_model_len = 32768

    def get_head_size(self):
        return self._head_size


class DummyCacheConfig:
    def __init__(self):
        # DeepSeek v3 系 block_size = 64
        self.block_size = 64


class DummyVllmConfig:
    def __init__(self):
        self.model_config = DummyModelConfig()
        self.cache_config = DummyCacheConfig()


@pytest.mark.cuda
def test_ucm_flashmla_deepseekv3_like_smoke():
    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    # --------- 1. 初始化 UcmOffloader 单例（worker 角色） ---------

    vllm_config = DummyVllmConfig()
    offloader = init_ucm_offloader(
        vllm_config=vllm_config,
        role=UcmSparseRole.WORKER,
        device_id=2,
        force_reinit=True,
    )
    assert isinstance(offloader, UcmOffloader)

    # DeepSeek v3.2 风格参数
    num_layers = offloader.num_layers          # 61
    block_size = offloader.block_size          # 64
    token_dim_bytes = offloader.token_dim_bytes  # 656
    head_dim_k = 576
    head_dim_v = 512
    num_heads_q = 16
    num_heads_k = 1
    batch_size = 1

    # kv_cache_u8 形状： [748, 64, 656]
    N_blk = 748
    kv_cache_u8 = torch.randint(
        0,
        256,
        (N_blk, block_size, token_dim_bytes),
        dtype=torch.uint8,
        device=device,
    )

    # flash_mla_with_kvcache 需要的 k_cache 形状为
    # [num_blocks, block_size, num_heads_k, bytes_per_token]
    k_cache = kv_cache_u8.view(N_blk, block_size, 1, token_dim_bytes)

    # --------- 2. Prefill 阶段：q shape = [1, 5012, 16, 576] ---------
    seqlen_q_prefill = 5012
    topk_prefill = 128  # prefill 随便用个较小 topk，以免 indices 太大

    # q: [B, Sq, Hq, Dk]
    q_prefill = torch.randn(
        (batch_size, seqlen_q_prefill, num_heads_q, head_dim_k),
        dtype=torch.bfloat16,
        device=device,
    )

    # cache_seqlens：当前 cache 内的有效长度，这里就等于 prefill 序列长
    cache_seqlens_prefill = torch.full(
        (batch_size,),
        fill_value=seqlen_q_prefill,
        dtype=torch.int32,
        device=device,
    )

    # MLA metadata
    q_seq_per_hk_prefill = seqlen_q_prefill * num_heads_q // num_heads_k
    tile_md_prefill, num_splits_prefill = fm.get_mla_metadata(
        cache_seqlens_prefill,
        q_seq_per_hk_prefill,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk_prefill,
        is_fp8_kvcache=True,
    )

    # block_table：shape [1, 79]，值 1~78，最后一个 0 作为 pad
    max_blocks_prefill = 79
    block_table_prefill = torch.zeros(
        (batch_size, max_blocks_prefill),
        dtype=torch.int32,
        device=device,
    )
    block_table_prefill[0, :78] = torch.arange(1, 79, dtype=torch.int32, device=device)

    # indices: [B, Sq, topk_prefill]，这里简单全 0
    indices_prefill = torch.randint(
        low=64,
        high=5055,   # 你需要的最大索引 + 1
        size=(batch_size, seqlen_q_prefill, topk_prefill),
        dtype=torch.int32,
        device=device,
    )

    # 先做一次 attention 计算，确保 FlashMLA FP8 kvcache 正常
    out_prefill, lse_prefill = fm.flash_mla_with_kvcache(
        q_prefill,
        k_cache,
        block_table_prefill,
        cache_seqlens_prefill,
        head_dim_v,
        tile_md_prefill,
        num_splits_prefill,
        indices=indices_prefill,
        is_fp8_kvcache=True,
    )
    
    new_block_table_prefill = block_table_prefill + 100

    out_prefill_1, lse_prefill_1 = fm.flash_mla_with_kvcache(
        q_prefill,
        k_cache,
        new_block_table_prefill,
        cache_seqlens_prefill,
        head_dim_v,
        tile_md_prefill,
        num_splits_prefill,
        indices=indices_prefill,
        is_fp8_kvcache=True,
    )
    diff_out_prefill = (out_prefill != out_prefill_1)
    num_diff_d2h = int(diff_out_prefill.sum().item())

    assert out_prefill.shape[0] == batch_size
    assert out_prefill.shape[-1] == head_dim_v
    assert lse_prefill.shape[0] == batch_size

    # UcmOffloader 在第 0 层进行统一 slab 规划 + block offload
    lid = 0
    offloader.prepare_unified_slab(
        layer_id=lid,
        num_actual_tokens=int(seqlen_q_prefill),
    )
    offloader.offload_blocks_prefill(
        lid=lid,
        kv_cache_u8=kv_cache_u8,
        block_table_cuda=block_table_prefill,
    )

    # --------- 3. Decode 阶段：q shape = [1, 1, 16, 576] ---------
    seqlen_q_decode = 1
    topk_decode = 2048  # 这里按你给的配置，用 2048

    q_decode = torch.zeros(
        (batch_size, seqlen_q_decode, num_heads_q, head_dim_k),
        dtype=torch.bfloat16,
        device=device,
    )

    cache_seqlens_decode = torch.full(
        (batch_size,),
        fill_value=seqlen_q_prefill,  # decode 时 cache 里已有的长度
        dtype=torch.int32,
        device=device,
    )

    q_seq_per_hk_decode = seqlen_q_decode * num_heads_q // num_heads_k
    tile_md_decode, num_splits_decode = fm.get_mla_metadata(
        cache_seqlens_decode,
        q_seq_per_hk_decode,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk_decode,
        is_fp8_kvcache=True,
    )

    # 新的 block_table：shape [1, 32]，值从 79 ~ 79+32-1
    num_new_blocks = 32
    new_block_table = torch.arange(
        79,
        79 + num_new_blocks,
        dtype=torch.int32,
        device=device,
    ).view(1, -1)

    # 作为 dst_blocks_1d 传入 reload_tokens_decode_into_blocks
    dst_blocks_1d = new_block_table.reshape(-1)  # [32]

    # topk_indices_global：shape [1, 2048]
    # 选取若干连续 block 的 token 区间，然后从中随机抽样 2048 个 token
    BLOCK_SIZE = block_size
    start_block = 1
    num_sel_blocks = 32  # 使用 32 个 block 覆盖 2048 个 token
    start_token = start_block * BLOCK_SIZE
    end_token = (start_block + num_sel_blocks) * BLOCK_SIZE
    total_tokens = end_token - start_token  # = 32*64 = 2048

    sel_token_indices = np.random.choice(
        total_tokens, size=topk_decode, replace=False
    ) + start_token
    topk_indices_global = torch.from_numpy(sel_token_indices).to(
        device=device, dtype=torch.int32
    ).view(1, -1)  # [1, 2048]

    # 通过 UcmOffloader 将这些 token 映射到新的 blocks（H2D）
    compact_bt = offloader.reload_tokens_decode_into_blocks(
        lid=lid,
        topk_indices_global=topk_indices_global,
        kv_cache_u8=kv_cache_u8,
        dst_blocks_1d=dst_blocks_1d,
    )
    # compact_bt 应该是一维 block_id 列表，作为 block_table 使用
    assert compact_bt.dim() == 1
    assert compact_bt.numel() > 0

    block_table_decode = compact_bt.view(1, -1).to(torch.int32)

    slot_mapping = (block_table_decode.unsqueeze(-1) * 64 + torch.arange(64, device=device, dtype=torch.int32)).reshape(-1)
    indices_decode = slot_mapping.unsqueeze(0)
    # decode 阶段的 FlashMLA 调用：
    # 这里模拟“重新排布后的 block_table + 无 indices”（token 已经 compact）
    out_decode, lse_decode = fm.flash_mla_with_kvcache(
        q_decode,
        k_cache,
        block_table_decode,
        cache_seqlens_decode,
        head_dim_v,
        tile_md_decode,
        num_splits_decode,
        indices=indices_decode.unsqueeze(0),
        is_fp8_kvcache=True,
    )


    out_decode_1, lse_decode_1 = fm.flash_mla_with_kvcache(
        q_decode,
        k_cache,
        block_table_decode,
        cache_seqlens_decode,
        head_dim_v,
        tile_md_decode,
        num_splits_decode,
        indices=indices_decode.unsqueeze(0),
        is_fp8_kvcache=True,
    )


    assert out_decode.shape[0] == batch_size
    assert out_decode.shape[-1] == head_dim_v
    assert lse_decode.shape[0] == batch_size

    # 将本次 decode 产生的新 token 回写 host slab（D2H）
    new_tok_ids = torch.tensor([7061], device=device, dtype=torch.int32)
    offloader.dump_new_tokens_decode(
        lid=lid,
        kv_cache_u8=kv_cache_u8,
        new_token_global_ids_1d=new_tok_ids,
    )

    # 如果跑到这里没抛异常，基本可以认为：
    # - UcmOffloader 的 prefill / decode 路径都能在 flashmla_sparse 场景下正常工作
    # - FP8 MLA sparse kernel 与 UcmOffloader 的集成在 DeepSeek v3.2 典型配置下可用
    print("[TEST] UcmOffloader + FlashMLA sparse (DeepSeek-v3.2-like) smoke test passed.")

test_ucm_flashmla_deepseekv3_like_smoke()
