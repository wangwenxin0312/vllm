"""
vllm.v1.ucm_offload.state
=========================
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import enum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from .ucm_offloader import UcmOffloader

# ---- 模块级常量 ----
INVALID_SLOT = -1  # 通用常量，用于表示无效 slot

class UcmSparseRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1

# ---- 单例缓存 ----
_ucm_singleton: Optional["UcmOffloader"] = None
_ucm_singleton_role: Optional[UcmSparseRole] = None

def init_ucm_offloader(
    vllm_config: "VllmConfig",
    *,
    role: UcmSparseRole,
    force_reinit: bool = False,
) -> "UcmOffloader":
    """
    初始化全局 UcmOffloader 单例。
    """
    global _ucm_singleton, _ucm_singleton_role

    # 懒加载：仅此时导入实现，避免循环依赖
    from .ucm_offloader import UcmOffloader  # type: ignore

    if _ucm_singleton is not None and not force_reinit:
        # 已有实例：角色需保持一致
        if _ucm_singleton_role != role:
            raise RuntimeError(
                f"UcmOffloader 已以角色 {_ucm_singleton_role} 初始化；"
                f"当前请求的角色为 {role}。若确需切换角色，请设置 force_reinit=True。"
            )
        return _ucm_singleton

    # 创建/重建单例
    _ucm_singleton = UcmOffloader(vllm_config=vllm_config, role=role)
    _ucm_singleton_role = role
    return _ucm_singleton


def get_ucm_offloader() -> Optional["UcmOffloader"]:
    return _ucm_singleton

def get_ucm_offloader_role() -> Optional[UcmSparseRole]:
    return _ucm_singleton_role

def ensure_ucm_offloader_initialized(
    vllm_config: "VllmConfig",
    role: UcmSparseRole,
) -> "UcmOffloader":
    offloader = get_ucm_offloader()
    if offloader is not None:
        return offloader
    return init_ucm_offloader(vllm_config=vllm_config, role=role)


__all__ = [
    "INVALID_SLOT",
    "UcmSparseRole",
    "init_ucm_offloader",
    "get_ucm_offloader",
    "get_ucm_offloader_role",
    "ensure_ucm_offloader_initialized",
]
