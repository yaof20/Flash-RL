from vllm.device_allocator.cumem import CuMemAllocator
from contextlib import contextmanager
import torch
from torch.cuda.memory import MemPool # new MemoryPool class still exported
from torch._C import (
    _cuda_beginAllocateCurrentThreadToPool,
    _cuda_endAllocateToPool,
)

def _get_pool_by_name(name: str) -> MemPool | None:
    """Best-effort fetch of a named pool (e.g., 'weights') from vLLM."""
    pools = CuMemAllocator.get_instance().allocator_and_pools
    entry = pools.get(name)
    if not entry:
        return None
    # vLLM stored (pool, allocator?) tuples/lists before; keep robust.
    cand = entry[0]
    return cand if isinstance(cand, MemPool) else None


@contextmanager
def disable_mem_pool(disable: bool = False, pool_name: str = "weights"):
    """
    Temporarily stop routing allocations from *this thread* to the given MemPool.
    On exit, restore routing to the same pool.

    Notes/pitfalls (new API):
      - This only affects the current thread (new behavior).
      - We cannot check the 'active' pool anymore (MemPoolContext removed),
        so we optimistically end routing and then restore it. If routing
        wasn't active, end is a no-op and restore is harmless.
    """
    device_index = torch.cuda.current_device()
    pool: MemPool | None = None
    did_end = False

    if disable:
        pool = _get_pool_by_name(pool_name)
        if pool is not None:
            # End routing of current thread to this pool (new API).
            _cuda_endAllocateToPool(device_index, pool.id)
            did_end = True

    try:
        yield
    finally:
        # Restore routing to the pool if we had ended it.
        if disable and did_end and pool is not None:
            _cuda_beginAllocateCurrentThreadToPool(device_index, pool.id)