"""
Optional NVML-based clock/power tuning for CMP cards, layered on top of
cmpext3's FFMA-avoidance kernels.

IMPORTANT SCOPE -- read this before expecting a miracle: this does NOT
bypass the CMP driver's FFMA/Tensor-Core instruction-pattern throttle. If
it did, none of cmpext3's kernel-rewriting work would have been necessary
in the first place. NVML's clock/power controls operate on a completely
different mechanism (board power limits, DVFS clock caps) that the driver
enforces independently of which instructions are actually running. This
module only helps with THAT layer: making sure the card isn't *also*
capped by a conservative default power limit, or paying an idle-to-boost
ramp-up delay between calls, on top of (not instead of) the
instruction-pattern throttle cmpext3's kernels avoid. Use
query_throttle_reasons() to see what NVML itself can observe -- a clean
report there does not mean the card isn't being throttled by the separate,
NVML-invisible mechanism.

Uses source/other/nvml/pynvml.py (the standard NVIDIA NVML Python
bindings) -- added to sys.path relative to this file so no extra install
step is needed.

What boost() does, best-effort, called automatically by cmpext3.enable()
(CMPEXT3_NVML_BOOST=1 by default -- see cmpext3/__init__.py):
  - Raises the power management limit to the board's maximum supported
    value (nvmlDeviceSetPowerManagementLimit), if currently set lower.
  - Locks GPU clocks to [max, max] (nvmlDeviceSetGpuLockedClocks) so the
    card doesn't idle-ramp-down between calls and pay a re-boost latency
    on the next one.
  - Both are restored to their original values by restore(), called from
    cmpext3.disable() (also registered via atexit, best-effort, in case
    the caller forgets).

Both Set* calls typically require elevated privileges (root on Linux /
Administrator on Windows) and a supported board (locked SKUs may reject
them outright). Every call is wrapped and failures are logged as warnings,
never raised -- this is a best-effort performance nudge, not a
correctness dependency, and cmpext3 must keep working identically whether
or not this module can apply anything.

SAFETY: nvmlDeviceSetGpuLockedClocks / nvmlDeviceSetPowerManagementLimit
change GLOBAL, system-wide GPU state -- not just for this process. On a
shared machine this affects every other process using the GPU too. Set
CMPEXT3_NVML_BOOST=0 before importing cmpext3 to opt out (e.g. on a
machine shared with other workloads/users), or call cmpext3.nvml_boost
.restore() manually at any time to release the lock/limit early.
"""
from __future__ import annotations

import atexit
import os
import sys
import warnings
from typing import Any, Dict, Optional

# source/other/nvml/pynvml.py lives two levels up from this file
# (cmp_ext_turing/cmpext3/nvml_boost.py -> source/other/nvml/pynvml.py).
_NVML_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "nvml"))
if os.path.isdir(_NVML_DIR) and _NVML_DIR not in sys.path:
    sys.path.insert(0, _NVML_DIR)

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    _PYNVML_AVAILABLE = False


_STATE: Dict[str, Any] = {
    "initialized": False,
    "handle": None,
    "original_power_limit_mw": None,
    "clocks_locked": False,
}


def _warn(msg: str) -> None:
    warnings.warn(f"cmpext3.nvml_boost: {msg}", stacklevel=2)


def _device_index() -> int:
    return int(os.environ.get("CMPEXT3_NVML_DEVICE_INDEX", "0"))


def is_available() -> bool:
    return _PYNVML_AVAILABLE


def query_throttle_reasons(device_index: Optional[int] = None) -> Dict[str, Any]:
    """Best-effort report of NVML-visible clocks/power/throttle state, for
    diagnostics. Returns {} if pynvml/NVML is unavailable or the query fails.

    NOTE: this can only see throttle reasons NVML itself tracks (power cap,
    thermal, sw/hw slowdown, etc.) -- it CANNOT see the CMP driver's
    FFMA/Tensor-Core instruction-pattern throttle, which is the whole
    reason cmpext3's kernels exist. A clean report here does not mean the
    card isn't being throttled by that separate mechanism.
    """
    if not _PYNVML_AVAILABLE:
        return {}
    idx = _device_index() if device_index is None else device_index
    shutdown_needed = False
    try:
        pynvml.nvmlInit()
        shutdown_needed = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        reasons_bitmask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        active_reasons = {
            name: True
            for name, value in vars(pynvml).items()
            if name.startswith("nvmlClocksThrottleReason")
            and isinstance(value, int)
            and (reasons_bitmask & value)
        }
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        _, power_limit_max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        return {
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "max_sm_clock_mhz": pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "power_usage_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
            "power_limit_w": power_limit_mw / 1000.0,
            "power_limit_max_w": power_limit_max_mw / 1000.0,
            "gpu_util_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "throttle_reasons_active": active_reasons,
        }
    except Exception as exc:
        _warn(f"could not query throttle reasons: {exc}")
        return {}
    finally:
        if shutdown_needed:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def boost(device_index: Optional[int] = None) -> bool:
    """Best-effort: raise the power limit to max and lock clocks to max.

    Returns True if at least one adjustment was successfully applied.
    Safe to call repeatedly (idempotent-ish: won't re-raise an
    already-max power limit); safe to call when pynvml/NVML is
    unavailable or the calling process lacks the privileges these Set*
    calls need -- failures are logged as warnings and this returns
    False, never raises.
    """
    if not _PYNVML_AVAILABLE:
        _warn("pynvml not available -- skipping (see source/other/nvml/pynvml.py)")
        return False

    idx = _device_index() if device_index is None else device_index
    applied = False
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        _STATE["handle"] = handle
        _STATE["initialized"] = True
    except Exception as exc:
        _warn(f"NVML initialization failed: {exc}")
        return False

    # 1. Power limit -> max supported.
    try:
        current_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        _, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        if current_mw < max_mw:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, max_mw)
            _STATE["original_power_limit_mw"] = current_mw
            applied = True
    except Exception as exc:
        _warn(f"could not raise power limit (often needs root/Administrator): {exc}")

    # 2. Lock SM clocks to [max, max] so the card doesn't idle-ramp down
    #    between calls and pay a re-boost delay on the next one.
    try:
        max_sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, max_sm_clock, max_sm_clock)
        _STATE["clocks_locked"] = True
        applied = True
    except Exception as exc:
        _warn(f"could not lock GPU clocks (often needs root/Administrator): {exc}")

    return applied


def restore(device_index: Optional[int] = None) -> None:
    """Undo whatever boost() applied. Safe to call even if boost() was
    never called or only partially succeeded."""
    if not _PYNVML_AVAILABLE or not _STATE["initialized"]:
        return

    idx = _device_index() if device_index is None else device_index
    handle = _STATE["handle"]
    if handle is None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception:
            _STATE["initialized"] = False
            return

    if _STATE["clocks_locked"]:
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        except Exception as exc:
            _warn(f"could not reset locked clocks: {exc}")
        _STATE["clocks_locked"] = False

    if _STATE["original_power_limit_mw"] is not None:
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, _STATE["original_power_limit_mw"])
        except Exception as exc:
            _warn(f"could not restore original power limit: {exc}")
        _STATE["original_power_limit_mw"] = None

    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    _STATE["initialized"] = False
    _STATE["handle"] = None


atexit.register(restore)
