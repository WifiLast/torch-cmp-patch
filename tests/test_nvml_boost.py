"""Tests for cmpext3/nvml_boost.py's Python-level dispatch logic.

These test boost()/restore()/query_throttle_reasons()'s control flow and
error handling by monkeypatching nvml_boost's internal ``pynvml`` reference
directly (a lightweight fake module, not a real import) -- they don't
require a real NVIDIA driver, a real GPU, or pynvml's native library calls
to actually succeed. ``vars()``-introspection of the throttle-reason
constants (see query_throttle_reasons()) needs a real module object rather
than a MagicMock, since MagicMock's own bookkeeping attributes would
otherwise leak into that iteration -- hence types.ModuleType here instead
of the MagicMock-stub approach used for cmpext3._native in conftest.py.

Run with:

    python -m pytest source/other/cmp_ext_turing/tests
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from cmpext3 import nvml_boost


@pytest.fixture(autouse=True)
def _reset_state():
    """Snapshot/restore nvml_boost._STATE around every test. This can't
    rely on monkeypatch's own revert tracking, since boost()/restore()
    mutate the dict in place directly rather than through monkeypatch.
    """
    original = dict(nvml_boost._STATE)
    yield
    nvml_boost._STATE.clear()
    nvml_boost._STATE.update(original)


def _fake_pynvml(**overrides) -> types.ModuleType:
    """A real module object standing in for pynvml, with sane defaults --
    real ModuleType (not MagicMock) so vars(fake) behaves exactly like
    vars(real_pynvml_module) for query_throttle_reasons()'s introspection.
    """
    fake = types.ModuleType("pynvml")
    fake.NVML_CLOCK_SM = 1
    fake.nvmlClocksThrottleReasonNone = 0x0
    fake.nvmlClocksThrottleReasonSwPowerCap = 0x4
    fake.nvmlInit = MagicMock(name="nvmlInit")
    fake.nvmlShutdown = MagicMock(name="nvmlShutdown")
    fake.nvmlDeviceGetHandleByIndex = MagicMock(name="nvmlDeviceGetHandleByIndex", return_value="handle")
    fake.nvmlDeviceGetPowerManagementLimit = MagicMock(name="nvmlDeviceGetPowerManagementLimit", return_value=150_000)
    fake.nvmlDeviceGetPowerManagementLimitConstraints = MagicMock(
        name="nvmlDeviceGetPowerManagementLimitConstraints", return_value=[50_000, 225_000])
    fake.nvmlDeviceSetPowerManagementLimit = MagicMock(name="nvmlDeviceSetPowerManagementLimit")
    fake.nvmlDeviceGetMaxClockInfo = MagicMock(name="nvmlDeviceGetMaxClockInfo", return_value=2100)
    fake.nvmlDeviceSetGpuLockedClocks = MagicMock(name="nvmlDeviceSetGpuLockedClocks")
    fake.nvmlDeviceResetGpuLockedClocks = MagicMock(name="nvmlDeviceResetGpuLockedClocks")
    fake.nvmlDeviceGetCurrentClocksThrottleReasons = MagicMock(
        name="nvmlDeviceGetCurrentClocksThrottleReasons", return_value=0x0)
    fake.nvmlDeviceGetClockInfo = MagicMock(name="nvmlDeviceGetClockInfo", return_value=1800)
    fake.nvmlDeviceGetPowerUsage = MagicMock(name="nvmlDeviceGetPowerUsage", return_value=150_000)
    util = MagicMock(name="utilization")
    util.gpu = 50
    fake.nvmlDeviceGetUtilizationRates = MagicMock(name="nvmlDeviceGetUtilizationRates", return_value=util)
    for name, value in overrides.items():
        setattr(fake, name, value)
    return fake


class TestIsAvailable:
    def test_true_when_pynvml_imported(self, monkeypatch):
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        assert nvml_boost.is_available() is True

    def test_false_when_pynvml_not_imported(self, monkeypatch):
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", False)
        assert nvml_boost.is_available() is False


class TestBoost:
    def test_returns_false_when_pynvml_unavailable(self, monkeypatch):
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", False)
        assert nvml_boost.boost() is False

    def test_raises_power_limit_and_locks_clocks_when_below_max(self, monkeypatch):
        fake = _fake_pynvml()
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)

        result = nvml_boost.boost()

        assert result is True
        fake.nvmlDeviceSetPowerManagementLimit.assert_called_once_with("handle", 225_000)
        fake.nvmlDeviceSetGpuLockedClocks.assert_called_once_with("handle", 2100, 2100)
        assert nvml_boost._STATE["original_power_limit_mw"] == 150_000
        assert nvml_boost._STATE["clocks_locked"] is True

    def test_does_not_lower_power_limit_when_already_at_max(self, monkeypatch):
        fake = _fake_pynvml(nvmlDeviceGetPowerManagementLimit=MagicMock(return_value=225_000))
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)

        nvml_boost.boost()

        fake.nvmlDeviceSetPowerManagementLimit.assert_not_called()
        assert nvml_boost._STATE["original_power_limit_mw"] is None

    def test_returns_false_when_nvml_init_fails(self, monkeypatch):
        fake = _fake_pynvml()
        fake.nvmlInit.side_effect = RuntimeError("driver not found")
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)

        assert nvml_boost.boost() is False

    def test_power_limit_failure_does_not_prevent_clock_lock(self, monkeypatch):
        # Common real-world case: the process lacks privilege for one Set*
        # call but not the other -- each is independent, one failing
        # shouldn't abort the other.
        fake = _fake_pynvml()
        fake.nvmlDeviceSetPowerManagementLimit.side_effect = RuntimeError("permission denied")
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)

        result = nvml_boost.boost()

        assert result is True  # clock lock still succeeded
        fake.nvmlDeviceSetGpuLockedClocks.assert_called_once()
        assert nvml_boost._STATE["original_power_limit_mw"] is None  # never recorded -- it failed

    def test_returns_false_when_both_adjustments_fail(self, monkeypatch):
        fake = _fake_pynvml()
        fake.nvmlDeviceSetPowerManagementLimit.side_effect = RuntimeError("no")
        fake.nvmlDeviceSetGpuLockedClocks.side_effect = RuntimeError("no")
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)

        assert nvml_boost.boost() is False


class TestRestore:
    def test_noop_when_never_initialized(self, monkeypatch):
        fake = _fake_pynvml()
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)

        nvml_boost.restore()  # must not raise

        fake.nvmlDeviceResetGpuLockedClocks.assert_not_called()

    def test_undoes_boost(self, monkeypatch):
        fake = _fake_pynvml()
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)
        nvml_boost.boost()
        fake.nvmlDeviceResetGpuLockedClocks.reset_mock()
        fake.nvmlDeviceSetPowerManagementLimit.reset_mock()

        nvml_boost.restore()

        fake.nvmlDeviceResetGpuLockedClocks.assert_called_once_with("handle")
        fake.nvmlDeviceSetPowerManagementLimit.assert_called_once_with("handle", 150_000)
        assert nvml_boost._STATE["initialized"] is False
        assert nvml_boost._STATE["clocks_locked"] is False
        assert nvml_boost._STATE["original_power_limit_mw"] is None


class TestQueryThrottleReasons:
    def test_returns_empty_dict_when_pynvml_unavailable(self, monkeypatch):
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", False)
        assert nvml_boost.query_throttle_reasons() == {}

    def test_returns_empty_dict_on_query_failure(self, monkeypatch):
        fake = _fake_pynvml()
        fake.nvmlInit.side_effect = RuntimeError("boom")
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)

        assert nvml_boost.query_throttle_reasons() == {}

    def test_reports_expected_fields(self, monkeypatch):
        fake = _fake_pynvml(
            nvmlDeviceGetCurrentClocksThrottleReasons=MagicMock(return_value=0x4),  # SwPowerCap bit set
        )
        monkeypatch.setattr(nvml_boost, "_PYNVML_AVAILABLE", True)
        monkeypatch.setattr(nvml_boost, "pynvml", fake)

        info = nvml_boost.query_throttle_reasons()

        assert info["sm_clock_mhz"] == 1800
        assert info["max_sm_clock_mhz"] == 2100
        assert info["power_usage_w"] == pytest.approx(150.0)
        assert info["power_limit_w"] == pytest.approx(150.0)
        assert info["power_limit_max_w"] == pytest.approx(225.0)
        assert info["gpu_util_pct"] == 50
        assert "nvmlClocksThrottleReasonSwPowerCap" in info["throttle_reasons_active"]
        assert "nvmlClocksThrottleReasonNone" not in info["throttle_reasons_active"]
