"""Tests for cmpext3's monkeypatch layer (source/other/cmp_ext_turing).

These test the *Python dispatch logic* in cmpext3/__init__.py -- eligibility
gating (grad-safety, dtype/device, shape/argument constraints), correct
wiring to the native kernel, and fallback-to-stock on ineligibility or a
RuntimeError/TypeError from the native call. They do not require a CUDA
GPU or a compiled extension: cmpext3._native is replaced with a MagicMock
stub (see conftest.py) and the CUDA/dtype gate is bypassed on demand via
``force_eligible`` so the branch logic can be exercised with plain CPU
tensors.

Run with:

    python -m pytest source/other/cmp_ext_turing/tests
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import cmpext3
from conftest import STOCK_UNPATCHED, force_eligible
from _helpers import assert_called_once_with_tensors


# ---------------------------------------------------------------------------
# _grad_safe / _usable helpers
# ---------------------------------------------------------------------------

class TestGradSafe:
    def test_true_when_grad_disabled(self):
        x = torch.randn(2, requires_grad=True)
        with torch.no_grad():
            assert cmpext3._grad_safe(x) is True

    def test_true_in_inference_mode(self):
        x = torch.randn(2, requires_grad=True)
        with torch.inference_mode():
            assert cmpext3._grad_safe(x) is True

    def test_false_when_any_tensor_requires_grad(self):
        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2)
        assert torch.is_grad_enabled()
        assert cmpext3._grad_safe(y, x) is False

    def test_true_when_no_tensor_requires_grad(self):
        x = torch.randn(2)
        y = torch.randn(2)
        assert cmpext3._grad_safe(x, y) is True

    def test_non_tensor_args_are_ignored(self):
        assert cmpext3._grad_safe(1, "x", None) is True


class TestUsable:
    def test_true_for_non_tensor_args(self):
        assert cmpext3._usable(1, "x", None) is True

    def test_false_for_cpu_tensor(self):
        x = torch.randn(2)
        assert x.is_cuda is False
        assert cmpext3._usable(x) is False

    def test_false_for_unsupported_dtype_on_fake_cuda_tensor(self, monkeypatch):
        # cmpext3 only supports fp16/fp32/bf16; fake is_cuda so the dtype
        # check (rather than the device check) is what's under test.
        x = torch.zeros(2, dtype=torch.int64)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert cmpext3._usable(x) is False

    def test_true_for_supported_dtype_on_fake_cuda_tensor(self, monkeypatch):
        x = torch.zeros(2, dtype=torch.float16)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert cmpext3._usable(x) is True

    def test_true_for_bfloat16_on_fake_cuda_tensor(self, monkeypatch):
        # bf16 has no native arithmetic on Turing, but the native kernels
        # handle it via fp32 boundary conversion (see src/main.cpp), so the
        # Python-level eligibility gate accepts it like fp16/fp32.
        x = torch.zeros(2, dtype=torch.bfloat16)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert cmpext3._usable(x) is True


# ---------------------------------------------------------------------------
# _install / _restore / enable / disable / is_enabled
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_enable_installs_expected_targets(self, patched):
        assert cmpext3.is_enabled() is True
        assert F.linear is cmpext3._patched_linear
        assert torch.matmul is cmpext3._patched_matmul
        assert torch.bmm is cmpext3._patched_bmm
        assert F.conv2d is cmpext3._patched_conv2d
        assert F.embedding is cmpext3._patched_embedding
        assert F.group_norm is cmpext3._patched_group_norm
        assert F.layer_norm is cmpext3._patched_layer_norm
        assert F.gelu.__name__ == "wrapper"
        assert F.silu.__name__ == "wrapper"
        assert F.softsign.__name__ == "wrapper"

    def test_every_op_installed_by_default(self):
        # conv3d/conv_transpose2d/interpolate used to be excluded here unless
        # CMPEXT3_ENABLE_UNVERIFIED_KERNELS=1 was set, on the theory that an
        # unbenchmarked kernel is a liability. They are installed
        # unconditionally now: autoselect measures each against stock the
        # first time it sees a shape and drops the loser, so installing the
        # wrapper no longer commits to using the kernel.
        assert not cmpext3.is_enabled(), "a previous test left cmpext3 enabled"
        cmpext3.enable()
        try:
            assert F.conv3d is cmpext3._patched_conv3d
            assert F.conv_transpose2d is cmpext3._patched_conv_transpose2d
            assert F.interpolate is cmpext3._patched_interpolate
            assert F.linear is cmpext3._patched_linear
        finally:
            cmpext3.disable()

    def test_legacy_unverified_opt_out_is_still_honored(self, monkeypatch):
        # Nobody who deliberately set CMPEXT3_ENABLE_UNVERIFIED_KERNELS=0
        # should find those kernels switched on under them by upgrading.
        # (_DISABLED_OPS is computed at import time from the environment, so
        # the test patches the parsed result rather than the variable.)
        assert not cmpext3.is_enabled(), "a previous test left cmpext3 enabled"
        monkeypatch.setattr(cmpext3, "_DISABLED_OPS",
                            frozenset({"conv3d", "conv_transpose2d", "interpolate"}))
        stock_conv3d, stock_interpolate = F.conv3d, F.interpolate
        cmpext3.enable()
        try:
            assert F.conv3d is stock_conv3d
            assert F.interpolate is stock_interpolate
            assert F.linear is cmpext3._patched_linear  # unrelated ops unaffected
        finally:
            cmpext3.disable()

    def test_disable_ops_pins_named_ops_to_stock(self, monkeypatch):
        assert not cmpext3.is_enabled(), "a previous test left cmpext3 enabled"
        monkeypatch.setattr(cmpext3, "_DISABLED_OPS", frozenset({"conv2d", "attention"}))
        stock_conv2d, stock_sdpa = F.conv2d, F.scaled_dot_product_attention
        cmpext3.enable()
        try:
            assert F.conv2d is stock_conv2d
            # "attention" is the op name; the torch function it patches is
            # called scaled_dot_product_attention (see _OP_ALIASES).
            assert F.scaled_dot_product_attention is stock_sdpa
            assert F.linear is cmpext3._patched_linear
        finally:
            cmpext3.disable()

    def test_disable_restores_original_callables(self, patched):
        orig_linear = cmpext3._ORIGINALS[(F, "linear")]
        orig_matmul = cmpext3._ORIGINALS[(torch, "matmul")]
        cmpext3.disable()
        assert F.linear is orig_linear
        assert torch.matmul is orig_matmul
        assert cmpext3.is_enabled() is False
        cmpext3.enable()  # so the fixture's teardown disable() is a no-op-safe call

    def test_enable_is_idempotent(self, patched):
        orig_linear_first = cmpext3._ORIGINALS[(F, "linear")]
        cmpext3.enable()  # second call while already enabled
        assert cmpext3._ORIGINALS[(F, "linear")] is orig_linear_first
        assert F.linear is cmpext3._patched_linear

    def test_disable_without_enable_is_noop(self):
        assert cmpext3.is_enabled() is False
        cmpext3.disable()  # must not raise
        assert cmpext3.is_enabled() is False

    def test_ops_alias_is_native_module(self, native):
        assert cmpext3.ops is native

    def test_deliberately_unpatched_ops_untouched(self, patched):
        for (target, name), stock_fn in STOCK_UNPATCHED.items():
            assert getattr(target, name) is stock_fn, f"{name} was patched"


# ---------------------------------------------------------------------------
# F.linear
# ---------------------------------------------------------------------------

class TestLinear:
    def test_calls_native_when_eligible(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(4, 4, dtype=torch.float16)  # even out-channels
        b = torch.randn(4, dtype=torch.float16)
        native.linear.return_value = "native-result"
        assert F.linear(x, w, b) == "native-result"
        assert_called_once_with_tensors(native.linear, x, w, b)

    def test_falls_back_when_not_cuda(self, patched, native):
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        b = torch.randn(3)
        got = F.linear(x, w, b)
        native.linear.assert_not_called()
        # Compare against the captured original directly (not a hand-rolled
        # x @ w.t() + b) so this can't flake on fused-kernel rounding
        # differences from a mathematically-equivalent but distinct formula.
        assert torch.equal(got, cmpext3._ORIGINALS[(F, "linear")](x, w, b))

    def test_falls_back_when_autograd_active(self, patched, native, monkeypatch):
        monkeypatch.setattr(cmpext3, "_usable", lambda *a, **k: True)
        x = torch.randn(2, 4, requires_grad=True)
        w = torch.randn(3, 4)
        assert torch.is_grad_enabled()
        F.linear(x, w)
        native.linear.assert_not_called()

    def test_falls_back_on_1d_input(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(4)
        w = torch.randn(3, 4)
        F.linear(x, w)
        native.linear.assert_not_called()

    def test_falls_back_on_odd_fp16_out_channels(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, dtype=torch.float16)
        w = torch.randn(5, 4, dtype=torch.float16)  # odd out-channels
        F.linear(x, w)
        native.linear.assert_not_called()

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        native.linear.side_effect = RuntimeError("unsupported shape")
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        b = torch.randn(3)
        got = F.linear(x, w, b)
        native.linear.assert_called_once()
        assert torch.equal(got, cmpext3._ORIGINALS[(F, "linear")](x, w, b))


# ---------------------------------------------------------------------------
# torch.matmul
# ---------------------------------------------------------------------------

class TestMatmul:
    def test_2d_calls_native_bmm_unsqueezed(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        native.bmm.return_value = torch.zeros(1, 2, 4)
        got = torch.matmul(a, b)
        assert_called_once_with_tensors(native.bmm, a.unsqueeze(0), b.unsqueeze(0))
        assert got.shape == (2, 4)

    def test_3d_calls_native_bmm_directly(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        sentinel = torch.zeros(2, 3, 5)
        native.bmm.return_value = sentinel
        got = torch.matmul(a, b)
        assert_called_once_with_tensors(native.bmm, a, b)
        assert got is sentinel

    def test_falls_back_when_not_cuda(self, patched, native):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        got = torch.matmul(a, b)
        native.bmm.assert_not_called()
        assert torch.equal(got, cmpext3._ORIGINALS[(torch, "matmul")](a, b))

    def test_falls_back_on_dtype_mismatch(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        a = torch.randn(2, 3, dtype=torch.float32)
        b = torch.randn(3, 4, dtype=torch.float16)
        with pytest.raises(RuntimeError):
            torch.matmul(a, b)
        native.bmm.assert_not_called()

    def test_falls_back_on_unsupported_ndim(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        a = torch.randn(4)
        b = torch.randn(4)
        got = torch.matmul(a, b)
        native.bmm.assert_not_called()
        assert torch.equal(got, cmpext3._ORIGINALS[(torch, "matmul")](a, b))

    def test_respects_out_kwarg(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        out = torch.empty(2, 4)
        expected = torch.empty(2, 4)
        torch.matmul(a, b, out=out)
        cmpext3._ORIGINALS[(torch, "matmul")](a, b, out=expected)
        native.bmm.assert_not_called()
        assert torch.equal(out, expected)

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        native.bmm.side_effect = RuntimeError("boom")
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        got = torch.matmul(a, b)
        native.bmm.assert_called_once()
        assert torch.equal(got, cmpext3._ORIGINALS[(torch, "matmul")](a, b))


# ---------------------------------------------------------------------------
# torch.bmm
# ---------------------------------------------------------------------------

class TestBmm:
    def test_calls_native_when_eligible(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        sentinel = torch.zeros(2, 3, 5)
        native.bmm.return_value = sentinel
        assert torch.bmm(a, b) is sentinel
        assert_called_once_with_tensors(native.bmm, a, b)

    def test_falls_back_when_not_cuda(self, patched, native):
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        got = torch.bmm(a, b)
        native.bmm.assert_not_called()
        assert torch.equal(got, cmpext3._ORIGINALS[(torch, "bmm")](a, b))

    def test_falls_back_on_dtype_mismatch(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        a = torch.randn(2, 3, 4, dtype=torch.float32)
        b = torch.randn(2, 4, 5, dtype=torch.float16)
        with pytest.raises(RuntimeError):
            torch.bmm(a, b)
        native.bmm.assert_not_called()

    def test_respects_out_kwarg(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        out = torch.empty(2, 3, 5)
        expected = torch.empty(2, 3, 5)
        torch.bmm(a, b, out=out)
        cmpext3._ORIGINALS[(torch, "bmm")](a, b, out=expected)
        native.bmm.assert_not_called()
        assert torch.equal(out, expected)

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        native.bmm.side_effect = RuntimeError("boom")
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        got = torch.bmm(a, b)
        native.bmm.assert_called_once()
        assert torch.equal(got, cmpext3._ORIGINALS[(torch, "bmm")](a, b))


# ---------------------------------------------------------------------------
# F.conv2d
# ---------------------------------------------------------------------------

class TestConv2d:
    def test_calls_native_when_eligible(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(4, 3, 3, 3)
        b = torch.randn(4)
        native.conv2d.return_value = "native-result"
        got = F.conv2d(x, w, b, stride=1, padding=1)
        assert got == "native-result"
        assert_called_once_with_tensors(native.conv2d, x, w, b, 1, 1, 1, 1)

    def test_falls_back_when_groups_not_1(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 4, 5, 5)
        w = torch.randn(4, 2, 3, 3)
        got = F.conv2d(x, w, groups=2, padding=1)
        native.conv2d.assert_not_called()
        assert got.shape == (1, 4, 5, 5)

    def test_falls_back_when_not_cuda(self, patched, native):
        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(4, 3, 3, 3)
        F.conv2d(x, w, padding=1)
        native.conv2d.assert_not_called()

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        native.conv2d.side_effect = RuntimeError("boom")
        x = torch.randn(1, 3, 8, 8)
        w = torch.randn(4, 3, 3, 3)
        got = F.conv2d(x, w, padding=1)
        native.conv2d.assert_called_once()
        assert got.shape == (1, 4, 8, 8)



# ---------------------------------------------------------------------------
# cmpext3/autoselect.py -- the runtime native-vs-stock choice
#
# Decisions are driven by wall-clock timing, which would make these tests
# flaky, so `_time` is replaced with a lookup table keyed on the callable.
# What's under test is the decision logic, not the clock.
# ---------------------------------------------------------------------------

class TestAutoselect:

    @staticmethod
    def _timed(monkeypatch, autoselect, timings):
        def fake_time(fn):
            return timings[fn], fn()
        monkeypatch.setattr(autoselect, "_time", fake_time)

    def test_picks_the_faster_implementation(self, autoselecting, monkeypatch):
        calls = {"native": 0, "stock": 0}

        def native():
            calls["native"] += 1
            return torch.zeros(4)

        def stock():
            calls["stock"] += 1
            return torch.zeros(4)

        self._timed(monkeypatch, autoselecting, {native: 0.001, stock: 0.010})
        assert autoselecting.choose("op", ("k",), native, stock) is not None
        decision = autoselecting._decisions[("k",)]
        assert decision.use_native
        assert decision.speedup == pytest.approx(10.0)

        # Later calls with the same signature run the winner and only the winner.
        before_stock = calls["stock"]
        for _ in range(3):
            autoselecting.choose("op", ("k",), native, stock)
        assert calls["stock"] == before_stock

    def test_picks_stock_when_the_kernel_is_slower(self, autoselecting, monkeypatch):
        calls = {"native": 0}

        def native():
            calls["native"] += 1
            return torch.zeros(4)

        def stock():
            return torch.zeros(4)

        self._timed(monkeypatch, autoselecting, {native: 0.010, stock: 0.001})
        autoselecting.choose("op", ("k",), native, stock)
        assert not autoselecting._decisions[("k",)].use_native

        before = calls["native"]
        for _ in range(3):
            autoselecting.choose("op", ("k",), native, stock)
        assert calls["native"] == before, "the losing kernel must never run again"

    def test_rejects_a_kernel_whose_output_disagrees_with_stock(self, autoselecting, monkeypatch):
        # The whole reason it is safe to install every kernel by default: a
        # wrong kernel is dropped on the evidence, however fast it is. Both
        # outputs already exist at probe time, so this costs nothing extra.
        def native():
            return torch.full((4,), 10.0)

        def stock():
            return torch.zeros(4)

        self._timed(monkeypatch, autoselecting, {native: 0.001, stock: 0.010})
        got = autoselecting.choose("op", ("k",), native, stock)
        decision = autoselecting._decisions[("k",)]
        assert not decision.use_native
        assert "mismatch" in decision.note
        assert torch.equal(got, torch.zeros(4)), "must return stock's result, not the bad one"

    def test_tolerates_rounding_differences(self, autoselecting, monkeypatch):
        def native():
            return torch.ones(4) * (1.0 + 1e-6)

        def stock():
            return torch.ones(4)

        self._timed(monkeypatch, autoselecting, {native: 0.001, stock: 0.010})
        autoselecting.choose("op", ("k",), native, stock)
        assert autoselecting._decisions[("k",)].use_native

    def test_rejects_non_finite_output(self, autoselecting, monkeypatch):
        def native():
            return torch.full((4,), float("nan"))

        def stock():
            return torch.zeros(4)

        self._timed(monkeypatch, autoselecting, {native: 0.001, stock: 0.010})
        autoselecting.choose("op", ("k",), native, stock)
        assert not autoselecting._decisions[("k",)].use_native

    def test_kernel_that_raises_is_never_probed_again(self, autoselecting):
        calls = {"native": 0}

        def native():
            calls["native"] += 1
            raise RuntimeError("unsupported shape")

        def stock():
            return torch.zeros(4)

        for _ in range(3):
            assert torch.equal(autoselecting.choose("op", ("k",), native, stock), torch.zeros(4))
        assert calls["native"] == 1
        assert "RuntimeError" in autoselecting._decisions[("k",)].note

    def test_one_off_failure_does_not_overturn_a_good_decision(self, autoselecting, monkeypatch):
        state = {"fail": False}
        calls = {"stock": 0}

        def native():
            if state["fail"]:
                raise RuntimeError("transient OOM")
            return torch.zeros(4)

        def stock():
            calls["stock"] += 1
            return torch.zeros(4)

        self._timed(monkeypatch, autoselecting, {native: 0.001, stock: 0.010})
        autoselecting.choose("op", ("k",), native, stock)
        assert autoselecting._decisions[("k",)].use_native
        settled = calls["stock"]

        state["fail"] = True
        autoselecting.choose("op", ("k",), native, stock)
        assert calls["stock"] == settled + 1, "the failed call should be served by stock"
        assert autoselecting._decisions[("k",)].use_native, "decision should survive one bad call"

        state["fail"] = False
        autoselecting.choose("op", ("k",), native, stock)
        assert calls["stock"] == settled + 1, "and the kernel should be tried again after"

    def test_stops_probing_after_too_many_signatures(self, autoselecting, monkeypatch):
        # An op whose shape changes every call (attention over a variable
        # sequence length) must not pay the probe cost forever, nor grow the
        # decision table without bound.
        def native():
            return torch.zeros(4)

        def stock():
            return torch.zeros(4)

        self._timed(monkeypatch, autoselecting, {native: 0.001, stock: 0.010})
        cap = autoselecting._MAX_SIGNATURES_PER_OP
        for i in range(cap + 5):
            autoselecting.choose("op", ("k", i), native, stock)

        assert autoselecting._probe_counts["op"] == cap
        # Everything past the cap inherits the majority verdict, unmeasured.
        beyond = autoselecting._decisions[("k", cap + 3)]
        assert beyond.use_native
        assert "not probed" in beyond.note

    def test_disabled_autoselect_always_uses_the_kernel(self, monkeypatch):
        from cmpext3 import autoselect
        calls = {"stock": 0}

        def native():
            return torch.zeros(4)

        def stock():
            calls["stock"] += 1
            return torch.ones(4)

        monkeypatch.setattr(autoselect, "_ENABLED", False)
        x = torch.zeros(4)
        got = cmpext3._dispatch("op", x, ("k",), native, stock)
        assert torch.equal(got, torch.zeros(4))
        assert calls["stock"] == 0, "stock must not even run when autoselect is off"


# ---------------------------------------------------------------------------
# F.conv3d
# (video model patch-embed / 3D causal VAE coverage; mirrors TestConv2d
#  above but with 5D NCDHW tensors)
# ---------------------------------------------------------------------------

class TestConv3d:
    """Off by default (see enable()'s comment) -- these use
    patched_with_unverified_kernels to exercise the dispatch logic anyway."""

    def test_calls_native_when_eligible(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 3, 4, 8, 8)
        w = torch.randn(4, 3, 2, 3, 3)
        b = torch.randn(4)
        native.conv3d.return_value = "native-result"
        got = F.conv3d(x, w, b, stride=1, padding=1)
        assert got == "native-result"
        assert_called_once_with_tensors(native.conv3d, x, w, b, 1, 1, 1, 1)

    def test_falls_back_when_groups_not_1(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 4, 4, 5, 5)
        w = torch.randn(4, 2, 2, 3, 3)
        got = F.conv3d(x, w, groups=2, padding=1)
        native.conv3d.assert_not_called()
        # D: (4 + 2*1 - 1*(2-1) - 1)//1 + 1 = 5; H/W: 3x3 kernel + pad=1 is "same" -> 5
        assert got.shape == (1, 4, 5, 5, 5)

    def test_falls_back_when_not_cuda(self, patched_with_unverified_kernels, native):
        x = torch.randn(1, 3, 4, 8, 8)
        w = torch.randn(4, 3, 2, 3, 3)
        F.conv3d(x, w, padding=1)
        native.conv3d.assert_not_called()

    def test_falls_back_on_native_runtime_error(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        native.conv3d.side_effect = RuntimeError("boom")
        x = torch.randn(1, 3, 4, 8, 8)
        w = torch.randn(4, 3, 2, 3, 3)
        got = F.conv3d(x, w, padding=1)
        native.conv3d.assert_called_once()
        # D: (4 + 2*1 - 1*(2-1) - 1)//1 + 1 = 5; H/W: 3x3 kernel + pad=1 is "same" -> 8
        assert got.shape == (1, 4, 5, 8, 8)

    def test_calls_native_for_patch_embed_style_conv(self, patched_with_unverified_kernels, native, monkeypatch):
        # kernel_size == stride (non-overlapping "patchify"), the shape
        # HunyuanVideo-style patch-embed layers use.
        force_eligible(monkeypatch)
        x = torch.randn(1, 16, 4, 8, 8)
        w = torch.randn(32, 16, 2, 4, 4)
        native.conv3d.return_value = "native-result"
        got = F.conv3d(x, w, stride=(2, 4, 4))
        assert got == "native-result"
        assert_called_once_with_tensors(native.conv3d, x, w, None, (2, 4, 4), 0, 1, 1)


# ---------------------------------------------------------------------------
# F.conv_transpose2d
# ---------------------------------------------------------------------------

class TestConvTranspose2d:
    """Off by default (see enable()'s comment) -- these use
    patched_with_unverified_kernels to exercise the dispatch logic anyway."""

    def test_calls_native_when_eligible(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 2, 4, 4)
        w = torch.randn(2, 3, 2, 2)
        native.conv_transpose2d.return_value = "native-result"
        got = F.conv_transpose2d(x, w, stride=2)
        assert got == "native-result"
        # wrapper forwards (input, weight, bias, stride, padding,
        # output_padding, dilation, groups) to the native kernel.
        assert_called_once_with_tensors(
            native.conv_transpose2d, x, w, None, 2, 0, 0, 1, 1
        )

    def test_falls_back_when_groups_not_1(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 4, 4, 4)
        w = torch.randn(4, 3, 2, 2)
        got = F.conv_transpose2d(x, w, stride=2, groups=2)
        native.conv_transpose2d.assert_not_called()
        assert got.shape[1] == 6

    def test_falls_back_when_not_cuda(self, patched_with_unverified_kernels, native):
        x = torch.randn(1, 2, 4, 4)
        w = torch.randn(2, 3, 2, 2)
        F.conv_transpose2d(x, w, stride=2)
        native.conv_transpose2d.assert_not_called()

    def test_falls_back_on_native_runtime_error(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        native.conv_transpose2d.side_effect = RuntimeError("boom")
        x = torch.randn(1, 2, 4, 4)
        w = torch.randn(2, 3, 2, 2)
        got = F.conv_transpose2d(x, w, stride=2)
        native.conv_transpose2d.assert_called_once()
        assert got.shape == (1, 3, 8, 8)


# ---------------------------------------------------------------------------
# F.interpolate
# ---------------------------------------------------------------------------

class TestInterpolate:
    """Off by default (see enable()'s comment) -- these use
    patched_with_unverified_kernels to exercise the dispatch logic anyway."""

    def test_calls_native_with_size(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 1, 2, 2)
        native.upsample_scaling.return_value = "native-result"
        got = F.interpolate(x, size=(4, 4))
        assert got == "native-result"
        assert_called_once_with_tensors(native.upsample_scaling, x, (4, 4))

    def test_calls_native_with_scale_factor(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 1, 2, 2)
        native.upsample_scaling.return_value = "native-result"
        got = F.interpolate(x, scale_factor=2)
        assert got == "native-result"
        assert_called_once_with_tensors(native.upsample_scaling, x, 2)

    def test_falls_back_on_non_nearest_mode(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 1, 4, 4)
        F.interpolate(x, size=(8, 8), mode="bilinear")
        native.upsample_scaling.assert_not_called()

    def test_falls_back_on_3d_input(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 1, 4)
        F.interpolate(x, size=(8,))
        native.upsample_scaling.assert_not_called()

    def test_falls_back_when_align_corners_given(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(1, 1, 4, 4)
        # align_corners is only valid for non-nearest modes, so falling back
        # to stock with mode="nearest" surfaces stock's own ValueError --
        # the point of this test is that the native kernel is never called.
        with pytest.raises(ValueError):
            F.interpolate(x, size=(8, 8), mode="nearest", align_corners=False)
        native.upsample_scaling.assert_not_called()

    def test_falls_back_when_not_cuda(self, patched_with_unverified_kernels, native):
        x = torch.randn(1, 1, 2, 2)
        F.interpolate(x, size=(4, 4))
        native.upsample_scaling.assert_not_called()

    def test_falls_back_on_native_runtime_error(self, patched_with_unverified_kernels, native, monkeypatch):
        force_eligible(monkeypatch)
        native.upsample_scaling.side_effect = RuntimeError("boom")
        x = torch.randn(1, 1, 2, 2)
        got = F.interpolate(x, size=(4, 4))
        native.upsample_scaling.assert_called_once()
        assert got.shape == (1, 1, 4, 4)


# ---------------------------------------------------------------------------
# F.scaled_dot_product_attention
# ---------------------------------------------------------------------------

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


@pytest.mark.skipif(not _HAS_SDPA, reason="torch build has no SDPA")
class TestScaledDotProductAttention:
    def test_calls_native_when_eligible(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        native.attention.return_value = "native-result"
        got = F.scaled_dot_product_attention(q, k, v)
        assert got == "native-result"
        assert_called_once_with_tensors(native.attention, q, k, v, None)

    def test_falls_back_when_attn_mask_given(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        mask = torch.zeros(3, 3, dtype=torch.bool)
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        native.attention.assert_not_called()

    def test_falls_back_when_dropout_nonzero(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        native.attention.assert_not_called()

    def test_falls_back_when_is_causal(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        native.attention.assert_not_called()

    def test_falls_back_on_non_4d_query(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        q = torch.randn(2, 3, 4)
        k = torch.randn(2, 3, 4)
        v = torch.randn(2, 3, 4)
        F.scaled_dot_product_attention(q, k, v)
        native.attention.assert_not_called()

    def test_falls_back_when_not_cuda(self, patched, native):
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        F.scaled_dot_product_attention(q, k, v)
        native.attention.assert_not_called()

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        native.attention.side_effect = RuntimeError("boom")
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        got = F.scaled_dot_product_attention(q, k, v)
        native.attention.assert_called_once()
        assert got.shape == q.shape


# ---------------------------------------------------------------------------
# F.embedding
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_calls_native_defaults_padding_idx_to_negative_one(
        self, patched, native, monkeypatch
    ):
        force_eligible(monkeypatch)
        idx = torch.tensor([0, 2, 1], dtype=torch.long)
        weight = torch.randn(5, 4)
        native.embedding.return_value = "native-result"
        got = F.embedding(idx, weight)
        assert got == "native-result"
        assert_called_once_with_tensors(native.embedding, idx, weight, -1, False, False)

    def test_calls_native_with_explicit_padding_idx(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        idx = torch.tensor([0, 2, 1], dtype=torch.long)
        weight = torch.randn(5, 4)
        native.embedding.return_value = "native-result"
        F.embedding(idx, weight, padding_idx=3)
        assert_called_once_with_tensors(native.embedding, idx, weight, 3, False, False)

    def test_falls_back_when_max_norm_given(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        idx = torch.tensor([0, 1], dtype=torch.long)
        weight = torch.randn(5, 4)
        F.embedding(idx, weight, max_norm=1.0)
        native.embedding.assert_not_called()

    def test_falls_back_when_not_cuda(self, patched, native):
        idx = torch.tensor([0, 1], dtype=torch.long)
        weight = torch.randn(5, 4)
        got = F.embedding(idx, weight)
        native.embedding.assert_not_called()
        assert torch.equal(got, weight[idx])

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        native.embedding.side_effect = RuntimeError("boom")
        idx = torch.tensor([0, 1], dtype=torch.long)
        weight = torch.randn(5, 4)
        got = F.embedding(idx, weight)
        native.embedding.assert_called_once()
        assert torch.equal(got, weight[idx])


# ---------------------------------------------------------------------------
# F.group_norm / F.layer_norm
# ---------------------------------------------------------------------------

class TestGroupNorm:
    def test_calls_native_when_eligible(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 3, 3)
        assert x.is_contiguous()
        native.group_norm.return_value = "native-result"
        got = F.group_norm(x, num_groups=2)
        assert got == "native-result"
        assert_called_once_with_tensors(native.group_norm, x, 2, None, None, 1e-5)

    def test_falls_back_on_non_contiguous_input(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 4, 3, 3).transpose(2, 3)
        assert not x.is_contiguous()
        F.group_norm(x, num_groups=2)
        native.group_norm.assert_not_called()

    def test_falls_back_when_not_cuda(self, patched, native):
        x = torch.randn(2, 4, 3, 3)
        F.group_norm(x, num_groups=2)
        native.group_norm.assert_not_called()

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        native.group_norm.side_effect = RuntimeError("boom")
        x = torch.randn(2, 4, 3, 3)
        got = F.group_norm(x, num_groups=2)
        native.group_norm.assert_called_once()
        assert got.shape == x.shape


class TestLayerNorm:
    def test_calls_native_with_normalized_shape_as_list(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 3, 4)
        assert x.is_contiguous()
        native.layer_norm.return_value = "native-result"
        got = F.layer_norm(x, (4,))
        assert got == "native-result"
        assert_called_once_with_tensors(native.layer_norm, x, [4], None, None, 1e-5)

    def test_falls_back_on_non_contiguous_input(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        x = torch.randn(2, 3, 4).transpose(0, 1)
        assert not x.is_contiguous()
        F.layer_norm(x, (4,))
        native.layer_norm.assert_not_called()

    def test_falls_back_when_not_cuda(self, patched, native):
        x = torch.randn(2, 3, 4)
        F.layer_norm(x, (4,))
        native.layer_norm.assert_not_called()

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch):
        force_eligible(monkeypatch)
        native.layer_norm.side_effect = RuntimeError("boom")
        x = torch.randn(2, 3, 4)
        got = F.layer_norm(x, (4,))
        native.layer_norm.assert_called_once()
        assert got.shape == x.shape


# ---------------------------------------------------------------------------
# Elementwise ops installed via _make_elementwise_patch (gelu/silu/softsign,
# and mish when present in this torch build).
# ---------------------------------------------------------------------------

_ELEMENTWISE_OPS = ["gelu", "silu", "softsign"] + (["mish"] if hasattr(F, "mish") else [])


@pytest.mark.parametrize("op_name", _ELEMENTWISE_OPS)
class TestElementwise:
    def test_calls_native_when_eligible(self, patched, native, monkeypatch, op_name):
        force_eligible(monkeypatch)
        x = torch.randn(8)
        native_fn = getattr(native, op_name)
        native_fn.return_value = "native-result"
        got = getattr(F, op_name)(x)
        assert got == "native-result"
        assert_called_once_with_tensors(native_fn, x)

    def test_falls_back_when_not_cuda(self, patched, native, op_name):
        x = torch.randn(8)
        got = getattr(F, op_name)(x)
        getattr(native, op_name).assert_not_called()
        assert torch.equal(got, cmpext3._ORIGINALS[(F, op_name)](x))

    def test_falls_back_on_native_runtime_error(self, patched, native, monkeypatch, op_name):
        force_eligible(monkeypatch)
        native_fn = getattr(native, op_name)
        native_fn.side_effect = RuntimeError("boom")
        x = torch.randn(8)
        got = getattr(F, op_name)(x)
        native_fn.assert_called_once()
        assert torch.equal(got, cmpext3._ORIGINALS[(F, op_name)](x))

    def test_falls_back_on_native_type_error(self, patched, native, monkeypatch, op_name):
        # Elementwise wrappers additionally catch TypeError (e.g. an
        # unexpected kwarg the native kernel doesn't accept), unlike the
        # other wrappers which only catch RuntimeError.
        force_eligible(monkeypatch)
        native_fn = getattr(native, op_name)
        native_fn.side_effect = TypeError("unexpected kwarg")
        x = torch.randn(8)
        got = getattr(F, op_name)(x)
        native_fn.assert_called_once()
        assert torch.equal(got, cmpext3._ORIGINALS[(F, op_name)](x))
