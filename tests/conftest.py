"""Pytest fixtures shared by the cmpext3 monkeypatch tests.

``cmpext3/__init__.py`` hard-imports its compiled CUDA extension
(``cmpext3._native``) at module load time and raises ``ImportError`` if it
isn't present. Building that extension requires nvcc and a CUDA toolchain,
neither of which is available on every machine that runs these tests, so
before ``cmpext3`` is ever imported we install a fake ``cmpext3._native``
module into ``sys.modules`` whose every kernel is a ``MagicMock``. This lets
the *Python-level dispatch logic* (grad-safety checks, dtype/shape
eligibility, fallback-on-error) be tested anywhere, independent of whether a
real GPU or compiled kernel is available.

``CMPEXT3_AUTOPATCH`` is forced to ``"0"`` before import so importing the
package doesn't silently patch global ``torch``/``torch.nn.functional``
state as a side effect of collecting these tests; individual tests opt in
via the ``patched`` fixture instead.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Every kernel exported by src/main.cpp / cmpext3.ops.
NATIVE_OPS = [
    "linear", "bmm", "conv2d", "conv3d", "conv_transpose2d", "upsample_scaling",
    "attention", "embedding", "group_norm", "layer_norm", "rmsnorm",
    "gelu", "silu", "swish", "mish", "softmax", "softplus", "softsign",
    "softshrink", "tanh", "erf",
]

os.environ.setdefault("CMPEXT3_AUTOPATCH", "0")

if "cmpext3._native" not in sys.modules:
    _fake_native = types.ModuleType("cmpext3._native")
    for _name in NATIVE_OPS:
        setattr(_fake_native, _name, MagicMock(name=_name))
    sys.modules["cmpext3._native"] = _fake_native

import cmpext3  # noqa: E402 -- must come after the sys.modules stub above
import torch.nn.functional as F  # noqa: E402

# Snapshot of the functions cmpext3 deliberately never patches, captured
# before any test has a chance to call enable().
STOCK_UNPATCHED = {
    (F, "softmax"): F.softmax,
    (F, "softplus"): F.softplus,
    (F, "softshrink"): F.softshrink,
}
import torch  # noqa: E402
STOCK_UNPATCHED[(torch, "tanh")] = torch.tanh
STOCK_UNPATCHED[(torch, "erf")] = torch.erf


@pytest.fixture(autouse=True)
def no_autoselect():
    """Turn the runtime kernel choice OFF for every test in this suite.

    cmpext3/autoselect.py decides native-vs-stock by actually RUNNING both and
    timing them. That is exactly wrong for these tests: the native side is a
    MagicMock returning a sentinel string, the stock side is real PyTorch on
    CPU tensors, and "which is faster" is meaningless -- the wrapper would
    dispatch to stock and every "did it reach the native kernel" assertion
    would fail for reasons that have nothing to do with the dispatch logic
    under test. With it off, _dispatch collapses to "native, stock only on
    exception", which is the routing these tests are about.

    autoselect's own logic is covered separately, by TestAutoselect, which
    opts back in via the `autoselecting` fixture.
    """
    from cmpext3 import autoselect
    previous = autoselect._ENABLED
    autoselect._ENABLED = False
    autoselect.reset()
    try:
        yield
    finally:
        autoselect._ENABLED = previous
        autoselect.reset()


@pytest.fixture
def autoselecting():
    """Turn the runtime kernel choice back on (see `no_autoselect`)."""
    from cmpext3 import autoselect
    autoselect._ENABLED = True
    autoselect.reset()
    try:
        yield autoselect
    finally:
        autoselect._ENABLED = False
        autoselect.reset()


@pytest.fixture
def native():
    """The fake native module (``cmpext3.ops``), reset before each test."""
    for name in NATIVE_OPS:
        getattr(cmpext3.ops, name).reset_mock(return_value=True, side_effect=True)
    return cmpext3.ops


@pytest.fixture
def patched():
    """Applies cmpext3.enable() for the test, then always reverts it."""
    assert not cmpext3.is_enabled(), "a previous test left cmpext3 enabled"
    cmpext3.enable()
    try:
        yield cmpext3
    finally:
        cmpext3.disable()


@pytest.fixture
def patched_with_unverified_kernels(monkeypatch):
    """Historical name for `patched`, kept so the conv3d/conv_transpose2d/
    interpolate tests that use it keep reading sensibly.

    Those three ops used to be excluded from enable() unless
    CMPEXT3_ENABLE_UNVERIFIED_KERNELS=1 was set. They are installed
    unconditionally now -- autoselect measures each of them against stock per
    shape instead of anyone having to guess up front (see enable()'s comment
    in cmpext3/__init__.py) -- so this fixture is equivalent to `patched`.
    The env var is still set here because it is still *read*: an explicit "0"
    remains an opt-out, and setting "1" documents which ops these tests are
    about.
    """
    assert not cmpext3.is_enabled(), "a previous test left cmpext3 enabled"
    monkeypatch.setenv("CMPEXT3_ENABLE_UNVERIFIED_KERNELS", "1")
    cmpext3.enable()
    try:
        yield cmpext3
    finally:
        cmpext3.disable()


def force_eligible(monkeypatch):
    """Makes every wrapper's grad-safety/dtype/device gate pass.

    cmpext3's eligibility checks (`_usable`) require a real CUDA fp16/fp32/bf16
    tensor. Forcing them to pass lets the *rest* of each wrapper's dispatch
    logic (shape/argument-specific fallbacks, native-call wiring, error
    fallback) be exercised with plain CPU tensors on any machine.
    """
    monkeypatch.setattr(cmpext3, "_usable", lambda *a, **k: True)
    monkeypatch.setattr(cmpext3, "_grad_safe", lambda *a, **k: True)
