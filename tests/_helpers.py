"""Small assertion helpers shared across the cmpext3 monkeypatch tests."""
from __future__ import annotations

import torch


def assert_called_once_with_tensors(mock_fn, *expected_args, **expected_kwargs):
    """Like ``mock.assert_called_once_with`` but safe for tensor arguments.

    ``Mock.assert_called_with`` compares call args with ``==``, which for
    multi-element tensors raises "the truth value of a tensor ... is
    ambiguous" instead of a useful assertion failure. Compare tensors with
    ``torch.equal`` and everything else with ``==``.
    """
    assert mock_fn.call_count == 1, (
        f"expected 1 call, got {mock_fn.call_count}"
    )
    got_args, got_kwargs = mock_fn.call_args
    assert len(got_args) == len(expected_args), (
        f"expected {len(expected_args)} positional args, got {len(got_args)}"
    )
    for got, want in zip(got_args, expected_args):
        if isinstance(want, torch.Tensor):
            assert isinstance(got, torch.Tensor) and torch.equal(got, want), (
                f"tensor arg mismatch: got {got!r} want {want!r}"
            )
        else:
            assert got == want, f"arg mismatch: got {got!r} want {want!r}"
    assert got_kwargs == expected_kwargs, (
        f"kwarg mismatch: got {got_kwargs!r} want {expected_kwargs!r}"
    )
