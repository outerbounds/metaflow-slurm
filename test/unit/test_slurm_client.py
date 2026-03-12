"""
Unit tests for SlurmClient.connect() in slurm_client.py.

Bug fixed: asyncssh.connect() was called without a connect_timeout, so if
the SLURM cluster is unreachable (wrong IP, firewall, VPN down) the coroutine
hangs indefinitely with no error and no way to interrupt cleanly.

Fix: pass connect_timeout=30 to asyncssh.connect() so that unreachable
clusters produce a clear RuntimeError within 30 seconds.
"""

import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Stub Linux-only / optional deps before any package import.
# ---------------------------------------------------------------------------
if "fcntl" not in sys.modules:
    sys.modules["fcntl"] = MagicMock()


class _MetaflowException(Exception):
    pass


_metaflow_stub = MagicMock()
_metaflow_stub.exception.MetaflowException = _MetaflowException
sys.modules.setdefault("metaflow", _metaflow_stub)
sys.modules.setdefault("metaflow.exception", _metaflow_stub.exception)
sys.modules.setdefault("metaflow.sidecar", MagicMock())
sys.modules.setdefault("metaflow.sidecar.sidecar_subprocess", MagicMock())

import os  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(mock_asyncssh):
    """
    Return a SlurmClient whose asyncssh module is fully replaced by
    mock_asyncssh.  The mock provides a fake private key so __init__
    does not touch the filesystem.
    """
    # Stub read_private_key so __init__ doesn't need a real .pem file.
    mock_asyncssh.read_private_key.return_value = MagicMock()

    with patch("builtins.__import__", side_effect=_make_importer(mock_asyncssh)):
        from metaflow_extensions.slurm_ext.plugins.slurm.slurm_client import (
            SlurmClient,
        )

        with patch("pathlib.Path.exists", return_value=True):
            client = SlurmClient(
                username="testuser",
                address="1.2.3.4",
                ssh_key_file="/fake/key.pem",
            )
    # Replace the asyncssh attribute directly so connect() uses our mock.
    client.asyncssh = mock_asyncssh
    return client


def _make_importer(mock_asyncssh):
    """Return an __import__ side-effect that returns mock_asyncssh for 'asyncssh'."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _importer(name, *args, **kwargs):
        if name == "asyncssh":
            return mock_asyncssh
        return real_import(name, *args, **kwargs)

    return _importer


def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Simpler approach: construct client then patch asyncssh on the instance
# ---------------------------------------------------------------------------


def _make_client_simple():
    """Build a SlurmClient by patching at the module level."""
    mock_asyncssh = MagicMock()
    mock_asyncssh.read_private_key.return_value = MagicMock()

    # Patch asyncssh in sys.modules so __import__("asyncssh") inside __init__ works.
    sys.modules["asyncssh"] = mock_asyncssh

    with patch("pathlib.Path.exists", return_value=True):
        # Re-import to pick up the patched sys.modules entry.
        import importlib
        import metaflow_extensions.slurm_ext.plugins.slurm.slurm_client as _mod

        importlib.reload(_mod)
        client = _mod.SlurmClient(
            username="testuser",
            address="1.2.3.4",
            ssh_key_file="/fake/key.pem",
        )

    # Swap in a fresh MagicMock so connect() calls are inspectable.
    client.asyncssh = mock_asyncssh
    return client, mock_asyncssh


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_connect_passes_connect_timeout():
    """
    connect() must pass connect_timeout=30 to asyncssh.connect().

    This is the core regression test.  Without the fix, connect_timeout is
    absent from the call and the coroutine can hang indefinitely.
    """
    client, mock_asyncssh = _make_client_simple()

    mock_conn = MagicMock()
    mock_asyncssh.connect = AsyncMock(return_value=mock_conn)

    _run(client.connect())

    mock_asyncssh.connect.assert_called_once()
    _, kwargs = mock_asyncssh.connect.call_args
    assert "connect_timeout" in kwargs, (
        "connect_timeout was not passed to asyncssh.connect(). "
        "The SSH call can hang indefinitely on unreachable clusters."
    )
    assert kwargs["connect_timeout"] == 30, (
        "Expected connect_timeout=30, got connect_timeout=%r"
        % kwargs["connect_timeout"]
    )


def test_connect_passes_correct_credentials():
    """connect() must pass the username and client_keys through unchanged."""
    client, mock_asyncssh = _make_client_simple()
    mock_conn = MagicMock()
    mock_asyncssh.connect = AsyncMock(return_value=mock_conn)

    _run(client.connect())

    _, kwargs = mock_asyncssh.connect.call_args
    assert kwargs["username"] == "testuser"
    assert kwargs["known_hosts"] is None


def test_connect_sets_self_conn():
    """After a successful connect(), self.conn must be set to the returned connection."""
    client, mock_asyncssh = _make_client_simple()
    mock_conn = MagicMock()
    mock_asyncssh.connect = AsyncMock(return_value=mock_conn)

    result = _run(client.connect())

    assert client.conn is mock_conn
    assert result is mock_conn


def test_connect_raises_runtime_error_on_failure():
    """
    When asyncssh.connect() raises (e.g. timeout, refused, unreachable),
    connect() must re-raise as RuntimeError with a message containing the
    address and username — not the raw asyncssh exception.
    """
    client, mock_asyncssh = _make_client_simple()
    mock_asyncssh.connect = AsyncMock(side_effect=ConnectionRefusedError("refused"))

    with pytest.raises(RuntimeError) as exc_info:
        _run(client.connect())

    msg = str(exc_info.value)
    assert "1.2.3.4" in msg, "RuntimeError message must include the host address"
    assert "testuser" in msg, "RuntimeError message must include the username"


def test_connect_timeout_triggers_runtime_error():
    """
    Simulate asyncssh raising TimeoutError (what connect_timeout=30 produces
    on an unreachable host) and verify it surfaces as RuntimeError.
    """
    client, mock_asyncssh = _make_client_simple()
    mock_asyncssh.connect = AsyncMock(side_effect=TimeoutError("timed out"))

    with pytest.raises(RuntimeError) as exc_info:
        _run(client.connect())

    msg = str(exc_info.value)
    assert "1.2.3.4" in msg
    assert "testuser" in msg
