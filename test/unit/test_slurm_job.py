"""
Unit tests for SlurmJob fluent setter methods in slurm_job.py.

Bug fixed: cpus_per_task(), memory(), and run_time_limit() wrote the wrong
keys into self.kwargs, so create_slurm_script() always saw None for those
fields and silently omitted them from the generated #SBATCH directives.

Root cause:
    cpus_per_task()  wrote kwargs["cpus-per-task"]  instead of "cpus_per_task"
    memory()         wrote kwargs["mem"]             instead of "memory"
    run_time_limit() wrote kwargs["time"]            instead of "run_time_limit"

create_slurm_script() reads "cpus_per_task", "memory", and "run_time_limit"
from kwargs — so all three values were silently dropped when set via the
fluent API.
"""

import sys
import os
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Make the package importable without installing it.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# asyncssh is an optional runtime dep — stub it so tests run without it.
if "asyncssh" not in sys.modules:
    sys.modules["asyncssh"] = MagicMock()


# The import chain slurm_job → slurm_client → slurm_exceptions →
# metaflow.exception triggers metaflow's full __init__.py, which runs its
# plugin loader and fails on Windows (fcntl, Linux-targeting cloud plugins).
# Stub just enough of metaflow to let the exception class be importable
# without pulling in the entire plugin system.
class _MetaflowException(Exception):
    pass


_metaflow_stub = MagicMock()
_metaflow_stub.exception.MetaflowException = _MetaflowException

sys.modules.setdefault("metaflow", _metaflow_stub)
sys.modules.setdefault("metaflow.exception", _metaflow_stub.exception)
sys.modules.setdefault("fcntl", MagicMock())
sys.modules.setdefault("metaflow.sidecar", MagicMock())
sys.modules.setdefault("metaflow.sidecar.sidecar_subprocess", MagicMock())

from metaflow_extensions.slurm_ext.plugins.slurm.slurm_job import SlurmJob  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(**extra_kwargs):
    """Return a SlurmJob with a minimal mock client."""
    mock_client = MagicMock()
    mock_client.cleanup = False
    mock_client.remote_workdir = "~/metaflow"
    return SlurmJob(
        client=mock_client,
        name="testjob",
        command=["python", "-c 'pass'"],
        loop=MagicMock(),
        **extra_kwargs,
    )


# ---------------------------------------------------------------------------
# Bug regression: kwargs keys written by setter methods
# ---------------------------------------------------------------------------


def test_cpus_per_task_setter_writes_correct_key():
    """
    .cpus_per_task(4) must write kwargs["cpus_per_task"], not "cpus-per-task".

    Fails without fix (wrong key silently discards the value),
    passes with fix.
    """
    job = _make_job()
    job.cpus_per_task(4)
    assert "cpus_per_task" in job.kwargs, (
        "cpus_per_task() wrote the wrong key into kwargs. "
        "Expected 'cpus_per_task', got keys: %s" % list(job.kwargs.keys())
    )
    assert job.kwargs["cpus_per_task"] == 4
    # The old wrong key must not be present.
    assert "cpus-per-task" not in job.kwargs


def test_memory_setter_writes_correct_key():
    """
    .memory("8G") must write kwargs["memory"], not "mem".

    Fails without fix, passes with fix.
    """
    job = _make_job()
    job.memory("8G")
    assert "memory" in job.kwargs, (
        "memory() wrote the wrong key into kwargs. "
        "Expected 'memory', got keys: %s" % list(job.kwargs.keys())
    )
    assert job.kwargs["memory"] == "8G"
    assert "mem" not in job.kwargs


def test_run_time_limit_setter_writes_correct_key():
    """
    .run_time_limit(60) must write kwargs["run_time_limit"], not "time".

    Fails without fix, passes with fix.
    """
    job = _make_job()
    job.run_time_limit(60)
    assert "run_time_limit" in job.kwargs, (
        "run_time_limit() wrote the wrong key into kwargs. "
        "Expected 'run_time_limit', got keys: %s" % list(job.kwargs.keys())
    )
    assert job.kwargs["run_time_limit"] == 60
    assert "time" not in job.kwargs


# ---------------------------------------------------------------------------
# End-to-end: verify the generated #SBATCH directives after fluent API calls
# ---------------------------------------------------------------------------


def test_create_slurm_script_includes_cpus_per_task_directive():
    """
    After .cpus_per_task(4).create_slurm_script(), the generated script
    must contain '#SBATCH --cpus-per-task=4'.

    Without the fix the directive is silently absent.
    """
    job = _make_job()
    job.cpus_per_task(4).create_slurm_script()
    script = job.slurm_job_script.generate_script(command="python -c 'pass'")
    assert "#SBATCH --cpus-per-task=4" in script, (
        "Expected '#SBATCH --cpus-per-task=4' in generated script.\n"
        "Actual sbatch_options: %s" % job.slurm_job_script.sbatch_options
    )


def test_create_slurm_script_includes_mem_directive():
    """
    After .memory("8G").create_slurm_script(), the generated script
    must contain '#SBATCH --mem=8G'.
    """
    job = _make_job()
    job.memory("8G").create_slurm_script()
    script = job.slurm_job_script.generate_script(command="python -c 'pass'")
    assert "#SBATCH --mem=8G" in script, (
        "Expected '#SBATCH --mem=8G' in generated script.\n"
        "Actual sbatch_options: %s" % job.slurm_job_script.sbatch_options
    )


def test_create_slurm_script_includes_time_directive():
    """
    After .run_time_limit(7200).create_slurm_script(), the generated script
    must contain '#SBATCH --time=7200'.
    """
    job = _make_job()
    job.run_time_limit(7200).create_slurm_script()
    script = job.slurm_job_script.generate_script(command="python -c 'pass'")
    assert "#SBATCH --time=7200" in script, (
        "Expected '#SBATCH --time=7200' in generated script.\n"
        "Actual sbatch_options: %s" % job.slurm_job_script.sbatch_options
    )


def test_create_slurm_script_chained_fluent_api():
    """
    Chaining all three setters at once must produce all three directives.
    This is the realistic usage pattern and the most important integration check.
    """
    job = _make_job()
    job.cpus_per_task(8).memory("16G").run_time_limit(7200).create_slurm_script()
    script = job.slurm_job_script.generate_script(command="python -c 'pass'")

    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=7200" in script


def test_kwargs_constructor_path_still_works():
    """
    Values supplied directly via **kwargs to SlurmJob.__init__ (the path
    used by SlurmDecorator) must still be picked up correctly by
    create_slurm_script() — this path was not broken by the bug.
    """
    job = _make_job(cpus_per_task=2, memory="4G", run_time_limit=3600)
    job.create_slurm_script()
    script = job.slurm_job_script.generate_script(command="python -c 'pass'")

    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --mem=4G" in script
    assert "#SBATCH --time=3600" in script
