"""Shared pytest fixtures.

`research_pipeline.checkpointer` memoizes its checkpointer and its node cache
for the life of the process — deliberately, so the eight graphs (each rebuilt
per agent call) share one connection and one cache instead of one per build.
That is right for a pipeline run and wrong for a test session, which is many
independent "processes" pretending to share one: the literature and
interdisciplinary graphs cache their paper-search nodes on the node's input
state, so without this reset one test's canned search results are served to the
next test that happens to invoke with the same input, and the fakes it injected
never run.

`research_pipeline.agents.coder.fix_pattern_store` memoizes its Store the same
way and needs the same reset, for the same isolation reason. It also needs one
thing checkpointer.py never did: CODER_FIX_STORE_BACKEND defaults to "sqlite",
not "memory" (deliberately — see that module's docstring), so unlike
CHECKPOINTER_BACKEND's already-safe default, a bare `pytest` invocation would
otherwise write a real coder_fix_patterns.db at the repo root the moment any
test constructs a CoderAgent without its own fix_store= override — which
several tests in test_coder_agent.py do (they're exercising unrelated
constructor params, e.g. SLURM settings, and have no reason to know this
setting exists). The environ override below must run before
research_pipeline.config is imported anywhere (settings is computed once, at
that module's own import time) — hence being the first thing in this file,
ahead of even the research_pipeline imports below it.
"""

import os

os.environ.setdefault("CODER_FIX_STORE_BACKEND", "memory")

import pytest  # noqa: E402 — see the module docstring for why this isn't at the top

from research_pipeline.agents.coder.fix_pattern_store import (  # noqa: E402
    reset_store as reset_fix_pattern_store,
)
from research_pipeline.checkpointer import reset_checkpointer  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_checkpointer_singletons():
    reset_checkpointer()
    reset_fix_pattern_store()
    yield
    reset_checkpointer()
    reset_fix_pattern_store()
