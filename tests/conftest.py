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
"""

import pytest

from research_pipeline.checkpointer import reset_checkpointer


@pytest.fixture(autouse=True)
def _isolate_checkpointer_singletons():
    reset_checkpointer()
    yield
    reset_checkpointer()
