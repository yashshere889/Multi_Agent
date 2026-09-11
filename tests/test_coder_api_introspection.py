"""A module that lacks a name is answered with the names it really has.

Barkla job 10496057 failed two consecutive fix attempts on
`from SALib.sample import sample_saltelli`: the traceback said the name was
missing, never what to write instead, and the model wrote it again.
"""

import sys
from pathlib import Path

import pytest

from research_pipeline.agents.coder import diagnose, sandbox
from research_pipeline.agents.coder.coder_agent import CoderAgent

SALIB = (
    "Traceback (most recent call last):\n"
    '  File "run_smoke.py", line 43, in <module>\n'
    "    from SALib.sample import sample_saltelli\n"
    "ImportError: cannot import name 'sample_saltelli' from 'SALib.sample' "
    "(/tmp/venv/lib/python3.14/site-packages/SALib/sample/__init__.py)"
)


def test_cannot_import_name_is_parsed_to_module_and_name():
    assert diagnose.missing_module_name(SALIB) == ("SALib.sample", "sample_saltelli")


def test_a_module_attribute_error_is_parsed():
    text = "AttributeError: module 'scipy.stats' has no attribute 'ttest'"
    assert diagnose.missing_module_name(text) == ("scipy.stats", "ttest")


def test_an_object_attribute_error_is_not_a_missing_module_name():
    assert diagnose.missing_module_name("AttributeError: 'DataFrame' object has no attribute 'foo'") is None


def test_module_exports_reads_the_interpreter_it_is_given(tmp_path):
    exports = sandbox.module_exports(Path(sys.executable), "json", tmp_path)
    assert "dumps" in exports["names"]
    assert "decoder" in exports["submodules"]


def test_module_exports_is_none_for_a_module_that_is_not_installed(tmp_path):
    assert sandbox.module_exports(Path(sys.executable), "no_such_module_for_this_test", tmp_path) is None


def test_module_exports_refuses_anything_that_is_not_a_module_path(tmp_path):
    assert sandbox.module_exports(Path(sys.executable), "os; import sys", tmp_path) is None


def test_the_failure_summary_names_what_the_module_really_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sandbox,
        "module_exports",
        lambda _python, module, _cwd: {"names": ["saltelli", "sobol"], "submodules": ["latin", "saltelli", "sobol"]},
    )
    failure = diagnose.classify_execution_failure("run.py exited with code 1: " + SALIB)

    enriched = CoderAgent._with_real_exports(failure, SALIB, Path(sys.executable), tmp_path)

    assert "`SALib.sample` does not provide `sample_saltelli`" in enriched.summary
    assert "Its submodules: latin, saltelli, sobol" in enriched.summary
    assert enriched.summary.startswith(failure.summary)
    assert (enriched.route, enriched.error_source) == (failure.route, failure.error_source)


def test_a_removed_api_keeps_its_own_guidance_and_is_not_probed(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "module_exports", lambda *_args: pytest.fail("probed a removed API"))
    message = "AttributeError: module 'numpy' has no attribute 'float'"
    failure = diagnose.classify_execution_failure(message)

    assert CoderAgent._with_real_exports(failure, message, Path(sys.executable), tmp_path) is failure


def test_an_unprobeable_module_leaves_the_failure_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "module_exports", lambda *_args: None)
    failure = diagnose.classify_execution_failure(SALIB)

    assert CoderAgent._with_real_exports(failure, SALIB, Path(sys.executable), tmp_path) is failure
