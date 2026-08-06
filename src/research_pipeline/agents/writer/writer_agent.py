"""Writer Agent.

Sits at the end of the pipeline, after the Literature, Hypothesis, Experiment
Planner, and Coder Agents: takes their combined output and drafts a full
academic paper as a PDF, plus a structured JSON summary for review.

Input contract
--------------
`run(literature_output, hypothesis_output, planner_output, coder_output)`:
- `hypothesis_output`, `planner_output`, `coder_output` — the exact outputs of
  the corresponding upstream agents (or their `outputs/*.json` files loaded
  from disk). Validated against their own schemas on entry — `planner_output`
  and `coder_output` are also checked to have an entry for every hypothesis id
  in `hypothesis_output`. If any of these fail, WriterAgentError is raised
  immediately — drafting against a malformed input is never attempted.
- `literature_output` — the Literature Agent's papers, in whichever shape the
  pipeline already produces them: a dict with a "papers" key (the
  `metadata.json` written to disk) or a "merged_papers" key (the in-memory
  LangGraph state), or a bare list of paper dicts. No dedicated schema exists
  for this today (see agents/literature/state.py), so it isn't schema-validated
  — only checked to be one of these shapes.

Writing model
--------------
Every section's prose is drafted by the LLM, not hand-templated — see
prompts.py's module docstring for what's fixed (deterministic facts the model
is told to state accurately) vs. what's actually written by the model (the
prose itself). The paper is styled after a NeurIPS submission, including
NeurIPS's default author-year in-text citations: papers are cited with
`[[cite:PAPER_ID]]` (parenthetical) / `[[citet:PAPER_ID]]` (narrative)
markers restricted to the ids actually present in `literature_output`
(agents.writer.citations); these are mechanically resolved to "(Smith et al.,
2020)" / "Smith et al. (2020)" style citations and a matching References list
after every section is drafted — never invented by the model post hoc. A
hypothesis's outcome (supported/refuted/inconclusive) is computed
deterministically in `compute_hypothesis_verdict` from the Coder Agent's own
`meets_success_criteria` / `status` fields, then handed to the model as a fact
it must write consistently with, not a judgment it makes itself.

Generation happens section by section (Introduction, Related Work, Hypotheses,
then Methods/Results/Discussion per hypothesis, then Limitations, Future Work,
and finally Abstract/Title once the body's content is settled) so no single
LLM call ever needs the full pipeline output in context — each call gets only
the slice of upstream data relevant to that section.

Output contract
----------------
A dict matching `agents.writer.schema.WriterAgentOutput`:
    {
      "paper_path": str,               # the rendered PDF
      "sections_generated": [str, ...],
      "hypotheses_supported": [str, ...],    # hypothesis ids
      "hypotheses_refuted": [str, ...],
      "hypotheses_inconclusive": [str, ...],
      "citations_used": [str, ...],    # paper ids from literature_output, in citation order
      "notes_for_review": [str, ...],  # honesty/consistency flags from the validation pass
      "generated_at": "<UTC ISO 8601>",
      "model": str,
    }
Validated (agents.writer.schema.validate_output) before being returned. Also
written to `<output_dir>/paper_summary_<UTC timestamp>.json`. The PDF is
written to `<output_dir>/paper_<UTC timestamp>.pdf` regardless of the JSON
summary's validation outcome (a rendered paper is useful even if the summary
needs fixing); on a summary schema validation failure, the raw (invalid)
summary is still written — suffixed `_invalid` — and WriterAgentError is
raised.

Entry points
------------
    from research_pipeline.agents.writer import run_writer_agent
    result = run_writer_agent(literature_output, hypothesis_output, planner_output, coder_output)

Or, to reuse one configured model/output dir across multiple calls:
    agent = WriterAgent()
    result = agent.run(literature_output, hypothesis_output, planner_output, coder_output)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.coder.schema import SchemaValidationError as CoderSchemaValidationError
from research_pipeline.agents.coder.schema import validate_output as validate_coder_output
from research_pipeline.agents.experiment_planner.schema import SchemaValidationError as PlannerSchemaValidationError
from research_pipeline.agents.experiment_planner.schema import validate_output as validate_planner_output
from research_pipeline.agents.hypothesis.papers import chunk_papers, normalize_papers, paper_to_text
from research_pipeline.agents.hypothesis.schema import SchemaValidationError as HypothesisSchemaValidationError
from research_pipeline.agents.hypothesis.schema import validate_output as validate_hypothesis_output
from research_pipeline.agents.writer import pdf_reader, prompts
from research_pipeline.agents.writer.citations import CitationRegistry, build_paper_index
from research_pipeline.agents.writer.pdf_builder import build_pdf
from research_pipeline.agents.writer.schema import SchemaValidationError, validate_output
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import strip_fences

logger = logging.getLogger(__name__)

STATUS_MEANINGS = {
    "completed": "the experiment ran and produced results",
    "code_generated_not_run": "code was generated but never executed (e.g. needs more compute, or is left for manual SLURM submission)",
    "skipped": "no code was generated at all (the plan was marked infeasible)",
    "submitted_to_slurm": "code was submitted to the cluster as a batch job, but had not finished when this paper was written — no results are available",
}

class WriterAgentError(RuntimeError):
    """Raised when the agent can't produce a valid paper/summary, even after retries."""


def extract_literature_papers(literature_output: object) -> Tuple[List[dict], Optional[str]]:
    """Accepts whichever shape the pipeline already produces for literature
    output: a bare list of papers, a disk metadata.json dict ("papers" key),
    or the in-memory LangGraph state dict ("merged_papers" key)."""
    if isinstance(literature_output, list):
        return literature_output, None
    if isinstance(literature_output, dict):
        papers = literature_output.get("papers")
        if papers is None:
            papers = literature_output.get("merged_papers")
        return list(papers or []), literature_output.get("research_question")
    raise WriterAgentError(
        "literature_output must be a list of papers, or a dict with a 'papers' or 'merged_papers' key "
        f"— got {type(literature_output).__name__}"
    )


def compute_hypothesis_verdict(hypothesis_id: str, experiment_by_id: Dict[str, dict]) -> Tuple[str, str]:
    """Deterministically decides supported/refuted/inconclusive from the Coder
    Agent's own status/results — never left to the LLM to judge."""
    experiment = experiment_by_id.get(hypothesis_id)
    if experiment is None:
        return "inconclusive", "No Coder Agent record exists for this hypothesis."

    status = experiment["status"]
    if status in ("skipped", "code_generated_not_run", "submitted_to_slurm"):
        return "inconclusive", f"Experiment status is '{status}': {experiment['reason']}"

    meets = experiment["results"]["meets_success_criteria"]
    if meets is True:
        return "supported", "The Coder Agent reported meets_success_criteria=true."
    if meets is False:
        return "refuted", "The Coder Agent reported meets_success_criteria=false."
    return "inconclusive", 'The Coder Agent reported meets_success_criteria="unknown".'


class WriterAgent:
    def __init__(
        self,
        chat_model: Optional[BaseChatModel] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        # Reuses the pipeline's existing LLM client/config, same as the other
        # agents, at a low temperature suited to grounded, non-embellished prose.
        self.chat_model = chat_model or get_chat_model(temperature=0.2)
        self.output_dir = Path(output_dir or settings.writer_output_dir)

    def run(
        self,
        literature_output: object,
        hypothesis_output: dict,
        planner_output: dict,
        coder_output: dict,
        revision: Optional[dict] = None,
        paper_filename: Optional[str] = None,
        summary_filename: Optional[str] = None,
    ) -> dict:
        """revision, when given, turns this into a revision pass rather than a
        fresh draft: {"previous_sections": {heading: text}, "previous_subsections":
        {heading: {hypothesis_id: text}}, "feedback_by_section": {heading:
        [actionable bullet, ...]}} — see revise() below, which builds this from
        a prior paper + a Reviewer Agent review and is the intended way to call
        this. Every section-drafting prompt gets its own slice of this (its
        previous text + feedback addressed to it specifically, plus anything
        filed under the "__general__" key) rather than the model seeing the
        whole previous paper + full feedback dump on every call.
        paper_filename/summary_filename override the default timestamped
        names — the Writer/Reviewer loop uses this for "v1.pdf", "v2.pdf", ...
        """
        try:
            validate_hypothesis_output(hypothesis_output)
        except HypothesisSchemaValidationError as exc:
            raise WriterAgentError(f"hypothesis_output doesn't match the Hypothesis Agent's output schema: {exc}") from exc

        hypotheses = hypothesis_output["hypotheses"]
        expected_ids = [h["id"] for h in hypotheses]

        try:
            validate_planner_output(planner_output, expected_hypothesis_ids=expected_ids)
        except PlannerSchemaValidationError as exc:
            raise WriterAgentError(f"planner_output doesn't match the Experiment Planner's output schema: {exc}") from exc

        try:
            validate_coder_output(coder_output, expected_hypothesis_ids=expected_ids)
        except CoderSchemaValidationError as exc:
            raise WriterAgentError(f"coder_output doesn't match the Coder Agent's output schema: {exc}") from exc

        raw_papers, _research_question = extract_literature_papers(literature_output)
        paper_index = build_paper_index(raw_papers)
        valid_paper_ids = list(paper_index.keys())
        registry = CitationRegistry(paper_index)
        logger.info("Drafting paper for %d hypotheses, %d indexed source paper(s)", len(expected_ids), len(paper_index))

        plan_by_id = {p["hypothesis_id"]: p for p in planner_output["experiment_plans"]}
        experiment_by_id = {e["hypothesis_id"]: e for e in coder_output["experiments"]}
        verdicts = {
            hid: {
                "hypothesis_id": hid,
                "statement": next(h["statement"] for h in hypotheses if h["id"] == hid),
                "verdict": (v := compute_hypothesis_verdict(hid, experiment_by_id))[0],
                "reason": v[1],
            }
            for hid in expected_ids
        }

        introduction = self._draft_introduction(hypothesis_output, hypotheses, valid_paper_ids, revision)
        related_work = self._draft_related_work(raw_papers, hypothesis_output, hypotheses, revision)
        hypotheses_section = self._draft_hypotheses_section(hypotheses, valid_paper_ids, revision)
        methods_by_id, results_by_id, discussion_by_id = self._draft_per_hypothesis_sections(
            hypotheses, plan_by_id, experiment_by_id, verdicts, revision
        )
        limitations = self._draft_limitations(hypotheses, plan_by_id, experiment_by_id, revision)
        future_work = self._draft_future_work(verdicts, expected_ids, hypothesis_output.get("gaps", []), hypotheses, revision)

        raw_sections = [
            ("Introduction", introduction),
            ("Related Work", related_work),
            ("Hypotheses", hypotheses_section),
            ("Methods", self._join_per_hypothesis(expected_ids, methods_by_id)),
            ("Results", self._join_per_hypothesis(expected_ids, results_by_id)),
            ("Discussion", self._join_per_hypothesis(expected_ids, discussion_by_id, verdicts)),
            ("Limitations", limitations),
            ("Future Work", future_work),
        ]
        raw_abstract = self._draft_abstract(hypothesis_output.get("literature_summary", ""), verdicts, expected_ids, revision)
        raw_title = self._draft_title(hypothesis_output.get("literature_summary", ""), verdicts, expected_ids, revision)

        # Author-year disambiguation (e.g. "2020a"/"2020b") needs the full set
        # of cited papers before any single marker can be formatted, so every
        # drafted text is scanned first, then the registry is finalized once,
        # then every text is resolved — see citations.py's module docstring.
        for heading, body in raw_sections:
            registry.scan(body, section=heading)
        registry.scan(raw_abstract, section="Abstract")
        registry.scan(raw_title, section="Title")
        registry.finalize()

        resolved_sections = [(heading, registry.resolve(body, section=heading)) for heading, body in raw_sections]
        abstract = registry.resolve(raw_abstract, section="Abstract")
        title = registry.resolve(raw_title, section="Title")

        references = registry.references()
        notes_for_review = self._validate_paper(resolved_sections, verdicts, experiment_by_id, registry)

        paper_path = self._render_pdf(title, abstract, resolved_sections, references, filename=paper_filename)

        result: dict = {
            "paper_path": str(paper_path),
            "sections_generated": ["Title", "Abstract"] + [heading for heading, _ in resolved_sections] + ["References"],
            "hypotheses_supported": [hid for hid in expected_ids if verdicts[hid]["verdict"] == "supported"],
            "hypotheses_refuted": [hid for hid in expected_ids if verdicts[hid]["verdict"] == "refuted"],
            "hypotheses_inconclusive": [hid for hid in expected_ids if verdicts[hid]["verdict"] == "inconclusive"],
            "citations_used": registry.citation_order(),
            "notes_for_review": notes_for_review,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": settings.llm_model,
        }

        try:
            validate_output(result)
        except SchemaValidationError as exc:
            debug_path = self._write_summary(result, suffix="_invalid")
            raise WriterAgentError(
                f"Assembled output failed schema validation: {exc}. Raw (invalid) output written to {debug_path} "
                f"for inspection. The rendered paper at {paper_path} is unaffected."
            ) from exc

        summary_path = self._write_summary(result, filename=summary_filename)
        logger.info("Wrote writer agent summary to %s (paper: %s)", summary_path, paper_path)
        return result

    def revise(
        self,
        literature_output: object,
        hypothesis_output: dict,
        planner_output: dict,
        coder_output: dict,
        previous_paper_path: str | Path,
        previous_summary: dict,
        feedback_by_section: Dict[str, List[str]],
        paper_filename: Optional[str] = None,
        summary_filename: Optional[str] = None,
    ) -> dict:
        """Produces a revised draft addressing Reviewer Agent feedback.
        Extracts the previous PDF's own text back out (agents.writer.pdf_reader
        — the same module the paper was rendered through, so headings/labels
        round-trip exactly) and re-drafts every section with that prior text
        plus the feedback filed against it as additional grounding context —
        see run()'s `revision` parameter. feedback_by_section maps a heading
        (matching `previous_summary["sections_generated"]`, e.g. "Introduction",
        "Results") to a list of actionable bullet strings; a "__general__" key
        is applied to every section (e.g. paper-wide quality-score feedback).
        Writer_reviewer_loop.py builds this from a ReviewerAgent review — see
        its `route_feedback_to_sections`."""
        previous_text = pdf_reader.extract_text(previous_paper_path)
        previous_sections = pdf_reader.split_into_sections(previous_text, previous_summary["sections_generated"])

        hypothesis_ids = [h["id"] for h in hypothesis_output["hypotheses"]]
        previous_subsections = {
            heading: pdf_reader.split_hypothesis_subsections(previous_sections[heading], hypothesis_ids)
            for heading in ("Methods", "Results", "Discussion")
            if heading in previous_sections
        }

        revision = {
            "previous_sections": previous_sections,
            "previous_subsections": previous_subsections,
            "feedback_by_section": feedback_by_section,
        }
        return self.run(
            literature_output,
            hypothesis_output,
            planner_output,
            coder_output,
            revision=revision,
            paper_filename=paper_filename,
            summary_filename=summary_filename,
        )

    # -- Section drafting ------------------------------------------------------

    def _call_text(self, user_prompt: str) -> str:
        response = self.chat_model.invoke([("system", prompts.SYSTEM_PROMPT), ("human", user_prompt)])
        return strip_fences(response.content).strip()

    @staticmethod
    def _revision_note(revision: Optional[dict], heading: str, hid: Optional[str] = None) -> str:
        """Builds the text appended to a section's drafting prompt when
        `revision` is given (see run()'s docstring for its shape): the
        section's (or per-hypothesis subsection's) own previous text, plus
        feedback filed against that heading specifically and anything under
        "__general__" (paper-wide feedback, e.g. a low tone/clarity score).
        Returns "" for a fresh (non-revision) draft, unchanged from before."""
        if not revision:
            return ""

        if hid is not None:
            previous_text = (revision.get("previous_subsections", {}).get(heading) or {}).get(hid)
        else:
            previous_text = (revision.get("previous_sections") or {}).get(heading)

        feedback_by_section = revision.get("feedback_by_section") or {}
        combined_feedback = list(feedback_by_section.get(heading, [])) + list(feedback_by_section.get("__general__", []))

        parts = []
        if previous_text:
            parts.append(f"You previously wrote this as:\n{previous_text}")
        if combined_feedback:
            feedback_lines = "\n".join(f"- {item}" for item in combined_feedback)
            parts.append(f"Reviewer feedback to address in this revision:\n{feedback_lines}")
        if not parts:
            return ""
        return (
            "\n\n" + "\n\n".join(parts) + "\n\nWrite a revised version that fixes the issues above. Keep everything "
            "that was already accurate and well-grounded — only change what the feedback requires."
        )

    def _draft_introduction(
        self, hypothesis_output: dict, hypotheses: List[dict], valid_paper_ids: List[str], revision: Optional[dict] = None
    ) -> str:
        hypotheses_brief = "\n".join(f"- {h['id']}: {h['statement']}" for h in hypotheses)
        prompt = prompts.INTRODUCTION_PROMPT.format(
            literature_summary=hypothesis_output.get("literature_summary", ""),
            gaps_block=json.dumps(hypothesis_output.get("gaps", []), indent=2),
            hypotheses_brief_block=hypotheses_brief,
            valid_paper_ids=json.dumps(valid_paper_ids),
        )
        return self._call_text(prompt + self._revision_note(revision, "Introduction"))

    def _draft_related_work(
        self, raw_papers: List[dict], hypothesis_output: dict, hypotheses: List[dict], revision: Optional[dict] = None
    ) -> str:
        normalized = normalize_papers(raw_papers)
        if not normalized:
            return (
                "No source papers were available from the Literature Agent for this run, so no related work "
                "could be summarized or cited here."
            )

        methods_overview_block = json.dumps(hypothesis_output.get("methods_overview", []), indent=2)
        gaps_block = json.dumps(hypothesis_output.get("gaps", []), indent=2)
        batches = chunk_papers(normalized, settings.writer_related_work_batch_max_chars)

        def _draft_batch(batch: list) -> str:
            papers_block = "\n\n".join(paper_to_text(p) for p in batch)
            prompt = prompts.RELATED_WORK_BATCH_PROMPT.format(
                papers_block=papers_block, methods_overview_block=methods_overview_block, gaps_block=gaps_block
            )
            return self._call_text(prompt)

        if len(batches) == 1:
            # single batch: this call IS the final Related Work text, so the revision note applies here
            papers_block = "\n\n".join(paper_to_text(p) for p in batches[0])
            prompt = prompts.RELATED_WORK_BATCH_PROMPT.format(
                papers_block=papers_block, methods_overview_block=methods_overview_block, gaps_block=gaps_block
            )
            return self._call_text(prompt + self._revision_note(revision, "Related Work"))

        with ThreadPoolExecutor(max_workers=len(batches)) as pool:
            drafts = list(pool.map(_draft_batch, batches))

        hypotheses_brief = "\n".join(f"- {h['id']}: {h['statement']}" for h in hypotheses)
        synthesis_prompt = prompts.RELATED_WORK_SYNTHESIS_PROMPT.format(
            partial_drafts_block="\n\n---\n\n".join(drafts), hypotheses_brief_block=hypotheses_brief
        )
        return self._call_text(synthesis_prompt + self._revision_note(revision, "Related Work"))

    def _draft_hypotheses_section(
        self, hypotheses: List[dict], valid_paper_ids: List[str], revision: Optional[dict] = None
    ) -> str:
        prompt = prompts.HYPOTHESES_PROMPT.format(
            n=len(hypotheses), hypotheses_block=json.dumps(hypotheses, indent=2), valid_paper_ids=json.dumps(valid_paper_ids)
        )
        return self._call_text(prompt + self._revision_note(revision, "Hypotheses"))

    def _draft_per_hypothesis_sections(
        self,
        hypotheses: List[dict],
        plan_by_id: Dict[str, dict],
        experiment_by_id: Dict[str, dict],
        verdicts: Dict[str, dict],
        revision: Optional[dict] = None,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        def _bundle(hypothesis: dict) -> Tuple[str, str, str, str]:
            hid = hypothesis["id"]
            plan = plan_by_id[hid]
            experiment = experiment_by_id[hid]
            methods_text = self._draft_methods(hid, plan, experiment, revision)
            results_text = self._draft_results(hid, hypothesis, experiment, revision)
            discussion_text = self._draft_discussion(hid, hypothesis, plan, verdicts[hid], revision)
            return hid, methods_text, results_text, discussion_text

        methods_by_id: Dict[str, str] = {}
        results_by_id: Dict[str, str] = {}
        discussion_by_id: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(hypotheses)) as pool:
            futures = [pool.submit(_bundle, h) for h in hypotheses]
            for future in as_completed(futures):
                hid, methods_text, results_text, discussion_text = future.result()
                methods_by_id[hid] = methods_text
                results_by_id[hid] = results_text
                discussion_by_id[hid] = discussion_text
        return methods_by_id, results_by_id, discussion_by_id

    def _draft_methods(self, hid: str, plan: dict, experiment: dict, revision: Optional[dict] = None) -> str:
        prompt = prompts.METHODS_PROMPT.format(
            hypothesis_id=hid,
            plan_block=json.dumps(plan, indent=2),
            status=experiment["status"],
            status_meaning=STATUS_MEANINGS[experiment["status"]],
            assumptions_block=json.dumps(experiment.get("assumptions_made", []), indent=2),
        )
        return self._call_text(prompt + self._revision_note(revision, "Methods", hid))

    def _draft_results(self, hid: str, hypothesis: dict, experiment: dict, revision: Optional[dict] = None) -> str:
        status = experiment["status"]
        if status == "completed":
            results_or_reason_block = f"Results (JSON): {json.dumps(experiment['results'], indent=2)}"
        else:
            results_or_reason_block = f"Reason this experiment was not executed: {experiment['reason']}"
        prompt = prompts.RESULTS_PROMPT.format(
            hypothesis_id=hid,
            hypothesis_statement=hypothesis["statement"],
            status=status,
            status_meaning=STATUS_MEANINGS[status],
            results_or_reason_block=results_or_reason_block,
        )
        return self._call_text(prompt + self._revision_note(revision, "Results", hid))

    def _draft_discussion(
        self, hid: str, hypothesis: dict, plan: dict, verdict: dict, revision: Optional[dict] = None
    ) -> str:
        feasibility_note_block = ""
        if not plan["feasible"]:
            feasibility_note_block = (
                f"Note: this experiment's design was already simplified by the Experiment Planner because the "
                f"original was judged infeasible: {plan['feasibility_notes']}\n"
            )
        prompt = prompts.DISCUSSION_PROMPT.format(
            hypothesis_id=hid,
            hypothesis_statement=hypothesis["statement"],
            rationale=hypothesis["rationale"],
            related_gaps_block=json.dumps(hypothesis.get("related_gaps", [])),
            feasibility_note_block=feasibility_note_block,
            verdict_label=verdict["verdict"],
            verdict_reason=verdict["reason"],
        )
        return self._call_text(prompt + self._revision_note(revision, "Discussion", hid))

    def _draft_limitations(
        self,
        hypotheses: List[dict],
        plan_by_id: Dict[str, dict],
        experiment_by_id: Dict[str, dict],
        revision: Optional[dict] = None,
    ) -> str:
        risks_by_id = {h["id"]: plan_by_id[h["id"]].get("risks", []) for h in hypotheses}
        assumptions_by_id = {h["id"]: experiment_by_id[h["id"]].get("assumptions_made", []) for h in hypotheses}
        not_run = {
            h["id"]: experiment_by_id[h["id"]]["reason"]
            for h in hypotheses
            if experiment_by_id[h["id"]]["status"] != "completed"
        }
        prompt = prompts.LIMITATIONS_PROMPT.format(
            n=len(hypotheses),
            risks_block=json.dumps(risks_by_id, indent=2),
            assumptions_block=json.dumps(assumptions_by_id, indent=2),
            not_run_block=json.dumps(not_run, indent=2),
        )
        return self._call_text(prompt + self._revision_note(revision, "Limitations"))

    def _draft_future_work(
        self,
        verdicts: Dict[str, dict],
        expected_ids: List[str],
        gaps: List[dict],
        hypotheses: List[dict],
        revision: Optional[dict] = None,
    ) -> str:
        covered_gaps = {gap for h in hypotheses for gap in h.get("related_gaps", [])}
        unresolved_gaps = [g for g in gaps if g.get("gap") not in covered_gaps]
        prompt = prompts.FUTURE_WORK_PROMPT.format(
            verdicts_block=json.dumps([verdicts[hid] for hid in expected_ids], indent=2),
            unresolved_gaps_block=json.dumps(unresolved_gaps, indent=2),
        )
        return self._call_text(prompt + self._revision_note(revision, "Future Work"))

    def _draft_abstract(
        self, literature_summary: str, verdicts: Dict[str, dict], expected_ids: List[str], revision: Optional[dict] = None
    ) -> str:
        prompt = prompts.ABSTRACT_PROMPT.format(
            literature_summary=literature_summary, verdicts_block=json.dumps([verdicts[hid] for hid in expected_ids], indent=2)
        )
        return self._call_text(prompt + self._revision_note(revision, "Abstract"))

    def _draft_title(
        self, literature_summary: str, verdicts: Dict[str, dict], expected_ids: List[str], revision: Optional[dict] = None
    ) -> str:
        prompt = prompts.TITLE_PROMPT.format(
            literature_summary=literature_summary, verdicts_block=json.dumps([verdicts[hid] for hid in expected_ids], indent=2)
        )
        # a title is one line; strip any stray quoting/markdown the model adds despite instructions
        return self._call_text(prompt + self._revision_note(revision, "Title")).strip().strip('"').strip("*# ")

    @staticmethod
    def _join_per_hypothesis(hypothesis_ids: List[str], body_by_id: Dict[str, str], verdicts: Optional[Dict[str, dict]] = None) -> str:
        """Joins per-hypothesis LLM-drafted paragraphs into one section body,
        inserting a deterministic "## H1 [-- verdict]" label before each — the
        label is data (hypothesis id, and the independently-computed verdict
        when given), not LLM prose, and pdf_builder.py renders "## " lines as
        sub-headings (see its module docstring)."""
        blocks: List[str] = []
        for hid in hypothesis_ids:
            label = f"## {hid}" if verdicts is None else f"## {hid} — {verdicts[hid]['verdict']}"
            blocks.append(label)
            blocks.append(body_by_id[hid].strip())
        return "\n\n".join(blocks)

    # -- Validation pass ---------------------------------------------------------

    def _validate_paper(
        self,
        resolved_sections: List[Tuple[str, str]],
        verdicts: Dict[str, dict],
        experiment_by_id: Dict[str, dict],
        registry: CitationRegistry,
    ) -> List[str]:
        """Checks the spec's three honesty invariants and returns a list of
        human-readable notes for anything that looks off — this never raises;
        a flagged paper is still handed back for review rather than discarded,
        since the underlying LLM calls (and any real experiment runs behind
        them) are expensive to redo over a text-level inconsistency."""
        notes: List[str] = []
        sections_by_heading = dict(resolved_sections)

        # 1. every hypothesis has a Results and Discussion mention
        for hid in verdicts:
            for heading in ("Results", "Discussion"):
                if hid not in sections_by_heading.get(heading, ""):
                    notes.append(f"{hid} does not appear to be mentioned by id in the {heading} section — check it was covered.")

        # 2. every in-text citation maps to a paper in the Literature Agent's output
        for unresolved in registry.unresolved:
            notes.append(
                f"Dropped a citation to unknown paper id '{unresolved['paper_id']}' in the {unresolved['section']} "
                "section (not present in the Literature Agent's output) — removed from the text rather than printed."
            )

        # 3. no skipped/code_generated_not_run experiment is described as completed
        not_completed_phrases = ("not executed", "not run", "no results", "not completed", "not implemented", "no empirical", "not been run", "were not executed")
        completed_claim_phrases = ("results show", "we found", "successfully", "demonstrates that", "confirms that")
        results_text = sections_by_heading.get("Results", "")
        for hid, experiment in experiment_by_id.items():
            if experiment["status"] == "completed":
                continue
            # crude per-hypothesis slice: from this hid's "## " label to the next one (or end of section)
            marker = f"## {hid}"
            start = results_text.find(marker)
            if start == -1:
                continue
            end = results_text.find("\n\n## ", start + 1)
            segment = results_text[start:end if end != -1 else None].lower()
            has_disclaimer = any(phrase in segment for phrase in not_completed_phrases)
            overstates = any(phrase in segment for phrase in completed_claim_phrases)
            if not has_disclaimer or overstates:
                notes.append(
                    f"{hid}'s Results text may not clearly state that status is '{experiment['status']}' "
                    "(not completed) — please review before submitting."
                )

        return notes

    # -- Output persistence ----------------------------------------------------

    def _render_pdf(
        self, title: str, abstract: str, sections: List[Tuple[str, str]], references: List[str], filename: Optional[str] = None
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / (filename or f"paper_{timestamp}.pdf")
        build_pdf(
            path,
            title=title,
            abstract=abstract,
            sections=sections,
            references=references,
            authors=settings.writer_paper_authors,
            affiliation=settings.writer_paper_affiliation,
        )
        return path

    def _write_summary(self, result: dict, suffix: str = "", filename: Optional[str] = None) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / (filename or f"paper_summary_{timestamp}{suffix}.json")
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path


def run_writer_agent(
    literature_output: object,
    hypothesis_output: dict,
    planner_output: dict,
    coder_output: dict,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Module-level entry point — the stable call the pipeline (or a manual
    trigger, once the Coder Agent has finished) should use. Builds a
    default-configured WriterAgent and runs it once; use the WriterAgent class
    directly to reuse one model/config across multiple calls."""
    return WriterAgent(output_dir=output_dir).run(literature_output, hypothesis_output, planner_output, coder_output)
