"""Reviewer Agent.

Sits after the Writer Agent and, in a loop orchestrated by
`research_pipeline.writer_reviewer_loop`, checks its paper against the same
upstream ground truth the Writer used — not just for internal consistency —
and produces structured, actionable feedback the Writer can revise against.

Input contract
--------------
`run(paper_path, paper_summary, literature_output, hypothesis_output,
planner_output, coder_output, iteration=1, quality_threshold=None)`:
- `paper_path` — the PDF the Writer Agent rendered (WriterAgentOutput's
  "paper_path"). Read back via `agents.writer.pdf_reader` — the Reviewer
  checks what actually got printed, not the Writer's own self-report.
- `paper_summary` — the Writer Agent's JSON output (WriterAgentOutput):
  used for its "sections_generated" (to split the PDF text back into
  sections) and "citations_used" (cross-checked, not trusted outright).
- `literature_output`, `hypothesis_output`, `planner_output`, `coder_output`
  — the exact same upstream outputs the Writer Agent was given. Validated
  against their own schemas on entry, same as the Writer Agent — a review
  against a malformed input is never attempted.

Check strategy
---------------
Where a check can be done exactly, it is (agents.reviewer.checks, no LLM
calls): every id in `citations_used` actually exists in the Literature
Agent's output; every "(Surname, YYYY)"-shaped citation actually printed
matches some real paper by surname+year; every reported metric's value
appears in some plausible textual form in its hypothesis's Results text; no
`skipped`/`code_generated_not_run` experiment's Results text reads as
completed; every hypothesis has a Results and Discussion subsection; a coarse
keyword scan flags the unambiguous cases where Discussion text contradicts
the true supported/refuted/inconclusive verdict outright.

The true verdict itself is never re-guessed by an LLM — it's computed the
same way the Writer Agent computed it in the first place
(`agents.writer.writer_agent.compute_hypothesis_verdict`, from the Coder
Agent's own `meets_success_criteria`/`status`), so the Writer and Reviewer
can never quietly disagree about what "supported" means.

What's left — hallucination nuance beyond bare fact-checking, and honesty of
framing/tone/flow — genuinely needs judgment, so those go to the LLM
(agents.reviewer.prompts), one call per section plus one holistic
quality-scoring call, each given only the ground-truth slice relevant to what
it's checking (mirroring how the Writer Agent grounded that section when
drafting it).

Output contract
----------------
A dict matching `agents.reviewer.schema.ReviewOutput` — see schema.py.
Validated before being returned; also written to
`<output_dir>/review_<UTC timestamp>.json` (the Writer/Reviewer loop keeps its
own consolidated `review_log.json` across iterations — see
writer_reviewer_loop.py — this per-call file is for standalone use/debugging).

Entry points
------------
    from research_pipeline.agents.reviewer import run_reviewer_agent
    result = run_reviewer_agent(paper_path, paper_summary, literature_output,
                                 hypothesis_output, planner_output, coder_output)

Or, to reuse one configured model/output dir across multiple calls (as the
Writer/Reviewer loop does):
    agent = ReviewerAgent()
    result = agent.run(paper_path, paper_summary, literature_output,
                        hypothesis_output, planner_output, coder_output, iteration=2)

Internals
---------
`run()` executes a LangGraph StateGraph (see graph.py / state.py) rather than a
straight-line method, so every step — each LLM call and each deterministic check
— is its own traceable node, and the three deterministic checks fan out
concurrently. Unlike the Hypothesis/Experiment Planner graphs the fan-out width
is fixed (there are always exactly those three checks, whatever the input
contains), so it's plain edges rather than `Send`, matching the Literature
Agent's two-way search fan-out. That's purely internal: the entry points above,
their signatures, the returned dict, and the raised ReviewerAgentError are all
unchanged.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from research_pipeline.agents.coder.schema import SchemaValidationError as CoderSchemaValidationError
from research_pipeline.agents.coder.schema import validate_output as validate_coder_output
from research_pipeline.agents.experiment_planner.schema import SchemaValidationError as PlannerSchemaValidationError
from research_pipeline.agents.experiment_planner.schema import validate_output as validate_planner_output
from research_pipeline.agents.hypothesis.schema import SchemaValidationError as HypothesisSchemaValidationError
from research_pipeline.agents.hypothesis.schema import validate_output as validate_hypothesis_output
from research_pipeline.agents.reviewer import checks, prompts
from research_pipeline.agents.reviewer.schema import SchemaValidationError, validate_output
from research_pipeline.agents.reviewer.state import ReviewerState
from research_pipeline.agents.writer import pdf_reader
from research_pipeline.agents.writer.citations import build_paper_index
from research_pipeline.agents.writer.writer_agent import compute_hypothesis_verdict, extract_literature_papers
from research_pipeline.config import settings
from research_pipeline.llm import get_chat_model
from research_pipeline.llm_json import LLMJSONError, invoke_json

logger = logging.getLogger(__name__)

# Body sections checked for hallucinations with a plain grounding block —
# Discussion is handled separately (prompts.DISCUSSION_REVIEW_PROMPT) since it
# also needs the framing/honesty check, and Hypotheses/Results/Methods need
# per-hypothesis grounding assembled specially — see _grounding_for_section.
_HALLUCINATION_SECTIONS = ("Introduction", "Related Work", "Hypotheses", "Methods", "Results", "Limitations", "Future Work")


class ReviewerAgentError(RuntimeError):
    """Raised when the agent can't produce a valid review, even after retries."""


class ReviewerAgent:
    def __init__(
        self,
        chat_model: Optional[BaseChatModel] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        # Reuses the pipeline's existing LLM client/config, at a low
        # temperature suited to strict, grounded judgment rather than a
        # separate client — same reasoning as hypothesis/experiment-planner/coder.
        self.chat_model = chat_model or get_chat_model(temperature=0.1)
        self.output_dir = Path(output_dir or settings.reviewer_output_dir)

    def run(
        self,
        paper_path: str | Path,
        paper_summary: dict,
        literature_output: object,
        hypothesis_output: dict,
        planner_output: dict,
        coder_output: dict,
        iteration: int = 1,
        quality_threshold: Optional[int] = None,
    ) -> dict:
        """Runs the agent's graph end to end. Same signature, same returned dict,
        and the same ReviewerAgentError on failure as the earlier sequential
        implementation — the graph is an internal detail, so callers (the
        Writer/Reviewer loop, the orchestrator, the CLI) are unaffected.

        Node exceptions aren't swallowed by LangGraph, so a ReviewerAgentError
        raised inside a node propagates out of `.invoke` unchanged.
        """
        # Imported here rather than at module scope: graph.py needs this module's
        # ReviewerAgent for typing, so a top-level import would be circular.
        from research_pipeline.agents.reviewer.graph import build_reviewer_graph

        graph = build_reviewer_graph(self)
        final_state = graph.invoke(
            {
                "paper_path": paper_path,
                "paper_summary": paper_summary,
                "literature_output": literature_output,
                "hypothesis_output": hypothesis_output,
                "planner_output": planner_output,
                "coder_output": coder_output,
                "iteration": iteration,
                "quality_threshold": (
                    quality_threshold if quality_threshold is not None else settings.writer_reviewer_quality_threshold
                ),
            },
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        return final_state["result"]

    # -- Graph nodes -----------------------------------------------------------
    # Thin adapters: each takes the graph state, delegates to the private helper
    # (or checks.py function) below that does the actual work, and returns only
    # the keys it produces.

    def _node_prepare_context(self, state: ReviewerState) -> dict:
        """Validates the three upstream outputs, indexes the ground truth every
        later node checks against, and reads the rendered PDF back into
        sections/per-hypothesis subsections."""
        hypothesis_output = state["hypothesis_output"]
        planner_output = state["planner_output"]
        coder_output = state["coder_output"]

        try:
            validate_hypothesis_output(hypothesis_output)
        except HypothesisSchemaValidationError as exc:
            raise ReviewerAgentError(f"hypothesis_output doesn't match the Hypothesis Agent's output schema: {exc}") from exc

        hypotheses = hypothesis_output["hypotheses"]
        # Not [h["id"] for h in hypotheses] — see the matching comment in
        # writer_agent.py._node_prepare_context: the orchestrator narrows
        # planner_output/coder_output to the selected hypothesis, so validating
        # against the full hypothesis_output id list always fails post-narrowing.
        expected_ids = planner_output.get("source_hypothesis_ids") or [h["id"] for h in hypotheses]

        try:
            validate_planner_output(planner_output, expected_hypothesis_ids=expected_ids)
        except PlannerSchemaValidationError as exc:
            raise ReviewerAgentError(f"planner_output doesn't match the Experiment Planner's output schema: {exc}") from exc

        try:
            validate_coder_output(coder_output, expected_hypothesis_ids=expected_ids)
        except CoderSchemaValidationError as exc:
            raise ReviewerAgentError(f"coder_output doesn't match the Coder Agent's output schema: {exc}") from exc

        raw_papers, _ = extract_literature_papers(state["literature_output"])
        paper_index = build_paper_index(raw_papers)
        plan_by_id = {p["hypothesis_id"]: p for p in planner_output["experiment_plans"]}
        experiment_by_id = {e["hypothesis_id"]: e for e in coder_output["experiments"]}
        verdicts = {
            hid: {
                "hypothesis_id": hid,
                "statement": next(h["statement"] for h in hypotheses if h["id"] == hid),
                "rationale": next(h["rationale"] for h in hypotheses if h["id"] == hid),
                "verdict": (v := compute_hypothesis_verdict(hid, experiment_by_id))[0],
                "reason": v[1],
            }
            for hid in expected_ids
        }

        logger.info("Reviewing %s (iteration %d)", state["paper_path"], state["iteration"])
        full_text = pdf_reader.extract_text(state["paper_path"])
        sections = pdf_reader.split_into_sections(full_text, state["paper_summary"].get("sections_generated", []))

        return {
            "hypotheses": hypotheses,
            "expected_ids": expected_ids,
            "raw_papers": raw_papers,
            "paper_index": paper_index,
            "plan_by_id": plan_by_id,
            "experiment_by_id": experiment_by_id,
            "verdicts": verdicts,
            "full_text": full_text,
            "sections": sections,
            "results_subsections": pdf_reader.split_hypothesis_subsections(sections.get("Results", ""), expected_ids),
            "discussion_subsections": pdf_reader.split_hypothesis_subsections(sections.get("Discussion", ""), expected_ids),
        }

    # -- Deterministic checks: fanned out from prepare_context with plain edges --
    # -- (fixed set of three, no LLM, disjoint state keys — see graph.py) --------

    def _node_check_citations(self, state: ReviewerState) -> dict:
        return {
            "citation_issues": checks.check_citations(
                state["sections"], state["paper_index"], state["paper_summary"].get("citations_used", [])
            )
        }

    def _node_check_results_accuracy(self, state: ReviewerState) -> dict:
        return {"results_accuracy_issues": checks.check_results_accuracy(state["results_subsections"], state["coder_output"])}

    def _node_check_hypothesis_coverage(self, state: ReviewerState) -> dict:
        return {
            "hypothesis_coverage_issues": checks.check_hypothesis_coverage(
                state["results_subsections"], state["discussion_subsections"], state["expected_ids"], state["verdicts"]
            )
        }

    # -- LLM checks: hallucinations per section + discussion framing ------------

    def _node_check_hallucinations(self, state: ReviewerState) -> dict:
        """One LLM call per printed body section. Kept as a single node running
        the existing loop: the sections are cheap in number and each call's
        grounding block is assembled from the same state, so fanning them out
        would add branches without adding traceability worth the complexity."""
        sections = state["sections"]
        hallucinations: List[dict] = []
        for heading in _HALLUCINATION_SECTIONS:
            if heading not in sections or not sections[heading].strip():
                continue
            grounding = self._grounding_for_section(
                heading,
                state["hypothesis_output"],
                state["hypotheses"],
                state["plan_by_id"],
                state["experiment_by_id"],
                state["verdicts"],
                state["raw_papers"],
            )
            hallucinations += self._check_hallucinations(heading, sections[heading], grounding)
        return {"hallucinations": hallucinations}

    def _node_check_discussion(self, state: ReviewerState) -> dict:
        """Extends — never replaces — the two lists its predecessors produced:
        the Discussion pass yields both further hallucinations and framing
        issues that belong with the deterministic coverage findings."""
        sections = state["sections"]
        if "Discussion" not in sections or not sections["Discussion"].strip():
            return {}

        disc_hallucinations, framing_issues = self._check_discussion(
            sections["Discussion"], state["verdicts"], state["expected_ids"]
        )
        return {
            "hallucinations": state["hallucinations"] + disc_hallucinations,
            "hypothesis_coverage_issues": state["hypothesis_coverage_issues"] + framing_issues,
        }

    def _node_score_quality(self, state: ReviewerState) -> dict:
        quality_scores, quality_notes = self._score_quality(
            state["full_text"], state["plan_by_id"], state["experiment_by_id"], state["expected_ids"]
        )
        return {"quality_scores": quality_scores, "quality_notes": quality_notes}

    # -- Verdict + feedback assembly (deterministic) ----------------------------

    def _node_compute_overall_pass(self, state: ReviewerState) -> dict:
        quality_threshold = state["quality_threshold"]
        return {
            "overall_pass": (
                not state["hallucinations"]
                and not state["citation_issues"]
                and not state["results_accuracy_issues"]
                and not state["hypothesis_coverage_issues"]
                and all(score >= quality_threshold for score in state["quality_scores"].values())
            )
        }

    def _node_build_feedback(self, state: ReviewerState) -> dict:
        return {
            "feedback_for_writer": self._build_feedback(
                state["hallucinations"],
                state["citation_issues"],
                state["results_accuracy_issues"],
                state["hypothesis_coverage_issues"],
                state["quality_scores"],
                state["quality_notes"],
                state["quality_threshold"],
            )
        }

    def _node_assemble_and_validate(self, state: ReviewerState) -> dict:
        result: dict = {
            "iteration": state["iteration"],
            "hallucinations": state["hallucinations"],
            "citation_issues": state["citation_issues"],
            "results_accuracy_issues": state["results_accuracy_issues"],
            "hypothesis_coverage_issues": state["hypothesis_coverage_issues"],
            "quality_scores": state["quality_scores"],
            "overall_pass": state["overall_pass"],
            "feedback_for_writer": state["feedback_for_writer"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": settings.llm_model,
        }

        try:
            validate_output(result)
        except SchemaValidationError as exc:
            debug_path = self._write_review(result, suffix="_invalid")
            raise ReviewerAgentError(
                f"Assembled review failed schema validation: {exc}. Raw (invalid) output written to {debug_path} for inspection."
            ) from exc

        review_path = self._write_review(result)
        logger.info("Wrote review to %s (overall_pass=%s)", review_path, result["overall_pass"])
        return {"result": result}

    # -- Grounding assembly (mirrors what the Writer Agent used to draft each section) --

    @staticmethod
    def _grounding_for_section(
        heading: str,
        hypothesis_output: dict,
        hypotheses: List[dict],
        plan_by_id: Dict[str, dict],
        experiment_by_id: Dict[str, dict],
        verdicts: Dict[str, dict],
        raw_papers: List[dict],
    ) -> dict:
        if heading == "Introduction":
            return {
                "literature_summary": hypothesis_output.get("literature_summary", ""),
                "gaps": hypothesis_output.get("gaps", []),
                "hypotheses": [{"id": h["id"], "statement": h["statement"]} for h in hypotheses],
            }
        if heading == "Related Work":
            return {
                "papers": [
                    {"title": p.get("title"), "authors": p.get("authors"), "year": p.get("year"), "abstract": p.get("abstract")}
                    for p in raw_papers
                ],
                "methods_overview": hypothesis_output.get("methods_overview", []),
                "gaps": hypothesis_output.get("gaps", []),
            }
        if heading == "Hypotheses":
            return {"hypotheses": hypotheses}
        if heading == "Methods":
            return {
                "plans_by_hypothesis": plan_by_id,
                "coder_assumptions_by_hypothesis": {
                    hid: exp.get("assumptions_made", []) for hid, exp in experiment_by_id.items()
                },
                "coder_status_by_hypothesis": {hid: exp["status"] for hid, exp in experiment_by_id.items()},
            }
        if heading == "Results":
            return {"experiments": experiment_by_id, "hypothesis_statements": {h["id"]: h["statement"] for h in hypotheses}}
        if heading == "Limitations":
            return {
                "risks_by_hypothesis": {hid: plan.get("risks", []) for hid, plan in plan_by_id.items()},
                "assumptions_by_hypothesis": {hid: exp.get("assumptions_made", []) for hid, exp in experiment_by_id.items()},
                "not_run_reasons": {
                    hid: exp["reason"] for hid, exp in experiment_by_id.items() if exp["status"] != "completed"
                },
            }
        if heading == "Future Work":
            return {"verdicts": list(verdicts.values()), "gaps": hypothesis_output.get("gaps", [])}
        return {}

    # -- LLM calls ---------------------------------------------------------------

    def _call_json(self, user_prompt: str) -> dict:
        try:
            return invoke_json(self.chat_model, prompts.SYSTEM_PROMPT, user_prompt)
        except LLMJSONError as exc:
            raise ReviewerAgentError(str(exc)) from exc

    def _check_hallucinations(self, heading: str, section_text: str, grounding: dict) -> List[dict]:
        prompt = prompts.HALLUCINATION_CHECK_PROMPT.format(
            section_name=heading, grounding_block=json.dumps(grounding, indent=2, default=str), section_text=section_text
        )
        response = self._call_json(prompt)
        return [{"location": heading, "claim": h.get("claim", ""), "issue": h.get("issue", "")} for h in response.get("hallucinations", [])]

    def _check_discussion(self, section_text: str, verdicts: Dict[str, dict], expected_ids: List[str]) -> Tuple[List[dict], List[dict]]:
        prompt = prompts.DISCUSSION_REVIEW_PROMPT.format(
            verdicts_block=json.dumps([verdicts[hid] for hid in expected_ids], indent=2), section_text=section_text
        )
        response = self._call_json(prompt)
        hallucinations = [
            {"location": "Discussion", "claim": h.get("claim", ""), "issue": h.get("issue", "")}
            for h in response.get("hallucinations", [])
        ]
        framing_issues = [
            {
                "hypothesis_id": f.get("hypothesis_id", ""),
                "location": f"Discussion > {f.get('hypothesis_id', '')}",
                "issue": f.get("issue", ""),
            }
            for f in response.get("framing_issues", [])
        ]
        return hallucinations, framing_issues

    def _score_quality(
        self, full_text: str, plan_by_id: Dict[str, dict], experiment_by_id: Dict[str, dict], expected_ids: List[str]
    ) -> Tuple[dict, dict]:
        ground_truth = {
            "risks_by_hypothesis": {hid: plan_by_id[hid].get("risks", []) for hid in expected_ids},
            "assumptions_by_hypothesis": {hid: experiment_by_id[hid].get("assumptions_made", []) for hid in expected_ids},
            "not_run_reasons": {
                hid: experiment_by_id[hid]["reason"] for hid in expected_ids if experiment_by_id[hid]["status"] != "completed"
            },
        }
        prompt = prompts.QUALITY_SCORE_PROMPT.format(ground_truth_block=json.dumps(ground_truth, indent=2), paper_text=full_text)
        response = self._call_json(prompt)
        scores = response.get("quality_scores", {})
        # never trust an out-of-range/missing score silently — clamp and default rather than let a
        # malformed model response produce an artificially-passing (or crashing) review
        clamped = {}
        for key in ("clarity", "flow", "tone", "structure", "limitations_honesty"):
            value = scores.get(key)
            clamped[key] = int(value) if isinstance(value, (int, float)) and 1 <= value <= 5 else 1
        return clamped, response.get("quality_notes", {})

    # -- Feedback assembly (deterministic — every issue must reach the Writer, --
    # -- so this isn't itself an LLM summarization pass that could drop one) -----

    @staticmethod
    def _build_feedback(
        hallucinations: List[dict],
        citation_issues: List[dict],
        results_accuracy_issues: List[dict],
        hypothesis_coverage_issues: List[dict],
        quality_scores: dict,
        quality_notes: dict,
        quality_threshold: int,
    ) -> str:
        low_scores = {k: v for k, v in quality_scores.items() if v < quality_threshold}
        if not (hallucinations or citation_issues or results_accuracy_issues or hypothesis_coverage_issues or low_scores):
            return "No issues found — the paper is grounded in the upstream data and meets the quality bar on every dimension."

        lines: List[str] = []
        if hallucinations:
            lines.append("Hallucinations / ungrounded claims:")
            lines += [f'- {h["location"]}: "{h["claim"]}" — {h["issue"]}' for h in hallucinations]
        if citation_issues:
            lines.append("Citation issues:")
            lines += [f'- {c["location"]}: {c["issue"]}' for c in citation_issues]
        if results_accuracy_issues:
            lines.append("Results accuracy issues:")
            lines += [f'- {r["location"]}: claimed — {r["claimed"]}; actual — {r["actual"]}. Correct this.' for r in results_accuracy_issues]
        if hypothesis_coverage_issues:
            lines.append("Hypothesis coverage issues:")
            lines += [f'- {h["location"]}: {h["issue"]}' for h in hypothesis_coverage_issues]
        if low_scores:
            lines.append(f"Quality scores below the {quality_threshold}/5 threshold:")
            lines += [f'- {dim}: {score}/5 — {quality_notes.get(dim, "no further detail given")}' for dim, score in low_scores.items()]
        return "\n".join(lines)

    # -- Output persistence ----------------------------------------------------

    def _write_review(self, result: dict, suffix: str = "") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"review_{timestamp}{suffix}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path


def run_reviewer_agent(
    paper_path: str | Path,
    paper_summary: dict,
    literature_output: object,
    hypothesis_output: dict,
    planner_output: dict,
    coder_output: dict,
    iteration: int = 1,
    quality_threshold: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Module-level entry point — the stable call the Writer/Reviewer loop (or
    a manual trigger) should use. Builds a default-configured ReviewerAgent
    and runs it once; use the ReviewerAgent class directly to reuse one
    model/config across multiple review iterations."""
    return ReviewerAgent(output_dir=output_dir).run(
        paper_path, paper_summary, literature_output, hypothesis_output, planner_output, coder_output,
        iteration=iteration, quality_threshold=quality_threshold,
    )
