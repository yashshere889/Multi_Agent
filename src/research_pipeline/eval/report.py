"""Terminal rendering for scored eval runs.

Plain text on purpose, matching the rest of cli.py: the harness is something you
run in a loop while changing a threshold, and a table you can read without
leaving the terminal beats a file you have to open.

Every renderer here is total — a run with a failed question, an absent gold set,
or no judge still prints. A missing number shows as "—" rather than 0.0, because
"the judge never ran" and "the judge rejected everything" are opposite findings
and must never look the same.
"""

from __future__ import annotations

from typing import Optional


def _num(value: Optional[float], places: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{places}f}{suffix}"


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))) for r in rows)
    return f"{line}\n{rule}\n{body}" if rows else f"{line}\n{rule}"


def format_run(scored: dict) -> str:
    out: list[str] = []
    name = scored.get("name") or "(unnamed)"
    out.append(f"\n=== {name} ===")

    config = scored.get("config") or {}
    if config:
        sources = [n for n, on in
                   (("semantic_scholar", config.get("semantic_scholar_enabled")),
                    ("core", config.get("core_enabled"))) if on]
        out.append(
            f"max_results/query={config.get('max_results_per_query')}  "
            f"model={config.get('llm_model')}  sources=arxiv+{'+'.join(sources) or 'none'}"
        )

    rows = []
    for q in scored.get("questions", []):
        rows.append([
            (q["question"][:48] + "…") if len(q["question"]) > 49 else q["question"],
            "FAILED" if q.get("error") else _num(q.get("pool_size")),
            f"{q.get('gold_found', 0)}/{q.get('gold_total', 0)}",
            _pct(q.get("recall")),
            _pct(q.get("abstract_coverage")),
            _num(q.get("mean_relevance_score")),
            _pct(q.get("judged_precision")),
        ])
    out.append("\n" + _table(
        ["question", "pool", "gold", "recall", "abstracts", "score", "precision"], rows
    ))

    agg = scored.get("aggregate") or {}
    if agg:
        out.append(
            f"\n{agg.get('questions', 0)} question(s)  "
            f"mean recall {_pct(agg.get('mean_recall'))}  "
            f"gold found {agg.get('total_gold_found', 0)}/{agg.get('total_gold', 0)}  "
            f"mean pool {_num(agg.get('mean_pool_size'), 1)}"
        )
        if agg.get("total_duplicate_groups"):
            # Non-zero means the pipeline's single doi-or-title dedupe key let
            # the same work through twice — worth chasing, not cosmetic.
            out.append(f"  ! {agg['total_duplicate_groups']} duplicate group(s) survived dedupe")

    sweep = scored.get("sweep") or []
    if sweep:
        out.append("\nRelevance threshold sweep (what each RELEVANCE_MIN_SCORE would do):")
        out.append(_table(
            ["threshold", "mean kept", "mean recall", "precision", "good papers lost"],
            [[
                str(r["threshold"]),
                _num(r.get("mean_kept"), 1),
                _pct(r.get("mean_recall")),
                _pct(r.get("judged_precision")),
                str(r.get("lost_good_papers", 0)),
            ] for r in sweep],
        ))
        out.append("  threshold 0 = keep everything, i.e. the pre-screen baseline.")

    misses = [(q["question"], q.get("missed_titles") or []) for q in scored.get("questions", [])]
    misses = [(q, m) for q, m in misses if m]
    if misses:
        out.append("\nGold papers the search missed (the most actionable output here —")
        out.append("a paper missed across several questions points straight at the queries):")
        for question, titles in misses[:5]:
            out.append(f"  {question[:60]}")
            for title in titles[:5]:
                out.append(f"    - {title}")
            if len(titles) > 5:
                out.append(f"    … and {len(titles) - 5} more")

    inter = [q["interdisciplinary"] for q in scored.get("questions", []) if q.get("interdisciplinary")]
    if inter:
        out.append("\nInterdisciplinary:")
        out.append(_table(
            ["fields", "cross-field", "off-field", "insights", "grounded"],
            [[
                ", ".join(str(f) for f in (m.get("fields_explored") or []))[:32],
                str(m.get("cross_field_papers", 0)),
                _pct(m.get("off_field_rate")),
                str(m.get("bridge_insights", 0)),
                _pct(m.get("grounded_insight_rate")),
            ] for m in inter],
        ))
        out.append("  off-field = cross-field papers whose reported subject really differs from the")
        out.append("  in-domain papers'; grounded = insights citing at least one cross-field paper.")

    return "\n".join(out) + "\n"


def format_comparison(comparison: dict) -> str:
    out = [f"\n=== {comparison.get('baseline')} → {comparison.get('candidate')} ==="]

    keys = ("recall", "pool_size", "abstract_coverage", "judged_precision")
    rows = []
    for row in comparison.get("questions", []):
        cells = [(row["question"][:40] + "…") if len(row["question"]) > 41 else row["question"]]
        for key in keys:
            entry = row.get(key) or {}
            delta = entry.get("delta")
            arrow = "" if delta is None else (" ▲" if delta > 0 else (" ▼" if delta < 0 else " ="))
            cells.append(f"{_num(entry.get('before'))} → {_num(entry.get('after'))}{arrow}")
        rows.append(cells)

    out.append("\n" + _table(["question", *keys], rows))

    if comparison.get("unmatched"):
        out.append(
            f"\n! {len(comparison['unmatched'])} question(s) had no baseline counterpart and were skipped."
        )
    return "\n".join(out) + "\n"
