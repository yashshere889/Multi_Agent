# Eval sets for the search agents

`gold/` holds the ground truth: research questions paired with papers a good
search *should* find. Gold sets are committed — they're the reference every
future measurement is made against, and a run is only comparable to another run
scored on the same set. `runs/` holds outputs and is gitignored.

## Where the papers come from

From real survey bibliographies, fetched through Semantic Scholar — never from a
model, and never from memory. A recent survey on topic X is a hand-curated,
peer-reviewed bibliography of topic X: ground truth someone else already
assembled.

```bash
uv run research-pipeline eval-bootstrap --survey "arXiv:2312.10997" --question "how does retrieval augmentation affect factual accuracy in LLMs?" --min-year 2015
```

This writes `gold/<slug>.json` with the survey's actual reference list. A gold
set of plausible-looking invented titles would still produce clean-looking
recall numbers, and those numbers would be worse than none — confidently wrong,
with every later decision made against them. Same rule the pipeline applies to
synthetic experiment data.

**Prune what you get.** A bibliography includes background and methods papers a
topical search has no business finding. `--min-year` cuts the long tail; the
rest is a few minutes of reading. Ten questions with fifty well-chosen papers
each beats fifty questions of unfiltered bibliography.

## Measuring

```bash
uv run research-pipeline eval-run --gold evals/gold --name baseline --judge
```

Searches every question once with the in-pipeline relevance screen disabled,
saves the whole raw pool, then replays the screen over it at every threshold.
One search, every threshold — searching is the slow, rate-limited part, and
running it twice to A/B one variable just adds a second sample of API flakiness
to the comparison.

Re-score a saved run — new threshold, new metric, no network:

```bash
uv run research-pipeline eval-score --run evals/runs/baseline.json --gold evals/gold
```

Then change something and compare:

```bash
uv run research-pipeline eval-compare evals/runs/baseline.scored.json evals/runs/candidate.scored.json
```

## Reading the numbers

**Recall is relative, not absolute.** A survey's bibliography holds a
hundred-odd papers and a default run returns a few dozen, so even a flawless
search scores far below 1.0. What matters is whether a change moves it.

**The sweep is the point.** It shows what each `RELEVANCE_MIN_SCORE` would keep,
what it costs in recall, and how many papers an independent judge would have
kept that the threshold throws away. That last column is the one to watch when
tightening the screen.

**`--judge` is the only irreproducible number here.** Everything else is
deterministic and free to recompute. The judge deliberately does not reuse the
screen's rubric — grading the screen with its own answer key would measure
nothing.

**Duplicate groups should be zero.** Anything else is the pipeline's single
doi-or-title dedupe key letting the same work through twice, usually as an arXiv
preprint alongside its published DOI.
