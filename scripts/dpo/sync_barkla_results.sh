#!/usr/bin/env bash
# Pull every benchmark and batch artefact this project needs for write-up from
# the cluster. Safe to re-run: it only ever adds files.
#
#   bash scripts/dpo/sync_barkla_results.sh [ssh-host]
#
# Needs the university network (or VPN) — Barkla's login node is not reachable
# from a general internet connection.
set -euo pipefail

HOST="${1:-barkla}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/dissertation/data"
REMOTE_USER="$(ssh -o BatchMode=yes "$HOST" 'echo $USER')"
S="/mnt/fastscratch/users/$REMOTE_USER"

echo "==> queue state"
ssh -o BatchMode=yes "$HOST" 'squeue -u $USER -o "%.12i %.22j %.3t %.8M %R"' || true

echo "==> archiving results on $HOST"
ssh -o BatchMode=yes "$HOST" "cd $S && rm -f /tmp/sync_results.tar.gz && \
  find coder-benchmark-runs pipeline-runs coder-only-runs coder-agent-runs \
       \( -name 'coder_agent_summary_*.json' \
       -o -name 'results.json' \
       -o -name 'data_provenance.json' \
       -o -name 'batch_manifest.json' \
       -o -name 'hypotheses_*.json' \
       -o -name 'experiment_plan_*.json' \
       -o -name 'interdisciplinary_*.json' \
       -o -name 'review_*.json' \
       -o -name 'review_log.json' \
       -o -name '*_summary.json' \
       -o -name 'v*.pdf' \
       -o -path '*/experiments/*' -name 'run.py' \) \
       -not -path '*/.venv/*' -not -path '*/site-packages/*' 2>/dev/null \
    | sort -u > /tmp/sync_files.txt && \
  wc -l < /tmp/sync_files.txt && \
  tar czf /tmp/sync_results.tar.gz -T /tmp/sync_files.txt 2>/dev/null || true"

echo "==> downloading into $DEST"
mkdir -p "$DEST"
scp -q "$HOST:/tmp/sync_results.tar.gz" "$DEST/"
tar xzf "$DEST/sync_results.tar.gz" -C "$DEST" 2>/dev/null || true
rm -f "$DEST/sync_results.tar.gz"
ssh -o BatchMode=yes "$HOST" 'rm -f /tmp/sync_results.tar.gz /tmp/sync_files.txt' || true

echo "==> dpo results"
ssh -o BatchMode=yes "$HOST" "cd $S && tar czf /tmp/sync_dpo.tar.gz dpo/run-*/result.json 2>/dev/null || true"
scp -q "$HOST:/tmp/sync_dpo.tar.gz" "$DEST/" 2>/dev/null || true
[ -f "$DEST/sync_dpo.tar.gz" ] && tar xzf "$DEST/sync_dpo.tar.gz" -C "$DEST" 2>/dev/null || true
rm -f "$DEST/sync_dpo.tar.gz"
ssh -o BatchMode=yes "$HOST" 'rm -f /tmp/sync_dpo.tar.gz' || true

echo "==> removing Desktop-sync duplicates (\"<name> 2.json\")"
find "$DEST" -name '* [0-9].*' -type f -delete 2>/dev/null || true

echo "==> local benchmark runs now held:"
ls -1 "$DEST/coder-benchmark-runs" 2>/dev/null || true
echo
echo "Next: rebuild the preference dataset and re-score the arms —"
echo "  uv run python scripts/dpo/build_preference_dataset.py dissertation/data \\"
echo "      --out dissertation/data/preference_pairs.jsonl"
