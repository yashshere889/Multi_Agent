"""LoRA + DPO on preference pairs harvested from the Coder Agent's fix loop.

    python scripts/dpo/train_dpo.py \
        --pairs dissertation/data/preference_pairs.jsonl \
        --output-dir /mnt/fastscratch/$USER/dpo/run1

This trains a *small* coder model, not the 30B the pipeline serves. The claim
being tested is that the pipeline's own failure history is usable preference
data at all — not that it improves the served model, which the available number
of pairs could not support and which no honest reading of the result would
license. Qwen2.5-Coder-1.5B-Instruct is chosen because it shares the vendor and
the code specialisation of the served model while fitting a single GPU with an
adapter.

The dataset is small by construction: a pair exists only where a deterministic
check rejected one generation and accepted its replacement, and the pipeline
produced a few dozen such events across every run ever recorded. The evaluation
is therefore held-out *preference accuracy* — on unseen pairs, does the policy
assign higher implicit reward to the generation the check accepted? — rather
than any claim about downstream experiment success.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def getattr_trl_version() -> str:
    try:
        import trl

        return getattr(trl, "__version__", "?")
    except Exception:
        return "?"


def load_pairs(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        # trl's standard (non-conversational) DPO format.
        row.pop("metadata", None)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=640)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    rows = load_pairs(args.pairs)
    random.Random(args.seed).shuffle(rows)
    split = max(1, int(len(rows) * args.test_fraction))
    test_rows, train_rows = rows[:split], rows[split:]
    logger.info("pairs: %d total -> %d train / %d held out", len(rows), len(train_rows), len(test_rows))
    if not train_rows:
        raise SystemExit("no training pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
    )

    # The adapter targets the attention and MLP projections. With a few dozen
    # pairs the rank is kept low deliberately: the aim is a measurable
    # preference shift, not a capable model.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # trl renames and removes DPOConfig fields between releases — 1.12 dropped
    # `max_prompt_length` and `max_completion_length`, keeping only `max_length`
    # — and passing an unknown one is a TypeError, not a warning. Filter against
    # the dataclass itself so this script survives the version it meets, and say
    # which settings were dropped rather than silently ignoring them.
    import dataclasses

    wanted = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "beta": args.beta,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "bf16": torch.cuda.is_bf16_supported(),
        "seed": args.seed,
    }
    supported = {field.name for field in dataclasses.fields(DPOConfig)}
    dropped = sorted(set(wanted) - supported)
    if dropped:
        logger.warning(
            "DPOConfig in trl %s does not accept %s - ignoring", getattr_trl_version(), dropped
        )
    config = DPOConfig(**{k: v for k, v in wanted.items() if k in supported})

    trainer_kwargs = dict(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        peft_config=peft_config,
    )
    # trl renamed `tokenizer` to `processing_class`; accept either.
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    before = evaluate_preference_accuracy(trainer, test_rows, "before")
    trainer.train()
    after = evaluate_preference_accuracy(trainer, test_rows, "after")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir / "adapter"))
    (args.output_dir / "result.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "pairs_total": len(rows),
                "pairs_train": len(train_rows),
                "pairs_held_out": len(test_rows),
                "held_out_before": before,
                "held_out_after": after,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "beta": args.beta,
                "lora_rank": 16,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    logger.info(
        "mean margin %+.5f -> %+.5f (change %+.5f)",
        before["mean_margin"],
        after["mean_margin"],
        after["mean_margin"] - before["mean_margin"],
    )


def evaluate_preference_accuracy(trainer, rows: list[dict], label: str) -> float:
    """Fraction of held-out pairs where the accepted generation scores higher.

    Scores the **completion tokens only**. The first version of this averaged
    loss over prompt-plus-completion, and the pilot (job 10439928) returned
    exactly the same 0.286 before and after training — which was a property of
    the metric, not of the model. A pair here shares a prompt of a few hundred
    tokens and two completions of one to two thousand that differ in a single
    section, so a mean over every token dilutes the part under test by roughly
    an order of magnitude, and masking the prompt is the minimum needed to see
    anything at all.

    Accuracy alone is not enough at this sample size: over seven pairs it can
    only take values in sevenths, so it cannot separate "training had no effect"
    from "the effect was too small to flip any single pair". The **margin** --
    the length-normalised log-probability of the accepted completion minus that
    of the rejected one, per pair -- is continuous, and is what makes a null
    result interpretable rather than merely uninformative.

    Both a summed and a length-normalised score are reported because they carry
    opposite biases — a sum favours the shorter completion, a mean is
    indifferent to length — and with completions of unequal length neither alone
    is trustworthy. The summed figure is returned, since that is the quantity
    DPO's objective is defined over.
    """
    import torch

    if not rows:
        return float("nan")
    model = trainer.model
    tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
    model.eval()
    correct_sum = 0
    correct_mean = 0
    margins: list[float] = []
    with torch.no_grad():
        for row in rows:
            scored = {}
            for key in ("chosen", "rejected"):
                prompt_ids = tokenizer(row["prompt"], return_tensors="pt").input_ids
                full_ids = tokenizer(
                    row["prompt"] + "\n" + row[key],
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).input_ids
                n_prompt = min(prompt_ids.shape[1], full_ids.shape[1] - 1)
                full_ids = full_ids.to(model.device)
                logits = model(input_ids=full_ids).logits[:, :-1]
                targets = full_ids[:, 1:]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                token_logprobs = logprobs.gather(2, targets.unsqueeze(2)).squeeze(2)
                # Drop the prompt: only the generated section is under test.
                completion = token_logprobs[:, n_prompt:]
                if completion.numel() == 0:
                    completion = token_logprobs
                scored[key] = (completion.sum().item(), completion.mean().item())
            correct_sum += scored["chosen"][0] > scored["rejected"][0]
            correct_mean += scored["chosen"][1] > scored["rejected"][1]
            margins.append(scored["chosen"][1] - scored["rejected"][1])
    total = len(rows)
    mean_margin = sum(margins) / total
    logger.info(
        "[%s] n=%d | accuracy: summed %.3f, length-normalised %.3f | "
        "mean margin %+.5f | margins %s",
        label,
        total,
        correct_sum / total,
        correct_mean / total,
        mean_margin,
        [round(m, 4) for m in margins],
    )
    model.train()
    return {
        "accuracy_summed": correct_sum / total,
        "accuracy_length_normalised": correct_mean / total,
        "mean_margin": mean_margin,
        "margins": margins,
    }


if __name__ == "__main__":
    main()
