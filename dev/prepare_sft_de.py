"""
Build a curated, mixed German SFT dataset for nanochat.

Sources (configurable below):
  1. seedboxai/multitask_german_examples_32k   -- filtered (MC answers + screenplays removed)
  2. OpenAssistant/oasst2                       -- German threads only, best-ranked path
  3. a German Dolly/Alpaca instruction set      -- VERIFY repo id on HF before use (see SOURCES)

Pipeline:
  load each source -> normalize to [{"role","content"}, ...] (strict user/assistant alternation)
  -> filter (MC-only answers, screenplays, too-long, malformed)
  -> dedup across sources -> shuffle -> cap to TARGET_TOTAL

Outputs:
  A) nanochat CustomJSON:  <base_dir>/sft_de_train.jsonl  +  sft_de_val.jsonl
       (one JSON array per line: [{"role":"user",...},{"role":"assistant",...}])
  B) HF dataset (optional, --hf-repo):  DatasetDict{train,val} with a single "messages" column

Run (from repo root, so `nanochat` is importable):
  python -m dev.build_sft_de_mix
  python -m dev.build_sft_de_mix --target 50000 --hf-repo your-username/german-sft-mix --push
"""

import os
import re
import json
import random
import hashlib
import argparse

from datasets import load_dataset, Dataset, DatasetDict

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Config

VAL_SIZE = 5000           # held-out conversations for validation
MAX_CHARS = 9000          # ~2048 tokens * 4.56 chars/token (your measured DE ratio) + headroom
MIN_CHARS = 8             # drop trivially short assistant answers
SEED = 42

# Per-source caps applied BEFORE the global cap, so no single source dominates.
# Set to None for "take all (clean)". Tune to taste.
SOURCES = {
    "seedboxai": {"cap": 30000},
    "oasst_de":  {"cap": 15000},
    # NOTE: verify the exact repo id on the HF Hub before relying on this one.
    # Candidates seen in the wild: "mayflowergmbh/dolly-15k_de", "mayflowergmbh/alpaca-gpt4_de".
    # Disable by setting "enabled": False if the repo id is wrong / unavailable.
    "dolly_de":  {"cap": 10000, "repo": "mayflowergmbh/dolly-15k_de", "enabled": True},
}

# -----------------------------------------------------------------------------
# Filters (shared across sources)

# Matches assistant answers that are just a multiple-choice letter, optionally in backticks:
#   "A", "```B```", "`C`", "D."  -> these teach MC behavior, bad for a chat model.
_MC_RE = re.compile(r"^`{0,3}\s*[A-D]\s*[.)]?\s*`{0,3}$")
# Screenplay name line: a short ALL-CAPS line that is just a character name (KAREN, EMILY).
_NAME_LINE_RE = re.compile(r"^[A-ZÄÖÜ][A-ZÄÖÜ' ]{1,24}$")
# Stage-direction / script markers.
_SCRIPT_MARKERS = ("INT.", "EXT.", "FADE IN", "FADE OUT", "CUT TO")


def is_mc_answer(text: str) -> bool:
    return bool(_MC_RE.match(text.strip()))


def is_screenplay(text: str) -> bool:
    """Heuristic: many ALL-CAPS name lines and/or script slug lines => screenplay/roleplay."""
    if any(text.lstrip().startswith(m) for m in _SCRIPT_MARKERS):
        return True
    name_lines = sum(1 for ln in text.splitlines() if _NAME_LINE_RE.match(ln.strip()))
    return name_lines >= 3


def clean_conversation(msgs):
    """
    msgs: list of {"role","content"}. Returns a validated/cleaned conversation or None.
    Rules: strict user/assistant alternation starting with user; no empty turns;
    reject if ANY assistant turn is a bare MC answer or screenplay; length-bounded.
    """
    if not msgs or len(msgs) < 2:
        return None

    for i, m in enumerate(msgs):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected or not content:
            return None
        if role == "assistant":
            if len(content) < MIN_CHARS:
                return None
            if is_mc_answer(content) or is_screenplay(content):
                return None

    if sum(len(m["content"]) for m in msgs) > MAX_CHARS:
        return None

    return [{"role": m["role"], "content": (m["content"]).strip()} for m in msgs]


def conv_hash(msgs) -> str:
    blob = "\n".join(f"{m['role']}:{m['content']}" for m in msgs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Source loaders -> each yields raw [{"role","content"}, ...] (pre-clean)

def load_seedboxai(cap=None):
    ds = load_dataset("seedboxai/multitask_german_examples_32k", split="train", streaming=True)
    n = 0
    for row in ds:
        msgs = list(row["prompt"]) + [row["completion"]]
        # fold optional leading system message into first user turn
        if msgs and msgs[0]["role"] == "system":
            sys_c = (msgs[0]["content"] or "").strip()
            msgs = msgs[1:]
            if msgs and msgs[0]["role"] == "user" and sys_c:
                msgs[0] = {"role": "user", "content": sys_c + "\n\n" + msgs[0]["content"]}
        yield [{"role": m["role"], "content": m.get("content", "")} for m in msgs]
        n += 1
        if cap and n >= cap * 3:   # over-fetch; many will be filtered out
            break


def load_oasst_de(cap=None):
    """
    OASST is a message tree. Reconstruct linear best-ranked German conversations.
    role: 'prompter' -> user, 'assistant' -> assistant. Pick rank-0 reply at each step.
    """
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    ds = ds.filter(lambda r: r.get("lang") == "de" and not r.get("deleted", False))

    by_id, children = {}, {}
    for r in ds:
        by_id[r["message_id"]] = r
        children.setdefault(r["parent_id"], []).append(r)

    def best_child(parent_id, role):
        kids = [c for c in children.get(parent_id, []) if c["role"] == role]
        if not kids:
            return None
        kids.sort(key=lambda c: (c["rank"] is None, c["rank"] if c["rank"] is not None else 1e9))
        return kids[0]

    roots = [r for r in by_id.values() if r["parent_id"] is None and r["role"] == "prompter"]
    n = 0
    for root in roots:
        conv, node, role = [], root, "prompter"
        while node is not None:
            conv.append({"role": "user" if node["role"] == "prompter" else "assistant",
                         "content": node["text"]})
            next_role = "assistant" if node["role"] == "prompter" else "prompter"
            node = best_child(node["message_id"], next_role)
        if len(conv) >= 2:
            yield conv
            n += 1
            if cap and n >= cap * 2:
                break


def load_instruction_de(repo, cap=None):
    """Generic instruction->output loader (Dolly/Alpaca style). Tries common field names."""
    ds = load_dataset(repo, split="train")
    n = 0
    for r in ds:
        instr = r.get("instruction") or r.get("prompt") or r.get("input_text") or ""
        ctx = r.get("input") or r.get("context") or ""
        out = r.get("output") or r.get("response") or r.get("completion") or ""
        user = (instr + ("\n\n" + ctx if ctx.strip() else "")).strip()
        if not user or not out.strip():
            continue
        yield [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
        n += 1
        if cap and n >= cap * 2:
            break


# -----------------------------------------------------------------------------

def collect(loader, cap, label, seen):
    kept = []
    for raw in loader:
        conv = clean_conversation(raw)
        if conv is None:
            continue
        h = conv_hash(conv)
        if h in seen:
            continue
        seen.add(h)
        kept.append(conv)
        if cap and len(kept) >= cap:
            break
    print(f"  [{label}] kept {len(kept):,}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50000, help="total conversations after mixing")
    ap.add_argument("--hf-repo", type=str, default=None, help="HF dataset repo id (e.g. user/german-sft-mix)")
    ap.add_argument("--push", action="store_true", help="actually push to the HF Hub")
    ap.add_argument("--private", action="store_true", help="push as a private dataset")
    args = ap.parse_args()

    random.seed(SEED)
    base_dir = get_base_dir()
    seen = set()
    pools = []

    print("Loading sources …")
    s = SOURCES["seedboxai"]
    pools += collect(load_seedboxai(s["cap"]), s["cap"], "seedboxai", seen)

    s = SOURCES["oasst_de"]
    pools += collect(load_oasst_de(s["cap"]), s["cap"], "oasst_de", seen)

    s = SOURCES["dolly_de"]
    if s.get("enabled", True):
        try:
            pools += collect(load_instruction_de(s["repo"], s["cap"]), s["cap"], "dolly_de", seen)
        except Exception as e:
            print(f"  [dolly_de] SKIPPED ({s['repo']}): {e}")

    print(f"\nCombined clean pool: {len(pools):,} conversations")
    random.shuffle(pools)
    if len(pools) > args.target:
        pools = pools[:args.target]
        print(f"Capped to target: {len(pools):,}")

    val = pools[:VAL_SIZE]
    train = pools[VAL_SIZE:]
    print(f"Split -> train {len(train):,} | val {len(val):,}")

    # A) nanochat CustomJSON JSONL
    train_path = os.path.join(base_dir, "sft_de_train.jsonl")
    val_path = os.path.join(base_dir, "sft_de_val.jsonl")
    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for conv in data:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        print(f"  wrote {path}  ({len(data):,} rows)")

    # B) HF dataset (messages column)
    if args.hf_repo:
        dd = DatasetDict({
            "train": Dataset.from_list([{"messages": c} for c in train]),
            "validation": Dataset.from_list([{"messages": c} for c in val]),
        })
        if args.push:
            dd.push_to_hub(args.hf_repo, private=args.private)
            print(f"  pushed to https://huggingface.co/datasets/{args.hf_repo}")
        else:
            local = os.path.join(base_dir, "german_sft_mix_hf")
            dd.save_to_disk(local)
            print(f"  saved HF dataset locally to {local} (use --push to upload)")


if __name__ == "__main__":
    main()