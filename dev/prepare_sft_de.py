"""
Build a curated, mixed German SFT dataset for nanochat.

Sources (configurable below):
  1. seedboxai/multitask_german_examples_32k   -- filtered (MC answers + screenplays removed)
  2. OpenAssistant/oasst2                       -- German threads only, best-ranked path
  3. a German Dolly/Alpaca instruction set      -- VERIFY repo id on HF before use
  4. fmorgens/smol-smoltalk-german              -- conversational; filtered for EN leakage + markers

Pipeline:
  load each source -> normalize to [{"role","content"}, ...] (strict user/assistant alternation)
  -> filter (MC-only answers, screenplays, pipeline markers, English turns, too-long, malformed)
  -> dedup across sources -> shuffle -> cap to target

Outputs:
  A) nanochat CustomJSON:  <base_dir>/sft_de_train.jsonl  +  sft_de_val.jsonl
  B) HF dataset (optional, --hf-repo):  DatasetDict{train,val} with a single "messages" column

Run (from repo root):
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

VAL_SIZE = 5000
MAX_CHARS = 9000
MIN_CHARS = 8
SEED = 42

SOURCES = {
    "seedboxai":  {"cap": 25000},
    "oasst_de":   {"cap": 15000},
    "dolly_de":   {"cap": 10000, "repo": "mayflowergmbh/dolly-15k_de", "enabled": True},
    # NEW: conversational German data (translated SmolTalk). Heavily filtered below.
    "smoltalk_de": {"cap": 20000, "repo": "fmorgens/smol-smoltalk-german", "enabled": True},
}

# -----------------------------------------------------------------------------
# Filters

_MC_RE = re.compile(r"^`{0,3}\s*[A-D]\s*[.)]?\s*`{0,3}$")
_NAME_LINE_RE = re.compile(r"^[A-ZÄÖÜ][A-ZÄÖÜ' ]{1,24}$")
_SCRIPT_MARKERS = ("INT.", "EXT.", "FADE IN", "FADE OUT", "CUT TO")
# Broken pipeline markers seen in smol-smoltalk-german, e.g. <<<MSG_0 role=user>>>
_PIPELINE_MARKER_RE = re.compile(r"<<<\s*(?:END_)?MSG_\d+|role\s*=\s*(?:user|assistant)\s*>>>", re.IGNORECASE)

_EN_STOPWORDS = {
    "the", "and", "is", "are", "was", "were", "of", "to", "in", "it", "that",
    "this", "with", "for", "you", "have", "has", "my", "your", "an", "their",
    "by", "from", "on", "as", "at", "be", "or", "but", "not", "they", "we", "he",
    "she", "would", "could", "should", "a", "i",
}

# Generic assistant-persona prefix folded into the first user turn, e.g.
# "Du bist ein KI-Assistent, der ... .\n\n<actual task>". nanochat has no system
# role, so these pseudo-system blurbs create a train/inference mismatch -> strip them.
_PERSONA_RE = re.compile(r"^(du bist\b|als\s+\w+).*\bassistent\w*\b.*$", re.IGNORECASE | re.DOTALL)


def strip_persona_prefix(content):
    """Strip a leading generic assistant-persona paragraph, keep the real task/context."""
    parts = content.split("\n\n", 1)
    if len(parts) == 2:
        first = parts[0].strip()
        # short first paragraph + persona phrasing => it's a system blurb, not the task
        if len(first) <= 700 and _PERSONA_RE.match(first):
            return parts[1].strip()
    return content

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False


def is_mc_answer(text):
    return bool(_MC_RE.match(text.strip()))


def is_screenplay(text):
    if any(text.lstrip().startswith(m) for m in _SCRIPT_MARKERS):
        return True
    name_lines = sum(1 for ln in text.splitlines() if _NAME_LINE_RE.match(ln.strip()))
    return name_lines >= 3


def has_pipeline_markers(text):
    return bool(_PIPELINE_MARKER_RE.search(text))


def is_mostly_english(text):
    """Reject turns that are predominantly English OR contain a long English block."""
    t = text.strip()
    if len(t) < 40:
        return False
    # langdetect handles whole-turn English
    if _HAS_LANGDETECT:
        try:
            if detect(t) == "en":
                return True
        except Exception:
            pass
    words = re.findall(r"[A-Za-zÄÖÜäöüß']+", t)
    # contiguous English run: catches German-framed answers with big English quotes
    run = 0
    for w in words:
        if w.lower() in _EN_STOPWORDS:
            run += 1
            if run >= 6:
                return True
        else:
            run = 0
    # global ratio fallback
    lw = [w.lower() for w in words][:200]
    if not lw:
        return False
    return sum(1 for w in lw if w in _EN_STOPWORDS) / len(lw) > 0.15


def clean_conversation(msgs):
    if not msgs or len(msgs) < 2:
        return None
    # strip a generic assistant-persona prefix from the very first user turn
    if msgs[0].get("role") == "user":
        stripped = strip_persona_prefix((msgs[0].get("content") or "").strip())
        msgs[0] = {"role": "user", "content": stripped}
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected or not content:
            return None
        if has_pipeline_markers(content):
            return None
        if role == "assistant":
            if len(content) < MIN_CHARS:
                return None
            if is_mc_answer(content) or is_screenplay(content):
                return None
            if is_mostly_english(content):
                return None
    if sum(len(m["content"]) for m in msgs) > MAX_CHARS:
        return None
    return [{"role": m["role"], "content": m["content"].strip()} for m in msgs]


def conv_hash(msgs):
    blob = "\n".join(f"{m['role']}:{m['content']}" for m in msgs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Source loaders

def load_seedboxai(cap=None):
    ds = load_dataset("seedboxai/multitask_german_examples_32k", split="train", streaming=True)
    n = 0
    for row in ds:
        msgs = list(row["prompt"]) + [row["completion"]]
        if msgs and msgs[0]["role"] == "system":
            sys_c = (msgs[0]["content"] or "").strip()
            msgs = msgs[1:]
            if msgs and msgs[0]["role"] == "user" and sys_c:
                msgs[0] = {"role": "user", "content": sys_c + "\n\n" + msgs[0]["content"]}
        yield [{"role": m["role"], "content": m.get("content", "")} for m in msgs]
        n += 1
        if cap and n >= cap * 3:
            break


def load_oasst_de(cap=None):
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
        conv, node = [], root
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


def load_smoltalk_de(repo, cap=None):
    """fmorgens/smol-smoltalk-german: HF 'messages' column. Normalize role names."""
    ds = load_dataset(repo, split="train", streaming=True)
    n = 0
    for row in ds:
        msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
        if not msgs:
            continue
        out = []
        for m in msgs:
            role = m.get("role") or m.get("from")
            if role in ("human", "prompter"):
                role = "user"
            elif role in ("gpt", "bot"):
                role = "assistant"
            out.append({"role": role, "content": m.get("content") or m.get("value") or ""})
        yield out
        n += 1
        if cap and n >= cap * 3:
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
    ap.add_argument("--target", type=int, default=50000)
    ap.add_argument("--hf-repo", type=str, default=None)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    random.seed(SEED)
    base_dir = get_base_dir()
    seen = set()
    pools = []

    if not _HAS_LANGDETECT:
        print("NOTE: langdetect not installed -> using stopword heuristic for English filtering.")
        print("      For better accuracy: pip install langdetect")

    print("Loading sources ...")
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

    s = SOURCES["smoltalk_de"]
    if s.get("enabled", True):
        try:
            pools += collect(load_smoltalk_de(s["repo"], s["cap"]), s["cap"], "smoltalk_de", seen)
        except Exception as e:
            print(f"  [smoltalk_de] SKIPPED ({s['repo']}): {e}")

    print(f"\nCombined clean pool: {len(pools):,} conversations")
    random.shuffle(pools)
    if len(pools) > args.target:
        pools = pools[:args.target]
        print(f"Capped to target: {len(pools):,}")

    val = pools[:VAL_SIZE]
    train = pools[VAL_SIZE:]
    print(f"Split -> train {len(train):,} | val {len(val):,}")

    train_path = os.path.join(base_dir, "sft_de_train.jsonl")
    val_path = os.path.join(base_dir, "sft_de_val.jsonl")
    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for conv in data:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        print(f"  wrote {path}  ({len(data):,} rows)")

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