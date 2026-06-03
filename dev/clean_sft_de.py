#!/usr/bin/env python3
"""Cleaning-Pipeline für die deutschen SFT-Datasets (nanochat).

Liest ~/.cache/nanochat/sft_de_{train,val}.jsonl, wendet vier Schritte an und
schreibt bereinigte Files (*_clean.jsonl) plus ein *_review.jsonl für unsichere
Fälle. Druckt einen Vorher/Nachher-Report auf stdout. KEIN Training.

Datenformat (bestätigt via dev/inspect_sft_de.py + Sichtung): jede Zeile ist
*direkt* ein JSON-Array von Messages [{"role","content"}, ...]. Kein
{"messages":[...]}-Wrapper, kein source-Feld, nur Rollen user/assistant.
Der nanochat-Tokenizer (render_conversation) erwartet dagegen den
{"messages":[...]}-Wrapper -> wird beim Rendern angelegt.

Schritt-Reihenfolge: 1 (trailing-user) -> 4 (müll) -> 2 (sprache) -> 3 (BPE-len,
nur Report, kein Drop). So wird erst strukturell/müll gedroppt, dann Sprache
gefiltert und schließlich die Längen auf dem finalen Set gemessen.

Abhängigkeiten: stdlib + langdetect (==1.0.9) + nanochat-Tokenizer.
"""

import json
import os
import re
import sys
import unicodedata

from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# deterministische Spracherkennung
DetectorFactory.seed = 0

# Repo-Root in den Pfad, damit `nanochat` importierbar ist, egal von wo gestartet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.tokenizer import get_tokenizer

CACHE = os.path.expanduser("~/.cache/nanochat")
SPLITS = {
    "train": os.path.join(CACHE, "sft_de_train.jsonl"),
    "val": os.path.join(CACHE, "sft_de_val.jsonl"),
}

CTX_LIMIT = 2048          # Trainings-Kontextlänge des nanochat-Modells
SAMPLE = 4                # wie viele Beispiel-Drops pro Grund zeigen

# --- Sprach-Schwellen (Schritt 2) ---
# "mischsprachig/unsicher": en UND de beide substanziell vorhanden -> Review.
MIX_MIN_PROB = 0.20       # beide Sprachen >= 20% Wahrscheinlichkeit => mixed

# --- Müll-Erkennung (Schritt 4) ---
BOILERPLATE_PATTERNS = [
    r"as an ai language model",
    r"as an ai\b",
    r"i'?m sorry,? but i can'?t",
    r"i cannot fulfill",
    r"i'?m just an ai",
    r"as a large language model",
    r"i don'?t have personal",
    r"knowledge cutoff",
    r"\bopenai\b",
    r"i'?m unable to provide",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

MOJIBAKE_PATTERNS = ["Ã¤", "Ã¶", "Ã¼", "ÃŸ", "Ã„", "Ã–", "Ãœ", "â€™", "â€œ",
                     "â€\x9d", "â€“", "â€”", "Ã©", "Ã¨", "ï¿½", "�"]

# Steuerzeichen, die erhalten bleiben (alle anderen C*-Kategorien werden gestrippt)
KEEP_CTRL = {"\n", "\t"}


# --------------------------------------------------------------------------- #
# Hilfen
# --------------------------------------------------------------------------- #
def pct(part, whole):
    return (100.0 * part / whole) if whole else 0.0


def quantiles(vals):
    """min, median, p95, max."""
    if not vals:
        return (0, 0, 0, 0)
    s = sorted(vals)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    p95 = s[min(n - 1, int(0.95 * n))]
    return (s[0], median, p95, s[-1])


def strip_ctrl(s):
    """Entferne Steuerzeichen außer \\n und \\t. Gibt (neu, geändert?) zurück."""
    out = []
    changed = False
    for ch in s:
        if ch in KEEP_CTRL:
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):  # Control / Format / Surrogate / ...
            changed = True
            continue  # strippen
        out.append(ch)
    return ("".join(out), changed)


def has_mojibake(s):
    return any(p in s for p in MOJIBAKE_PATTERNS)


def assistant_text(msgs):
    """Konkatenierte Assistant-Antworten (für Spracherkennung)."""
    return "\n".join(
        m["content"] for m in msgs
        if isinstance(m, dict) and m.get("role") == "assistant"
        and isinstance(m.get("content"), str)
    )


def lang_probs(text):
    """dict lang->prob via langdetect; {} bei Fehler/leer."""
    if not text or not text.strip():
        return {}
    try:
        return {dl.lang: dl.prob for dl in detect_langs(text)}
    except LangDetectException:
        return {}


# --------------------------------------------------------------------------- #
# Per-Split-Pipeline
# --------------------------------------------------------------------------- #
class Counts:
    def __init__(self):
        self.n_in = 0
        self.invalid = 0
        self.drop_trailing_user = 0          # Schritt 1
        self.mod_ctrl = 0                    # Schritt 4 (geändert, nicht gedroppt)
        self.drop_mojibake = 0               # Schritt 4
        self.drop_boilerplate = 0            # Schritt 4
        # Schritt 2 — Sprachverteilung (ehrlich, über alle die Schritt 2 erreichen)
        self.lang_total = 0
        self.lang_de = 0
        self.lang_en = 0
        self.lang_other = 0
        self.lang_undetected = 0
        # Schritt 2 — Routing-Ergebnis
        self.drop_en = 0
        self.review_mixed = 0
        # final
        self.kept = 0
        # Schritt 3 — BPE
        self.bpe_lens = []
        self.over_ctx = 0
        self.over_ctx_trunc = []             # abgeschnittene Tokens pro betroffener Conv
        self.render_err = 0
        # Beispiele pro Grund: reason -> [(lineno, hint), ...]
        self.samples = {}

    def sample(self, reason, lineno, hint=""):
        self.samples.setdefault(reason, [])
        if len(self.samples[reason]) < SAMPLE:
            self.samples[reason].append((lineno, hint))


def process_split(name, path, tok):
    c = Counts()
    kept_convs = []      # list[ (lineno, msgs) ]
    review_convs = []    # list[ (lineno, msgs) ]

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            c.n_in += 1

            # --- parse + minimale Schema-Prüfung ---
            try:
                msgs = json.loads(raw)
            except Exception as e:  # noqa: BLE001
                c.invalid += 1
                c.sample("invalid_json", lineno, str(e)[:60])
                continue
            if not (isinstance(msgs, list) and len(msgs) > 0 and all(
                isinstance(m, dict) and isinstance(m.get("content"), str)
                and m.get("role") in ("user", "assistant") for m in msgs
            )):
                c.invalid += 1
                c.sample("invalid_schema", lineno)
                continue

            # ============ SCHRITT 1: trailing-user droppen ============
            if msgs[-1].get("role") != "assistant":
                c.drop_trailing_user += 1
                c.sample("trailing_user", lineno,
                         "letzte Rolle=%s" % msgs[-1].get("role"))
                continue

            # ============ SCHRITT 4: müll (ctrl-strip / mojibake / boilerplate) ===
            # 4a) Steuerzeichen strippen (modifiziert, droppt nicht)
            conv_changed = False
            for m in msgs:
                new_c, changed = strip_ctrl(m["content"])
                if changed:
                    m["content"] = new_c
                    conv_changed = True
            if conv_changed:
                c.mod_ctrl += 1
                c.sample("ctrl_stripped", lineno)

            # 4b) Mojibake -> droppen
            if any(has_mojibake(m["content"]) for m in msgs):
                c.drop_mojibake += 1
                c.sample("mojibake", lineno)
                continue
            # 4c) EN-Boilerplate -> droppen (in irgendeiner Message)
            bp_hit = None
            for m in msgs:
                mo = BOILERPLATE_RE.search(m["content"])
                if mo:
                    bp_hit = mo.group(0)
                    break
            if bp_hit is not None:
                c.drop_boilerplate += 1
                c.sample("boilerplate", lineno, repr(bp_hit))
                continue

            # ============ SCHRITT 2: Sprache messen + filtern ============
            text = assistant_text(msgs)
            probs = lang_probs(text)
            c.lang_total += 1
            if not probs:
                c.lang_undetected += 1
                top = None
            else:
                top = max(probs, key=probs.get)
                if top == "de":
                    c.lang_de += 1
                elif top == "en":
                    c.lang_en += 1
                else:
                    c.lang_other += 1

            p_en = probs.get("en", 0.0)
            p_de = probs.get("de", 0.0)
            mixed = (p_en >= MIX_MIN_PROB and p_de >= MIX_MIN_PROB)

            if mixed:
                # nicht blind droppen -> Review
                c.review_mixed += 1
                c.sample("mixed_review", lineno,
                         "de=%.2f en=%.2f" % (p_de, p_en))
                review_convs.append((lineno, msgs))
                continue
            if top == "en":
                # eindeutig englische Assistant-Antwort -> droppen
                c.drop_en += 1
                c.sample("english", lineno, "en=%.2f" % p_en)
                continue

            # behalten (de / other / undetected)
            kept_convs.append((lineno, msgs))

    # ============ SCHRITT 3: echte BPE-Längen auf dem finalen (kept) Set ====
    for lineno, msgs in kept_convs:
        try:
            ids, _ = tok.render_conversation({"messages": msgs},
                                             max_tokens=10 ** 9)
        except Exception:  # noqa: BLE001  (z.B. Alternations-Assertion)
            c.render_err += 1
            continue
        n = len(ids)
        c.bpe_lens.append(n)
        if n > CTX_LIMIT:
            c.over_ctx += 1
            c.over_ctx_trunc.append(n - CTX_LIMIT)

    c.kept = len(kept_convs)

    # ============ Files schreiben ============
    base = os.path.join(CACHE, "sft_de_%s" % name)
    clean_path = base + "_clean.jsonl"
    review_path = base + "_review.jsonl"
    with open(clean_path, "w", encoding="utf-8") as fh:
        for _, msgs in kept_convs:
            fh.write(json.dumps(msgs, ensure_ascii=False) + "\n")
    with open(review_path, "w", encoding="utf-8") as fh:
        for _, msgs in review_convs:
            fh.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    return c, clean_path, review_path


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def hr(title=""):
    if title:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
    else:
        print("-" * 78)


def row(label, *vals):
    cells = "  ".join("%14s" % v for v in vals)
    print("%-32s %s" % (label, cells))


def print_samples(results, reason, header):
    lines = []
    for name, c in results.items():
        for lineno, hint in c.samples.get(reason, []):
            lines.append("    [%s] L%-7d %s" % (name, lineno, hint))
    if lines:
        print("\n  %s:" % header)
        for ln in lines:
            print(ln)


def main():
    for p in SPLITS.values():
        if not os.path.exists(p):
            sys.exit("FEHLT: %s" % p)

    print("Lade nanochat-Tokenizer ...")
    tok = get_tokenizer()

    results = {}
    paths = {}
    for name, path in SPLITS.items():
        print("Verarbeite %-5s (%s) ..." % (name, path))
        c, cp, rp = process_split(name, path, tok)
        results[name] = c
        paths[name] = (cp, rp)

    train = results["train"]
    val = results["val"]

    # ===== Schritt 1 + 4: Drops =====
    hr("DROPS — SCHRITT 1 (trailing-user) + SCHRITT 4 (müll)")
    row("", "train", "val")
    row("eingelesen", train.n_in, val.n_in)
    row("invalide (geskippt)", train.invalid, val.invalid)
    row("[S1] trailing-user gedroppt",
        train.drop_trailing_user, val.drop_trailing_user)
    row("[S4] ctrl-zeichen gestrippt (mod)", train.mod_ctrl, val.mod_ctrl)
    row("[S4] mojibake gedroppt", train.drop_mojibake, val.drop_mojibake)
    row("[S4] boilerplate gedroppt",
        train.drop_boilerplate, val.drop_boilerplate)
    print_samples(results, "trailing_user", "Beispiele trailing-user")
    print_samples(results, "mojibake", "Beispiele mojibake")
    print_samples(results, "boilerplate", "Beispiele boilerplate (Treffer)")

    # ===== Schritt 2: Sprachverteilung =====
    hr("SCHRITT 2 — SPRACHE DER ASSISTANT-ANTWORTEN (ehrliche Verteilung)")
    print("  (gemessen NACH Schritt 1+4, also auf strukturell sauberem Set)\n")
    row("", "train", "val", "train %", "val %")
    for key, lbl in [("lang_de", "de (top-lang)"),
                     ("lang_en", "en (top-lang)"),
                     ("lang_other", "andere sprache"),
                     ("lang_undetected", "nicht erkennbar")]:
        tv, vv = getattr(train, key), getattr(val, key)
        row(lbl, tv, vv,
            "%.1f%%" % pct(tv, train.lang_total),
            "%.1f%%" % pct(vv, val.lang_total))
    row("gesamt (Schritt-2-Eingang)", train.lang_total, val.lang_total)

    hr("SCHRITT 2 — FILTER-ROUTING")
    row("", "train", "val")
    row("englisch gedroppt", train.drop_en, val.drop_en)
    row("mixed/unsicher -> review", train.review_mixed, val.review_mixed)
    row("  (Schwelle: de>=%.0f%% UND en>=%.0f%%)"
        % (MIX_MIN_PROB * 100, MIX_MIN_PROB * 100), "", "")
    print_samples(results, "english", "Beispiele englisch (gedroppt)")
    print_samples(results, "mixed_review", "Beispiele mixed (-> review)")

    # ===== Schritt 3: BPE-Längen =====
    hr("SCHRITT 3 — ECHTE BPE-LÄNGEN (nanochat render_conversation, finales Set)")
    print("  Chat-Format inkl. Special-Tokens (<|bos|>, <|user_start|> ...),")
    print("  ohne Truncation gemessen. NICHT gedroppt — nur Report.\n")
    row("", "min", "median", "p95", "max")
    for name, c in results.items():
        mn, md, p95, mx = quantiles(c.bpe_lens)
        row("  %s" % name, mn, md, p95, mx)
    print()
    row("", "train", "val")
    row("Conv. > %d Tokens" % CTX_LIMIT, train.over_ctx, val.over_ctx)
    row("  %% des finalen Sets",
        "%.2f%%" % pct(train.over_ctx, len(train.bpe_lens) or 1),
        "%.2f%%" % pct(val.over_ctx, len(val.bpe_lens) or 1))
    t_avg = (sum(train.over_ctx_trunc) / len(train.over_ctx_trunc)
             if train.over_ctx_trunc else 0)
    v_avg = (sum(val.over_ctx_trunc) / len(val.over_ctx_trunc)
             if val.over_ctx_trunc else 0)
    row("  Ø abgeschnittene Tokens", "%.0f" % t_avg, "%.0f" % v_avg)
    row("  max abgeschnitten",
        max(train.over_ctx_trunc) if train.over_ctx_trunc else 0,
        max(val.over_ctx_trunc) if val.over_ctx_trunc else 0)
    if train.render_err or val.render_err:
        row("render-Fehler (übersprungen)", train.render_err, val.render_err)
    print("\n  -> NICHT automatisch gedroppt. Entscheidung liegt bei dir.")

    # ===== Endbilanz =====
    hr("ENDBILANZ — ÜBRIG PRO SPLIT")
    row("", "train", "val")
    row("vorher (eingelesen)", train.n_in, val.n_in)
    total_drop_t = (train.invalid + train.drop_trailing_user
                    + train.drop_mojibake + train.drop_boilerplate
                    + train.drop_en)
    total_drop_v = (val.invalid + val.drop_trailing_user
                    + val.drop_mojibake + val.drop_boilerplate + val.drop_en)
    row("gedroppt gesamt", total_drop_t, total_drop_v)
    row("-> review (separat)", train.review_mixed, val.review_mixed)
    row("CLEAN übrig", train.kept, val.kept)
    row("  Anteil behalten",
        "%.1f%%" % pct(train.kept, train.n_in),
        "%.1f%%" % pct(val.kept, val.n_in))

    hr("GESCHRIEBENE FILES")
    for name in SPLITS:
        cp, rp = paths[name]
        print("  %-5s clean : %s  (%d conv.)"
              % (name, cp, results[name].kept))
        print("  %-5s review: %s  (%d conv.)"
              % (name, rp, results[name].review_mixed))

    hr()
    print("Hinweis: Schritt 3 misst die echten BPE-Längen, droppt aber nichts.")
    print("Fertig.")


if __name__ == "__main__":
    main()
