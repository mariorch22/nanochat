#!/usr/bin/env python3
"""Standalone-Qualitätsanalyse für die deutschen SFT-Datasets.

Liest ~/.cache/nanochat/sft_de_{train,val}.jsonl und druckt einen kompakten
Report auf stdout. Kein Training, nur Analyse. Nur stdlib.

Datenformat (bestätigt durch Inspektion): jede Zeile ist *direkt* ein JSON-Array
von Message-Objekten, z.B.
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
Es gibt KEIN {"messages": [...]}-Wrapper und KEIN source-Feld. Der Loader
unten akzeptiert beide Formen, falls sich das spätere ändert.
"""

import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from hashlib import sha1

CACHE = os.path.expanduser("~/.cache/nanochat")
TRAIN = os.path.join(CACHE, "sft_de_train.jsonl")
VAL = os.path.join(CACHE, "sft_de_val.jsonl")

VALID_ROLES = {"system", "user", "assistant"}
CTX_LIMIT = 2048          # angenommene Kontextlänge des nanochat-Modells
SAMPLE_BAD = 5            # wie viele fehlerhafte Zeilen zeigen
TOP_CLUSTERS = 10         # wie viele Near-Duplicate-Cluster zeigen

# Boilerplate-/Übersetzungsartefakte, die in einem deutschen Datensatz nichts
# zu suchen haben (zurückgelassene englische LLM-Floskeln etc.).
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

# Typische Mojibake-Sequenzen (UTF-8 als Latin-1 fehlinterpretiert).
MOJIBAKE_PATTERNS = ["Ã¤", "Ã¶", "Ã¼", "ÃŸ", "Ã„", "Ã–", "Ãœ", "â€™", "â€œ",
                     "â€\x9d", "â€“", "â€”", "Ã©", "Ã¨", "ï¿½", "�"]

# erlaubte Steuerzeichen
ALLOWED_CTRL = {"\n", "\r", "\t"}


# --------------------------------------------------------------------------- #
# Laden / Parsen
# --------------------------------------------------------------------------- #
def load(path):
    """Yield (lineno, raw_line, parsed_or_None, error_or_None)."""
    with open(path, "r", encoding="utf-8") as fh:
        for i, raw in enumerate(fh, 1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                yield i, raw, None, "empty line"
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:  # noqa: BLE001
                yield i, raw, None, "JSON: %s" % e
                continue
            yield i, raw, obj, None


def as_messages(obj):
    """Normiere ein geparstes Objekt zu einer Message-Liste (oder None)."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        return obj["messages"]
    return None


def get_source(obj):
    if isinstance(obj, dict):
        return obj.get("source")
    return None


# --------------------------------------------------------------------------- #
# Validierung einer Conversation
# --------------------------------------------------------------------------- #
def validate_messages(msgs):
    """Gib Liste von Fehlerstrings zurück (leer = ok)."""
    errs = []
    if not isinstance(msgs, list):
        return ["top-level nicht Liste/messages"]
    if len(msgs) == 0:
        errs.append("leere conversation")
        return errs
    for j, m in enumerate(msgs):
        if not isinstance(m, dict):
            errs.append("msg[%d] kein dict" % j)
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in VALID_ROLES:
            errs.append("msg[%d] role=%r" % (j, role))
        if not isinstance(content, str):
            errs.append("msg[%d] content kein str" % j)
        elif content.strip() == "":
            errs.append("msg[%d] content leer" % j)
    return errs


# --------------------------------------------------------------------------- #
# Hilfen
# --------------------------------------------------------------------------- #
def norm_text(s):
    """lowercase + whitespace kollabiert — für Near-Dup-Vergleich."""
    return re.sub(r"\s+", " ", s.strip().lower())


def first_user_msg(msgs):
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
    return None


def conv_text(msgs):
    parts = []
    for m in msgs:
        if isinstance(m, dict):
            parts.append("%s:%s" % (m.get("role"), m.get("content")))
    return "\n".join(parts)


def h(s):
    return sha1(s.encode("utf-8", "replace")).hexdigest()


def approx_tokens(s):
    """grobe Token-Schätzung via Whitespace-Split."""
    return len(s.split())


def has_mojibake(s):
    return any(p in s for p in MOJIBAKE_PATTERNS)


def has_ctrl(s):
    for ch in s:
        if ch in ALLOWED_CTRL:
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch != "�":  # C* = Control/Format/...
            return True
    return False


def pct(part, whole):
    return (100.0 * part / whole) if whole else 0.0


def quantiles(vals):
    """min, median, p95, max für eine nicht-leere Werteliste."""
    if not vals:
        return (0, 0, 0, 0)
    s = sorted(vals)
    n = len(s)

    def q(p):
        idx = min(n - 1, int(p * n))
        return s[idx]

    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return (s[0], median, q(0.95), s[-1])


# --------------------------------------------------------------------------- #
# Pro-Split-Analyse
# --------------------------------------------------------------------------- #
class SplitStats:
    def __init__(self, name):
        self.name = name
        self.n_lines = 0
        self.n_valid = 0
        self.bad = []                      # (lineno, error, raw-preview)
        self.first_user_hashes = {}        # hash -> lineno
        self.conv_hashes = {}              # hash -> lineno
        self.first_user_norm = Counter()   # normalisierte erste user-msg
        self.first_user_examples = {}      # norm -> beispieltext
        self.sources = Counter()
        self.has_source_field = False

        self.conv_tok = []                 # tokens pro conversation
        self.msg_tok = []                  # tokens pro message
        self.over_ctx = 0                  # conversations > CTX_LIMIT

        self.single_turn = 0
        self.multi_turn = 0
        self.starts_user = 0
        self.bad_alternation = 0
        self.no_final_assistant = 0    # endet nicht auf assistant (kein Lernziel)

        self.empty_assistant = 0
        self.mojibake = 0
        self.ctrlchars = 0
        self.boilerplate = 0
        self.boilerplate_ex = []


def analyze(path, name):
    st = SplitStats(name)
    for lineno, raw, obj, err in load(path):
        st.n_lines += 1
        if err is not None:
            if len(st.bad) < SAMPLE_BAD:
                st.bad.append((lineno, err, raw[:120]))
            continue

        msgs = as_messages(obj)
        verrs = validate_messages(msgs)
        if verrs:
            if len(st.bad) < SAMPLE_BAD:
                st.bad.append((lineno, "; ".join(verrs[:3]), raw[:120]))
            # trotzdem so viel wie möglich weiter auswerten? Nein: skip strukturell kaputte.
            if msgs is None or len(msgs) == 0:
                continue

        st.n_valid += 1

        src = get_source(obj)
        if src is not None:
            st.has_source_field = True
            st.sources[src] += 1

        # ---- Leak-Hashes ----
        fu = first_user_msg(msgs)
        if fu is not None:
            fh = h(norm_text(fu))
            st.first_user_hashes.setdefault(fh, lineno)
            nf = norm_text(fu)
            st.first_user_norm[nf] += 1
            st.first_user_examples.setdefault(nf, fu)
        ch = h(conv_text(msgs))
        st.conv_hashes.setdefault(ch, lineno)

        # ---- Längen ----
        conv_t = 0
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str):
                t = approx_tokens(c)
                st.msg_tok.append(t)
                conv_t += t
                # ---- Encoding/Müll ----
                if has_mojibake(c):
                    st.mojibake += 1
                if has_ctrl(c):
                    st.ctrlchars += 1
                if m.get("role") == "assistant":
                    if c.strip() == "":
                        st.empty_assistant += 1
                    if BOILERPLATE_RE.search(c):
                        st.boilerplate += 1
                        if len(st.boilerplate_ex) < 3:
                            mobj = BOILERPLATE_RE.search(c)
                            st.boilerplate_ex.append(
                                (lineno, mobj.group(0)))
        st.conv_tok.append(conv_t)
        if conv_t > CTX_LIMIT:
            st.over_ctx += 1

        # ---- Rollen-Struktur ----
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        if roles and roles[0] == "user":
            st.starts_user += 1
        # saubere Alternation user/assistant/user/...
        ok_alt = bool(roles) and roles[0] == "user"
        for k in range(1, len(roles)):
            if roles[k] == roles[k - 1]:
                ok_alt = False
                break
        if not ok_alt:
            st.bad_alternation += 1
        if not roles or roles[-1] != "assistant":
            st.no_final_assistant += 1
        # turns = anzahl user-messages
        n_user = sum(1 for r in roles if r == "user")
        if n_user <= 1:
            st.single_turn += 1
        else:
            st.multi_turn += 1

    return st


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
    cells = "  ".join("%12s" % v for v in vals)
    print("%-34s %s" % (label, cells))


def main():
    for p in (TRAIN, VAL):
        if not os.path.exists(p):
            sys.exit("FEHLT: %s" % p)

    train = analyze(TRAIN, "train")
    val = analyze(VAL, "val")

    # ===== 1. Schema-Validität =====
    hr("1. SCHEMA-VALIDITÄT")
    row("", "train", "val")
    row("Zeilen gesamt", train.n_lines, val.n_lines)
    row("valide Conversations", train.n_valid, val.n_valid)
    row("fehlerhaft", train.n_lines - train.n_valid,
        val.n_lines - val.n_valid)
    for st in (train, val):
        if st.bad:
            print("\n  erste fehlerhafte Zeilen [%s]:" % st.name)
            for lineno, err, prev in st.bad:
                print("    L%-7d %s" % (lineno, err))
                print("            | %s" % prev)

    # ===== 2. Train/Val-Leak =====
    hr("2. TRAIN/VAL-LEAK")
    fu_overlap = set(train.first_user_hashes) & set(val.first_user_hashes)
    conv_overlap = set(train.conv_hashes) & set(val.conv_hashes)
    row("", "count", "% von val")
    row("gleiche erste User-Msg (hash)", len(fu_overlap),
        "%.2f%%" % pct(len(fu_overlap), val.n_valid))
    row("identische Conversation (hash)", len(conv_overlap),
        "%.2f%%" % pct(len(conv_overlap), val.n_valid))
    if conv_overlap:
        print("\n  Beispiel-Leaks (val-Zeile / train-Zeile):")
        for chash in list(conv_overlap)[:SAMPLE_BAD]:
            print("    val L%-7d  <->  train L%d"
                  % (val.conv_hashes[chash], train.conv_hashes[chash]))

    # ===== 3. Near-Duplicates in train =====
    hr("3. NEAR-DUPLICATES (train, normalisierte erste User-Msg)")
    dup_clusters = [(c, n) for c, n in train.first_user_norm.items() if n > 1]
    dup_clusters.sort(key=lambda x: -x[1])
    n_dup_conv = sum(n for _, n in dup_clusters)
    row("Cluster mit >1 Vorkommen", len(dup_clusters))
    row("betroffene Conversations", n_dup_conv,
        "%.2f%%" % pct(n_dup_conv, train.n_valid))
    if dup_clusters:
        print("\n  häufigste Cluster:")
        print("    %5s  %s" % ("count", "erste User-Msg (gekürzt)"))
        for norm, cnt in dup_clusters[:TOP_CLUSTERS]:
            ex = train.first_user_examples.get(norm, "")
            ex = re.sub(r"\s+", " ", ex)[:70]
            print("    %5d  %s" % (cnt, ex))

    # ===== 4. Quellenverteilung =====
    hr("4. QUELLENVERTEILUNG")
    if not train.has_source_field and not val.has_source_field:
        print("  Kein 'source'-Feld vorhanden -> übersprungen.")
    else:
        row("", "train", "val")
        keys = set(train.sources) | set(val.sources)
        for k in sorted(keys, key=lambda x: -(train.sources[x] + val.sources[x])):
            row(str(k), train.sources.get(k, 0), val.sources.get(k, 0))

    # ===== 5. Längen-Verteilung =====
    hr("5. LÄNGEN-VERTEILUNG (Tokens ~ Whitespace-Split)")
    print("  pro Conversation:")
    row("", "min", "median", "p95", "max")
    cmn, cmd, cp95, cmx = quantiles(train.conv_tok)
    row("  train", cmn, cmd, cp95, cmx)
    cmn, cmd, cp95, cmx = quantiles(val.conv_tok)
    row("  val", cmn, cmd, cp95, cmx)
    print("\n  pro Message:")
    row("", "min", "median", "p95", "max")
    mmn, mmd, mp95, mmx = quantiles(train.msg_tok)
    row("  train", mmn, mmd, mp95, mmx)
    mmn, mmd, mp95, mmx = quantiles(val.msg_tok)
    row("  val", mmn, mmd, mp95, mmx)
    print("\n  Conversations vermutlich > %d Tokens (Kontext-Risiko):" % CTX_LIMIT)
    row("", "count", "% des Splits")
    row("  train", train.over_ctx, "%.2f%%" % pct(train.over_ctx, train.n_valid))
    row("  val", val.over_ctx, "%.2f%%" % pct(val.over_ctx, val.n_valid))

    # ===== 6. Rollen-Struktur =====
    hr("6. ROLLEN-STRUKTUR")
    row("", "train", "val")
    row("startet mit user", train.starts_user, val.starts_user)
    row("  davon nicht-user-Start",
        train.n_valid - train.starts_user, val.n_valid - val.starts_user)
    row("unsaubere Alternation", train.bad_alternation, val.bad_alternation)
    row("endet NICHT auf assistant", train.no_final_assistant,
        val.no_final_assistant)
    row("single-turn (1 user-msg)", train.single_turn, val.single_turn)
    row("multi-turn (>1 user-msg)", train.multi_turn, val.multi_turn)

    # ===== 7. Encoding/Müll =====
    hr("7. ENCODING / MÜLL")
    row("", "train", "val")
    row("Conv. mit Mojibake-Msg", train.mojibake, val.mojibake)
    row("Conv. mit Steuerzeichen", train.ctrlchars, val.ctrlchars)
    row("leere Assistant-Antworten", train.empty_assistant, val.empty_assistant)
    row("EN-Boilerplate/Artefakte", train.boilerplate, val.boilerplate)
    for st in (train, val):
        if st.boilerplate_ex:
            print("\n  Boilerplate-Beispiele [%s]:" % st.name)
            for lineno, frag in st.boilerplate_ex:
                print("    L%-7d %r" % (lineno, frag))

    hr()
    print("Hinweis: Token-Zahlen sind grobe Whitespace-Schätzungen, kein BPE.")
    print("Fertig.")


if __name__ == "__main__":
    main()
