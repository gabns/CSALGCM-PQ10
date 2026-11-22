"""
T10 - End-to-end pipeline  ->  Variants.csv
Columns: word1, word2, word1_count, word2_count, character_difference

  count (t01, parallel D&C) -> prune vocab (greedy) -> candidates (t04 deletion hashing)
  -> verify (t02 DP alignment: only spelling alternations / o-u, e-i swaps allowed)
  -> sort (t05) -> context filter (same lexical item = used in similar contexts) -> CSV

character_difference is the readable list of edits from t02.lev_ops, word1 -> word2:
  "o vs. u"  |  "e vs. i"  |  "ph vs. p"  |  "ch vs. ts"  |  "c vs. k; o vs. u"
Rows are sorted by the letter-level edit distance d (t02.lev_two_row on the raw words;
a digraph such as "ch vs. ts" is shown as one item but counts as 2 edits), then by total frequency.

Run:  python t10_pipeline_csv.py datasets/Raw.txt Variants.csv [max_dist=2] [min_count=5]
Needs t01, t02, t04, t05 in the same folder.
"""
import csv, math, re, sys
from collections import Counter
from t01_counting import count_file_parallel, tokenize
from t02_levenshtein_dp import lev_two_row, lev_ops
from t04_candidate_generation import prune_vocab, hybrid_candidate_pairs
from t05_merge_sort_dc import merge_sort_iter

# Spellings that sound the same in Filipino; order matters (ch before c, ph before f)
_ORTHO = (("ch", "ts"), ("qu", "k"), ("ph", "f"), ("ñ", "ny"), ("x", "ks"), ("v", "b"), ("f", "p"), ("z", "s"))
_SOFT_C = re.compile(r"c(?=[ei])")
_VOWEL_SWAPS = {("o", "u"), ("u", "o"), ("e", "i"), ("i", "e")}

# Digraphs are swapped for one placeholder character before alignment so they print as one unit
_DIGRAPH_TOKENS = (
    ("ph", "\x01"),
    ("ch", "\x02"),
    ("ts", "\x03"),
    ("qu", "\x04"),
    ("ny", "\x05"),
)
_REV_TOKENS = {tok: dg for dg, tok in _DIGRAPH_TOKENS}


def spelling_key(w):
    """Map orthographic alternatives to one form: cuento->kuento, familya->pamilya, señor->senyor."""
    s = w
    for old, new in _ORTHO:
        s = s.replace(old, new)
    return _SOFT_C.sub("s", s).replace("c", "k")


def is_valid_filipino_variant(a, b):
    """
    Variant only if, after normalising spelling alternatives, every remaining edit
    (found by DP alignment, t02.lev_ops) is an o/u or e/i vowel swap.
      politiko/pulitiko -> one o/u swap            -> variant
      polittico/pulitiko -> extra 't' (deletion)   -> rejected
      opo/upo, mesa/misa -> too short; short words collide into different lexical items
      isang/ibang        -> s/b consonant change   -> rejected
    """
    if min(len(a), len(b)) < 5:
        return False
    ka, kb = spelling_key(a), spelling_key(b)
    if len(ka) != len(kb):                    # a swap-only alignment needs equal length: skip the DP
        return False
    _, ops = lev_ops(ka, kb)
    return all(op[0] == "sub" and (op[1], op[2]) in _VOWEL_SWAPS for op in ops)


def _pack_digraphs(a, b):
    """Replace a digraph by one placeholder only where exactly one of the two words has it."""
    for dg, tok in _DIGRAPH_TOKENS:
        if dg in a and dg not in b:
            a = a.replace(dg, tok)
        if dg in b and dg not in a:
            b = b.replace(dg, tok)
    return a, b


def character_difference(a, b):
    """Readable edits a -> b: 'o vs. u', 'ph vs. p', 'f vs. ph', 'c vs. k; o vs. u'."""
    pa, pb = _pack_digraphs(a, b)
    parts = []
    for op in lev_ops(pa, pb)[1]:
        if op[0] == "sub":
            x = _REV_TOKENS.get(op[1], op[1])
            y = _REV_TOKENS.get(op[2], op[2])
            parts.append(f"{x} vs. {y}")
        elif op[0] == "ins":
            parts.append(f"+{_REV_TOKENS.get(op[1], op[1])}")
        else:
            parts.append(f"-{_REV_TOKENS.get(op[1], op[1])}")
    return "; ".join(parts)


def find_variants(counts, max_dist=2, min_count=5, ratio=None):
    words = prune_vocab(counts, min_count=min_count)
    out = []
    for a, b in hybrid_candidate_pairs(words):
        d = lev_two_row(a, b)
        if d == 0 or d > max_dist:
            continue
        if not is_valid_filipino_variant(a, b):
            continue
        ca, cb = counts[a], counts[b]
        if ratio and min(ca, cb) * ratio < max(ca, cb):
            continue
        out.append((a, b, ca, cb, character_difference(a, b), d))
    out = merge_sort_iter(out, key=lambda r: (r[5], -(r[2] + r[3]), r[0]))   # sort by integer d
    return [r[:5] for r in out]                                              # drop d: CSV has 5 columns


# Filipino/Taglish alphabet only: drops Arabic/Cyrillic/CJK tokens from the Wikipedia dump
_LATIN_WORD = re.compile(r"[a-zñ]+(?:'[a-zñ]+)*")


def clean_counts(counts):
    return {w: c for w, c in counts.items() if _LATIN_WORD.fullmatch(w.replace("’", "'"))}


def context_filter(rows, path, min_sim=0.05, window=3):
    """
    Drop pairs whose words are used in different contexts (romano/rumano, petso/pitso).
    One streaming pass collects +-window neighbour counts for the pair words only;
    neighbours are idf-weighted so 'ang', 'ng', 'sa' count for ~nothing; then cosine.
    shortcut: rare words (count ~5) have sparse contexts, so some true variants
    (placebo/plasebo) are lost; lower min_sim to trade precision for recall.
    """
    ctx = {w: Counter() for r in rows for w in r[:2]}
    with open(path, "r", encoding="utf-8", errors="ignore", buffering=1 << 20) as f:
        for line in f:
            t = tokenize(line)
            for i, w in enumerate(t):
                c = ctx.get(w)
                if c is not None:
                    c.update(t[max(0, i - window):i])
                    c.update(t[i + 1:i + 1 + window])
    df = Counter(k for c in ctx.values() for k in c)
    n = len(ctx)

    def cosine(a, b):
        va = {k: v * math.log((n + 1) / df[k]) for k, v in ctx[a].items()}
        vb = {k: v * math.log((n + 1) / df[k]) for k, v in ctx[b].items()}
        dot = sum(v * vb.get(k, 0.0) for k, v in va.items())
        norm = math.sqrt(sum(v * v for v in va.values())) * math.sqrt(sum(v * v for v in vb.values()))
        return dot / norm if norm else 0.0

    return [r for r in rows if cosine(r[0], r[1]) >= min_sim]


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word1", "word2", "word1_count", "word2_count", "character_difference"])
        w.writerows(rows)


if __name__ == "__main__":                    # needed for multiprocessing on Windows/macOS
    if len(sys.argv) < 3:
        sys.exit("usage: python t10_pipeline_csv.py corpus.txt Variants.csv [max_dist=2] [min_count=5]")
    max_dist = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    min_count = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    counts = clean_counts(count_file_parallel(sys.argv[1]))
    rows = context_filter(find_variants(counts, max_dist=max_dist, min_count=min_count), sys.argv[1])
    write_csv(rows, sys.argv[2])
    print(f"{len(counts)} clean words -> {len(rows)} variant pairs -> {sys.argv[2]}")
