"""
T04 - Candidate pair generation (avoid the O(V^2) all-pairs comparison)

With V up to millions of distinct words, V^2 is impossible. Two filters:
  1. length_buckets  : only compare words whose lengths differ by <= k
  2. deletion_pairs  : SymSpell idea. Two words within distance k share a string
                       obtained by deleting <= k characters from each. Hash those
                       "deletion keys" -> words that collide are candidates.
Always VERIFY candidates afterwards with lev_bounded (collisions are necessary, not sufficient).

Cost (k=1): O(sum of word lengths) keys; expected near-linear in vocabulary size.
`max_bucket` skips pathological buckets so memory/time stay bounded.
"""
from collections import defaultdict
from itertools import combinations


def prune_vocab(counts, min_count=1, min_len=2, max_len=40):
    """Greedy shrink of the vocabulary before any pairing."""
    return sorted(w for w, c in counts.items() if c >= min_count and min_len <= len(w) <= max_len)


def length_buckets(words):
    b = defaultdict(list)
    for w in words:
        b[len(w)].append(w)
    return b


def _deletions(w, k):
    """All strings made by deleting 0..k characters from w (iterative, no recursion)."""
    out = {w}
    frontier = {w}
    for _ in range(k):
        nxt = set()
        for s in frontier:
            for i in range(len(s)):
                nxt.add(s[:i] + s[i + 1:])
        out |= nxt
        frontier = nxt
    return out


def deletion_pairs(words, k=1, max_bucket=2000):
    """Yield unique candidate (w1, w2) with w1 < w2. Words should be unique."""
    index = defaultdict(list)                 # deletion key -> word ids
    for wid, w in enumerate(words):
        for key in _deletions(w, k):
            index[key].append(wid)
    seen = set()
    n = len(words)
    for ids in index.values():
        if len(ids) < 2 or len(ids) > max_bucket:
            continue
        for i, j in combinations(sorted(ids), 2):
            code = i * n + j
            if code in seen:
                continue
            seen.add(code)
            a, b = words[i], words[j]
            if abs(len(a) - len(b)) <= k:
                yield (a, b) if a < b else (b, a)
    index.clear()


def length_bucket_pairs(words, k=2):
    """Fallback / baseline: all pairs with |len diff| <= k (still quadratic per bucket)."""
    b = length_buckets(words)
    for L in sorted(b):
        for M in range(L, L + k + 1):
            if M not in b:
                continue
            if M == L:
                yield from combinations(b[L], 2)
            else:
                for x in b[L]:
                    for y in b[M]:
                        yield (x, y)
