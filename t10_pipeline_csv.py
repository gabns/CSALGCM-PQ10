"""
T10 - End-to-end pipeline  ->  Variants.csv
Columns: word1, word2, word1_count, word2_count, character_difference

  count (t01, parallel D&C) -> prune vocab (greedy) -> candidates (t04 deletion hashing)
  -> verify (t02 bounded DP) -> optional British/American rule filter
  -> sort (t05) -> stream-write CSV

Run:  python t10_pipeline_csv.py corpus.txt Variants.csv [max_dist=2] [min_count=5]
Needs t01, t02, t04, t05 in the same folder.
"""
import csv, sys
from t01_counting import count_file_parallel
from t02_levenshtein_dp import lev_bounded
from t04_candidate_generation import prune_vocab, deletion_pairs
from t05_merge_sort_dc import merge_sort_iter

# (British, American) substring rules; pair is a "spelling variant" if one rule maps a -> b
RULES = [("our", "or"), ("ise", "ize"), ("isation", "ization"), ("yse", "yze"), ("re", "er"),
         ("ence", "ense"), ("ogue", "og"), ("ll", "l"), ("ae", "e"), ("oe", "e"),
         ("ou", "o"), ("mme", "m"), ("ck", "k"), ("ce", "se")]


def rule_variant(a, b):
    for x, y in RULES:
        if a.replace(x, y) == b or b.replace(x, y) == a:
            return True
        if a.endswith(x) and a[:-len(x)] + y == b or b.endswith(x) and b[:-len(x)] + y == a:
            return True
    return False


def find_variants(counts, max_dist=2, min_count=5, use_rules=True, ratio=None):
    words = prune_vocab(counts, min_count=min_count)
    out = []
    for a, b in deletion_pairs(words, k=max_dist):
        d = lev_bounded(a, b, max_dist)
        if d == 0 or d > max_dist:
            continue
        if use_rules and not rule_variant(a, b):
            continue
        ca, cb = counts[a], counts[b]
        if ratio and min(ca, cb) * ratio < max(ca, cb):
            continue                          # optional: drop typos (rare word vs very common word)
        out.append((a, b, ca, cb, d))
    return merge_sort_iter(out, key=lambda r: (r[4], -(r[2] + r[3]), r[0]))


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8", buffering=1 << 20) as f:
        w = csv.writer(f)
        w.writerow(["word1", "word2", "word1_count", "word2_count", "character_difference"])
        w.writerows(rows)                     # rows: (word1, word2, c1, c2, dist)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    md = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    mc = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    cnt = count_file_parallel(src)
    write_csv(find_variants(cnt, max_dist=md, min_count=mc), dst)
    print("done")
