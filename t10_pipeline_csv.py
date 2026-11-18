"""
T10 - End-to-end pipeline  ->  Variants.csv
Columns: word1, word2, word1_count, word2_count, character_difference

  count (t01, parallel D&C) -> prune vocab (greedy) -> candidates (t04 deletion hashing)
  -> verify (t02 bounded DP) -> optional British/American rule filter
  -> sort (t05) -> stream-write CSV

Run:  python t10_pipeline_csv.py corpus.txt Variants.csv [max_dist=2] [min_count=5]
Needs t01, t02, t04, t05 in the same folder.
"""
import csv, re, sys
from t01_counting import count_file_parallel
from t02_levenshtein_dp import lev_two_row, filipino_weighted_dist
from t04_candidate_generation import prune_vocab, hybrid_candidate_pairs, filipino_skeleton, squeeze_repeats
from t05_merge_sort_dc import merge_sort_iter

_INTERNAL_DOUBLE_CONS = re.compile(r"(?<=.)([bcdfghjklmnpqrstvwxyz])\1(?=.)")


def is_valid_filipino_variant(a, b):
    la, lb = len(a), len(b)
    min_len, max_len = min(la, lb), max(la, lb)

    # 1. Block short-word root collisions (e.g., opo vs. upo)
    if min_len < 4:
        return False

    # 2. Block foreign/unmotivated internal double consonants (e.g., polittico vs. pulitiko)
    if bool(_INTERNAL_DOUBLE_CONS.search(a)) != bool(_INTERNAL_DOUBLE_CONS.search(b)):
        if squeeze_repeats(a) != squeeze_repeats(b):
            return False

    # 3. Check weighted Filipino phonetic/orthographic cost
    if filipino_weighted_dist(a, b, max_cost=1.6) > 1.6:
        return False

    # 4. Enforce proportional similarity on raw characters
    raw_dist = lev_two_row(a, b)
    same_skeleton = (filipino_skeleton(a) == filipino_skeleton(b))
    max_ratio = 0.50 if same_skeleton else 0.25
    if (raw_dist / max_len) > max_ratio:
        return False

    return True, raw_dist


def find_variants(counts, max_dist=3, min_count=5, ratio=None):
    words = prune_vocab(counts, min_count=min_count)
    out = []
    for a, b in hybrid_candidate_pairs(words):
        res = is_valid_filipino_variant(a, b)
        if not res:
            continue
        _, d = res
        if d == 0 or d > max_dist:
            continue
        ca, cb = counts[a], counts[b]
        if ratio and min(ca, cb) * ratio < max(ca, cb):
            continue
        out.append((a, b, ca, cb, d))
    return merge_sort_iter(out, key=lambda r: (r[4], -(r[2] + r[3]), r[0]))
