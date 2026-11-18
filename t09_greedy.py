"""
T09 - Greedy toolkit

  canonical(w1,c1,w2,c2)          more frequent word is the 'standard' spelling (greedy by count)
  best_partner(pairs)             for every word keep only its closest partner (min dist, then max count)
  greedy_one_to_one(pairs)        sort pairs by (dist, -count), take a pair if both words unused
  top_k(iterable, k, key)         bounded min-heap, O(n log k) time, O(k) memory
  interval_schedule(intervals)    classic activity selection (sort by finish time)
  fractional_knapsack(items, W)   sort by value/weight

pairs are tuples: (word1, word2, count1, count2, dist)
"""
import heapq


def canonical(w1, c1, w2, c2):
    """Return (standard, variant, c_standard, c_variant)."""
    if (c1, w2) >= (c2, w1):                  # ties broken alphabetically
        return w1, w2, c1, c2
    return w2, w1, c2, c1


def best_partner(pairs):
    best = {}
    for w1, w2, c1, c2, d in pairs:
        for w, o, co in ((w1, w2, c2), (w2, w1, c1)):
            cur = best.get(w)
            if cur is None or (d, -co) < (cur[0], -cur[1]):
                best[w] = (d, co, o)
    return {w: (o, d, co) for w, (d, co, o) in best.items()}


def greedy_one_to_one(pairs):
    used, chosen = set(), []
    for p in sorted(pairs, key=lambda p: (p[4], -(p[2] + p[3]), p[0], p[1])):
        if p[0] not in used and p[1] not in used:
            used.add(p[0]); used.add(p[1]); chosen.append(p)
    return chosen


def top_k(iterable, k, key=lambda x: x):
    heap = []                                 # min-heap of (key, tiebreak, item)
    for n, x in enumerate(iterable):
        item = (key(x), -n, x)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [x for _, _, x in sorted(heap, reverse=True)]


def interval_schedule(intervals):
    """intervals: [(start, finish)]. Max number of non-overlapping intervals."""
    out, end = [], float("-inf")
    for s, f in sorted(intervals, key=lambda t: t[1]):
        if s >= end:
            out.append((s, f)); end = f
    return out


def fractional_knapsack(items, W):
    """items: [(value, weight)]; returns max value."""
    total = 0.0
    for v, w in sorted(items, key=lambda t: t[0] / t[1], reverse=True):
        if W <= 0:
            break
        take = min(w, W)
        total += v * take / w
        W -= take
    return total
