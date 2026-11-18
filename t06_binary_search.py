"""
T06 - Binary search family (Divide & Conquer, O(log n))

  lower_bound / upper_bound     first index with a[i] >= x / > x
  find(a, x)                    exact lookup in a sorted list
  find_range(a, lo, hi)         all items in [lo, hi]
  search_answer(lo, hi, ok)     smallest value in [lo,hi] with ok(v) True (monotone predicate)
  prefix_range(sorted_words, p) every word starting with p
"""


def lower_bound(a, x, key=lambda v: v):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if key(a[mid]) < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(a, x, key=lambda v: v):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if key(a[mid]) <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def find(a, x, key=lambda v: v):
    i = lower_bound(a, x, key)
    return i if i < len(a) and key(a[i]) == x else -1


def find_range(a, lo, hi, key=lambda v: v):
    return a[lower_bound(a, lo, key):upper_bound(a, hi, key)]


def search_answer(lo, hi, ok):
    """Smallest v in [lo, hi] with ok(v); returns hi+1 if none."""
    while lo <= hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid - 1
        else:
            lo = mid + 1
    return lo


def prefix_range(sorted_words, p):
    i = lower_bound(sorted_words, p)
    j = lower_bound(sorted_words, p + "\uffff")
    return sorted_words[i:j]
