"""
T02 - Edit distance (Dynamic Programming)

  lev_full(a, b)          O(nm) time/space, returns table (for backtracking)
  lev_ops(a, b)           backtrack -> list of edit operations
  lev_two_row(a, b)       O(nm) time, O(min(n,m)) space
  lev_bounded(a, b, k)    Ukkonen band: O(k * min(n,m)) time, returns k+1 if dist > k
  differing_chars(a, b)   the characters inserted/deleted/substituted

Recurrence: D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1] + (a[i-1]!=b[j-1]))
"""

_VOWEL_SWAPS = {("o", "u"), ("u", "o"), ("e", "i"), ("i", "e")}
_PHONETIC_SWAPS = {("p", "f"), ("f", "p"), ("b", "v"), ("v", "b"), ("k", "c"), ("c", "k"), ("s", "z"), ("z", "s")}

def filipino_weighted_dist(a, b, max_cost=2.0):
    """
    DP plausibility distance for Filipino/Taglish.
    Returns float cost; returns > max_cost early if implausible.
    """
    if a == b:
        return 0.0
    n, m = len(a), len(b)
    prev = [float(j) for j in range(m + 1)]

    for i in range(1, n + 1):
        ai = a[i - 1]
        cur = [float(i)] + [0.0] * m
        row_min = cur[0]
        for j in range(1, m + 1):
            bj = b[j - 1]
            # Substitution cost
            if ai == bj:
                sub_cost = 0.0
            elif (ai, bj) in _VOWEL_SWAPS:
                sub_cost = 0.25          # politiko <-> pulitiko, lalake <-> lalaki
            elif (ai, bj) in _PHONETIC_SWAPS:
                sub_cost = 0.35          # pamilya <-> familya, kwento <-> cuento
            else:
                sub_cost = 1.5           # heavy penalty for distinct root consonants (bata vs. pato)

            # Deletion / Insertion cost (cheap if repeating adjacent char or dropping internal vowel)
            del_cost = 0.2 if (i > 1 and ai == a[i - 2]) or (1 < i < n and ai in "aeiou") else 1.0
            ins_cost = 0.2 if (j > 1 and bj == b[j - 2]) or (1 < j < m and bj in "aeiou") else 1.0

            v = min(prev[j] + del_cost, cur[j - 1] + ins_cost, prev[j - 1] + sub_cost)
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min > max_cost:
            return max_cost + 1.0
        prev = cur
    return prev[m]

def lev_full(a, b):
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        Di, Dp = D[i], D[i - 1]
        for j in range(1, m + 1):
            Di[j] = min(Dp[j] + 1, Di[j - 1] + 1, Dp[j - 1] + (ai != b[j - 1]))
    return D


def lev_ops(a, b):
    """Return (distance, ops); ops are ('sub',x,y) ('ins',y) ('del',x) in order a->b."""
    D = lev_full(a, b)
    i, j, ops = len(a), len(b), []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                ops.append(("sub", a[i - 1], b[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", a[i - 1])); i -= 1
        else:
            ops.append(("ins", b[j - 1])); j -= 1
    ops.reverse()
    return D[len(a)][len(b)], ops


def differing_chars(a, b):
    """e.g. color/colour -> 'u'; analyse/analyze -> 's','z'."""
    out = []
    for op in lev_ops(a, b)[1]:
        out.extend(op[1:])
    return out


def lev_two_row(a, b):
    if len(a) < len(b):
        a, b = b, a                           # b is the shorter -> row length min+1
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj))
        prev = cur
    return prev[-1]


def lev_bounded(a, b, k):
    """Exact distance if <= k else k+1. Use this in the hot loop over candidate pairs."""
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    la, lb = len(a), len(b)
    if lb - la > k:
        return k + 1
    INF = k + 1
    prev = [j if j <= k else INF for j in range(lb + 1)]
    for i in range(1, la + 1):
        lo, hi = max(1, i - k), min(lb, i + k)
        cur = [INF] * (lb + 1)
        if i <= k:
            cur[0] = i
        rowmin = cur[0]
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            v = min(prev[j - 1] + (ai != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
            if v > INF:
                v = INF
            cur[j] = v
            if v < rowmin:
                rowmin = v
        if rowmin > k:                        # whole band exceeds k -> early exit
            return INF
        prev = cur
    return min(prev[lb], INF)
