"""
T03 - LCS and Damerau-Levenshtein (Dynamic Programming)

  lcs_length(a, b)        O(nm) time, O(min) space
  lcs_string(a, b)        O(nm) time/space with reconstruction
  lcs_distance(a, b)      insert/delete-only distance = n + m - 2*LCS
  damerau_osa(a, b)       edit distance with adjacent transposition (teh -> the = 1)
  similarity(a, b)        1 - lev/maxlen style score in [0,1]
"""


def lcs_length(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ai in a:
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if ai == bj else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def lcs_string(a, b):
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            L[i][j] = L[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(L[i - 1][j], L[i][j - 1])
    i, j, out = n, m, []
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1]); i -= 1; j -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))


def lcs_distance(a, b):
    return len(a) + len(b) - 2 * lcs_length(a, b)


def damerau_osa(a, b):
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = a[i - 1] != b[j - 1]
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)
    return D[n][m]


def similarity(a, b, dist_fn=None):
    if dist_fn is None:
        from t02_levenshtein_dp import lev_two_row as dist_fn
    mx = max(len(a), len(b))
    return 1.0 if mx == 0 else 1 - dist_fn(a, b) / mx
