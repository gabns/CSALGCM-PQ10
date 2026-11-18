"""
T08 - Union-Find / Disjoint Set (clusters of variants: grey/gray/greye)

Path halving + union by size -> amortised O(alpha(n)) per op, iterative.
  DSU(n)                       integer ids
  WordDSU()                    maps arbitrary strings to ids on the fly
  .groups()                    {root: [members]} (only sets with size > 1 by default)
"""


class DSU:
    def __init__(self, n=0):
        self.p = list(range(n))
        self.sz = [1] * n

    def add(self):
        self.p.append(len(self.p)); self.sz.append(1)
        return len(self.p) - 1

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return True

    def same(self, a, b):
        return self.find(a) == self.find(b)


class WordDSU:
    def __init__(self):
        self.id, self.words, self.d = {}, [], DSU()

    def _get(self, w):
        i = self.id.get(w)
        if i is None:
            i = self.id[w] = self.d.add()
            self.words.append(w)
        return i

    def union(self, a, b):
        return self.d.union(self._get(a), self._get(b))

    def same(self, a, b):
        return self.d.same(self._get(a), self._get(b))

    def groups(self, min_size=2):
        g = {}
        for w, i in self.id.items():
            g.setdefault(self.d.find(i), []).append(w)
        return {r: sorted(m) for r, m in g.items() if len(m) >= min_size}
