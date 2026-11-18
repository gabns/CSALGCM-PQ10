"""
T07 - Trie + DP edit-distance search

Each trie node carries one DP row (Levenshtein row for the word prefix), so words sharing
a prefix share work, and whole subtrees are pruned when min(row) > k.
Iterative DFS (no recursion limit problems on long words / big vocabularies).

  Trie.insert(word, count)       O(len)
  Trie.get(word)                 count or 0
  Trie.with_prefix(p)            all words with prefix
  Trie.fuzzy(word, k)            [(word, dist, count)] within k edits
"""


class Node:
    __slots__ = ("kids", "count")

    def __init__(self):
        self.kids = {}
        self.count = 0


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word, count=1):
        n = self.root
        for ch in word:
            n = n.kids.setdefault(ch, Node())
        n.count += count

    def get(self, word):
        n = self.root
        for ch in word:
            n = n.kids.get(ch)
            if n is None:
                return 0
        return n.count

    def with_prefix(self, p):
        n = self.root
        for ch in p:
            n = n.kids.get(ch)
            if n is None:
                return []
        out, stack = [], [(n, p)]
        while stack:
            node, s = stack.pop()
            if node.count:
                out.append(s)
            for ch, kid in node.kids.items():
                stack.append((kid, s + ch))
        return sorted(out)

    def fuzzy(self, word, k):
        res = []
        first = list(range(len(word) + 1))
        stack = [(self.root, "", first)]
        while stack:
            node, s, row = stack.pop()
            if node.count and row[-1] <= k:
                res.append((s, row[-1], node.count))
            for ch, kid in node.kids.items():
                new = [row[0] + 1]
                for j in range(1, len(word) + 1):
                    new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (word[j - 1] != ch)))
                if min(new) <= k:             # prune subtree
                    stack.append((kid, s + ch, new))
        return sorted(res)
