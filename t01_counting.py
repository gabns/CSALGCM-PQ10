"""
T01 - Streaming tokenizer + word counter (Divide & Conquer over a huge file)

Assumes the input can be many GB / hundreds of millions of tokens:
  * never calls f.read() on the whole file (streams line by line)
  * splits the file into byte ranges aligned to newlines (DIVIDE),
    counts each range in its own process (CONQUER),
    then merges the Counters pairwise like a merge tree (COMBINE).

Complexity: O(N) time over N characters, O(V) memory per worker (V = distinct words).
Usage:
    counts = count_file_parallel("corpus.txt")
    counts = count_csv_column("data.csv", "text")
"""
import csv, os, re, sys
from collections import Counter
from multiprocessing import Pool, cpu_count

# letters only (unicode aware); inner apostrophes kept, hyphens/digits split words
WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")


def tokenize(text):
    return WORD_RE.findall(text.lower())


# ---------- plain streaming (single process) ----------
def count_file(path, encoding="utf-8"):
    c = Counter()
    with open(path, "r", encoding=encoding, errors="ignore", buffering=1 << 20) as f:
        for line in f:
            c.update(WORD_RE.findall(line.lower()))
    return c


def count_csv_column(path, column, encoding="utf-8", delimiter=","):
    """Stream a CSV with a header and count words of one column (name or 0-based index)."""
    csv.field_size_limit(sys.maxsize)          # huge cells
    c = Counter()
    with open(path, "r", encoding=encoding, errors="ignore", newline="", buffering=1 << 20) as f:
        r = csv.reader(f, delimiter=delimiter)
        header = next(r)
        idx = column if isinstance(column, int) else header.index(column)
        for row in r:
            if idx < len(row):
                c.update(WORD_RE.findall(row[idx].lower()))
    return c


# ---------- divide & conquer over byte ranges ----------
def chunk_ranges(path, n_chunks):
    """DIVIDE: byte ranges [start, end) that begin/end on line boundaries."""
    size = os.path.getsize(path)
    step = max(1, size // n_chunks)
    cuts = [0]
    with open(path, "rb") as f:
        for i in range(1, n_chunks):
            pos = i * step
            if pos >= size:
                break
            f.seek(pos)
            f.readline()                      # advance to the next line start
            p = f.tell()
            if cuts[-1] < p < size:
                cuts.append(p)
    cuts.append(size)
    return list(zip(cuts[:-1], cuts[1:]))


def _count_range(args):
    path, start, end = args
    c = Counter()
    pos = start
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        while pos < end:
            line = f.readline()
            if not line:
                break
            pos += len(line)
            c.update(WORD_RE.findall(line.decode("utf-8", "ignore").lower()))
    return c


def merge_counters(a, b):
    if len(a) < len(b):                       # always fold the smaller into the larger
        a, b = b, a
    a.update(b)
    return a


def merge_tree(counters):
    """COMBINE: recursive pairwise merge (balanced, O(total log parts))."""
    if not counters:
        return Counter()
    if len(counters) == 1:
        return counters[0]
    mid = len(counters) // 2
    return merge_counters(merge_tree(counters[:mid]), merge_tree(counters[mid:]))


def count_file_parallel(path, workers=None, chunks_per_worker=4):
    workers = workers or cpu_count()
    ranges = chunk_ranges(path, workers * chunks_per_worker)
    with Pool(workers) as pool:
        parts = pool.map(_count_range, [(path, s, e) for s, e in ranges])
    return merge_tree(parts)


if __name__ == "__main__":                    # needed for multiprocessing on Windows/macOS
    if len(sys.argv) > 1:
        cnt = count_file_parallel(sys.argv[1])
        print(len(cnt), "distinct words;", sum(cnt.values()), "tokens")
        print(cnt.most_common(10))
