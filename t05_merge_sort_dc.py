"""
T05 - Divide & Conquer sorting

  merge_sort(a, key, reverse)   stable top-down merge sort, O(n log n), O(n) extra
  merge_sort_iter(a, key)       bottom-up (no recursion -> safe for n = 10^7)
  count_inversions(a)           D&C inversion count, O(n log n)
  external_sort(in, out, key)   file larger than RAM: sort chunks, then k-way heap merge
"""
import heapq, os, tempfile


def _merge(L, R, key, reverse):
    out, i, j = [], 0, 0
    while i < len(L) and j < len(R):
        a, b = key(L[i]), key(R[j])
        if (b < a) if not reverse else (b > a):
            out.append(R[j]); j += 1
        else:
            out.append(L[i]); i += 1          # '<=' keeps it stable
    out.extend(L[i:]); out.extend(R[j:])
    return out


def merge_sort(a, key=lambda x: x, reverse=False):
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    return _merge(merge_sort(a[:mid], key, reverse), merge_sort(a[mid:], key, reverse), key, reverse)


def merge_sort_iter(a, key=lambda x: x, reverse=False):
    runs = [[x] for x in a]
    while len(runs) > 1:
        nxt = [_merge(runs[i], runs[i + 1], key, reverse) for i in range(0, len(runs) - 1, 2)]
        if len(runs) % 2:
            nxt.append(runs[-1])
        runs = nxt
    return runs[0] if runs else []


def count_inversions(a):
    """Returns (sorted_list, inversions)."""
    if len(a) <= 1:
        return list(a), 0
    mid = len(a) // 2
    L, x = count_inversions(a[:mid])
    R, y = count_inversions(a[mid:])
    out, i, j, z = [], 0, 0, 0
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            out.append(L[i]); i += 1
        else:
            out.append(R[j]); j += 1
            z += len(L) - i
    out.extend(L[i:]); out.extend(R[j:])
    return out, x + y + z


def external_sort(in_path, out_path, key=lambda line: line, chunk_lines=2_000_000):
    """Line-based external merge sort. key receives the line (with newline)."""
    tmp_files = []

    def flush(buf):
        buf.sort(key=key)
        t = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        t.writelines(buf); t.close()
        tmp_files.append(t.name)

    with open(in_path, "r", encoding="utf-8", errors="ignore", buffering=1 << 20) as f:
        buf = []
        for line in f:
            buf.append(line if line.endswith("\n") else line + "\n")
            if len(buf) >= chunk_lines:
                flush(buf); buf = []
        if buf:
            flush(buf)
    handles = [open(p, "r", encoding="utf-8") for p in tmp_files]
    try:
        with open(out_path, "w", encoding="utf-8", buffering=1 << 20) as out:
            out.writelines(heapq.merge(*handles, key=key))
    finally:
        for h in handles:
            h.close()
        for p in tmp_files:
            os.remove(p)
