"""Run:  python -m pytest test_pipeline.py   (or just:  python test_pipeline.py)"""
import os, tempfile
from t01_counting import count_file
from t10_pipeline_csv import is_valid_filipino_variant, clean_counts, find_variants, context_filter


def test_quiz_examples():
    assert is_valid_filipino_variant("politiko", "pulitiko") == (True, 1)
    assert not is_valid_filipino_variant("opo", "upo")
    assert not is_valid_filipino_variant("polittico", "pulitiko")


def test_spelling_rules():
    assert is_valid_filipino_variant("familya", "pamilya")      # f/p
    assert is_valid_filipino_variant("lalake", "lalaki")        # e/i
    assert not is_valid_filipino_variant("maaalat", "maalat")   # plural form, not a spelling variant
    assert not is_valid_filipino_variant("isang", "ibang")      # consonant change = different word
    assert not is_valid_filipino_variant("kaaway", "kaway")     # enemy vs wave


def test_clean_drops_non_latin():
    assert clean_counts({"bata": 1, "niño": 1, "isa't": 1, "中文": 1, "русский": 1}) == {"bata": 1, "niño": 1, "isa't": 1}


def test_pipeline_end_to_end():
    text = ("Ang pulitiko ay bumoto sa halalan. Ang politiko ay bumoto sa halalan. "
            "Kumain ng mesa ang bata. Nagsimba sa misa ang lola. 中文\n") * 6
    fd, path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        rows = context_filter(find_variants(clean_counts(count_file(path)), max_dist=2), path)
    finally:
        os.remove(path)
    assert [r[:2] for r in rows] == [("politiko", "pulitiko")]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
