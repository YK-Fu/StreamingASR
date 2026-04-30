"""Verify multilingual MER correctness after extending the per-character
script range to cover CJK Ext-A, Japanese kana, Korean Hangul, and Thai.

Coverage:
  * Per-character script detection at every range boundary (general + edge).
  * Tokenisation per supported language (zh / ja / ko / th / yue / en /
    fr / de / es / ru / vi).
  * Mixed-script sentences (zh+en, ja+en, ko+en).
  * Edge cases: empty strings, punctuation-only, apostrophe / hyphen,
    numbers, and the guarantee that opencc t2s does NOT mangle non-CJK
    per-char scripts.
  * DDP correctness: 2-process gloo run on multilingual data must give
    the same global (rate, err, nref) as a single-process run.

Run from the repo root:
    python tests/test_mer_multilingual.py
"""
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.evaluation import MixErrorRate, MixErrorRateMetric


# ============================================================
# 1. Per-character script detection — boundaries and adjacency
# ============================================================
def test_per_char_script_boundaries():
    """Walk every range boundary; one inside, one just outside."""
    mer = MixErrorRate(to_simplified_chinese=False)
    cases = [
        # (codepoint, expected, label)
        (0x4DFF, False, "just before CJK Unified"),
        (0x4E00, True,  "start of CJK Unified"),
        (0x9FFF, True,  "end of CJK Unified"),
        (0xA000, False, "just after CJK Unified (Yi syllables)"),
        (0x33FF, False, "just before CJK Ext A"),
        (0x3400, True,  "start of CJK Ext A"),
        (0x4DBF, True,  "end of CJK Ext A"),
        (0x303F, False, "just before Hiragana (Ideographic half-fill space)"),
        (0x3040, True,  "start of Hiragana block"),
        (0x309F, True,  "end of Hiragana"),
        (0x30A0, True,  "start of Katakana"),
        (0x30FF, True,  "end of Katakana"),
        (0x3100, False, "Bopomofo block (not included)"),
        (0xABFF, False, "just before Hangul Syllables"),
        (0xAC00, True,  "start of Hangul Syllables"),
        (0xD7AF, True,  "end of Hangul Syllables"),
        (0xD7B0, False, "Hangul Jamo Extended-B (not included)"),
        (0x0DFF, False, "just before Thai (Sinhala)"),
        (0x0E00, True,  "start of Thai"),
        (0x0E7F, True,  "end of Thai"),
        (0x0E80, False, "just after Thai (Lao)"),
        # Latin / Cyrillic must stay False (handled by alnum branch)
        (ord("a"),    False, "Latin a"),
        (ord("Ñ"),    False, "Latin-1 supplement Ñ"),
        (ord("ä"),    False, "Latin-1 ä"),
        (ord("п"),    False, "Cyrillic п"),
    ]
    for cp, expected, label in cases:
        actual = mer._is_per_char_script(chr(cp))
        assert actual == expected, f"U+{cp:04X} ({label}): expected {expected}, got {actual}"
    print("[OK] per-char script boundaries")


# ============================================================
# 2. Per-language tokenisation
# ============================================================
def test_chinese_traditional_simplified():
    """opencc t2s preserved for CJK Unified range."""
    mer = MixErrorRate(to_simplified_chinese=True)
    # 國 (traditional) -> 国 (simplified). Pred simplified, ref traditional,
    # opencc t2s normalises both -> exact match.
    assert mer.compute_num_denom(["国家"], ["國家"]) == (0, 2)
    # Substitution survives normalisation
    assert mer.compute_num_denom(["国家"], ["国旗"]) == (1, 2)

    # Without converter: 国 vs 國 are different code points -> 1 sub
    mer_no = MixErrorRate(to_simplified_chinese=False)
    assert mer_no.compute_num_denom(["国家"], ["國家"]) == (1, 2)
    print("[OK] Chinese t2s")


def test_japanese():
    """Hiragana / katakana / kanji all tokenise per-char."""
    mer = MixErrorRate(to_simplified_chinese=False)

    # Pure hiragana, identical: 5 chars
    assert mer.compute_num_denom(["こんにちは"], ["こんにちは"]) == (0, 5)

    # Pure katakana, identical: 6 chars
    assert mer.compute_num_denom(["コンピュータ"], ["コンピュータ"]) == (0, 6)

    # Mixed kanji + hiragana (6 chars: 私 は 学 生 で す)
    assert mer.compute_num_denom(["私は学生です"], ["私は学生です"]) == (0, 6)

    # 1 deletion: missing trailing す
    assert mer.compute_num_denom(["私は学生で"], ["私は学生です"]) == (1, 6)

    # Mixed katakana + hiragana + kanji: コーヒーを飲みます (9 chars)
    sentence = "コーヒーを飲みます"
    assert mer.compute_num_denom([sentence], [sentence]) == (0, len(sentence))
    print("[OK] Japanese per-char")


def test_korean():
    """Hangul Syllables tokenise per-char (not via the alnum branch)."""
    mer = MixErrorRate(to_simplified_chinese=False)

    assert mer.compute_num_denom(["안녕하세요"], ["안녕하세요"]) == (0, 5)
    # 1 sub: 세 -> 새
    assert mer.compute_num_denom(["안녕하새요"], ["안녕하세요"]) == (1, 5)

    # Korean + Latin: 'AI 안녕하세요' -> ['AI','안','녕','하','세','요'] (6 tokens)
    assert mer.compute_num_denom(["AI 안녕하세요"], ["AI 안녕하세요"]) == (0, 6)
    print("[OK] Korean per-char")


def test_thai():
    """Thai has no inter-word whitespace; per-char counting is critical."""
    mer = MixErrorRate(to_simplified_chinese=False)
    sentence = "สวัสดีครับ"  # 10 code points, all in Thai block
    assert mer.compute_num_denom([sentence], [sentence]) == (0, len(sentence))

    # Pre-extension behaviour would treat the whole sentence as 1 alnum word,
    # giving (1, 1) on any difference. Post-extension: per-char editdistance.
    pred = "สวัสดีครุบ"  # last vowel changed
    ref  = "สวัสดีครับ"
    err, n = mer.compute_num_denom([pred], [ref])
    assert n == len(ref), n
    assert 1 <= err <= 2, err  # exact ed depends on code-point alignment
    print("[OK] Thai per-char")


def test_cantonese():
    """Cantonese: most chars in CJK Unified; CJK Ext A also covered."""
    mer = MixErrorRate(to_simplified_chinese=False)
    # Common Cantonese (all in U+4E00-9FFF)
    assert mer.compute_num_denom(["佢喺度食緊飯"], ["佢喺度食緊飯"]) == (0, 6)
    # Insertion of 緊 inside the sentence
    assert mer.compute_num_denom(["佢喺度食飯"], ["佢喺度食緊飯"]) == (1, 6)
    # Synthetic CJK Ext A char
    ext_a = chr(0x3401)
    assert mer.compute_num_denom([ext_a], [ext_a]) == (0, 1)
    print("[OK] Cantonese / CJK Ext A")


def test_european_languages():
    """Latin / Cyrillic stay word-level via the alnum branch."""
    mer = MixErrorRate(to_simplified_chinese=False)

    # English
    assert mer.compute_num_denom(["hello world"], ["hello world"]) == (0, 2)
    # French apostrophe (l'eau is one token because ' is in the alnum-keep list)
    assert mer.compute_num_denom(["c'est l'eau"], ["c'est l'eau"]) == (0, 2)
    # German umlauts + ß
    assert mer.compute_num_denom(["schöne grüße"], ["schöne grüße"]) == (0, 2)
    # Spanish ñ + accents
    assert mer.compute_num_denom(["mañana señor"], ["mañana señor"]) == (0, 2)
    # Russian Cyrillic
    assert mer.compute_num_denom(["привет мир"], ["привет мир"]) == (0, 2)
    # Vietnamese (precomposed NFC chars)
    assert mer.compute_num_denom(["tôi yêu bạn"], ["tôi yêu bạn"]) == (0, 3)
    # Italian, Portuguese, Dutch, Indonesian — all plain Latin
    assert mer.compute_num_denom(["ciao mondo"], ["ciao mondo"]) == (0, 2)
    assert mer.compute_num_denom(["olá mundo"], ["olá mundo"]) == (0, 2)
    assert mer.compute_num_denom(["hallo wereld"], ["hallo wereld"]) == (0, 2)
    assert mer.compute_num_denom(["halo dunia"], ["halo dunia"]) == (0, 2)
    print("[OK] European/Latin/Cyrillic word-level")


def test_mixed_scripts():
    """Cross-script sentences tokenise correctly."""
    mer = MixErrorRate(to_simplified_chinese=False)

    # zh + en: ['今','天','weather','very','nice'] = 5
    assert mer.compute_num_denom(
        ["今天 weather very nice"], ["今天 weather very nice"]
    ) == (0, 5)

    # ja + en: ['私','は','AI','を','使','う'] = 6
    assert mer.compute_num_denom(
        ["私は AI を使う"], ["私は AI を使う"]
    ) == (0, 6)

    # ko + en: ['안','녕','hello'] vs ['안','녕','world'] -> 1 sub
    assert mer.compute_num_denom(
        ["안녕 hello"], ["안녕 world"]
    ) == (1, 3)
    print("[OK] mixed scripts")


# ============================================================
# 3. Edge cases
# ============================================================
def test_edge_empty():
    mer = MixErrorRate(to_simplified_chinese=False)
    # Both empty
    assert mer.compute_num_denom([""], [""]) == (0, 0)
    # Pred empty, ref non-empty -> 2 deletions (2 ref tokens)
    assert mer.compute_num_denom([""], ["hello world"]) == (2, 2)
    # Pred non-empty, ref empty -> insertions, ref_len 0
    err, n = mer.compute_num_denom(["hello world"], [""])
    assert err == 2 and n == 0
    # Mix: one empty + one Chinese
    assert mer.compute_num_denom(["", "你好"], ["", "你好"]) == (0, 2)
    print("[OK] empty strings")


def test_edge_punctuation_only():
    mer = MixErrorRate(to_simplified_chinese=False)
    # All ASCII punctuation -> all skipped
    assert mer.compute_num_denom(["!!!,..."], ["???.,!"]) == (0, 0)
    # Full-width Chinese punctuation -> all skipped
    assert mer.compute_num_denom(["。。。"], ["，，，"]) == (0, 0)
    print("[OK] punctuation-only")


def test_edge_apostrophe_hyphen():
    mer = MixErrorRate(to_simplified_chinese=False)
    # Apostrophe stays inside word
    assert mer.compute_num_denom(["don't go"], ["don't go"]) == (0, 2)
    # Different contractions: ['don't','go'] vs ['do','not','go'] -> 2 errs / 3 ref
    assert mer.compute_num_denom(["don't go"], ["do not go"]) == (2, 3)
    # Hyphen kept inside compound
    assert mer.compute_num_denom(["well-known fact"], ["well-known fact"]) == (0, 2)
    print("[OK] apostrophe/hyphen")


def test_edge_numbers_alnum():
    mer = MixErrorRate(to_simplified_chinese=False)
    # Numbers go through the alnum branch
    assert mer.compute_num_denom(["test 123"], ["test 124"]) == (1, 2)
    # No whitespace between letters and digits -> single token
    assert mer.compute_num_denom(["abc123"], ["abc124"]) == (1, 1)
    print("[OK] numbers/alnum")


def test_edge_opencc_only_on_chinese_range():
    """opencc t2s must NOT mangle hiragana / katakana / hangul / thai
    even when ``to_simplified_chinese=True`` is set."""
    mer = MixErrorRate(to_simplified_chinese=True)
    # If opencc were applied to hiragana, chars could shift; they should not.
    assert mer.compute_num_denom(["あいうえお"], ["あいうえお"]) == (0, 5)
    # Hangul similarly
    assert mer.compute_num_denom(["안녕"], ["안녕"]) == (0, 2)
    # Thai
    assert mer.compute_num_denom(["สวัส"], ["สวัส"]) == (0, 4)
    # Mixed: only the Chinese portion gets normalised
    # pred '国' (already simplified) + 'あ', ref '國' (traditional) + 'あ'
    # After t2s on the kanji part: pred=['国','あ'], ref=['国','あ'] -> match
    assert mer.compute_num_denom(["国あ"], ["國あ"]) == (0, 2)
    print("[OK] opencc gated to CJK Unified only")


def test_edge_ext_a_chars():
    """CJK Extension A chars now tokenise per-char (used to fall through)."""
    mer = MixErrorRate(to_simplified_chinese=False)
    a, b = chr(0x3401), chr(0x3402)
    # Two distinct Ext-A chars in a row
    assert mer.compute_num_denom([a + b], [a + b]) == (0, 2)
    # Substitution between two Ext-A chars
    assert mer.compute_num_denom([a + b], [b + a]) == (2, 2)
    print("[OK] CJK Ext A chars")


# ============================================================
# 4. DDP correctness on multilingual data
# ============================================================
def _ddp_worker(rank, world_size, preds_split, refs_split, port, out_q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    metric = MixErrorRateMetric(MixErrorRate(to_simplified_chinese=False))
    metric.update(preds_split[rank], refs_split[rank])
    rate, err, n = metric.compute()
    out_q.put((rank, float(rate), int(err), int(n)))
    dist.destroy_process_group()


def test_ddp_multilingual():
    """Mix of all supported scripts split across 2 ranks must reduce to the
    same global (rate, err, nref) as a single-process run."""
    preds = [
        "hello world",          # en
        "私は学生です",         # ja kanji+hiragana
        "안녕하세요",           # ko hangul
        "สวัสดีครับ",           # th
        "今天 weather nice",    # zh+en
        "佢喺度食飯",           # yue
        "schöne grüße",         # de umlauts
        "привет мир",           # ru cyrillic
    ]
    refs = [
        "hello world",
        "私は学生でした",       # 2 char diff
        "안녕하새요",           # 1 char sub
        "สวัสดีครับ",
        "今天 weather is nice",  # 1 ins
        "佢喺度食緊飯",          # 1 ins
        "schöne grüße",
        "привет мир",
    ]

    mer = MixErrorRate(to_simplified_chinese=False)
    err_ref, n_ref = mer.compute_num_denom(preds, refs)
    expected = err_ref / n_ref

    half = len(preds) // 2
    preds_split = [preds[:half], preds[half:]]
    refs_split  = [refs[:half],  refs[half:]]

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_ddp_worker,
            args=(r, 2, preds_split, refs_split, 29501, out_q),
        )
        for r in range(2)
    ]
    for p in procs: p.start()
    for p in procs: p.join()
    results = sorted([out_q.get() for _ in range(2)])

    for rank, rate, err, n in results:
        assert err == err_ref, (rank, err, err_ref)
        assert n == n_ref, (rank, n, n_ref)
        assert math.isclose(rate, expected), (rank, rate, expected)
    print(f"[OK] DDP multilingual (2 ranks): mer={expected:.4f}, err={err_ref}, n={n_ref}")


if __name__ == "__main__":
    # 1. boundary
    test_per_char_script_boundaries()
    # 2. per-language
    test_chinese_traditional_simplified()
    test_japanese()
    test_korean()
    test_thai()
    test_cantonese()
    test_european_languages()
    test_mixed_scripts()
    # 3. edge
    test_edge_empty()
    test_edge_punctuation_only()
    test_edge_apostrophe_hyphen()
    test_edge_numbers_alnum()
    test_edge_opencc_only_on_chinese_range()
    test_edge_ext_a_chars()
    # 4. DDP
    test_ddp_multilingual()
    print("\nall multilingual checks passed.")
