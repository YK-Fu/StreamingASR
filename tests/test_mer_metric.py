"""Verify MixErrorRateMetric correctness.

Three layers:
  1. Sanity: hand-crafted (pred, ref) pairs match manual edit-distance counts.
  2. Accumulation: multi-update == single-update on the concatenated data.
  3. DDP: 2-process gloo run gives the same global (rate, err, nref) on every
     rank as a single-process run on the full data.

Run from the repo root:
    python tests/test_mer_metric.py
"""
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.evaluation import MixErrorRate, MixErrorRateMetric


# ---------- 1. sanity ----------
def test_sanity():
    mer = MixErrorRate(to_simplified_chinese=False)

    # English-only: 1 substitution out of 3 words
    assert mer.compute_num_denom(["hello world foo"], ["hello world bar"]) == (1, 3)

    # Chinese-only: per-character; 1 sub out of 3 chars
    assert mer.compute_num_denom(["你好嗎"], ["你好啊"]) == (1, 3)

    # Mixed: ref tokenizes to ['你','好','hello','world'] (len 4); 1 sub
    assert mer.compute_num_denom(["你好 hello bar"], ["你好 hello world"]) == (1, 4)

    # Multi-sample sums correctly
    err, n = mer.compute_num_denom(
        ["hello world foo", "你好嗎"],
        ["hello world bar", "你好啊"],
    )
    assert (err, n) == (2, 6), (err, n)
    print("[OK] sanity")


# ---------- 2. accumulation across batches ----------
def test_accumulation_matches_single():
    mer = MixErrorRate(to_simplified_chinese=False)
    metric = MixErrorRateMetric(mer)

    preds_a = ["hello world foo", "你好嗎"]
    refs_a  = ["hello world bar", "你好啊"]
    preds_b = ["this is good", "今天 weather very nice"]
    refs_b  = ["this was good", "今天 weather is nice"]

    metric.update(preds_a, refs_a)
    metric.update(preds_b, refs_b)
    rate, err, n = metric.compute()

    err_ref, n_ref = mer.compute_num_denom(preds_a + preds_b, refs_a + refs_b)
    assert int(err) == err_ref and int(n) == n_ref, (int(err), int(n), err_ref, n_ref)
    assert math.isclose(float(rate), err_ref / n_ref)
    print(f"[OK] accumulation: err={int(err)}, n={int(n)}, mer={float(rate):.4f}")


# ---------- 3. DDP across 2 ranks ----------
def _ddp_worker(rank, world_size, preds_split, refs_split, out_q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    metric = MixErrorRateMetric(MixErrorRate(to_simplified_chinese=False))
    metric.update(preds_split[rank], refs_split[rank])
    rate, err, n = metric.compute()
    out_q.put((rank, float(rate), int(err), int(n)))
    dist.destroy_process_group()


def test_ddp_matches_single():
    preds = ["hello world foo", "你好嗎", "this is good", "今天 weather very nice"]
    refs  = ["hello world bar", "你好啊", "this was good", "今天 weather is nice"]

    mer = MixErrorRate(to_simplified_chinese=False)
    err_ref, n_ref = mer.compute_num_denom(preds, refs)
    expected = err_ref / n_ref

    half = len(preds) // 2
    preds_split = [preds[:half], preds[half:]]
    refs_split  = [refs[:half],  refs[half:]]

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=_ddp_worker, args=(r, 2, preds_split, refs_split, out_q))
        for r in range(2)
    ]
    for p in procs: p.start()
    for p in procs: p.join()
    results = sorted([out_q.get() for _ in range(2)])

    for rank, rate, err, n in results:
        assert err == err_ref, (rank, err, err_ref)
        assert n == n_ref, (rank, n, n_ref)
        assert math.isclose(rate, expected), (rank, rate, expected)
    print(f"[OK] DDP 2-rank: each rank sees mer={expected:.4f}, err={err_ref}, n={n_ref}")


if __name__ == "__main__":
    test_sanity()
    test_accumulation_matches_single()
    test_ddp_matches_single()
    print("all checks passed.")
