"""One-off diff analysis script comparing ledger vs baseline (M3-Bench robot).

Usage: python scripts/compare_ledger_vs_baseline.py > /tmp/diff.txt
Produces:
  - overall accuracy / round distribution
  - typical cases where BASELINE is correct and LEDGER is wrong (regression analysis)
  - typical cases where LEDGER is correct and BASELINE is wrong (improvement analysis)
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

BL = Path('/path/to/lv_harness/data/results/m3bench_robot_all_kfon_evooff_gemini-2.5-flash_20260427_192245/m3bench_robot_all-gemini-2.5-flash-lv_harness.jsonl')
LG = Path('/path/to/lv_harness/data/results/m3bench_robot_all_kfon_evooff_agledger_gemini-2.5-flash_20260508_164746/m3bench_robot_all-gemini-2.5-flash-lv_harness.jsonl')


def load(p: Path):
    with p.open(encoding='utf-8') as f:
        return {json.loads(l)['id']: json.loads(l) for l in f}


def main() -> None:
    b = load(BL)
    g = load(LG)
    keys = set(b) & set(g)
    bc = {k for k in keys if b[k].get('gpt_eval')}
    gc = {k for k in keys if g[k].get('gpt_eval')}
    b_only = bc - gc
    g_only = gc - bc
    both = bc & gc

    print(f"Total: {len(keys)}  BASE correct: {len(bc)}({len(bc)/len(keys):.3f})  "
          f"LEDGER correct: {len(gc)}({len(gc)/len(keys):.3f})")
    print(f"  Both correct: {len(both)}  BASE only: {len(b_only)}  LEDGER only: {len(g_only)}")
    avgb = sum(b[k].get('num_rounds', 0) for k in keys) / len(keys)
    avgl = sum(g[k].get('num_rounds', 0) for k in keys) / len(keys)
    print(f"  Avg rounds: BASE={avgb:.2f}  LEDGER={avgl:.2f}")
    or_b = sum(1 for k in keys if b[k].get('num_rounds') == 1)
    or_l = sum(1 for k in keys if g[k].get('num_rounds') == 1)
    print(f"  Single round (early answer): BASE={or_b}  LEDGER={or_l}")
    print("  BASE num_rounds distribution:",
          dict(sorted(Counter(b[k].get('num_rounds', 0) for k in keys).items())))
    print("  LEDGER num_rounds distribution:",
          dict(sorted(Counter(g[k].get('num_rounds', 0) for k in keys).items())))

    print('\n=== Typical 12 cases: BASE correct, LEDGER wrong ===')
    random.seed(0)
    samp = random.sample(sorted(b_only), min(12, len(b_only)))
    for k in samp:
        print(f"  [{k}] BL_r={b[k]['num_rounds']} LG_r={g[k]['num_rounds']}")
        print(f"    Q : {b[k]['question'][:100]}")
        print(f"    GT: {b[k]['answer'][:100]}")
        print(f"    LG: {g[k]['response'][:180]}")
        print(f"    BL: {b[k]['response'][:180]}")

    print('\n=== Typical 8 cases: LEDGER correct, BASE wrong ===')
    samp2 = random.sample(sorted(g_only), min(8, len(g_only)))
    for k in samp2:
        print(f"  [{k}] BL_r={b[k]['num_rounds']} LG_r={g[k]['num_rounds']}")
        print(f"    Q : {g[k]['question'][:100]}")
        print(f"    GT: {g[k]['answer'][:100]}")
        print(f"    LG: {g[k]['response'][:180]}")
        print(f"    BL: {b[k]['response'][:180]}")


if __name__ == '__main__':
    main()
