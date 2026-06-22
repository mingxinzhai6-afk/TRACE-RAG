"""
NewG results.jsonl analysis script.
Analyzes routing, Critic rounds, Commendor decisions, and per-round accuracy.
"""
import json
import ast
import os
import glob
from collections import defaultdict, Counter

# ── locate all NewG result files ──────────────────────────────────────────────
BASE = "output/datasets"
DATASET = "Popqa"

result_files = sorted(glob.glob(f"{BASE}/{DATASET}/NewG_*/Results/results.jsonl"))
if not result_files:
    print(f"No result files found under {BASE}/{DATASET}/NewG_*/Results/results.jsonl")
    exit(1)

print(f"Found {len(result_files)} result files:\n")
for f in result_files:
    print(" ", f)
print()

# ── helpers ───────────────────────────────────────────────────────────────────
def get_accuracy(rec):
    acc = rec.get("accuracy")
    if acc is not None:
        return float(acc)
    # compute from output vs answers when field is missing
    output = str(rec.get("output", "")).lower().strip()
    answers = [str(a).lower().strip() for a in rec.get("answers", [])]
    if not answers or not output:
        return 0.0
    return 1.0 if any(a in output or output in a for a in answers) else 0.0

def get_em(rec):
    em = rec.get("em")
    if em is not None:
        return float(em)
    output = str(rec.get("output", "")).lower().strip()
    answers = [str(a).lower().strip() for a in rec.get("answers", [])]
    return 1.0 if output in answers else 0.0

def safe_parse(s):
    if not s or s == "[]":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def analyze_file(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # ── per-record stats ──────────────────────────────────────────────────────
    rounds_counter = Counter()
    routing_counter = Counter()       # graph / text
    critic_verdict_counter = Counter()
    commendor_kind_counter = Counter()

    # accuracy split by number of rounds
    acc_by_rounds = defaultdict(list)

    # multi-round: did accuracy improve vs round-1 candidate?
    multi_round_improved = 0
    multi_round_total = 0

    for rec in records:
        r = rec.get("rounds", 1)
        acc = get_accuracy(rec)
        rounds_counter[r] += 1
        acc_by_rounds[r].append(acc)

        # routing: count every round's selection
        sel_hist = safe_parse(rec.get("selection_history", "[]"))
        for s in sel_hist:
            routing_counter[s] += 1

        # round_details
        rd = safe_parse(rec.get("round_details", "[]"))
        for d in rd:
            cv = d.get("critic_verdict", "unknown")
            critic_verdict_counter[cv] += 1
            ck = d.get("commendor_kind", "")
            if ck:
                commendor_kind_counter[ck] += 1

        # multi-round improvement: if rounds>1 and final output correct
        if r > 1:
            multi_round_total += 1
            if acc == 1:
                multi_round_improved += 1

    n = len(records)
    if n == 0:
        return None

    # ── summary ───────────────────────────────────────────────────────────────
    overall_acc = sum(get_accuracy(rec) for rec in records) / n
    overall_em  = sum(get_em(rec)       for rec in records) / n
    overall_f1  = sum(rec.get("f1", 0) or 0 for rec in records) / n

    total_sel = sum(routing_counter.values()) or 1
    total_verdicts = sum(critic_verdict_counter.values()) or 1
    total_com = sum(commendor_kind_counter.values()) or 1

    result = {
        "n": n,
        "overall_acc": overall_acc,
        "overall_em":  overall_em,
        "overall_f1":  overall_f1,
        # rounds distribution
        "rounds_dist": {k: rounds_counter[k] for k in sorted(rounds_counter)},
        # accuracy by rounds
        "acc_by_rounds": {
            k: round(sum(v) / len(v) * 100, 2)
            for k, v in sorted(acc_by_rounds.items())
        },
        # routing
        "routing": {
            "graph": routing_counter.get("graph", 0),
            "text":  routing_counter.get("text",  0),
            "graph_pct": round(routing_counter.get("graph", 0) / total_sel * 100, 1),
            "text_pct":  round(routing_counter.get("text",  0) / total_sel * 100, 1),
        },
        # critic verdicts
        "critic_verdicts": {
            k: {"count": v, "pct": round(v / total_verdicts * 100, 1)}
            for k, v in sorted(critic_verdict_counter.items(), key=lambda x: -x[1])
        },
        # commendor decisions
        "commendor_kinds": {
            k: {"count": v, "pct": round(v / total_com * 100, 1)}
            for k, v in sorted(commendor_kind_counter.items(), key=lambda x: -x[1])
        },
        # multi-round improvement
        "multi_round": {
            "total": multi_round_total,
            "final_correct": multi_round_improved,
            "success_rate": round(multi_round_improved / multi_round_total * 100, 1)
                            if multi_round_total else 0,
        },
    }
    return result


# ── run analysis ──────────────────────────────────────────────────────────────
for path in result_files:
    tag = path.split("/")[3]   # e.g. NewG_hipporag_bm25_deepseek-v3.2
    print("=" * 70)
    print(f"  {tag}")
    print("=" * 70)

    r = analyze_file(path)
    if r is None:
        print("  [empty file]")
        continue

    print(f"  Records : {r['n']}")
    print(f"  Accuracy: {r['overall_acc']*100:.2f}%  EM: {r['overall_em']*100:.2f}%  F1: {r['overall_f1']*100:.2f}%")

    print("\n  [Routing — selections across all rounds]")
    rt = r["routing"]
    print(f"    graph : {rt['graph']:4d}  ({rt['graph_pct']}%)")
    print(f"    text  : {rt['text']:4d}  ({rt['text_pct']}%)")

    print("\n  [Critic rounds distribution & accuracy]")
    for rn, cnt in r["rounds_dist"].items():
        acc_r = r["acc_by_rounds"].get(rn, 0)
        print(f"    rounds={rn} : {cnt:4d} samples  acc={acc_r}%")

    print("\n  [Critic verdict distribution]")
    for v, info in r["critic_verdicts"].items():
        print(f"    {v:<20s}: {info['count']:4d}  ({info['pct']}%)")

    print("\n  [Commendor decision distribution]")
    for k, info in r["commendor_kinds"].items():
        print(f"    {k:<25s}: {info['count']:4d}  ({info['pct']}%)")

    mr = r["multi_round"]
    print(f"\n  [Multi-round cases]  total={mr['total']}  "
          f"final_correct={mr['final_correct']}  "
          f"success_rate={mr['success_rate']}%")

    print()

# ── aggregate across all 6 combinations ──────────────────────────────────────
print("=" * 70)
print("  AGGREGATE across all NewG combinations (PopQA)")
print("=" * 70)

agg_routing = Counter()
agg_verdicts = Counter()
agg_commendor = Counter()
agg_rounds = Counter()
agg_acc_by_rounds = defaultdict(list)
agg_multi_total = 0
agg_multi_correct = 0

for path in result_files:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rn = rec.get("rounds", 1)
            acc = get_accuracy(rec)
            agg_rounds[rn] += 1
            agg_acc_by_rounds[rn].append(acc)

            for s in safe_parse(rec.get("selection_history", "[]")):
                agg_routing[s] += 1

            for d in safe_parse(rec.get("round_details", "[]")):
                agg_verdicts[d.get("critic_verdict", "unknown")] += 1
                ck = d.get("commendor_kind", "")
                if ck:
                    agg_commendor[ck] += 1

            if rn > 1:
                agg_multi_total += 1
                if acc == 1.0:
                    agg_multi_correct += 1

total_sel = sum(agg_routing.values()) or 1
total_ver = sum(agg_verdicts.values()) or 1
total_com = sum(agg_commendor.values()) or 1

print("\n  [Routing]")
for k in ("graph", "text"):
    cnt = agg_routing[k]
    print(f"    {k:<6s}: {cnt:5d}  ({cnt/total_sel*100:.1f}%)")

print("\n  [Critic rounds]")
for rn in sorted(agg_rounds):
    cnt = agg_rounds[rn]
    avg_acc = sum(agg_acc_by_rounds[rn]) / len(agg_acc_by_rounds[rn]) * 100
    print(f"    rounds={rn}: {cnt:5d} samples  avg_acc={avg_acc:.2f}%")

print("\n  [Critic verdicts]")
for v, cnt in sorted(agg_verdicts.items(), key=lambda x: -x[1]):
    print(f"    {v:<20s}: {cnt:5d}  ({cnt/total_ver*100:.1f}%)")

print("\n  [Commendor decisions]")
for k, cnt in sorted(agg_commendor.items(), key=lambda x: -x[1]):
    print(f"    {k:<25s}: {cnt:5d}  ({cnt/total_com*100:.1f}%)")

print(f"\n  [Multi-round] total={agg_multi_total}  "
      f"correct={agg_multi_correct}  "
      f"success_rate={agg_multi_correct/agg_multi_total*100:.1f}%"
      if agg_multi_total else "\n  [Multi-round] total=0")
