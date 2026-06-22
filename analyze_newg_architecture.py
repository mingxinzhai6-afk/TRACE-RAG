"""
NewG architecture analysis script.

Reads results.jsonl files from either the historical `newg_logs/` layout or the
runtime `output/datasets/...` layout and summarizes the current NewG control
flow:

    QueryUnderstanding -> Retriever Bank -> Evidence Fusion -> ReGeneration
    -> Critic -> optional Commendor -> Answer Normalizer

The script is backward-compatible with older result files that only contain
stringified Python literals and do not expose the newer `route_source`,
`low_confidence`, or `round_outcome` fields.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ACTUAL_COMMENDOR_DECISIONS = {
    "pass",
    "wrong_retriever",
    "insufficient_evidence",
    "poor_generation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NewG architecture behavior")
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help=(
            "One or more glob patterns for results.jsonl files. "
            "Defaults to auto-discovery under newg_logs/ and output/datasets/."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional dataset name when auto-discovering under output/datasets/.",
    )
    parser.add_argument(
        "--judge-threshold",
        type=float,
        default=3.0,
        help="Judge aggregate score threshold used to infer low-confidence legacy rounds.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().strip(".,")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_parse_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", "[]"):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def discover_result_files(root: Path, patterns: List[str] | None, dataset: str) -> List[str]:
    if patterns:
        globs = patterns
    else:
        globs = ["newg_logs/NewG_*/Results/results.jsonl"]
        if dataset:
            globs.append(f"output/datasets/{dataset}/NewG_*/Results/results.jsonl")
        else:
            globs.append("output/datasets/*/NewG_*/Results/results.jsonl")

    found: List[str] = []
    for pattern in globs:
        full_pattern = pattern if Path(pattern).is_absolute() else str(root / pattern)
        found.extend(glob.glob(full_pattern))
    return sorted(set(found))


def load_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_answers(record: Dict[str, Any]) -> List[str]:
    answers = record.get("answers")
    if isinstance(answers, list) and answers:
        normalized = [normalize_text(item) for item in answers if normalize_text(item)]
        if normalized:
            return normalized

    answer = str(record.get("answer", "") or "")
    parts = answer.split("|") if "|" in answer else [answer]
    return [normalize_text(part) for part in parts if normalize_text(part)]


def value_matches_answers(value: Any, answers: List[str]) -> bool:
    candidate = normalize_text(value)
    if not candidate or not answers:
        return False
    return any(answer in candidate or candidate in answer for answer in answers)


def get_accuracy(record: Dict[str, Any]) -> float:
    output = normalize_text(record.get("output", ""))
    answers = extract_answers(record)
    return 1.0 if value_matches_answers(output, answers) else 0.0


def get_em(record: Dict[str, Any]) -> float:
    output = normalize_text(record.get("output", ""))
    answers = extract_answers(record)
    return 1.0 if output and output in answers else 0.0


def get_metric(record: Dict[str, Any], metric: str) -> float | None:
    """Use explicit scorer metrics when a scored result file provides them."""
    if metric not in record:
        return None
    try:
        return float(record.get(metric))
    except Exception:
        return None


def infer_round_outcome(detail: Dict[str, Any]) -> str:
    explicit = str(detail.get("round_outcome", "") or "").strip()
    if explicit:
        return explicit

    kind = str(detail.get("commendor_kind", "") or "").strip().lower()
    if kind == "pass(critic)":
        return "stop_critic_pass"
    if kind == "pass(revise)":
        return "stop_revise_pass"
    if kind == "max_rounds":
        return "stop_max_rounds"
    if kind == "critic_iterate":
        return "continue_critic_iterate"
    if kind == "pass":
        return "stop_commendor_pass"
    if kind == "wrong_retriever":
        return "continue_commendor_switch"
    if kind == "insufficient_evidence":
        return "continue_commendor_more_evidence"
    if kind == "poor_generation":
        regen_verdict = str(detail.get("regenerated_critic_verdict", "") or "").lower()
        return "stop_commendor_regen_pass" if regen_verdict == "pass" else "continue_commendor_regen"

    verdict = str(detail.get("critic_verdict", "") or "").strip().lower()
    if verdict == "pass":
        return "stop_critic_pass"
    if verdict == "revise":
        return "continue_critic_revise"
    if verdict == "retrieve_more":
        return "continue_critic_iterate"
    return "unknown"


def infer_final_outcome(record: Dict[str, Any], round_details: List[Dict[str, Any]]) -> str:
    if not round_details:
        return "unknown"

    outcome = infer_round_outcome(round_details[-1])
    if outcome.startswith("continue_"):
        total_rounds = int(record.get("rounds", 0) or 0)
        final_round = int(round_details[-1].get("round", 0) or 0)
        if total_rounds and final_round >= total_rounds:
            return "stop_max_rounds_legacy"
    return outcome


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_records(
    records: List[Dict[str, Any]],
    judge_threshold: float = 3.0,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n": len(records),
        "acc": 0.0,
        "em": 0.0,
        "f1": 0.0,
        "has_explicit_f1": False,
        "avg_rounds": 0.0,
        "initial_selection": Counter(),
        "round_selection": Counter(),
        "route_source": Counter(),
        "round_route_source": Counter(),
        "critic_verdicts": Counter(),
        "critic_by_round": defaultdict(Counter),
        "round_outcomes": Counter(),
        "stop_outcomes": Counter(),
        "commendor_decisions": Counter(),
        "acc_by_rounds": defaultdict(list),
        "acc_by_stop": defaultdict(list),
        "judge_scores": [],
        "judge_scores_by_round": defaultdict(list),
        "agg_scores": [],
        "switch_queries": 0,
        "critic_override_queries": 0,
        "commendor_queries": 0,
        "low_conf_queries": 0,
        "low_conf_rounds": 0,
        "total_rounds": 0,
        "regen_attempts": 0,
        "regen_passes": 0,
        "pass_guard_queries": 0,
        "pass_guard_rounds": 0,
        "normalized_feedback_rounds": 0,
        "candidate_helped": 0,
        "candidate_hurt": 0,
        "candidate_no_change": 0,
    }

    if not records:
        return summary

    accuracies: List[float] = []
    em_scores: List[float] = []
    f1_scores: List[float] = []
    rounds_used: List[float] = []

    for record in records:
        answers = extract_answers(record)
        acc = get_metric(record, "accuracy")
        if acc is None:
            acc = get_accuracy(record)
        em = get_metric(record, "em")
        if em is None:
            em = get_em(record)
        f1 = get_metric(record, "f1")
        rounds = int(record.get("rounds", 0) or 0)

        accuracies.append(acc)
        em_scores.append(em)
        if f1 is not None:
            f1_scores.append(f1)
        rounds_used.append(rounds)
        summary["acc_by_rounds"][rounds].append(acc)

        initial_selection = str(record.get("initial_selection", "") or "unknown")
        summary["initial_selection"][initial_selection] += 1

        top_route_source = str(record.get("route_source", "") or "unknown")
        summary["route_source"][top_route_source] += 1

        history = safe_parse_list(record.get("selection_history", []))
        if not history and initial_selection:
            history = [initial_selection]
        for selection in history:
            summary["round_selection"][str(selection or "unknown")] += 1
        if len(set(history)) > 1:
            summary["switch_queries"] += 1

        round_details = safe_parse_list(record.get("round_details", []))
        if any(str(detail.get("route_source", "") or "") == "critic_override" for detail in round_details):
            summary["critic_override_queries"] += 1

        low_conf_triggered = False
        pass_guard_triggered = False
        for detail in round_details:
            summary["total_rounds"] += 1
            round_id = int(detail.get("round", 0) or 0)

            round_source = str(detail.get("route_source", "") or "").strip()
            if round_source:
                summary["round_route_source"][round_source] += 1

            verdict = str(detail.get("critic_verdict", "unknown") or "unknown").strip().lower()
            summary["critic_verdicts"][verdict] += 1
            summary["critic_by_round"][round_id][verdict] += 1

            outcome = infer_round_outcome(detail)
            summary["round_outcomes"][outcome] += 1
            if detail.get("pass_guard_reason") or outcome == "continue_pass_guard":
                summary["pass_guard_rounds"] += 1
                pass_guard_triggered = True

            scores = safe_parse_list(detail.get("judge_scores", []))
            agg_score = safe_float(detail.get("agg_score", 0.0))
            if "agg_score" in detail:
                summary["agg_scores"].append(agg_score)

            explicit_low_conf = detail.get("low_confidence")
            inferred_low_conf = (
                explicit_low_conf is None
                and "agg_score" in detail
                and scores
                and agg_score < judge_threshold
            )
            if explicit_low_conf is True or inferred_low_conf:
                summary["low_conf_rounds"] += 1
                low_conf_triggered = True

            if scores:
                normalized_scores = [safe_float(score) for score in scores]
                summary["judge_scores"].extend(normalized_scores)
                summary["judge_scores_by_round"][round_id].extend(normalized_scores)

            if detail.get("regenerated_answer"):
                summary["regen_attempts"] += 1
                regen_verdict = str(detail.get("regenerated_critic_verdict", "") or "").strip().lower()
                if regen_verdict == "pass":
                    summary["regen_passes"] += 1

            if normalize_text(detail.get("normalized_for_critic")) and (
                normalize_text(detail.get("normalized_for_critic")) !=
                normalize_text(detail.get("intermediate_answer"))
            ):
                summary["normalized_feedback_rounds"] += 1

        if low_conf_triggered:
            summary["low_conf_queries"] += 1
        if pass_guard_triggered:
            summary["pass_guard_queries"] += 1

        commendor_decisions = safe_parse_list(record.get("commendor_decisions", []))
        actual_decisions: List[str] = []
        for item in commendor_decisions:
            if isinstance(item, dict):
                kind = str(item.get("kind", "") or "").strip().lower()
            else:
                kind = str(item or "").strip().lower()
            if kind in ACTUAL_COMMENDOR_DECISIONS:
                actual_decisions.append(kind)

        if not actual_decisions:
            for detail in round_details:
                kind = str(detail.get("commendor_kind", "") or "").strip().lower()
                if kind in ACTUAL_COMMENDOR_DECISIONS:
                    actual_decisions.append(kind)

        if actual_decisions:
            summary["commendor_queries"] += 1
            for kind in actual_decisions:
                summary["commendor_decisions"][kind] += 1

        if round_details:
            final_outcome = infer_final_outcome(record, round_details)
            summary["stop_outcomes"][final_outcome] += 1
            summary["acc_by_stop"][final_outcome].append(acc)

            first_candidate = round_details[0].get("candidate", "")
            candidate_correct = value_matches_answers(first_candidate, answers)
            final_correct = bool(acc)
            if not candidate_correct and final_correct:
                summary["candidate_helped"] += 1
            elif candidate_correct and not final_correct:
                summary["candidate_hurt"] += 1
            else:
                summary["candidate_no_change"] += 1

    summary["acc"] = average(accuracies)
    summary["em"] = average(em_scores)
    summary["f1"] = average(f1_scores)
    summary["has_explicit_f1"] = bool(f1_scores)
    summary["avg_rounds"] = average(rounds_used)
    return summary


def print_counter(title: str, counter: Counter, total: int | None = None) -> None:
    print(f"\n  [{title}]")
    if not counter:
        print("    (none)")
        return
    total = total if total is not None else sum(counter.values()) or 1
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        print(f"    {key:<30s} {count:5d}  ({count / total * 100:.1f}%)")


def print_accuracy_table(title: str, table: Dict[Any, List[float]]) -> None:
    print(f"\n  [{title}]")
    if not table:
        print("    (none)")
        return
    for key in sorted(table):
        values = table[key]
        print(f"    {str(key):<30s} {average(values) * 100:5.1f}%  ({len(values)} queries)")


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    if summary["n"] == 0:
        print("No records found.")
        return

    print(
        f"Samples={summary['n']}  Accuracy={summary['acc'] * 100:.2f}%  "
        f"EM={summary['em'] * 100:.2f}%  Avg rounds={summary['avg_rounds']:.2f}"
    )
    if summary.get("has_explicit_f1"):
        print(f"F1={summary['f1'] * 100:.2f}%")
    print(
        f"Switch queries={summary['switch_queries']}  "
        f"Critic overrides={summary['critic_override_queries']}  "
        f"Commendor queries={summary['commendor_queries']}"
    )
    print(
        f"Low-confidence queries={summary['low_conf_queries']}  "
        f"Low-confidence rounds={summary['low_conf_rounds']}/{summary['total_rounds'] or 1}"
    )
    print(
        f"Regen attempts={summary['regen_attempts']}  "
        f"Regen passes={summary['regen_passes']}  "
        f"Pass-guard queries={summary['pass_guard_queries']}  "
        f"rounds={summary['pass_guard_rounds']}  "
        f"Feedback-normalized rounds={summary['normalized_feedback_rounds']}"
    )
    print(
        f"Candidate->Final helped={summary['candidate_helped']}  "
        f"hurt={summary['candidate_hurt']}  "
        f"unchanged={summary['candidate_no_change']}"
    )

    print_counter("Initial Selection", summary["initial_selection"], summary["n"])
    print_counter("Round Selection", summary["round_selection"])
    print_counter("Top-level Route Source", summary["route_source"], summary["n"])
    print_counter("Round Route Source", summary["round_route_source"])
    print_counter("Critic Verdicts", summary["critic_verdicts"])
    print_counter("Round Outcomes", summary["round_outcomes"])
    print_counter("Final Stop Outcomes", summary["stop_outcomes"], summary["n"])
    print_counter("Commendor Decisions", summary["commendor_decisions"])

    print("\n  [Judge Score Summary]")
    print(
        f"    avg_judge_score={average(summary['judge_scores']):.2f}/10  "
        f"avg_agg_score={average(summary['agg_scores']):.2f}/10"
    )
    if summary["judge_scores_by_round"]:
        for round_id in sorted(summary["judge_scores_by_round"]):
            values = summary["judge_scores_by_round"][round_id]
            print(f"    round {round_id:<24d} {average(values):.2f}/10  ({len(values)} scores)")

    print_accuracy_table("Accuracy by Total Rounds", summary["acc_by_rounds"])
    print_accuracy_table("Accuracy by Final Stop Outcome", summary["acc_by_stop"])

    if summary["critic_by_round"]:
        print("\n  [Critic Verdict by Round]")
        for round_id in sorted(summary["critic_by_round"]):
            row = summary["critic_by_round"][round_id]
            parts = [f"{verdict}={count}" for verdict, count in sorted(row.items(), key=lambda item: (-item[1], item[0]))]
            print(f"    round {round_id}: " + ", ".join(parts))


def result_label(path: str) -> str:
    """Return an experiment-like label for both repo outputs and ad-hoc files."""
    result_path = Path(path)
    if result_path.parent.name == "Results" and len(result_path.parents) > 1:
        return result_path.parents[1].name
    return result_path.stem


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    result_files = discover_result_files(root, args.results, args.dataset)

    if not result_files:
        print("No results.jsonl files found.")
        return

    print(f"Found {len(result_files)} result files:\n")
    for path in result_files:
        print(" ", path)

    all_records: List[Dict[str, Any]] = []
    for path in result_files:
        records = load_records(path)
        all_records.extend(records)
        tag = result_label(path)
        print_summary(tag, summarize_records(records, args.judge_threshold))

    print_summary("AGGREGATE", summarize_records(all_records, args.judge_threshold))


if __name__ == "__main__":
    main()
