"""
生成单条 NewG 结果记录的详细诊断报告（Markdown 格式）。
适合人工逐条检查错误模式。
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().strip(".,")


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


def value_matches_answers(value: Any, answers: List[str]) -> bool:
    candidate = normalize_text(value)
    if not candidate or not answers:
        return False
    return any(answer in candidate or candidate in answer for answer in answers)


def judge_score_stdev(scores: List[float]) -> float:
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    return variance ** 0.5


def diagnose(record: Dict[str, Any], idx: int) -> str:
    answers = [normalize_text(a) for a in safe_parse_list(record.get("answers"))]
    if not answers:
        answers = [normalize_text(record.get("answer", ""))]
    answers = [a for a in answers if a]

    output = normalize_text(record.get("output", ""))
    correct = value_matches_answers(output, answers)

    lines: List[str] = []
    lines.append(f"## Case #{idx} | ID={record.get('id', '?')} | {'✅ CORRECT' if correct else '❌ WRONG'}")
    lines.append("")
    lines.append(f"**Question:** {record.get('question', '')}")
    lines.append(f"**Ground Truth:** {record.get('answer', '')}")
    lines.append(f"**Model Output:** `{record.get('output', '')}`")
    lines.append(f"**Domain:** {record.get('domain', 'unknown')} | **Initial Selection:** {record.get('initial_selection', 'unknown')}")
    lines.append("")

    # 多跳分解
    decomposition = safe_parse_list(record.get("question_decomposition"))
    if decomposition:
        lines.append("### Question Decomposition")
        for step in decomposition:
            lines.append(f"- Step {step.get('id', '?')}: {step.get('question', '')} → **{step.get('answer', '')}**")
        lines.append("")

    round_details = safe_parse_list(record.get("round_details", []))
    total_rounds = int(record.get("rounds", 0) or 0)

    for rd in round_details:
        r = int(rd.get("round", 0))
        lines.append(f"### Round {r}")
        lines.append("")

        # 路由
        selection = rd.get("selection", "unknown")
        route_source = rd.get("route_source", "unknown")
        lines.append(f"| 字段 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| Selection | `{selection}` |")
        lines.append(f"| Route Source | `{route_source}` |")
        lines.append(f"| Retrieval Query | `{rd.get('retrieval_query', '')}` |")
        lines.append("")

        # 证据
        evidence = str(rd.get("evidence_snippet", "") or "")[:500]
        lines.append("**Evidence Snippet (truncated):**")
        lines.append(f"```\n{evidence}{'...' if len(str(rd.get('evidence_snippet', ''))) > 500 else ''}\n```")
        lines.append("")

        # Candidate -> Intermediate -> Output 链条
        candidate = rd.get("candidate", "")
        intermediate = rd.get("intermediate_answer", "")
        normalized = rd.get("normalized_for_critic", "")
        lines.append("**Answer Pipeline:**")
        lines.append(f"- Candidate: `{candidate}`")
        lines.append(f"- Intermediate: `{intermediate}`")
        lines.append(f"- Normalized for Critic: `{normalized}`")
        lines.append("")

        # Judge 评分
        scores = safe_parse_list(rd.get("judge_scores", []))
        candidates = safe_parse_list(rd.get("judge_candidates", []))
        agg = rd.get("agg_score", 0.0)
        votes = safe_parse_list(rd.get("votes", []))
        low_conf = rd.get("low_confidence", False)

        lines.append("**Judge Panel:**")
        lines.append(f"| Judge | Score | Candidate |")
        lines.append(f"|-------|-------|-----------|")
        for i, (s, c) in enumerate(zip(scores, candidates)):
            lines.append(f"| #{i+1} | {s} | `{c}` |")
        lines.append("")
        lines.append(f"- **Aggregate Score:** {agg}")
        lines.append(f"- **Consensus Votes:** {votes}")
        lines.append(f"- **Score StdDev:** {judge_score_stdev([float(s) for s in scores]):.2f} (0 = 完全一致)")
        lines.append(f"- **Low Confidence Flag:** {low_conf}")
        lines.append("")

        # Critic
        verdict = rd.get("critic_verdict", "unknown")
        conf = rd.get("critic_confidence", 0.0)
        actions = rd.get("critic_actions", "")
        lines.append("**Critic Review:**")
        lines.append(f"- Verdict: `{verdict}` (confidence: {conf})")
        lines.append(f"- Action/Suggestion: {actions}")
        lines.append("")

        # Commendor
        commendor_kind = rd.get("commendor_kind", "")
        if commendor_kind:
            lines.append(f"**Commendor:** `{commendor_kind}`")
            lines.append("")

        # Outcome
        outcome = rd.get("round_outcome", "unknown")
        lines.append(f"**Round Outcome:** `{outcome}`")
        lines.append("")

        # 诊断（单轮内）
        lines.append("#### 🔍 Diagnosis")
        diagnoses = []

        if not correct and verdict == "pass":
            diagnoses.append("- **Critic false positive:** Critic passed a wrong answer. Check if critic prompt is too lenient or if evidence misled the critic.")

        if len(set(candidates)) <= 1 and len(candidates) >= 2:
            diagnoses.append("- **Judge degeneracy:** All judges produced identical candidates. Multi-agent consensus is not providing diversity.")

        if scores and all(float(s) >= 6.0 for s in scores) and not correct:
            diagnoses.append("- **High judge score, wrong answer:** Judges are miscalibrated. Scores do not correlate with correctness on this sample.")

        if candidate and not value_matches_answers(candidate, answers) and value_matches_answers(intermediate, answers):
            diagnoses.append("- **Answer corruption:** Candidate was correct but intermediate_answer/normalization lost it.")

        if candidate and value_matches_answers(candidate, answers) and not value_matches_answers(intermediate, answers):
            diagnoses.append("- **Regeneration/regression:** Candidate was correct but later pipeline step changed it to wrong.")

        if output and answers and not any(a in output or output in a for a in answers):
            # 答非所问检测
            q_lower = normalize_text(record.get("question", ""))
            if "where" in q_lower and output:
                # 简单启发：如果答案看起来像机构/人名而不是地点
                if any(x in output for x in ["university", "school", "college", "institute"]):
                    diagnoses.append("- **Question-type mismatch:** Question asks for location ('where'), but model returned an institution name. Query understanding or retriever may not respect the question type.")
            if "who" in q_lower and output:
                if any(x in output for x in ["serbia", "ottoman", "empire", "country"]):
                    diagnoses.append("- **Entity-type mismatch:** Question asks for a person ('who'), but model returned a country/empire. Entity type resolution failed.")
            if "when" in q_lower and output:
                if not any(c.isdigit() for c in output):
                    diagnoses.append("- **Temporal mismatch:** Question asks for time ('when'), but output contains no date.")

        # 时态/时间范围错误（针对历史题）
        if "kosovo" in q_lower and "was" in q_lower and "in charge" in q_lower:
            if "lazar" in output:
                diagnoses.append("- **Temporal scope error:** Question asks about present leadership (implied by 'was in charge of the country [Serbia]'), but graph retriever returned medieval leader. Graph evidence lacks temporal grounding.")

        if not diagnoses:
            diagnoses.append("- No automated diagnosis triggered. Manual review needed.")

        lines.extend(diagnoses)
        lines.append("")

    # 全局诊断
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-case diagnosis report")
    parser.add_argument("input", help="Path to results.jsonl")
    parser.add_argument("-o", "--output", default="case_report.md", help="Output markdown file")
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    lines: List[str] = []
    lines.append("# NewG Experiment Case Report")
    lines.append("")
    lines.append(f"**Total records:** {len(records)}")
    lines.append("")

    for idx, record in enumerate(records, 1):
        lines.append(diagnose(record, idx))

    # 汇总
    wrong = [r for r in records if not value_matches_answers(r.get("output", ""), [normalize_text(r.get("answer", ""))])]
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- Total: {len(records)}")
    lines.append(f"- Correct: {len(records) - len(wrong)}")
    lines.append(f"- Wrong: {len(wrong)}")
    lines.append("")

    # 统计最常见的 stop outcome
    from collections import Counter
    outcomes = Counter()
    for r in records:
        rd = safe_parse_list(r.get("round_details", []))
        if rd:
            outcomes[rd[-1].get("round_outcome", "unknown")] += 1
    if outcomes:
        lines.append("### Stop Outcome Distribution")
        for k, v in outcomes.most_common():
            lines.append(f"- {k}: {v}")
        lines.append("")

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
