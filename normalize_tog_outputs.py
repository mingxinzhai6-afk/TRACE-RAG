import argparse
import asyncio
import json
import re
import string
from collections import Counter
from pathlib import Path


METRICS = ("accuracy", "em", "precision", "recall", "f1")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return " ".join(remove_articles(remove_punc(str(text).lower())).split())


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)

    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
        return zero
    if normalized_ground_truth in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
        return zero

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    if not prediction_tokens or not ground_truth_tokens:
        return zero

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def contains_accuracy(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(ground_truth) in normalize_answer(prediction))


def score_rows(rows: list[dict]) -> tuple[list[dict], dict[str, float]]:
    scored = []
    sums = {metric: 0.0 for metric in METRICS}
    for row in rows:
        prediction = str(row.get("output") or "").replace("|", " ")
        answer = str(row.get("answer") or "")
        answer_list = answer.split("|") if answer else [""]
        answer_str = " ".join(answer_list)

        accuracy = int(any(contains_accuracy(prediction, ans) for ans in answer_list))
        em = int(any(exact_match(prediction, ans) for ans in answer_list))
        f1, precision, recall = f1_score(prediction, answer_str)

        scored_row = dict(row)
        scored_row.update(
            {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "em": em,
            }
        )
        scored.append(scored_row)
        for metric in METRICS:
            sums[metric] += float(scored_row[metric])

    n = len(scored) or 1
    metrics = {metric: sums[metric] / n for metric in METRICS}
    return scored, metrics


async def normalize_with_llm(rows: list[dict], model: str | None = None, limit: int = 0) -> list[dict]:
    import Core.Provider  # noqa: F401 - registers OpenAI-compatible provider
    from Core.Common.LLM import LLM
    from Core.Query.AnswerNormalizer import AnswerNormalizer
    from Option.Config2 import default_config

    if model:
        default_config.llm.model = model

    normalizer = AnswerNormalizer(LLM(default_config.llm))
    normalized_rows = []
    total = min(len(rows), limit) if limit > 0 else len(rows)
    for idx, row in enumerate(rows[:total], start=1):
        raw_output = str(row.get("output") or "")
        normalized = await normalizer.normalize(str(row.get("question") or ""), raw_output)
        new_row = dict(row)
        new_row["raw_output"] = raw_output
        new_row["output"] = normalized
        normalized_rows.append(new_row)
        print(f"[{idx}/{total}] {new_row.get('id', idx - 1)}\t{normalized}", flush=True)
    return normalized_rows


def normalize_rule_only(rows: list[dict], limit: int = 0) -> list[dict]:
    normalized_rows = []
    total_rows = rows[:limit] if limit > 0 else rows
    for row in total_rows:
        raw_output = str(row.get("output") or "")
        normalized = rule_extract(str(row.get("question") or ""), raw_output)
        new_row = dict(row)
        new_row["raw_output"] = raw_output
        new_row["output"] = normalized
        normalized_rows.append(new_row)
    return normalized_rows


def rule_extract(question: str, raw_output: str) -> str:
    text = raw_output.strip()
    for pattern in (
        r"(?:so\s+)?the\s+answer\s+is[:\s{]+(.+?)(?:[}.]|$)",
        r"Answer:\s*(.+?)(?:[}.]|$)",
        r"\{([^{}]+)\}",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            break

    text = text.strip().strip("\"'{} ").rstrip(".")
    if "," in text and not re.search(r"\b[A-Z][a-z]+ \d{1,2}, \d{4}\b", text):
        pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
        if pieces:
            text = choose_occupation_piece(question, pieces)
    return clean_short_answer(question, text)


def choose_occupation_piece(question: str, pieces: list[str]) -> str:
    if not asks_occupation(question):
        return min(pieces, key=lambda item: (len(item.split()), len(item)))

    priority = (
        "composer", "politician", "lawyer", "attorney", "actor", "actress",
        "journalist", "writer", "novelist", "photographer", "diplomat",
        "chef", "dentist", "physician", "doctor", "professor", "footballer",
        "athlete", "singer", "musician",
    )
    lowered = [(piece.lower(), piece) for piece in pieces]
    for key in priority:
        for low, original in lowered:
            if re.search(rf"\b{re.escape(key)}\b", low):
                return key
    return min(pieces, key=lambda item: (len(item.split()), len(item)))


def clean_short_answer(question: str, text: str) -> str:
    text = re.sub(r"^(?:the\s+)?answer\s+is\s+", "", text, flags=re.IGNORECASE).strip()
    text = text.strip().strip("\"'{} ").rstrip(".")
    if asks_occupation(question):
        text = re.sub(
            r"\b(american|british|chilean|german|french|norwegian|japanese|mexican|"
            r"polish|irish|australian|canadian|indian|italian|spanish|conservative|"
            r"liberal|professional|former)\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = " ".join(text.split())
    return text or "Insufficient information"


def asks_occupation(question: str) -> bool:
    question_l = question.lower()
    return any(term in question_l for term in ("occupation", "profession", "job"))


def write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize existing ToG outputs without re-running retrieval."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input results.json JSONL file.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for normalized Results files.")
    parser.add_argument("--mode", choices=["llm", "rule"], default="llm")
    parser.add_argument("--model", default=None, help="Optional LLM model override for --mode llm.")
    parser.add_argument("--limit", type=int, default=0, help="Normalize only the first N rows; 0 = all rows.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.mode == "llm":
        normalized_rows = await normalize_with_llm(rows, model=args.model, limit=args.limit)
    else:
        normalized_rows = normalize_rule_only(rows, limit=args.limit)

    scored_rows, metrics = score_rows(normalized_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "results.json", normalized_rows)
    write_jsonl(args.out_dir / "results.score.json", scored_rows)
    write_metrics(args.out_dir / "metrics.json", metrics)

    print("dataset-free normalized metrics")
    for metric in METRICS:
        print(f"{metric}: {metrics[metric]:.4f} | {metrics[metric] * 100:.2f}%")
    print(args.out_dir)


if __name__ == "__main__":
    asyncio.run(main_async())
