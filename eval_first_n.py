"""
Recompute evaluation metrics on the first N lines of an existing results.json.

Usage:
    python eval_first_n.py \
        --result_path output/datasets/Popqa/RAPTOR_deepseek-v3.2/Results/results.json \
        --limit 200

The script is self-contained and does NOT import any DIGIMON modules.
"""

import argparse
import re
import string
from collections import Counter

import pandas as pd


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def eval_accuracy(prediction: str, ground_truth: str) -> int:
    return 1 if normalize_answer(ground_truth) in normalize_answer(prediction) else 0


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0


def f1_score(prediction: str, ground_truth: str):
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    ZERO = (0.0, 0.0, 0.0)
    if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return ZERO
    if norm_gt in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return ZERO

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def short_eval(df: pd.DataFrame):
    accuracy_list, f1_list, prec_list, recall_list, em_list = [], [], [], [], []

    for _, row in df.iterrows():
        prediction = row["output"]
        answer = row["answer"]

        if not isinstance(prediction, str):
            prediction = "" if prediction is None or pd.isna(prediction) else str(prediction)
        if not isinstance(answer, str):
            answer = "" if answer is None or pd.isna(answer) else str(answer)

        prediction_str = prediction.replace("|", " ")
        answer_list = answer.split("|")
        answer_str = " ".join(answer_list)

        accuracy = 1 if any(eval_accuracy(prediction_str, ans) for ans in answer_list) else 0
        f1, prec, recall = f1_score(prediction_str, answer_str)
        em = 1 if any(exact_match_score(prediction_str, ans) for ans in answer_list) else 0

        accuracy_list.append(accuracy)
        f1_list.append(f1)
        prec_list.append(prec)
        recall_list.append(recall)
        em_list.append(em)

    n = len(df)
    acc = sum(accuracy_list) / n
    f1 = sum(f1_list) / n
    pre = sum(prec_list) / n
    rec = sum(recall_list) / n
    em = sum(em_list) / n

    print(f"\n=== Evaluation on first {n} records ===")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"EM        : {em * 100:.2f}%")
    print(f"Precision : {pre * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1        : {f1 * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True,
                        help="Path to results.json (JSONL)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Number of records to evaluate (default: 200)")
    args = parser.parse_args()

    df = pd.read_json(args.result_path, lines=True)
    print(f"Total records in file: {len(df)}")

    if args.limit > 0:
        df = df.head(args.limit)

    short_eval(df)


if __name__ == "__main__":
    main()
