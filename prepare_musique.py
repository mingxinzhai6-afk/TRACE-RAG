"""
Convert MuSiQue dataset to GraphRAG format.

Downloads from HuggingFace and writes:
  datasets/MuSiQue/Corpus.json   -- jsonlines: {title, context}
  datasets/MuSiQue/Question.json -- jsonlines: {question, answer}
"""

import json
import os

OUTPUT_DIR = "datasets/MuSiQue"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading MuSiQue from HuggingFace...")
from datasets import load_dataset
ds = load_dataset("StonyBrookNLP/musique", split="validation")

corpus_dict = {}  # title -> paragraph_text (deduplicate)
questions = []

for item in ds:
    if not item.get("answerable", True):
        continue
    for para in item["paragraphs"]:
        title = para["title"]
        if title not in corpus_dict:
            corpus_dict[title] = para["paragraph_text"]
    questions.append({
        "question": item["question"],
        "answer": item["answer"],
    })

corpus_path = os.path.join(OUTPUT_DIR, "Corpus.json")
with open(corpus_path, "w", encoding="utf-8") as f:
    for title, context in corpus_dict.items():
        f.write(json.dumps({"title": title, "context": context}, ensure_ascii=False) + "\n")

question_path = os.path.join(OUTPUT_DIR, "Question.json")
with open(question_path, "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"Done. Corpus: {len(corpus_dict)} docs, Questions: {len(questions)}")
