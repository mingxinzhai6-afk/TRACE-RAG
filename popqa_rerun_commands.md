# PopQA + MuSiQue rerun commands

Run these commands on the server:

```bash
ssh -p 45423 root@region-41.seetacloud.com
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
mkdir -p logs
```

Before starting, check whether another experiment is still active:

```bash
ps -ef | grep -E 'run_20_main|main.py|newg_main.py' | grep -v grep || true
```

If `run_20_main.sh` is still running, wait for it to finish unless you intentionally want to share API/GPU resources.

## Recommended: Use The Script

The active script is:

```bash
bash run_popqa_reruns.sh
```

It backs up and archives affected output directories, switches `Option/Config2.yaml` to the requested model before each run, writes logs to `logs/rerun_*.log`, and prints a metric summary at the end.

Set `ALLOW_PARALLEL=1` only if you intentionally want to run while another `run_20_main`, `main.py`, or `newg_main.py` process is active:

```bash
ALLOW_PARALLEL=1 bash run_popqa_reruns.sh
```

## Reruns Included

PopQA:

```text
gemini-2.5-flash-lite:
- ToG

deepseek-v3.2:
- BM25
- VDB
- HippoRAG

gpt-4o-mini:
- BM25
- VDB
```

MuSiQue:

```text
deepseek-v3.2:
- BM25
- VDB

gpt-4o-mini:
- BM25
- VDB
```

Total: 10 reruns.

## Output Directories

These reruns write to the standard method/model output directories:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite
output/datasets/Popqa/BM25_deepseek-v3.2
output/datasets/Popqa/VDB_deepseek-v3.2
output/datasets/Popqa/HippoRAG_deepseek-v3.2
output/datasets/Popqa/BM25_gpt-4o-mini
output/datasets/Popqa/VDB_gpt-4o-mini
output/datasets/musique/BM25_deepseek-v3.2
output/datasets/musique/VDB_deepseek-v3.2
output/datasets/musique/BM25_gpt-4o-mini
output/datasets/musique/VDB_gpt-4o-mini
```

The script creates a backup archive and then moves existing affected output directories into an archive folder before rerunning, so the new result directories are clean:

```text
.popqa_rerun_backup_<timestamp>.tar.gz
.popqa_rerun_archived_<timestamp>/
```

## Logs

```text
logs/rerun_popqa_tog_gemini.log
logs/rerun_popqa_bm25_deepseek.log
logs/rerun_popqa_vdb_deepseek.log
logs/rerun_popqa_hipporag_deepseek.log
logs/rerun_popqa_bm25_gpt4omini.log
logs/rerun_popqa_vdb_gpt4omini.log
logs/rerun_musique_bm25_deepseek.log
logs/rerun_musique_vdb_deepseek.log
logs/rerun_musique_bm25_gpt4omini.log
logs/rerun_musique_vdb_gpt4omini.log
```

Watch the current run with:

```bash
tail -f logs/<log_name>.log
```

## Check Results

Expected row count for every listed run: `200`.

```bash
python3 - <<'PY'
import json
import os

paths = [
    ("PopQA", "gemini", "ToG", "output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.score.json"),
    ("PopQA", "deepseek", "BM25", "output/datasets/Popqa/BM25_deepseek-v3.2/Results/results.score.json"),
    ("PopQA", "deepseek", "VDB", "output/datasets/Popqa/VDB_deepseek-v3.2/Results/results.score.json"),
    ("PopQA", "deepseek", "HippoRAG", "output/datasets/Popqa/HippoRAG_deepseek-v3.2/Results/results.score.json"),
    ("PopQA", "gpt-4o-mini", "BM25", "output/datasets/Popqa/BM25_gpt-4o-mini/Results/results.score.json"),
    ("PopQA", "gpt-4o-mini", "VDB", "output/datasets/Popqa/VDB_gpt-4o-mini/Results/results.score.json"),
    ("MuSiQue", "deepseek", "BM25", "output/datasets/musique/BM25_deepseek-v3.2/Results/results.score.json"),
    ("MuSiQue", "deepseek", "VDB", "output/datasets/musique/VDB_deepseek-v3.2/Results/results.score.json"),
    ("MuSiQue", "gpt-4o-mini", "BM25", "output/datasets/musique/BM25_gpt-4o-mini/Results/results.score.json"),
    ("MuSiQue", "gpt-4o-mini", "VDB", "output/datasets/musique/VDB_gpt-4o-mini/Results/results.score.json"),
]
metrics = ["accuracy", "em", "precision", "recall", "f1"]

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

print("dataset\tmodel\tmethod\tn\taccuracy\tem\tprecision\trecall\tf1")
for dataset, model, method, path in paths:
    if not os.path.exists(path):
        print(f"{dataset}\t{model}\t{method}\tMISSING")
        continue
    rows = load_rows(path)
    vals = [
        sum(float(r.get(metric, 0) or 0) for r in rows) / len(rows)
        for metric in metrics
    ]
    print(
        f"{dataset}\t{model}\t{method}\t{len(rows)}\t"
        + "\t".join(f"{v * 100:.2f}%" for v in vals)
    )
PY
```
