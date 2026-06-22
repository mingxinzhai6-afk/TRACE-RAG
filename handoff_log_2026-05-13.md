# NewG / GraphRAG Handoff Log

Last updated: 2026-05-16 17:15:09 +08:00, Asia/Shanghai

This file is a detailed handoff for continuing the NewG / GraphRAG experiment work with a fresh Codex session or a new GPT account.

Sensitive-value handling:

- The user explicitly requested that the server password be stored in this handoff file.
- Do not write API keys into this file.
- Do not copy the password into unrelated files or public output unless the user asks again.

## 2026-05-15 11:30 Resume Update

Continuation was requested with "继续上次任务".

Read-only remote check first found no active experiment processes, but the previous
watcher logs showed the workflow had exited during MuSiQue ablation step 1:

```text
logs/watch_popqa_then_musique_tog_20260515_103253.log
logs/abl_musique_step1_simple.log
```

Failure signature in `logs/abl_musique_step1_simple.log`:

```text
Cannot assign requested address while requesting HEAD https://huggingface.co/BAAI/bge-small-en-v1.5/...
RuntimeError: Cannot send a request, as the client has been closed.
```

Root cause:

- `BAAI/bge-small-en-v1.5` was cached under `/root/.cache/llama_index`.
- Default Hugging Face cache lookup did not find it.
- If the model loader tried the network HEAD request, it could fail under transient remote networking limits.

Remote cache fix applied:

```text
/root/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5
  -> /root/.cache/llama_index/models--BAAI--bge-small-en-v1.5
```

Verified after the symlink:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 /root/miniconda3/envs/graphrag/bin/python -c 'from sentence_transformers import SentenceTransformer; print(SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu").get_sentence_embedding_dimension())'
```

Output was:

```text
OK 384
```

Current remote experiment state after this resume:

```text
PID 402569  bash run_ablation_musique.sh
PID 402571  /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/NewG_abl_simple.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200
```

At the last check, MuSiQue ablation step 1 was actively progressing and had written
checkpoint output to:

```text
output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_simple/Results/results.jsonl
```

Added locally and synced to the server:

```text
watch_musique_then_tog.sh
```

Purpose:

- Waits for the currently running `run_ablation_musique.sh` and MuSiQue `newg_main.py` child to finish.
- Verifies all six MuSiQue ablation `results.score.json` files exist and each has 200 rows.
- Only then runs `bash run_tog_gemini_v3.sh`.
- If any score file is missing or incomplete, it logs the problem and does not start ToG.

Remote syntax check:

```bash
bash -n watch_musique_then_tog.sh
```

Passed on the server.

Started the new watcher at `2026-05-15 11:27:54 +08:00`:

```text
PID 433655  bash watch_musique_then_tog.sh
```

Watcher log:

```text
logs/watch_musique_then_tog_20260515_112754.log
logs/watch_musique_then_tog.nohup.log
```

Next check should be read-only:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
ps -ef | grep -E 'watch_musique_then_tog|run_ablation_musique|run_tog_gemini_v3|newg_main.py|main.py' | grep -v grep || true
tail -80 logs/watch_musique_then_tog_20260515_112754.log
tail -80 logs/abl_musique_step1_simple.log
```

Do not start another MuSiQue or ToG watcher while PID 433655, PID 402569,
or their child experiment processes are active.

## 2026-05-15 12:45 Leave-One-Out Ablation Code Update

The user asked to redesign the ablation experiment as Full-minus-one / leave-one-out.

Added locally and synced to the server:

```text
generate_newg_leave_one_out_configs.py
run_leave_one_out_ablation.sh
collect_leave_one_out_ablation.py
plot_leave_one_out_ablation.py
```

Design:

- `Option/Method/NewG.yaml` is the only Full NewG source of truth.
- `generate_newg_leave_one_out_configs.py` generates derived configs under:

```text
Option/Method/generated_leave_one_out/
```

- Variants:

```text
full
no_router
no_regen
no_critic
no_commendor
no_normalizer
no_disambiguation
single_agent
```

- The generated configs produce result tags like:

```text
NewG_hipporag_bm25_gemini-2.5-flash-lite_loo_no_router
```

- `run_leave_one_out_ablation.sh` defaults to:

```text
MODEL=gemini-2.5-flash-lite
GRAPH=hipporag
TEXT=bm25
DATA=datasets/Popqa
LIMIT=200
```

- It refuses to start if another experiment is active unless `ALLOW_PARALLEL=1`.
- It exports `SENTENCE_TRANSFORMERS_HOME=/root/.cache/llama_index` to avoid unnecessary Hugging Face network lookups.
- It summarizes each variant after completion and accepts both `results.score.json` and `results.score.jsonl`.

Validation:

```bash
/root/miniconda3/envs/graphrag/bin/python -m py_compile generate_newg_leave_one_out_configs.py collect_leave_one_out_ablation.py plot_leave_one_out_ablation.py
bash -n run_leave_one_out_ablation.sh
```

Both passed on the server. No leave-one-out experiment was started.

Important watcher fix:

- `watch_musique_then_tog.sh` previously checked only `results.score.json`.
- Actual evaluator output is `results.score.jsonl`.
- Updated `watch_musique_then_tog.sh` to accept both `results.score.json` and `results.score.jsonl`.
- Because the old watcher had already loaded the old function into memory, it was restarted without touching the running MuSiQue experiment.

Old watcher:

```text
PID 433655  bash watch_musique_then_tog.sh
```

New fixed watcher:

```text
PID 466297  bash watch_musique_then_tog.sh
```

New fixed watcher log:

```text
logs/watch_musique_then_tog_20260515_124115.log
```

Current remote experiment process at the time of this update:

```text
PID 402569  bash run_ablation_musique.sh
PID 435273  /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/NewG_abl_routing.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200
```

Do not run `run_leave_one_out_ablation.sh` until the current MuSiQue ablation and automatic ToG rerun are finished, unless the user explicitly approves parallel execution.

## 2026-05-15 12:48 Watcher Stopped

The user asked to stop the MuSiQue-to-ToG watcher because it is no longer needed.

Stopped only the watcher processes:

```text
PID 466297  bash watch_musique_then_tog.sh
PID 466302  bash watch_musique_then_tog.sh
PID 466303  tee -a logs/watch_musique_then_tog_20260515_124115.log
```

Removed stale watcher lock:

```text
.watch_musique_then_tog.lock
```

Confirmed the actual MuSiQue ablation was not stopped and remained active:

```text
PID 402569  bash run_ablation_musique.sh
PID 435273  /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/NewG_abl_routing.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200
```

There is now no automatic watcher that will start `run_tog_gemini_v3.sh` after MuSiQue ablation finishes. Start ToG manually later if still needed.

## 2026-05-15 12:58 Both-Dataset Leave-One-Out Commands

The user asked to combine the new leave-one-out ablation commands for both datasets and avoid rerunning variants already covered by previous ablations.

Added and synced:

```text
run_leave_one_out_ablation_both_datasets.sh
leave_one_out_ablation_commands.md
```

Updated:

```text
run_leave_one_out_ablation.sh
collect_leave_one_out_ablation.py
```

Skip/equivalence logic:

```text
full          -> existing main NewG result
no_commendor  -> existing NewG_abl_no_commendor result
no_normalizer -> existing NewG_abl_critic result
single_agent  -> existing NewG_abl_single_agent result
```

The old `Simple`, `+Router`, and `+Re-Generator` incremental results are not reused as leave-one-out results because they disable multiple modules at once.

The variants still needing true leave-one-out runs are:

```text
no_router
no_regen
no_critic
no_disambiguation
```

The full command after the server is idle:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
screen -S loo_both
bash run_leave_one_out_ablation_both_datasets.sh
```

Detached option:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
nohup bash run_leave_one_out_ablation_both_datasets.sh > logs/loo_both.nohup.log 2>&1 &
```

If the user wants to explicitly run only the truly new variants:

```bash
VARIANTS="no_router no_regen no_critic no_disambiguation" bash run_leave_one_out_ablation_both_datasets.sh
```

Validation passed on the server:

```bash
/root/miniconda3/envs/graphrag/bin/python -m py_compile generate_newg_leave_one_out_configs.py collect_leave_one_out_ablation.py plot_leave_one_out_ablation.py
bash -n run_leave_one_out_ablation.sh
bash -n run_leave_one_out_ablation_both_datasets.sh
```

Read-only collection currently recognizes:

```text
PopQA: full, no_commendor, no_normalizer, single_agent already available.
MuSiQue: full available; no_commendor/no_normalizer/single_agent expected after current old MuSiQue ablation finishes.
```

No leave-one-out experiments were started.

## 2026-05-15 15:01 Conversation Preservation Update

The user explicitly asked to preserve the current conversation so that saying
"继续上次任务" in a future session will not lose context.

Current server state from the latest read-only check at `2026-05-15 14:58 CST`:

```text
PID 402569  bash run_ablation_musique.sh
PID 467611  /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/NewG_abl_regen.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200
```

Current MuSiQue old/incremental ablation progress:

```text
Step 1 Simple             complete, 200 rows, results.score.jsonl
Step 2 +Router            complete, 200 rows, results.score.jsonl
Step 3 +Re-Generator      running, around Q53/200 at 14:58 CST, checkpoint had 50 rows
Step 4 +Critic & Commendor not started
Step 5 w/o Commendor      not started
Step 6 Single judge/voter not started
```

Important: the `watch_musique_then_tog.sh` watcher was stopped at the user's
request. There is currently no watcher that will automatically start
`run_tog_gemini_v3.sh` after MuSiQue ablation finishes.

New leave-one-out ablation design and code:

```text
generate_newg_leave_one_out_configs.py
run_leave_one_out_ablation.sh
run_leave_one_out_ablation_both_datasets.sh
collect_leave_one_out_ablation.py
plot_leave_one_out_ablation.py
leave_one_out_ablation_commands.md
```

The new leave-one-out variants are:

```text
full
no_router
no_regen
no_critic
no_commendor
no_normalizer
no_disambiguation
single_agent
```

Already-covered equivalents are skipped by default:

```text
full          -> existing main NewG result
no_commendor  -> existing NewG_abl_no_commendor result
no_normalizer -> existing NewG_abl_critic result
single_agent  -> existing NewG_abl_single_agent result
```

Do not reuse old `Simple`, `+Router`, or `+Re-Generator` as leave-one-out rows,
because those disable multiple modules at once and are not equivalent.

When the current MuSiQue ablation finishes and the server is idle, the main
command for the new two-dataset leave-one-out experiment is:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
screen -S loo_both
bash run_leave_one_out_ablation_both_datasets.sh
```

To explicitly run only truly new variants:

```bash
VARIANTS="no_router no_regen no_critic no_disambiguation" bash run_leave_one_out_ablation_both_datasets.sh
```

Next session instructions when the user says "继续上次任务":

1. Read this handoff file first.
2. Do a read-only server process check:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
ps -ef | grep -E 'run_ablation_musique|run_leave_one_out|newg_main.py|main.py' | grep -v grep || true
```

3. Check current MuSiQue ablation log and result counts:

```bash
ls -1t logs/abl_musique_step*.log | head
tail -80 "$(ls -1t logs/abl_musique_step*.log | head -1)"
```

4. If `run_ablation_musique.sh` is still active, do not start
`run_leave_one_out_ablation_both_datasets.sh`.
5. If `run_ablation_musique.sh` has completed, summarize the six old MuSiQue
ablation rows, then decide whether to start the new leave-one-out run.

## 2026-05-15 10:33 Resume Update

Continuation was requested with "continue last task".

Read-only remote check found:

- No active experiment process at the start of this resume.
- The previous watcher had exited after starting MuSiQue ablation step 1.
- Failure in `logs/abl_musique_step1_simple.log` was first `python: command not found`.
- After switching to system `python3`, the next failure was `ModuleNotFoundError: No module named 'pandas'`.

Root cause:

- Non-interactive watcher shell does not activate Conda.
- The working project interpreter is:

```text
/root/miniconda3/envs/graphrag/bin/python
```

Verified on the server:

```text
pandas=2.3.3
```

Updated locally and synced to the server:

```text
run_ablation_musique.sh
run_ablation_popqa.sh
run_tog_gemini_v3.sh
```

Each script now uses:

```bash
PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"
```

Remote syntax checks passed:

```bash
bash -n run_ablation_musique.sh
bash -n run_ablation_popqa.sh
bash -n run_tog_gemini_v3.sh
```

Restarted the watcher at `2026-05-15 10:32:53 +08:00`:

```text
PID 400851  bash watch_popqa_then_musique_tog.sh
```

Current child process at the post-start check:

```text
/root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/NewG_abl_simple.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200
```

New watcher logs:

```text
logs/watch_popqa_then_musique_tog.resume_20260515_103253.nohup.log
logs/watch_popqa_then_musique_tog_20260515_103253.log
```

Next check should be read-only:

```bash
ps -ef | grep -E 'watch_popqa_then_musique_tog|run_ablation_musique|run_tog_gemini_v3|newg_main.py|main.py' | grep -v grep
tail -80 logs/watch_popqa_then_musique_tog_20260515_103253.log
tail -80 logs/abl_musique_step1_simple.log
```

Do not start a duplicate watcher while PID 400851 or its child experiment is active.

## Workspace And Server

Local project path:

```text
D:\Tools\Downloads\GraphRAG-master1\root\autodl-tmp\GraphRAG-master\GraphRAG-master
```

Remote server SSH:

```bash
ssh -p 45423 root@region-41.seetacloud.com
```

Remote server password, stored at user request:

```text
m0bXX1jW3rzP
```

Remote project path:

```bash
/root/autodl-tmp/GraphRAG-master/GraphRAG-master
```

The server login was verified during the conversation.

## 2026-05-14 23:47 Auto Watcher Update

The user asked for a script that monitors `bash run_ablation_popqa.sh` and immediately starts the next queued experiments after PopQA ablation finishes.

Added and synced to the server:

```text
watch_popqa_then_musique_tog.sh
```

Behavior:

1. Polls every 30 seconds by default.
2. Waits until both `run_ablation_popqa.sh` and its PopQA `newg_main.py` child are gone.
3. Sets `Option/Config2.yaml` first `llm.model` back to `gemini-2.5-flash-lite`.
4. By default runs sequentially to avoid API/GPU contention:

```bash
bash run_ablation_musique.sh
bash run_tog_gemini_v3.sh
```

5. Optional true parallel mode exists but should be used only if the user explicitly accepts resource/API contention:

```bash
RUN_PARALLEL=1 bash watch_popqa_then_musique_tog.sh
```

Started on the server at `2026-05-14 23:47:16 +08:00`:

```text
PID 316457  bash watch_popqa_then_musique_tog.sh
```

Watcher logs:

```text
logs/watch_popqa_then_musique_tog.nohup.log
logs/watch_popqa_then_musique_tog_20260514_234716.log
```

At watcher startup, PopQA ablation was still active:

```text
bash run_ablation_popqa.sh
python newg_main.py -opt Option/Method/NewG_abl_critic.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200
```

Next session should first check:

```bash
ps -ef | grep -E 'watch_popqa_then_musique_tog|run_ablation_popqa|run_ablation_musique|run_tog_gemini_v3|newg_main.py|main.py' | grep -v grep
tail -80 logs/watch_popqa_then_musique_tog_20260514_234716.log
```

## 2026-05-14 23:54 Shutdown And Resume Note

The user asked whether closing Codex, closing the browser, or powering off the local computer would affect the running jobs.

Answer given:

- It will not affect the server-side watcher because `watch_popqa_then_musique_tog.sh` is running as a detached background process on the remote server.
- The user confirmed `run_ablation_popqa.sh` is running inside `screen`, so closing the local computer should not interrupt that PopQA ablation process either.
- The only expected interruption risk is stopping/rebooting/releasing the remote server/container itself.

Current continuation rule:

- If the user says "continue last task" or "继续上次任务", do not restart from scratch.
- Read this handoff first.
- Check the watcher log and current processes first.
- If PopQA ablation has finished, verify whether the watcher has already started or completed `run_ablation_musique.sh` and `run_tog_gemini_v3.sh`.
- Do not kill or restart existing experiment jobs unless the user explicitly asks.

## 2026-05-14 20:06 Conversation Preservation Update

The user asked to preserve the latest discussion so future sessions can continue from here when they say "继续上次任务".

Important decisions and clarifications:

1. Gemini ToG rerun target
   - The user wants only the normal Gemini ToG result, not diagnostic side branches.
   - Final desired result path:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.score.json
```

   - Do not use `ToG_old_normalized_gemini-2.5-flash-lite` in paper tables.
   - Do not use `ToG_v3_gemini-2.5-flash-lite` as a final method/result name.
   - `normalize_tog_outputs.py` is diagnostic only.
   - `run_tog_gemini_v3.sh` was corrected to run normal `Option/Method/ToG.yaml` with the updated ToG prompt and to archive the old normal ToG directory before writing a new normal result.

2. Why the stricter Gemini ToG rerun got worse
   - Old Gemini ToG:

```text
Accuracy 40.50 | EM 10.00 | Precision 23.06 | Recall 15.22 | F1 14.75
```

   - New stricter rerun:

```text
Accuracy 29.00 | EM 20.50 | Precision 24.18 | Recall 09.28 | F1 11.42
```

   - Analysis: EM rose, but Accuracy/F1 fell because old long/multi-candidate answers often matched PopQA aliases by containment, while stricter prompting produced many `Insufficient information` answers and short single-label answers.
   - The updated ToG prompt keeps short-answer constraints but forbids occupation questions from abstaining with `Insufficient information`.

3. Ablation naming
   - The user noted that `abl_*` naming is misleading because "ablation" usually implies missing/removing a module.
   - Use clearer table names:

```text
Simple
+ Router
+ Re-Generator
+ Critic & Commendor
w/o Commendor
Single judge/voter
Full NewG
```

   - `Simple` means fixed `hipporag + bm25`, Evidence Fusion/Candidate Generator only, no Router, Re-Generator, Critic, Commendor, or Answer Normalizer.
   - `+ Critic & Commendor` is more accurate than just `+ Critic` because `NewG_abl_critic.yaml` enables both.
   - `Full NewG` includes Answer Normalizer.

4. Ablation design clarification
   - The current experiment is a mixed design:
     - Incremental module analysis:

```text
Simple -> + Router -> + Re-Generator -> + Critic & Commendor -> Full NewG
```

     - Component/sensitivity analysis:

```text
Full NewG vs w/o Commendor
Full NewG vs Single judge/voter
```

   - Do not describe all rows as "each module removed" because the first four rows are cumulative additions.
   - If writing paper text, describe it as "incremental module analysis plus component sensitivity analysis".

5. Full vs normalizer clarification
   - The earlier phrase "`abl_normalizer / full`" was misleading.
   - Better wording:

```text
Full NewG = + Answer Normalizer / normalized feedback on top of + Critic & Commendor
```

   - Comparing `+ Critic & Commendor` to `Full NewG` estimates the effect of Answer Normalizer and normalized critic feedback.
   - `NewG_abl_normalizer.yaml` exists but is redundant if it is configuration-equivalent to the main full NewG run.

6. Current known PopQA ablation status from the last check
   - `run_ablation_popqa.sh` was running on the server.
   - It was on the second step:

```text
NewG_abl_routing.yaml
```

   - At that check, `Simple` had completed with:

```text
Accuracy 66.00 | EM 24.00 | Precision 42.16 | Recall 26.49 | F1 27.15
```

   - The large Accuracy/EM gap for `Simple` is expected because `use_normalizer: false`. There were 84 examples with `accuracy=1` and `em=0`, e.g. `Conservative politician` matching `politician` by containment but not exact match.

Next session instructions:

1. First read this handoff file.
2. Check server process status for `run_ablation_popqa.sh`, `run_ablation_musique.sh`, `newg_main.py`, and `main.py`.
3. Continue monitoring/filling the ablation table using the naming above.
4. Do not start the Gemini ToG normal rerun until ablation is idle, unless the user explicitly accepts parallel resource/API contention.

## 2026-05-14 19:22 Gemini ToG V3 Preparation

The Gemini ToG rerun with stricter short-answer prompting lowered Accuracy/F1 because it produced many `Insufficient information` answers and removed the older long-answer/multi-candidate behavior that often matched PopQA aliases by containment.

Implemented a third ToG prompt variant in:

```text
Core/Prompt/TogPrompt.py
```

The new prompt keeps short-answer constraints but weakens abstention for occupation questions:

- occupation/profession/job questions must not answer `Insufficient information`;
- if relation/entity clues include titles such as Member of Parliament, minister, judge, lawyer, actor, musician, composer, etc., map to a generic occupation label;
- do not copy KG triplets or relation names as the final occupation answer;
- `Insufficient information` is reserved for non-occupation questions with no plausible answer.

Added:

```text
normalize_tog_outputs.py
Option/Method/ToG_v3.yaml
run_tog_gemini_v3.sh
```

Purpose:

- `normalize_tog_outputs.py` was created as a diagnostic utility for old ToG raw outputs, but it is not part of the final normal Gemini ToG rerun.
- `Option/Method/ToG_v3.yaml` was created as a non-overwrite scratch option, but the user clarified that only the normal Gemini ToG result is wanted.
- `run_tog_gemini_v3.sh` was corrected to run normal `Option/Method/ToG.yaml` with the updated ToG prompt, archive the previous `ToG_gemini-2.5-flash-lite` directory, and write the new result back to the normal path.

Synced to the server and checked:

```bash
python3 -m py_compile normalize_tog_outputs.py
bash -n run_tog_gemini_v3.sh
bash -n run_ablation_popqa.sh
```

All passed on the server.

Do not start `run_tog_gemini_v3.sh` while ablation is active unless the user explicitly accepts parallel resource/API contention. At the time of this update, the server was running:

```text
bash run_ablation_popqa.sh
python newg_main.py -opt Option/Method/NewG_abl_routing.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200
```

When the server is idle, run:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
bash run_tog_gemini_v3.sh
```

Expected normal ToG output:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.score.json
logs/rerun_popqa_tog_gemini_v3.log
```

Do not use `ToG_old_normalized_*` in the main result table; it is a diagnostic branch only.

## 2026-05-14 18:01 Rerun Completion Update

Checked at `2026-05-14 17:47:07 +08:00`.

Server process status:

```text
No run_20_main, run_popqa_reruns, main.py, or newg_main.py processes were running.
```

The 10-run `run_popqa_reruns.sh` batch has completed. All 10 target rerun score files have exactly 200 rows:

```text
PopQA   gemini-2.5-flash-lite  ToG       Acc 29.00 | EM 20.50 | Precision 24.18 | Recall 09.28 | F1 11.42
PopQA   deepseek-v3.2          BM25      Acc 65.50 | EM 65.50 | Precision 66.58 | Recall 21.95 | F1 28.86
PopQA   deepseek-v3.2          VDB       Acc 55.50 | EM 55.50 | Precision 56.00 | Recall 17.82 | F1 23.59
PopQA   deepseek-v3.2          HippoRAG  Acc 61.00 | EM 61.00 | Precision 61.50 | Recall 21.09 | F1 27.34
PopQA   gpt-4o-mini            BM25      Acc 65.00 | EM 64.50 | Precision 65.83 | Recall 22.17 | F1 28.86
PopQA   gpt-4o-mini            VDB       Acc 54.00 | EM 53.50 | Precision 54.25 | Recall 17.48 | F1 23.02
MuSiQue deepseek-v3.2          BM25      Acc 13.50 | EM 13.00 | Precision 18.59 | Recall 16.93 | F1 17.24
MuSiQue deepseek-v3.2          VDB       Acc 08.00 | EM 08.00 | Precision 11.33 | Recall 10.33 | F1 10.65
MuSiQue gpt-4o-mini            BM25      Acc 16.50 | EM 16.00 | Precision 23.91 | Recall 21.77 | F1 22.22
MuSiQue gpt-4o-mini            VDB       Acc 10.50 | EM 09.50 | Precision 14.42 | Recall 13.33 | F1 13.40
```

Generated on the server:

```text
results_summary.tsv
results_popqa.md
results_musique.md
```

The server environment did not have `matplotlib`, so plotting was regenerated locally and uploaded back to the server. Updated figures:

```text
figures/popqa_cross_model_by_method.png
figures/popqa_cross_model_f1.png
figures/popqa_deepseek_baseline_vs_newg.png
figures/popqa_gpt4omini_baseline_vs_newg.png
figures/popqa_gemini_baseline_vs_newg.png
figures/popqa_f1_baseline_vs_newg_summary.png
figures/musique_cross_model_by_method.png
figures/musique_cross_model_f1.png
figures/musique_deepseek_baseline_vs_newg.png
figures/musique_gpt4omini_baseline_vs_newg.png
figures/musique_gemini_baseline_vs_newg.png
figures/musique_f1_baseline_vs_newg_summary.png
```

Local copies of the three result tables and the updated figures were also refreshed in the project root.

`Option/Config2.yaml` on the server was reset after reruns so the first `llm.model` entry is back to:

```yaml
model: "gemini-2.5-flash-lite"
```

## 2026-05-14 Continuation Update

Checked at `2026-05-14 10:25:25 +08:00`.

Main Gemini experiment status:

```text
MuSiQue / NewG_tog_vdb_gemini-2.5-flash-lite      200 rows
MuSiQue / NewG_raptor_bm25_gemini-2.5-flash-lite  200 rows
MuSiQue / NewG_raptor_vdb_gemini-2.5-flash-lite   200 rows
```

So the previous 24-command main experiment has completed.

Rerun status:

```text
PID 853320  bash run_popqa_reruns.sh
PID 853341  python3 main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

The 10-run rerun script is already running on the server and is currently on the first run:

```text
PopQA / gemini-2.5-flash-lite / ToG
```

The script created:

```text
.popqa_rerun_backup_20260514_100415.tar.gz
.popqa_rerun_archived_20260514_100415/
```

Archived old target outputs currently include:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite
output/datasets/Popqa/HippoRAG_deepseek-v3.2
```

The first rerun had not yet written its score file at the 10:25 check. Its log was still loading PopQA graph/vector indexes:

```text
logs/rerun_popqa_tog_gemini.log
```

New helper scripts added and synced to the server:

```text
collect_experiment_results.py
plot_popqa_model_comparison.py
plot_popqa_by_model_baseline_vs_newg.py
plot_musique_model_comparison.py
plot_musique_by_model_baseline_vs_newg.py
```

`collect_experiment_results.py` reads real score files from `output/datasets/.../Results`, handles `.json` and `.jsonl` score files, reports row counts and source paths, and can optionally use PopQA working-note fallback values for missing cells:

```bash
python3 collect_experiment_results.py --dataset PopQA --format markdown
python3 collect_experiment_results.py --dataset MuSiQue --format markdown
python3 collect_experiment_results.py --dataset all --format tsv --out results_summary.tsv
```

For plotting before all local files are present, the two PopQA plotting scripts now prefer real score files but fall back to existing PopQA working-note values. If a result file exists with a row count other than 200, the plotting scripts keep the fallback value instead of silently plotting a contaminated/incomplete file.

MuSiQue plotting was added from the user-provided Table 2 values. DeepSeek/GPT BM25 and VDB remain missing in the static table and are intentionally left blank in plots until reruns finish.

Generated and synced MuSiQue figures:

```text
figures/musique_cross_model_by_method.png
figures/musique_cross_model_f1.png
figures/musique_deepseek_baseline_vs_newg.png
figures/musique_gpt4omini_baseline_vs_newg.png
figures/musique_gemini_baseline_vs_newg.png
figures/musique_f1_baseline_vs_newg_summary.png
```

Remote syntax check passed:

```bash
python3 -m py_compile collect_experiment_results.py plot_popqa_model_comparison.py plot_popqa_by_model_baseline_vs_newg.py
```

Next steps:

1. Keep monitoring `run_popqa_reruns.sh`.
2. After it finishes, run:

```bash
python3 collect_experiment_results.py --dataset all --truncate-overlong --format tsv --out results_summary.tsv
python3 plot_popqa_model_comparison.py
python3 plot_popqa_by_model_baseline_vs_newg.py
```

3. Sync or inspect `results_summary.tsv` and regenerated `figures/*.png`.
4. Reconsider a clean 200-row DeepSeek RAPTOR rerun before final paper tables.

User decision after this update:

- Do not rerun PopQA / DeepSeek / RAPTOR.
- Use the first 200 rows from the overlong RAPTOR score file for final reporting.
- Use `collect_experiment_results.py --truncate-overlong` for final summaries so overlong files are explicitly truncated and annotated instead of averaged in full.
- Run ablation experiments with `gemini-2.5-flash-lite`.

Before ablation, after `run_popqa_reruns.sh` finishes, switch `Option/Config2.yaml` back to Gemini:

```bash
python3 - <<'PY'
from pathlib import Path
import re

p = Path("Option/Config2.yaml")
text = p.read_text()
new_text, n = re.subn(
    r'(?m)^(\s*model:\s*)"[^"]+"',
    r'\1"gemini-2.5-flash-lite"',
    text,
    count=1,
)
if n != 1:
    raise SystemExit("failed to update llm.model")
p.write_text(new_text)
PY
grep -n 'model:' Option/Config2.yaml | head -2
```

Then run:

```bash
bash run_ablation_popqa.sh
bash run_ablation_musique.sh
```

## Latest Server Status Check

Checked read-only at `2026-05-13 23:07:49 +08:00`.

Main experiment was still running:

```text
PID 3925    bash run_20_main.sh
PID 740460  python newg_main.py -opt Option/Method/NewG.yaml -graph_method tog -text_method vdb -dataset_name datasets/musique --eval_limit 200
```

Do not start `run_popqa_reruns.sh` yet unless the user explicitly accepts parallel API/GPU contention. The script itself will refuse to start while this process is active unless `ALLOW_PARALLEL=1` is set.

Current target rerun result counts:

```text
PopQA    gemini        ToG       200
PopQA    deepseek      BM25      MISSING
PopQA    deepseek      VDB       MISSING
PopQA    deepseek      HippoRAG  20
PopQA    gpt-4o-mini   BM25      MISSING
PopQA    gpt-4o-mini   VDB       MISSING
MuSiQue  deepseek      BM25      MISSING
MuSiQue  deepseek      VDB       MISSING
MuSiQue  gpt-4o-mini   BM25      MISSING
MuSiQue  gpt-4o-mini   VDB       MISSING
```

Current active model in local `Option/Config2.yaml` at the time of this log:

```yaml
llm.model: gemini-2.5-flash-lite
base_url: https://api.chatanywhere.tech/v1
```

Do not print or copy the API key.

## What We Did In This Session

1. Reconstructed where the previous task stopped.
   - The project directory itself is not a Git repository.
   - Recent files showed that the previous work had focused on experiment scripts, NewG code, prompts, and result analysis.
   - There were `.codex_remote_compare_...` snapshot directories from previous sync/compare work.

2. Confirmed main experiment scripts and their purpose.
   - `run_20_main.sh` now actually contains 24 main commands despite the old filename:
     - PopQA: BM25, VDB, HippoRAG, RAPTOR, ToG, AgentG, NewG x6.
     - MuSiQue: BM25, VDB, HippoRAG, RAPTOR, ToG, AgentG, NewG x6.
   - `run_all_20_main.sh` is a 20-run serial script that excludes BM25/VDB.
   - `run_ablation_popqa.sh` and `run_ablation_musique.sh` now contain 6-step NewG ablation ladders after removing the full-method step.

3. Verified SSH access to the server.
   - Initial normal SSH attempt failed because the local environment had no configured key.
   - Installed Python `paramiko` locally to use password-based SSH.
   - Login succeeded and remote project directory was verified.

4. Checked remote main experiment status.
   - At the time of the check, `run_20_main.sh` was still running.
   - 21 out of 24 main experiments had completed.
   - Running at that time:
     - `MuSiQue / NewG_tog_vdb_gemini-2.5-flash-lite`
     - raw result had 30/200 rows when checked.
   - Pending at that time:
     - `MuSiQue / NewG_raptor_bm25_gemini-2.5-flash-lite`
     - `MuSiQue / NewG_raptor_vdb_gemini-2.5-flash-lite`
   - This status may be stale. The first thing to do in the next session is to re-check it.

5. Parsed remote PopQA and MuSiQue results.
   - Wrote no files during read-only checks.
   - Found that some baseline result files are JSONL even when named `.json`.
   - Confirmed that `results.score.json` can contain one JSON object per line.

6. Analyzed suspicious PopQA results.
   - The user provided PopQA tables for `deepseek-v3.2`, `gpt-4o-mini`, and `gemini-2.5-flash-lite`.
   - Important finding:
     - `Popqa/HippoRAG_deepseek-v3.2/Results/results.score.json` on the server currently has only 20 rows, not 200.
     - `Popqa/RAPTOR_deepseek-v3.2/Results/results.score.json` has 1399 rows, not a clean 200.
     - The user's RAPTOR deepseek row matches the first 200 rows of that 1399-row file.
   - Conclusion:
     - DeepSeek baseline rows are not all clean/same-round/same-code 200-row results.
     - DeepSeek HippoRAG must be rerun.
     - DeepSeek RAPTOR should ideally be rerun too, but the user decided to first rerun only DeepSeek BM25/VDB/HippoRAG.

7. Investigated why Gemini ToG looked abnormal.
   - Gemini ToG had `Accuracy=40.50%`, `EM=10.00%`, `F1=14.75%`.
   - Output format was abnormal: many answers were long, with average output length around 89 words in the checked result file.
   - This explains very low EM/F1 despite some correct evidence.
   - User decided Gemini ToG should be rerun.

8. Created rerun documentation and script.
   - Created `popqa_rerun_commands.md`.
   - Created executable server script `run_popqa_reruns.sh`.
   - Synced both to the server.
   - Server-side `bash -n run_popqa_reruns.sh` passed.
   - The script was not started by Codex.

9. Created visualization scripts and figures.
   - Created `plot_popqa_model_comparison.py`.
   - Created `plot_popqa_by_model_baseline_vs_newg.py`.
   - Generated and synced cross-model comparison plots.
   - Generated and synced per-model baseline-vs-NewG plots.

## Current Result Summary From Remote Checks

### PopQA Gemini Main Results

These were clean 200-row results at the time of checking:

```text
BM25                 Acc 62.00 | EM 58.00 | Precision 61.08 | Recall 21.73 | F1 26.98
VDB                  Acc 56.00 | EM 51.50 | Precision 54.00 | Recall 19.27 | F1 23.57
HippoRAG             Acc 61.50 | EM 58.50 | Precision 60.92 | Recall 22.51 | F1 27.85
RAPTOR               Acc 38.50 | EM 32.50 | Precision 36.04 | Recall 13.83 | F1 16.55
ToG                  Acc 40.50 | EM 10.00 | Precision 23.06 | Recall 15.22 | F1 14.75
AgentG               Acc 48.00 | EM 27.50 | Precision 37.92 | Recall 16.80 | F1 19.53
NewG_hipporag_bm25   Acc 59.50 | EM 58.50 | Precision 59.83 | Recall 21.69 | F1 27.50
NewG_hipporag_vdb    Acc 57.50 | EM 56.50 | Precision 57.83 | Recall 20.93 | F1 26.57
NewG_tog_bm25        Acc 57.00 | EM 56.00 | Precision 57.25 | Recall 20.32 | F1 25.97
NewG_tog_vdb         Acc 54.00 | EM 53.00 | Precision 54.25 | Recall 19.46 | F1 24.71
NewG_raptor_bm25     Acc 58.00 | EM 57.00 | Precision 58.25 | Recall 19.37 | F1 25.32
NewG_raptor_vdb      Acc 53.00 | EM 52.00 | Precision 53.25 | Recall 16.95 | F1 22.41
```

Gemini ToG is suspicious and is included in the rerun script.

### PopQA GPT-4o-mini Results From User Table

```text
HippoRAG             Acc 65.50 | EM 36.00 | Precision 48.36 | Recall 25.46 | F1 28.24
RAPTOR               Acc 40.50 | EM 39.00 | Precision 41.42 | Recall 13.28 | F1 17.33
ToG                  Acc 50.50 | EM 27.00 | Precision 38.30 | Recall 19.84 | F1 22.32
AgentG               Acc 51.00 | EM 38.50 | Precision 45.63 | Recall 19.68 | F1 23.37
NewG_hipporag_bm25   Acc 64.00 | EM 64.00 | Precision 64.83 | Recall 22.73 | F1 29.06
NewG_hipporag_vdb    Acc 61.50 | EM 61.50 | Precision 62.33 | Recall 21.75 | F1 27.95
NewG_tog_bm25        Acc 49.50 | EM 49.50 | Precision 50.33 | Recall 16.34 | F1 21.43
NewG_tog_vdb         Acc 51.00 | EM 50.50 | Precision 51.83 | Recall 16.92 | F1 22.16
NewG_raptor_bm25     Acc 58.00 | EM 58.00 | Precision 58.50 | Recall 19.64 | F1 25.50
NewG_raptor_vdb      Acc 55.00 | EM 55.00 | Precision 55.50 | Recall 17.83 | F1 23.50
```

GPT BM25/VDB are missing and are included in the rerun script.

### PopQA DeepSeek Results From User Table

The user table contains:

```text
HippoRAG             Acc 47.75 | EM 42.82 | Precision 45.98 | Recall 15.47 | F1 19.69
RAPTOR               Acc 41.00 | EM 40.50 | Precision 41.67 | Recall 13.76 | F1 17.78
ToG                  Acc 63.00 | EM 38.00 | Precision 49.59 | Recall 23.19 | F1 26.16
AgentG               Acc 53.50 | EM 31.50 | Precision 42.43 | Recall 19.60 | F1 22.90
NewG_hipporag_bm25   Acc 65.50 | EM 65.00 | Precision 66.83 | Recall 23.26 | F1 29.78
NewG_hipporag_vdb    Acc 64.00 | EM 63.50 | Precision 64.83 | Recall 22.00 | F1 28.39
NewG_tog_bm25        Acc 58.00 | EM 58.00 | Precision 58.83 | Recall 20.13 | F1 25.96
NewG_tog_vdb         Acc 58.50 | EM 58.50 | Precision 59.33 | Recall 19.98 | F1 25.80
NewG_raptor_bm25     Acc 64.50 | EM 64.00 | Precision 64.92 | Recall 20.78 | F1 27.41
NewG_raptor_vdb      Acc 59.50 | EM 59.50 | Precision 60.75 | Recall 18.95 | F1 25.09
```

But important caveats:

- Server currently has DeepSeek HippoRAG score file with only 20 rows.
- Server currently has DeepSeek RAPTOR score file with 1399 rows.
- The RAPTOR table row matches the first 200 rows of that 1399-row file.
- Therefore DeepSeek baseline conclusions must be treated cautiously until reruns finish.

DeepSeek BM25/VDB/HippoRAG are included in the rerun script.

### MuSiQue Gemini Main Results Checked

At the time of checking:

```text
BM25                 Acc 09.00 | EM 09.00 | Precision 13.62 | Recall 12.46 | F1 12.78
VDB                  Acc 04.50 | EM 04.50 | Precision 07.42 | Recall 06.83 | F1 06.98
HippoRAG             Acc 16.00 | EM 14.00 | Precision 19.32 | Recall 18.13 | F1 18.13
RAPTOR               Acc 14.00 | EM 14.00 | Precision 23.08 | Recall 20.14 | F1 20.99
ToG                  Acc 05.50 | EM 04.50 | Precision 08.72 | Recall 08.36 | F1 07.71
AgentG               Acc 06.00 | EM 06.00 | Precision 11.67 | Recall 09.23 | F1 09.92
NewG_hipporag_bm25   Acc 19.50 | EM 18.50 | Precision 27.79 | Recall 25.95 | F1 26.37
NewG_hipporag_vdb    Acc 18.00 | EM 16.00 | Precision 24.91 | Recall 23.10 | F1 23.40
NewG_tog_bm25        Acc 24.00 | EM 21.50 | Precision 32.09 | Recall 30.07 | F1 30.26
NewG_tog_vdb         was still running at last check
NewG_raptor_bm25     pending at last check
NewG_raptor_vdb      pending at last check
```

Re-check current server status before making any conclusion.

## Scripts Created

### Ablation Scripts

Updated on 2026-05-13:

```text
run_ablation_popqa.sh
run_ablation_musique.sh
```

Both scripts now skip the Full NewG / `NewG_abl_normalizer.yaml` step. The Full NewG result should be reused from the main experiment directory:

```text
output/datasets/Popqa/NewG_hipporag_bm25_<model>/
output/datasets/musique/NewG_hipporag_bm25_<model>/
```

Each ablation script now runs 6 experiments instead of 7:

```text
abl_simple
abl_routing
abl_regen
abl_critic
abl_no_commendor
abl_single_agent
```

The modified scripts were synced to the server and passed:

```bash
bash -n run_ablation_popqa.sh
bash -n run_ablation_musique.sh
```

### Rerun Commands Document

Local:

```text
popqa_rerun_commands.md
```

Remote:

```text
/root/autodl-tmp/GraphRAG-master/GraphRAG-master/popqa_rerun_commands.md
```

Purpose:

- Documents how to run the current 10 reruns across PopQA and MuSiQue.
- Includes backup, model switch helper, run commands, and result check command.

### Rerun Script

Local:

```text
run_popqa_reruns.sh
```

Remote:

```text
/root/autodl-tmp/GraphRAG-master/GraphRAG-master/run_popqa_reruns.sh
```

Server syntax check passed:

```bash
bash -n run_popqa_reruns.sh
```

The script originally ran these six PopQA experiments:

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

Updated later on 2026-05-13: the same script now also runs missing MuSiQue BM25/VDB baselines for the first two models:

```text
MuSiQue / deepseek-v3.2:
- BM25
- VDB

MuSiQue / gpt-4o-mini:
- BM25
- VDB
```

So `run_popqa_reruns.sh` now runs 10 total reruns:

```text
PopQA:
- gemini ToG
- deepseek BM25 / VDB / HippoRAG
- gpt-4o-mini BM25 / VDB

MuSiQue:
- deepseek BM25 / VDB
- gpt-4o-mini BM25 / VDB
```

The script name is now historically inaccurate, but it is still the active rerun script.

The script:

- backs up existing affected output directories into `.popqa_rerun_backup_<timestamp>.tar.gz`;
- moves those existing directories into `.popqa_rerun_archived_<timestamp>/` before rerunning, so new rerun outputs are clean directories;
- changes only the first `llm.model` entry in `Option/Config2.yaml`;
- writes logs into `logs/rerun_*.log`;
- prints final metric summaries at the end;
- refuses to start if another `run_20_main`, `main.py`, or `newg_main.py` process is active, unless `ALLOW_PARALLEL=1` is set.

Run command:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
bash run_popqa_reruns.sh
```

If intentionally running in parallel:

```bash
ALLOW_PARALLEL=1 bash run_popqa_reruns.sh
```

Do not run in parallel unless the user explicitly accepts resource/API contention.

### Cross-Model Plot Script

Local and remote:

```text
plot_popqa_model_comparison.py
```

Outputs:

```text
figures/popqa_cross_model_by_method.png
figures/popqa_cross_model_f1.png
```

Purpose:

- Compares the same method across `deepseek-v3.2`, `gpt-4o-mini`, and `gemini-2.5-flash-lite`.
- Uses Accuracy, EM, and F1.
- BM25/VDB are omitted for now because deepseek/gpt BM25/VDB are missing until reruns finish.

### Per-Model Baseline-vs-NewG Plot Script

Local and remote:

```text
plot_popqa_by_model_baseline_vs_newg.py
```

Outputs:

```text
figures/popqa_deepseek_baseline_vs_newg.png
figures/popqa_gpt4omini_baseline_vs_newg.png
figures/popqa_gemini_baseline_vs_newg.png
figures/popqa_f1_baseline_vs_newg_summary.png
```

Purpose:

- For each model, compares baseline methods vs NewG variants.
- Baseline side is shaded gray.
- NewG side is shaded green.
- Shows Accuracy, EM, and F1 in each model-specific chart.
- Summary chart compares baseline average F1, NewG average F1, best baseline F1, and best NewG F1 per model.

## Figure Files Synced To Server

Remote files:

```text
/root/autodl-tmp/GraphRAG-master/GraphRAG-master/figures/popqa_cross_model_by_method.png
/root/autodl-tmp/GraphRAG-master/GraphRAG-master/figures/popqa_cross_model_f1.png
/root/autodl-tmp/GraphRAG-master/GraphRAG-master/figures/popqa_deepseek_baseline_vs_newg.png
/root/autodl-tmp/GraphRAG-master/GraphRAG-master/figures/popqa_gpt4omini_baseline_vs_newg.png
/root/autodl-tmp/GraphRAG-master/GraphRAG-master/figures/popqa_gemini_baseline_vs_newg.png
/root/autodl-tmp/GraphRAG-master/GraphRAG-master/figures/popqa_f1_baseline_vs_newg_summary.png
```

Local files are under:

```text
D:\Tools\Downloads\GraphRAG-master1\root\autodl-tmp\GraphRAG-master\GraphRAG-master\figures
```

## Interpretations From The Plots

Key interpretation from cross-model comparison:

1. NewG variants are more stable across models than several old baselines.
   - `NewG hippo+bm25` F1 is close across all three models:
     - DeepSeek: 29.78
     - GPT: 29.06
     - Gemini: 27.50
   - `NewG hippo+vdb` is also stable:
     - DeepSeek: 28.39
     - GPT: 27.95
     - Gemini: 26.57

2. ToG is unstable across models.
   - DeepSeek ToG looks strong in the current table.
   - Gemini ToG is abnormal because output format is too long and EM/F1 collapse.
   - Gemini ToG must be rerun before final claims.

3. NewG helps correct weak ToG/RAPTOR behavior.
   - Gemini ToG F1 is 14.75, while Gemini NewG tog+bm25 F1 is 25.97.
   - Gemini RAPTOR F1 is 16.55, while Gemini NewG raptor+bm25 F1 is 25.32.

4. BM25/VDB baselines are important.
   - Gemini BM25 is strong on PopQA:
     - Accuracy 62.00
     - F1 26.98
   - Because PopQA has many single-hop entity-attribute questions, BM25 can be a very strong text baseline.
   - Need deepseek/gpt BM25/VDB reruns before finalizing claims.

Safe paper-style claim for now:

```text
On PopQA, NewG shows stronger cross-model robustness than traditional graph baselines. Its HippoRAG-based variants achieve consistently high F1 across DeepSeek, GPT-4o-mini, and Gemini, while traditional ToG and RAPTOR are more sensitive to the underlying LLM and output format. The improvement is especially clear for ToG/RAPTOR backbones, where NewG substantially reduces generation-format failures and improves exact-match-oriented metrics.
```

Avoid claiming:

```text
NewG fully beats every baseline on PopQA.
```

Reason:

- Gemini BM25 is competitive.
- DeepSeek HippoRAG baseline is not yet clean.
- Gemini ToG needs rerun.

## Important Evaluation Detail

For short-form QA, `Core/Utils/Evaluation.py` uses:

- `accuracy`: true if any answer alias is contained in the normalized prediction;
- `EM`: true only if normalized prediction exactly matches any answer alias;
- `F1`: token overlap between prediction and all answer aliases joined.

This explains why outputs like `Conservative politician` may count for accuracy when the gold alias contains `politician`, but not necessarily for EM.

This also explains why long outputs hurt EM/F1 even when they contain the right keyword.

## Pending Tasks

### Highest Priority

1. Re-check server main experiment status.

Use:

```bash
ssh -p 45423 root@region-41.seetacloud.com
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
ps -ef | grep -E 'run_20_main|main.py|newg_main.py' | grep -v grep || true
```

Also check current MuSiQue NewG outputs:

```bash
find output/datasets/musique -path '*/Results/results.score.jsonl' -o -path '*/Results/results.jsonl' | sort | grep 'gemini-2.5-flash-lite'
```

2. If the main 24-run script has finished, run the PopQA rerun script:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
bash run_popqa_reruns.sh
```

3. Watch logs:

```bash
tail -f logs/rerun_popqa_tog_gemini.log
```

Then later:

```bash
tail -f logs/rerun_popqa_bm25_deepseek.log
tail -f logs/rerun_popqa_vdb_deepseek.log
tail -f logs/rerun_popqa_hipporag_deepseek.log
tail -f logs/rerun_popqa_bm25_gpt4omini.log
tail -f logs/rerun_popqa_vdb_gpt4omini.log
```

4. After reruns finish, update PopQA tables and regenerate figures.

Need to add BM25/VDB into cross-model and per-model charts once deepseek/gpt BM25/VDB are available.

5. Also update MuSiQue tables after the newly added MuSiQue BM25/VDB reruns finish.

The user's current MuSiQue table has missing BM25/VDB rows for:

```text
deepseek-v3.2:
- BM25
- VDB

gpt-4o-mini:
- BM25
- VDB
```

Gemini MuSiQue BM25/VDB already exist in the table:

```text
BM25: Accuracy 9.00, EM 9.00, Precision 13.63, Recall 12.46, F1 12.78
VDB:  Accuracy 4.50, EM 4.50, Precision 7.42, Recall 6.83, F1 6.98
```

6. Re-evaluate whether DeepSeek RAPTOR should also be rerun.

Even though the user did not include it in the immediate rerun list, the current server file has 1399 rows, which is not clean. For a final paper table, rerun DeepSeek RAPTOR 200 rows too.

### Medium Priority

7. Update figure scripts to read result files automatically rather than using hardcoded table values.

Currently the plotting scripts use hardcoded values from the user-provided table and checked Gemini results.

8. Add an updated result table after reruns.

Recommended format:

```text
model | method | architecture | n | accuracy | em | precision | recall | f1 | source path | notes
```

Include notes for:

- rerun results;
- old contaminated results;
- suspicious output-format failures.

9. Check whether `run_20_main.sh` should be renamed or copied to a clearer filename.

It now runs 24 commands, not 20. Do not rename without user approval because the user may already have command habits around the old filename.

### Lower Priority

10. Create MuSiQue plots analogous to PopQA once all MuSiQue Gemini main runs are done.

11. Consider adding a clean result collector script:

```text
collect_results.py
```

It should parse JSONL score files, handle `.json` files that are actually JSONL, and output CSV/Markdown tables.

## Useful Commands For Next Session

Check remote running jobs:

```bash
ssh -p 45423 root@region-41.seetacloud.com
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
ps -ef | grep -E 'run_20_main|main.py|newg_main.py' | grep -v grep || true
```

Check all 10 rerun result files:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
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

Regenerate plots locally:

```powershell
cd "D:\Tools\Downloads\GraphRAG-master1\root\autodl-tmp\GraphRAG-master\GraphRAG-master"
python plot_popqa_model_comparison.py
python plot_popqa_by_model_baseline_vs_newg.py
```

Regenerate plots on server:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
python3 plot_popqa_model_comparison.py
python3 plot_popqa_by_model_baseline_vs_newg.py
```

## What To Tell A New Codex Tomorrow

Copy this prompt into the new Codex session:

```text
我们要继续 GraphRAG/NewG 实验任务。请先读取项目根目录的 handoff_log_2026-05-13.md，不要从头猜。

本地项目路径：
D:\Tools\Downloads\GraphRAG-master1\root\autodl-tmp\GraphRAG-master\GraphRAG-master

远程服务器：
ssh -p 45423 root@region-41.seetacloud.com

远程项目路径：
/root/autodl-tmp/GraphRAG-master/GraphRAG-master

注意：服务器密码已经按用户要求写在 handoff_log_2026-05-13.md 顶部；不要把密码或 API key 再复制到其它文件或公开输出。

请先做三件事：
1. 读取 handoff_log_2026-05-13.md。
2. 只读检查服务器当前是否还有 run_20_main/main.py/newg_main.py 在跑，以及 MuSiQue 主实验是否完成。
3. 如果主实验已结束，准备运行或指导我运行 run_popqa_reruns.sh；如果已经运行过，则汇总 10 个 rerun 结果，并更新 PopQA/MuSiQue 表格和后续图。

当前已创建的关键文件：
- popqa_rerun_commands.md
- run_popqa_reruns.sh
- plot_popqa_model_comparison.py
- plot_popqa_by_model_baseline_vs_newg.py
- figures/popqa_cross_model_by_method.png
- figures/popqa_cross_model_f1.png
- figures/popqa_deepseek_baseline_vs_newg.png
- figures/popqa_gpt4omini_baseline_vs_newg.png
- figures/popqa_gemini_baseline_vs_newg.png
- figures/popqa_f1_baseline_vs_newg_summary.png

重要背景：
- DeepSeek HippoRAG 当前服务器结果只有 20 条，不是干净 200 条。
- DeepSeek RAPTOR 当前服务器文件有 1399 条，用户表里用的是前 200 条；最终论文最好也重跑。
- Gemini ToG 输出格式异常，EM/F1 崩，需要重跑。
- run_popqa_reruns.sh 当前一共跑 10 项：PopQA 的 gemini ToG、deepseek BM25/VDB/HippoRAG、gpt BM25/VDB；以及 MuSiQue 的 deepseek BM25/VDB、gpt BM25/VDB。
```

## Final Notes

- The new account/session should not assume previous conversation memory exists.
- The handoff file is the source of truth for next steps.
- First action tomorrow should be read-only status verification on the server.
- Do not start reruns while the main 24-run script is still running unless the user explicitly approves parallel execution.

## 2026-05-15 20:20 Server Progress Check

User asked to continue and check which task is currently running.

Latest read-only server check:

```text
DATE 2026-05-15 20:20:43 CST
active:
402569 1404 S+ 08:58:36 bash run_ablation_musique.sh
538742 402569 Sl+ 00:05:36 /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/NewG_abl_no_commendor.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200
```

Current old MuSiQue incremental ablation status:

```text
Step 1 Simple              complete, 200 rows
Step 2 +Router             complete, 200 rows
Step 3 +Re-Generator       complete, 200 rows
Step 4 +Critic & Commendor complete, 200 rows
Step 5 w/o Commendor       running, log at Q3/200 around 20:19:45
Step 6 Single judge/voter  not started
```

Actual completed result paths:

```text
output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_simple/Results/results.score.jsonl
output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_routing/Results/results.score.jsonl
output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_regen/Results/results.score.jsonl
output/datasets/musique/NewG_hipporag_bm25_gemini-2.5-flash-lite_abl_critic/Results/results.score.jsonl
```

Notes:

- Watcher remains stopped. No automatic ToG run should start after this ablation.
- New leave-one-out ablation scripts exist but have not been started.
- Current bottleneck is LLM/API calls, not GPU.

## 2026-05-15 21:54 Server Progress Check

User wants the MuSiQue ablation result table formatted like the screenshot after completion, with percentages rounded to two decimals.

Latest read-only server check:

```text
DATE 2026-05-15 21:54:20 CST
active:
402569 1404 S+ 10:33:44 bash run_ablation_musique.sh
538742 402569 Sl+ 01:40:44 /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/NewG_abl_no_commendor.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200
```

Progress:

```text
Step 1 Simple              complete, 200 rows
Step 2 +Router             complete, 200 rows
Step 3 +Re-Generator       complete, 200 rows
Step 4 +Critic & Commendor complete, 200 rows
Step 5 w/o Commendor       running, latest log marker [Q71/200]
Step 6 Single judge/voter  not started
```

Current metrics for completed rows:

```text
method                  n    Accuracy  EM      Precision  Recall  F1
Simple                  200  16.50%    14.00%  20.58%     20.38%  20.07%
+Router                 200  17.00%    14.50%  21.52%     21.68%  21.22%
+Re-Generator           200  18.00%    9.00%   17.28%     20.77%  17.78%
+ Critic & Commendor    200  23.00%    12.00%  22.62%     25.27%  22.78%
w/o Commendor           pending
Single judge/voter      pending
Full NewG               200  19.50%    18.50%  27.79%     25.95%  26.37%
```

When Step 5 and Step 6 finish, fill the final table in this order:

```text
Simple
+Router
+Re-Generator
+ Critic & Commendor
w/o Commendor
Single judge/voter
Full NewG
```

## 2026-05-15 22:39 Conversation Preservation Update

User explicitly asked again to preserve the conversation context so that saying "继续上次任务" in a later session resumes without losing state.

Latest known server status from read-only check:

```text
DATE 2026-05-15 22:39:17 CST
active:
402569 1404 S+ 11:18:41 bash run_ablation_musique.sh
538742 402569 Sl+ 02:25:41 /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/NewG_abl_no_commendor.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/musique --eval_limit 200
```

Current running log:

```text
logs/abl_musique_step5_no_commendor.log
last marker: [Q99/200]
tail context: Q99 asks "Who was in charge of the country beaten at the Battle of Kosovo?", Round 1/3, selection=graph.
```

Current MuSiQue old incremental ablation state:

```text
Step 1 Simple              complete, [Q200/200], result rows 200
Step 2 +Router             complete, [Q200/200], result rows 200
Step 3 +Re-Generator       complete, [Q200/200], result rows 200
Step 4 +Critic & Commendor complete, [Q200/200], result rows 200
Step 5 w/o Commendor       running, [Q99/200] as of 22:39
Step 6 Single judge/voter  not started, no log file yet
Full NewG                  existing result rows 200
```

User wants final MuSiQue table formatted like the screenshot after Step 5 and Step 6 finish:

```text
方法 | 规模 | Accuracy | EM | Precision | Recall | F1
Simple
+Router
+Re-Generator
+ Critic & Commendor
w/o Commendor
Single judge/voter
Full NewG
```

Percentages must be rounded to two decimals, e.g. `16.50%`.

Already computed completed metrics:

```text
Simple                  200  16.50%  14.00%  20.58%  20.38%  20.07%
+Router                 200  17.00%  14.50%  21.52%  21.68%  21.22%
+Re-Generator           200  18.00%  9.00%   17.28%  20.77%  17.78%
+ Critic & Commendor    200  23.00%  12.00%  22.62%  25.27%  22.78%
Full NewG               200  19.50%  18.50%  27.79%  25.95%  26.37%
```

Operational notes:

- Watcher was stopped earlier at user request. Do not assume any automatic ToG job will start.
- New leave-one-out ablation scripts were created and synced, but have not been started.
- Before taking action next time, first do a read-only server status check.
- Do not print or copy the stored server password/API keys in user-facing output.

## 2026-05-16 13:57 Resume Update

User asked to continue the previous task. A read-only remote status check was
performed first.

Remote status:

```text
DATE 2026-05-16 13:57:23 CST
active:
626333   1404 S+         33:10 bash run_tog_gemini_v3.sh
626347 626333 Rl+        33:10 /root/miniconda3/envs/graphrag/bin/python main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

MuSiQue old incremental ablation is complete. All six ablation steps and the
Full NewG reference have 200 scored rows:

```text
Method                   n    Accuracy  EM      Precision  Recall  F1
Simple                   200  16.50%    14.00%  20.58%     20.38%  20.07%
+Router                  200  17.00%    14.50%  21.52%     21.68%  21.22%
+Re-Generator            200  18.00%    9.00%   17.28%     20.77%  17.78%
+ Critic & Commendor     200  23.00%    12.00%  22.62%     25.27%  22.78%
w/o Commendor            200  20.00%    19.50%  29.54%     26.87%  27.62%
Single judge/voter       200  18.50%    18.00%  26.58%     25.20%  25.47%
Full NewG                200  19.50%    18.50%  27.79%     25.95%  26.37%
```

Latest MuSiQue log markers:

```text
logs/abl_musique_step6_single_agent.log  [Q200/200]
logs/abl_musique_step5_no_commendor.log  [Q200/200]
logs/abl_musique_step4_critic.log        [Q200/200]
```

Operational notes:

- The previous note saying no automatic ToG should start is now obsolete:
  `run_tog_gemini_v3.sh` is active and running PopQA ToG.
- Do not start leave-one-out ablation while ToG is active unless the user
  explicitly approves parallel API usage.
- Next read-only check should inspect `run_tog_gemini_v3.sh`, `main.py`, and
  `output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/`.

Follow-up read-only ToG progress check:

```text
DATE 2026-05-16 13:59:49 CST
active:
626333   1404 S+         35:36 bash run_tog_gemini_v3.sh
626347 626333 Rl+        35:36 /root/miniconda3/envs/graphrag/bin/python main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

At this point `output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/` had no
`results.json[l]` or `results.score.json[l]` yet. The log was still in startup
work after loading the PopQA graph and entity index, most recently loading:

```text
./output/datasets/Popqa/er_graph/relations_vdb
```

## 2026-05-16 14:19 ToG Progress Check

User asked for current progress. A read-only remote status check found PopQA
Gemini ToG still active:

```text
DATE 2026-05-16 14:19:25 CST
active:
626333   1404 S+         55:12 bash run_tog_gemini_v3.sh
626347 626333 Sl+        55:12 /root/miniconda3/envs/graphrag/bin/python main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

No `results.json[l]` or `results.score.json[l]` exists yet under:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/
```

The run has entered the GraphRAG/ToG QA loop. Latest visible progress in
`logs/rerun_popqa_tog_gemini_v3.log` was around `6/200` questions, with recent
log lines at `2026-05-16 14:17:50`. No MuSiQue ablation, leave-one-out ablation,
or `run_popqa_reruns.sh` process was active.

## 2026-05-16 14:54 ToG Progress Check

User asked for current progress again. A read-only remote status check found
PopQA Gemini ToG still active:

```text
DATE 2026-05-16 14:54:33 CST
active:
626333   1404 S+      01:30:20 bash run_tog_gemini_v3.sh
626347 626333 Sl+     01:30:20 /root/miniconda3/envs/graphrag/bin/python main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

Latest visible tqdm progress in `logs/rerun_popqa_tog_gemini_v3.log`:

```text
GraphRAG:  32%|...| 63/200 [45:09<1:23:17, 36.48s/q, What is Walter de la Pole's occupation?]
LAST_TS 2026-05-16 14:54:33.036 ... ToG still not find the answer at depth 1.
```

No `results.json[l]` or `results.score.json[l]` exists yet under:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/
```

This is expected for this runner because final score files may not appear until
the run finishes. No MuSiQue ablation, leave-one-out ablation, or
`run_popqa_reruns.sh` process was active.

## 2026-05-16 15:33 ToG Progress Check

User asked whether the PopQA Gemini ToG rerun had finished. A read-only remote
status check found it still active:

```text
DATE 2026-05-16 15:33:10 CST
active:
626333   1404 S+      02:08:57 bash run_tog_gemini_v3.sh
626347 626333 Sl+     02:08:57 /root/miniconda3/envs/graphrag/bin/python main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

Latest visible tqdm progress in `logs/rerun_popqa_tog_gemini_v3.log`:

```text
GraphRAG:  76%|...| 151/200 [1:23:49<17:19, 21.22s/q, In what city was Clarence Beck born?]
```

No `results.json[l]` or `results.score.json[l]` exists yet under:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/
```

This runner appears to write final result/score files only after completion.

## 2026-05-16 16:18 PopQA Gemini ToG Complete

User asked for current status again. A read-only remote check found no active
`run_tog_gemini_v3.sh` or ToG `main.py` process.

```text
DATE 2026-05-16 16:18:13 CST
active:
```

The PopQA Gemini ToG rerun completed at `2026-05-16 15:44:21` with 200 rows:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.json        200 rows
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.score.json  200 rows
```

Final metrics:

```text
Accuracy   31.50%
EM         26.50%
Precision  27.90%
Recall     10.01%
F1         12.09%
```

Log tail confirms completion:

```text
GraphRAG: 100%|...| 200/200 [1:35:10<00:00, 28.55s/q, In what city was John Robinson born?]
Loaded 200 records from ./output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.json
Evaluating short-form mode.
accuracy: 0.3150 (ratio) | 31.5000%
Precision: 0.2790 (ratio) | 27.8974%
Recall: 0.1001 (ratio) | 10.0074%
F1: 0.1209 (ratio) | 12.0912%
EM: 0.2650 (ratio) | 26.5000%
```

Next logical step, if continuing the planned experiment workflow, is to decide
whether to start the new leave-one-out ablation now that ToG is idle.

## 2026-05-16 16:32 Final Table And Figure Refresh

User provided final 200-row PopQA and MuSiQue result tables for all three
models and asked to update all existing charts.

Local updates completed:

```text
collect_experiment_results.py
plot_popqa_model_comparison.py
plot_popqa_by_model_baseline_vs_newg.py
plot_musique_model_comparison.py
plot_musique_by_model_baseline_vs_newg.py
results_summary.tsv
results_popqa.md
results_musique.md
```

`collect_experiment_results.py` now supports `--prefer-static` and the plotting
scripts use the final table values as the authoritative source, even when older
local result files exist.

Regenerated figures:

```text
figures/popqa_cross_model_by_method.png
figures/popqa_cross_model_f1.png
figures/popqa_deepseek_baseline_vs_newg.png
figures/popqa_gpt4omini_baseline_vs_newg.png
figures/popqa_gemini_baseline_vs_newg.png
figures/popqa_f1_baseline_vs_newg_summary.png
figures/musique_cross_model_by_method.png
figures/musique_cross_model_f1.png
figures/musique_deepseek_baseline_vs_newg.png
figures/musique_gpt4omini_baseline_vs_newg.png
figures/musique_gemini_baseline_vs_newg.png
figures/musique_f1_baseline_vs_newg_summary.png
figures/popqa_ablation_metric_profile.png
figures/popqa_ablation_delta_vs_simple.png
figures/popqa_ablation_accuracy_em_tradeoff.png
```

Validation:

```text
results_summary.tsv row count: 72
PopQA/gemini-2.5-flash-lite/ToG uses final table row:
31.50 accuracy, 26.50 EM, 27.90 precision, 10.01 recall, 14.75 F1
```

Note: the user-provided final table uses PopQA Gemini ToG F1 `14.75%`, which
differs from the remote evaluator output previously observed (`12.09%`). Charts
were generated from the user-provided final table for consistency.

## 2026-05-16 16:40 Gemini ToG Matched-Config Setup

User questioned why Gemini ToG is much worse than GPT-4o-mini and DeepSeek and
asked to set Gemini ToG using the same configuration as the first two models.

Finding:

- `Option/Method/ToG.yaml` and `Option/Method/ToG_v3.yaml` are identical.
- The previous Gemini ToG rerun already used `Option/Method/ToG.yaml`, but the
  old script name/comment was confusing.
- To remove ambiguity, a dedicated matched-config rerun script was created.

Local and remote file:

```text
run_tog_gemini_matched_config.sh
```

The script intentionally runs the same command pattern as the DeepSeek/GPT
baseline reruns:

```bash
python3 main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

It only changes `Option/Config2.yaml` model to:

```text
gemini-2.5-flash-lite
```

Remote sync and syntax check passed:

```text
synced_and_syntax_ok run_tog_gemini_matched_config.sh
```

The script was not started because the server was busy with leave-one-out
ablation:

```text
DATE 2026-05-16 16:39:25 CST
active:
663326   1404 S+         17:25 bash run_leave_one_out_ablation_both_datasets.sh
663327 663326 S+         17:25 bash run_leave_one_out_ablation.sh
663357 663327 Rl+        17:25 /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/generated_leave_one_out/NewG_loo_no_router.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200
```

When the server is idle, start the matched-config Gemini ToG rerun with:

```bash
cd /root/autodl-tmp/GraphRAG-master/GraphRAG-master
bash run_tog_gemini_matched_config.sh
```

Do not start this while `run_leave_one_out_ablation_both_datasets.sh` or any
`newg_main.py` process is active unless the user explicitly approves parallel
API usage.

## 2026-05-16 16:42 PopQA Gemini Table Correction

User corrected the PopQA `gemini-2.5-flash-lite` table. The previous final-table
data source had two F1 values swapped/wrong.

Correct values now applied:

```text
PopQA gemini-2.5-flash-lite RAPTOR  F1 = 16.55%
PopQA gemini-2.5-flash-lite ToG     F1 = 12.09%
```

Updated:

```text
collect_experiment_results.py
results_summary.tsv
results_popqa.md
figures/popqa_cross_model_by_method.png
figures/popqa_cross_model_f1.png
figures/popqa_deepseek_baseline_vs_newg.png
figures/popqa_gpt4omini_baseline_vs_newg.png
figures/popqa_gemini_baseline_vs_newg.png
figures/popqa_f1_baseline_vs_newg_summary.png
```

Validation:

```text
PopQA gemini RAPTOR  38.50 32.50 36.04 13.83 16.55
PopQA gemini ToG     31.50 26.50 27.90 10.01 12.09
results_summary.tsv row count: 72
```

## 2026-05-16 17:06 Server Progress Check

User asked which experiment is currently running. A read-only remote check found
the leave-one-out ablation workflow active:

```text
DATE 2026-05-16 17:06:44 CST
active:
663326   1404 S+         44:44 bash run_leave_one_out_ablation_both_datasets.sh
663327 663326 S+         44:44 bash run_leave_one_out_ablation.sh
663357 663327 Sl+        44:44 /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/generated_leave_one_out/NewG_loo_no_router.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200
```

Current running variant:

```text
dataset=PopQA
variant=no_router
latest log=logs/loo_Popqa_hipporag_bm25_gemini-2.5-flash-lite_no_router.log
latest marker=[Q84/200]
```

Leave-one-out result state:

```text
Popqa full              complete, 200 scored rows
Popqa no_router         running, partial results.jsonl 80 rows at check time
Popqa no_regen          missing/not started
Popqa no_critic         missing/not started
Popqa no_commendor      complete via old equivalent, 200 scored rows
Popqa no_normalizer     complete via old equivalent, 200 scored rows
Popqa no_disambiguation missing/not started
Popqa single_agent      complete via old equivalent, 200 scored rows

musique full              complete, 200 scored rows
musique no_router         missing/not started
musique no_regen          missing/not started
musique no_critic         missing/not started
musique no_commendor      complete via old equivalent, 200 scored rows
musique no_normalizer     complete via old equivalent, 200 scored rows
musique no_disambiguation missing/not started
musique single_agent      complete via old equivalent, 200 scored rows
```

Do not start the Gemini ToG matched-config rerun while this leave-one-out
workflow is active unless the user explicitly approves parallel API usage.

Follow-up estimate check:

```text
DATE 2026-05-16 17:11:17 CST
active:
663326   1404 S+         49:17 bash run_leave_one_out_ablation_both_datasets.sh
663327 663326 S+         49:17 bash run_leave_one_out_ablation.sh
663357 663327 Sl+        49:17 /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/generated_leave_one_out/NewG_loo_no_router.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200
```

Current log:

```text
logs/loo_Popqa_hipporag_bm25_gemini-2.5-flash-lite_no_router.log
last marker: [Q90/200]
tqdm around Q89 showed ETA about 1:44:53 for the current variant.
```

Estimated timing:

- Current PopQA `no_router` variant likely finishes around `2026-05-16 18:50-19:10 CST` if the current speed holds.
- Full both-dataset leave-one-out workflow still has several missing variants
  after this, so complete finish is more likely on `2026-05-17`, not tonight
  unless later variants run much faster.

## 2026-05-16 17:15 Conversation Preservation Update

User explicitly asked to preserve the conversation context so that saying
"继续上次任务" later resumes without losing state, and asked when tomorrow the
current server workflow may finish.

Latest read-only remote status:

```text
DATE 2026-05-16 17:15:09 CST
active:
663326   1404 S+         53:09 bash run_leave_one_out_ablation_both_datasets.sh
663327 663326 S+         53:09 bash run_leave_one_out_ablation.sh
663357 663327 Sl+        53:09 /root/miniconda3/envs/graphrag/bin/python newg_main.py -opt Option/Method/generated_leave_one_out/NewG_loo_no_router.yaml -graph_method hipporag -text_method bm25 -dataset_name datasets/Popqa --eval_limit 200
```

Current running item:

```text
dataset=PopQA
variant=no_router
log=logs/loo_Popqa_hipporag_bm25_gemini-2.5-flash-lite_no_router.log
last marker=[Q102/200]
checkpoint saved 100 partial rows to:
output/datasets/Popqa/NewG_hipporag_bm25_gemini-2.5-flash-lite_loo_no_router/Results/results.jsonl
```

Current state summary:

```text
Popqa full              done
Popqa no_router         running, partial 100 rows
Popqa no_regen          missing/not started
Popqa no_critic         missing/not started
Popqa no_commendor      done
Popqa no_normalizer     done
Popqa no_disambiguation missing/not started
Popqa single_agent      done

musique full              done
musique no_router         missing/not started
musique no_regen          missing/not started
musique no_critic         missing/not started
musique no_commendor      done
musique no_normalizer     done
musique no_disambiguation missing/not started
musique single_agent      done
```

Estimate:

- Current PopQA `no_router` variant may finish around `2026-05-16 17:45-18:00 CST`
  if the latest speed holds.
- Remaining missing variants after that: PopQA `no_regen`, `no_critic`,
  `no_disambiguation`; MuSiQue `no_router`, `no_regen`, `no_critic`,
  `no_disambiguation`.
- Full both-dataset leave-one-out workflow is most likely to finish on
  `2026-05-17` morning to midday CST. A conservative check window is
  `2026-05-17 10:00-12:00 CST`; if variants are slow, it may slip later.

Important next-session instruction:

- First action should be a read-only server status check.
- Do not start `run_tog_gemini_matched_config.sh` while leave-one-out is active
  unless the user explicitly approves parallel API usage.
- Do not print or copy the stored server password/API keys in user-facing output.

## 2026-05-17 11:21 Leave-One-Out Complete And Matched ToG Started

User asked to continue the previous task. A read-only remote status check found
the server idle and the leave-one-out workflow complete for both datasets.

Completion state:

```text
PopQA:  full, no_router, no_regen, no_critic, no_commendor,
        no_normalizer, no_disambiguation, single_agent all 200 scored rows
MuSiQue: full, no_router, no_regen, no_critic, no_commendor,
        no_normalizer, no_disambiguation, single_agent all 200 scored rows
```

Synced from the server to local:

```text
figures/leave_one_out_Popqa_gemini-2.5-flash-lite_hipporag_bm25.*
figures/leave_one_out_musique_gemini-2.5-flash-lite_hipporag_bm25.*
logs/loo_*_hipporag_bm25_gemini-2.5-flash-lite_*.log
output/datasets/{Popqa,musique}/NewG_hipporag_bm25_gemini-2.5-flash-lite*/Results/results.score.jsonl
```

Local validation passed with:

```bash
python collect_leave_one_out_ablation.py --dataset datasets/Popqa --graph hipporag --text bm25 --model gemini-2.5-flash-lite --variants full no_router no_regen no_critic no_commendor no_normalizer no_disambiguation single_agent
python collect_leave_one_out_ablation.py --dataset datasets/musique --graph hipporag --text bm25 --model gemini-2.5-flash-lite --variants full no_router no_regen no_critic no_commendor no_normalizer no_disambiguation single_agent
```

The first attempt to start `run_tog_gemini_matched_config.sh` failed immediately
because the script defaulted to system `python3`, which lacked `pyfiglet` on the
server. The script was fixed locally and synced to the server so the default is:

```bash
PYTHON="${PYTHON:-/root/miniconda3/envs/graphrag/bin/python}"
```

The failed attempt archived the old PopQA Gemini ToG output to:

```text
.tog_gemini_matched_config_archived_20260517_112015/output/datasets/Popqa/ToG_gemini-2.5-flash-lite
```

The corrected matched-config PopQA Gemini ToG rerun was started on the server:

```text
DATE 2026-05-17 11:21:33 CST
root     942204 ... bash run_tog_gemini_matched_config.sh
root     942215 ... /root/miniconda3/envs/graphrag/bin/python main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

Logs to monitor:

```text
logs/rerun_popqa_tog_gemini_matched_config.nohup.log
logs/rerun_popqa_tog_gemini_matched_config.log
```

Follow-up check found the 11:21 run hit the known Hugging Face HEAD-request
problem while loading `BAAI/bge-small-en-v1.5`. The script was updated locally
and synced to the server with:

```bash
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/root/.cache/llama_index}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
```

The corrected run was restarted:

```text
DATE 2026-05-17 11:26:07 CST
root     942539 ... bash run_tog_gemini_matched_config.sh
root     942550 ... /root/miniconda3/envs/graphrag/bin/python main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

Status at `2026-05-17 11:30:52 CST`: process still running, embedding loaded
from local cache, graph loaded, and the log was at vector index loading:

```text
Loading index from the file ./output/datasets/Popqa/er_graph/entities_vdb
```

User saw the terminal showing `Both-dataset leave-one-out ablation workflow
finished` and asked if the current job had ended because their API account had
previously run out of balance. A read-only remote check at
`2026-05-17 11:39:11 CST` confirmed that the screenshot was the completed
leave-one-out terminal, while the matched-config ToG rerun was still active:

```text
root     942539 ... bash run_tog_gemini_matched_config.sh
root     942550 ... /root/miniconda3/envs/graphrag/bin/python main.py -opt Option/Method/ToG.yaml -dataset_name datasets/Popqa --eval_limit 200
```

No quota/billing/API error was present in
`logs/rerun_popqa_tog_gemini_matched_config.log` or the nohup wrapper log.
The process was CPU-active and still at local index loading; no current result
files had been produced yet.

## 2026-05-17 13:12 PopQA Gemini ToG Matched-Config Complete

The PopQA Gemini ToG matched-config rerun finished successfully on the server.
No active `run_tog_gemini_matched_config.sh` or ToG `main.py` process remained
at the follow-up check.

Runtime:

```text
started  2026-05-17 11:26:07 CST
finished 2026-05-17 13:12:00 CST
```

Result files:

```text
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.json
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.score.json
output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/metrics.json
```

Metrics:

```text
n=200
accuracy  31.0000%
EM        26.0000%
precision 27.3974%
recall     9.9359%
F1        11.9662%
```

This is essentially consistent with the previous Gemini ToG result
(`accuracy=31.50`, `EM=26.50`, `F1=12.09`) and does not explain the Gemini ToG
weakness as a config mismatch.

## 2026-05-17 Final Data Audit And Resume Notes

The user asked to preserve the conversation/task state so that "continue previous
task" can resume without losing context. A long-term resume note was written to:

```text
C:\Users\zhaim\.codex\memories\graphrag_newg_resume.md
```

Current local project path:

```text
D:\Tools\Downloads\GraphRAG-master1\root\autodl-tmp\GraphRAG-master\GraphRAG-master
```

Final audit status:

```text
results_summary.tsv rows: 72
grid: 2 datasets x 3 models x 12 methods
missing rows: 0
duplicate rows: 0
bad n in summary: 0
bad metric range: 0
```

The final PopQA/Gemini/ToG row was corrected to the matched-config raw result:

```text
PopQA / gemini-2.5-flash-lite / ToG
n=200
accuracy  31.00
EM        26.00
precision 27.40
recall     9.94
F1        11.97
source: output/datasets/Popqa/ToG_gemini-2.5-flash-lite/Results/results.score.json
```

Updated files:

```text
collect_experiment_results.py
results_summary.tsv
results_popqa.md
figures/positive_showcase/*
experiment_results_organized.md
figures/positive_*.png
```

Leave-one-out audit:

```text
figures/leave_one_out_Popqa_gemini-2.5-flash-lite_hipporag_bm25.tsv
figures/leave_one_out_musique_gemini-2.5-flash-lite_hipporag_bm25.tsv
```

Both have 8 variants and all rows have `n=200`:

```text
full
no_router
no_regen
no_critic
no_commendor
no_normalizer
no_disambiguation
single_agent
```

The `full` row in each leave-one-out TSV matches the corresponding main-table
Gemini/NewG hippo+bm25 result.

Curated positive figure directory:

```text
figures/positive_showcase
```

Current PNGs:

```text
panel_largest_positive_paired_retriever_gains.png
panel_leave_one_out_removal_hurts_f1.png
panel_musique_best_f1_lift.png
panel_musique_best_newg_metric_gains.png
showcase_musique_bm25_vdb_multimetric_radar.png
showcase_musique_leaderboard.png
showcase_popqa_bm25_vdb_multimetric_radar.png
showcase_positive_main_experiment_cards.png
showcase_positive_paired_gain_heatmap.png
```

Recent figure logic:

- The paired radar charts use a 3x3 layout, not one best architecture per model.
- Rows are models; columns are HippoRAG, ToG, RAPTOR.
- Each panel compares original baseline vs paired NewG+BM25 vs paired NewG+VDB.
- MuSiQue radar uses 0..40 axis.
- PopQA radar uses 0..70 axis.
- `showcase_musique_leaderboard.png` was changed from Top-K ranking to fixed
  paired baseline display so every model shows all three gray baselines.

Important caveat:

```text
The final tables are internally consistent, but local raw output files are not
a complete reproducibility archive.
```

Latest raw-file audit found:

```text
directly available clean/usable score files for 72-row grid: 23
rows relying on curated final table values: 49
bad local raw audit sources:
  PopQA / DeepSeek / RAPTOR: raw score file has 1399 rows
  PopQA / GPT-4o-mini / RAPTOR: raw score file has 0 rows
```

If the next task is final archival/reproducibility packaging, sync all clean
remote 200-row `results.score.json` files and replace/fix the two bad local raw
files. If the next task is paper writing/figure polishing, the curated summary
tables and positive_showcase figures are the current working source.

## 2026-05-29 continuation: paper results narrative

Continuation was requested again with "continue previous task".
The active branch of work is paper writing and figure packaging, not more
experiment execution.

Created:

```text
paper_results_narrative.md
```

It contains:

- conservative paper claim framing;
- MuSiQue and PopQA result summaries;
- paired-retriever evidence;
- leave-one-out ablation interpretation;
- recommended main-paper vs appendix figure placement;
- draft Results paragraphs and figure captions.

Key narrative choices preserved there:

- MuSiQue is the primary positive result: best NewG beats the strongest
  baseline for DeepSeek, GPT-4o-mini, and Gemini.
- PopQA should be described as mixed/competitive robustness evidence, not as a
  universal win.
- Paired-retriever comparison: NewG wins 25/36 pairs (69.4%), average all-pair
  F1 change +4.78, average positive-pair gain +7.67.
- Ablation story should focus on Critic and Answer normalizer.

Next likely task:

- polish `paper_results_narrative.md` into the target paper style, or convert
  the selected figure/table plan into LaTeX.

## 2026-05-29 continuation: LaTeX results fragment

Continuation was requested with "continue previous task". The paper-writing
branch was continued by creating:

```text
paper_results_latex.tex
```

This is a standalone LaTeX Results-section fragment, not a full paper template.
It contains:

- MuSiQue main-results prose, table, and figure block;
- PopQA conservative robustness table;
- paired-retriever analysis prose, top-gain table, and heatmap figure block;
- leave-one-out ablation prose, key component table, and figure block;
- short final summary paragraph.

Required LaTeX packages noted at the top of the file:

```text
booktabs
graphicx
```

The fragment uses the curated `n=200` tables and figures from:

```text
results_summary.tsv
figures/positive_showcase/
```

## 2026-05-29 continuation: complete paired-delta heatmap

The user noted that `showcase_positive_paired_gain_heatmap.png` was incomplete
because it only displayed positive paired-retriever gains and blanked
non-positive cells. Updated:

```text
plot_positive_showcase_figures.py
figures/positive_showcase/showcase_positive_paired_gain_heatmap.png
figures/positive_showcase/paired_newg_deltas.tsv
figures/positive_showcase/README_positive_showcase.md
paper_results_latex.tex
paper_results_narrative.md
```

The heatmap now contains the full 36-cell paired-retriever delta grid:

```text
positive cells: 25
non-positive cells: 11
```

The old positive-only TSV is still generated as:

```text
figures/positive_showcase/positive_paired_newg_deltas.tsv
```

## 2026-05-29 continuation: radar figure layout polish

The user asked to enlarge the visual icons in the PopQA and MuSiQue paired
retriever radar grids without overlap. Updated:

```text
plot_positive_showcase_figures.py
figures/positive_showcase/showcase_popqa_bm25_vdb_multimetric_radar.png
figures/positive_showcase/showcase_musique_bm25_vdb_multimetric_radar.png
```

Changes:

- enlarged the radar-grid figure canvas;
- switched the 3x3 radar grids to manual subplot positioning so the radar
  circles occupy more of each cell;
- increased radar line widths and axis-label font sizes;
- increased the bottom legend font/handle sizes;
- moved the legend lower and reserved extra bottom margin so it does not
  overlap the F1 annotations.

## 2026-05-29 continuation: figures directory organization

The user asked to organize `figures/` while leaving the positive result folder
untouched. Created:

```text
figures/main_experiments/
figures/ablation_experiments/
figures/ablation_experiments/incremental_ablation/
figures/ablation_experiments/leave_one_out_ablation/
```

Moved root-level main experiment plots into:

```text
figures/main_experiments/
```

Moved PopQA incremental ablation plots into:

```text
figures/ablation_experiments/incremental_ablation/
```

Moved leave-one-out ablation plots/tables into:

```text
figures/ablation_experiments/leave_one_out_ablation/
```

Did not move or rename:

```text
figures/positive_showcase/
```

Updated plotting defaults so reruns write to the organized directories:

```text
plot_popqa_model_comparison.py
plot_musique_model_comparison.py
plot_popqa_ablation.py
plot_leave_one_out_ablation.py
plot_positive_story_figures.py
```

## 2026-06-17 Session Continuation Note

User asked to preserve the conversation so that saying "继续上次任务" in a
future session does not lose context.

Current state at the end of this session:

- The working focus is the NewG / GraphRAG paper-writing and repository
  packaging task.
- The latest edits were to the paper-facing result artifacts:
  - `paper_results_latex.tex`
  - `paper_results_narrative.md`
- The figure references were aligned with the current `positive_showcase`
  assets:
  - `panel_musique_best_f1_lift.png`
  - `panel_largest_positive_paired_retriever_gains.png`
  - `panel_leave_one_out_removal_hurts_f1.png`
- The referenced files exist in the repository and the main result numbers were
  checked against `results_summary.tsv`.

Safe next step when the user says "继续上次任务":

1. Read this handoff log first.
2. Continue from the paper results section and integrate the finalized
   narrative into the main manuscript, if needed.
3. Do not overwrite unrelated user changes.
