from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator
from Core.Common.Logger import logger

try:
    from tqdm import tqdm as _tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm as _logging_redirect_tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    _logging_redirect_tqdm = None
def check_dirs(opt):
    result_dir = os.path.join(opt.working_dir, opt.exp_name, "Results")
    config_dir = os.path.join(opt.working_dir, opt.exp_name, "Configs")
    metric_dir = os.path.join(opt.working_dir, opt.exp_name, "Metrics")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    opt_name = Path(args.opt).name
    basic_name = str(Path(args.opt).parent.parent / "Config2.yaml")
    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))
    return result_dir


async def wrapper_query(query_dataset, digimon, result_dir, eval_limit: int):
    all_res = []
    dataset_len = len(query_dataset)
    # eval_limit <= 0: use full dataset; else cap for cost/smoke tests
    if eval_limit > 0:
        dataset_len = min(dataset_len, eval_limit)

    async def _loop(pbar):
        for i in range(dataset_len):
            query = query_dataset[i]
            short_q = query["question"][:60].replace("\n", " ")
            if pbar is not None:
                pbar.set_postfix_str(short_q, refresh=True)
            else:
                print(f"[{i+1}/{dataset_len}] {short_q}", flush=True)
            prompt = (
                f"{query['question']}\n"
                "Answer with a short factual span only (no explanation, no full sentence)."
            )
            res = await digimon.query(prompt)
            query["output"] = res
            all_res.append(query)
            if pbar is not None:
                pbar.update(1)

    if _HAS_TQDM:
        with _tqdm(total=dataset_len, desc="GraphRAG", unit="q") as pbar:
            with _logging_redirect_tqdm():
                await _loop(pbar)
    else:
        await _loop(None)

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    return save_path


async def wrapper_evaluation(path, opt, result_dir):
    eval = Evaluator(path, opt.dataset_name)
    res_dict = await eval.evaluate()
    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        f.write(str(res_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument(
        "-eval_limit",
        "--eval_limit",
        type=int,
        default=0,
        help="Max queries to run; 0 or negative = full Question.json (paper-style). "
        "Set e.g. 100 for a quick/cheap smoke test.",
    )
    args = parser.parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)

    digimon = GraphRAG(config=opt)
    result_dir = check_dirs(opt)

    query_dataset = RAGQueryDataset(data_dir=opt.dataset_name)
    corpus = query_dataset.get_corpus()

    asyncio.run(digimon.insert(corpus))
    save_path = asyncio.run(
        wrapper_query(query_dataset, digimon, result_dir, eval_limit=args.eval_limit)
    )
    asyncio.run(wrapper_evaluation(save_path, opt, result_dir))
