"""
Agentic GraphRAG Pipeline (agentic_main.py)

Supports TWO modes:

1. Per-method Agentic mode (legacy):
   python agentic_main.py -opt Option/Method/AgenticHippoRAG.yaml \
       -dataset_name datasets/Popqa --eval_limit 20

2. AgenticNewG mode (unified architecture):
   python agentic_main.py -opt Option/Method/AgenticNewG.yaml \
       -dataset_name datasets/Popqa --eval_limit 20

   AgenticNewG automatically loads multiple retriever instances and
   dynamically routes each query to the best-fit method.
"""

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config, parse, merge_dict
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator
from Core.Common.Logger import logger
from Core.Utils.YamlModel import YamlModel

try:
    from tqdm import tqdm as _tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm as _logging_redirect_tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    _logging_redirect_tqdm = None


def check_dirs(opt, args):
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


def is_agentic_newg(opt_path) -> bool:
    """Check if the YAML is an AgenticNewG config (has instance_ppr section)."""
    raw = YamlModel.read_yaml(Path(opt_path))
    return "instance_ppr" in raw or "instance_tog" in raw


def build_instance_config(raw_yaml: dict, instance_name: str, dataset_name: str):
    """
    Build a Config object for a single retriever instance.

    Merges: Config2.yaml defaults + global llm/embedding + instance-specific overrides.
    """
    from Core.Common.Constants import GRAPHRAG_ROOT, CONFIG_ROOT

    # Start with Config2.yaml defaults
    default_config_paths = [
        GRAPHRAG_ROOT / "Option/Config2.yaml",
        CONFIG_ROOT / "Config2.yaml",
    ]
    dicts = [Config.read_yaml(path) for path in default_config_paths]

    # Add global llm/embedding from AgenticNewG.yaml
    global_overrides = {}
    if "llm" in raw_yaml:
        global_overrides["llm"] = raw_yaml["llm"]
    if "embedding" in raw_yaml:
        global_overrides["embedding"] = raw_yaml["embedding"]
    if "chunk" in raw_yaml:
        global_overrides["chunk"] = raw_yaml["chunk"]
    global_overrides["llm_model_max_token_size"] = raw_yaml.get(
        "llm_model_max_token_size", 32768
    )
    dicts.append(global_overrides)

    # Add instance-specific config (this wins over everything)
    instance_cfg = raw_yaml.get(instance_name, {})
    dicts.append(instance_cfg)

    final = merge_dict(dicts)
    final["dataset_name"] = dataset_name
    final["working_dir"] = os.path.join(
        final.get("working_dir", "./output"), dataset_name
    )

    # Generate exp_name for this instance
    model_short = final.get("llm", {}).get("model", "unknown").split("/")[-1]
    final["exp_name"] = f"AgenticNewG_{model_short}"

    return Config(**final)


# ===========================================================================
# Legacy per-method Agentic mode
# ===========================================================================

async def wrapper_query(query_dataset, digimon, result_dir, eval_limit: int,
                        router=None):
    """
    Query loop with optional per-query routing.

    When router is None, behaves identically to the standard main.py.
    When router is provided, classifies each query and dynamically
    switches the DIGIMON internal querier to the recommended method.
    """
    all_res = []
    dataset_len = len(query_dataset)
    if eval_limit > 0:
        dataset_len = min(dataset_len, eval_limit)

    # Cache the default querier so we can restore after routing
    default_query_type = digimon.config.retriever.query_type
    default_querier = getattr(digimon, '_querier', None)
    switched = False

    async def _loop(pbar):
        nonlocal switched
        for i in range(dataset_len):
            query = query_dataset[i]
            question = query["question"]
            short_q = question[:60].replace("\n", " ")
            if pbar is not None:
                pbar.set_postfix_str(short_q, refresh=True)
            else:
                print(f"[{i+1}/{dataset_len}] {short_q}", flush=True)

            if router:
                route_info = await router.classify(question)
                query["route_category"] = route_info.category
                query["route_confidence"] = route_info.confidence
                query["route_method"] = route_info.recommended_method
                logger.info(f"Q{i}: {route_info}")
                if route_info.recommended_method != default_query_type:
                    try:
                        from Core.Query import get_query
                        digimon._querier = get_query(
                            route_info.recommended_method,
                            digimon.config.query,
                            digimon.retriever_context,
                        )
                        switched = True
                        logger.info(f"Switched querier to: {route_info.recommended_method}")
                    except Exception as e:
                        logger.warning(f"Failed to switch querier to {route_info.recommended_method}: {e}, "
                                       f"keeping default ({default_query_type})")
                elif switched:
                    digimon._querier = default_querier
                    switched = False

            res = await digimon.query(question)
            query["output"] = res
            all_res.append(query)
            if pbar is not None:
                pbar.update(1)

    if _HAS_TQDM:
        with _tqdm(total=dataset_len, desc="AgenticGR", unit="q") as pbar:
            with _logging_redirect_tqdm():
                await _loop(pbar)
    else:
        await _loop(None)

    # Restore default querier after loop
    if switched and default_querier is not None:
        digimon._querier = default_querier

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.jsonl")
    all_res_df.to_json(save_path, orient="records", lines=True)
    return save_path


# ===========================================================================
# AgenticNewG mode — multi-instance dynamic routing
# ===========================================================================

async def newg_load_instances(raw_yaml: dict, dataset_name: str, corpus):
    """
    Load multiple GraphRAG instances from AgenticNewG.yaml.
    Returns dict of {instance_key: querier}.
    """
    instance_names = {
        "ppr": "instance_ppr",
        "tog": "instance_tog",
        "raptor": "instance_raptor",
    }

    instances = {}
    for key, yaml_key in instance_names.items():
        if yaml_key not in raw_yaml:
            logger.info(f"Instance '{yaml_key}' not defined in YAML, skipping")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"Loading instance: {key} ({yaml_key})")
        logger.info(f"{'='*60}")

        cfg = build_instance_config(raw_yaml, yaml_key, dataset_name)
        digimon = GraphRAG(config=cfg)

        # Insert corpus (builds graph / loads indexes)
        await digimon.insert(corpus)

        # Extract the querier
        instances[key] = digimon._querier
        logger.info(f"Instance '{key}' loaded successfully "
                     f"(querier={type(digimon._querier).__name__})")

    return instances


async def newg_query_loop(query_dataset, engine, result_dir, eval_limit: int):
    """
    AgenticNewG query loop — delegates everything to AgenticNewGEngine.
    """
    all_res = []
    dataset_len = len(query_dataset)
    if eval_limit > 0:
        dataset_len = min(dataset_len, eval_limit)

    async def _loop(pbar):
        for i in range(dataset_len):
            query = query_dataset[i]
            question = query["question"]
            short_q = question[:60].replace("\n", " ")
            if pbar is not None:
                pbar.set_postfix_str(short_q, refresh=True)
            else:
                print(f"[{i+1}/{dataset_len}] {short_q}", flush=True)
            logger.info(f"[Q{i+1}/{dataset_len}] {question[:80]}...")

            result = await engine.query(question)

            query["output"] = result.get("output") or ""
            query["route_category"] = result.get("route_category", "")
            query["route_confidence"] = result.get("route_confidence", 0.0)
            query["route_method"] = result.get("route_method", "")
            query["critic_rounds"] = result.get("critic_rounds", 0)
            query["methods_tried"] = str(result.get("methods_tried", []))
            all_res.append(query)
            if pbar is not None:
                pbar.update(1)

    if _HAS_TQDM:
        with _tqdm(total=dataset_len, desc="AgenticNewG", unit="q") as pbar:
            with _logging_redirect_tqdm():
                await _loop(pbar)
    else:
        await _loop(None)

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.jsonl")
    all_res_df.to_json(save_path, orient="records", lines=True)
    return save_path


# ===========================================================================
# Shared evaluation
# ===========================================================================

async def wrapper_evaluation(path, dataset_name, result_dir):
    evaluator = Evaluator(path, dataset_name)
    res_dict = await evaluator.evaluate()
    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        f.write(str(res_dict))
    return res_dict


# ===========================================================================
# Main entry points
# ===========================================================================

async def main_newg(args, raw_yaml, result_dir):
    """AgenticNewG mode: load all instances, then route queries dynamically."""
    from Core.Query.AgenticNewGQuery import AgenticNewGEngine

    dataset_name = args.dataset_name
    query_dataset = RAGQueryDataset(data_dir=dataset_name)
    corpus = query_dataset.get_corpus()

    # Phase 1: Load all retriever instances
    instances = await newg_load_instances(raw_yaml, dataset_name, corpus)

    if not instances:
        logger.error("No instances loaded! Check AgenticNewG.yaml config.")
        return

    logger.info(f"Loaded {len(instances)} instances: {list(instances.keys())}")

    # Phase 2: Build the unified engine
    # Get LLM from any loaded instance
    any_querier = next(iter(instances.values()))
    llm = any_querier.llm

    agentic_config = raw_yaml.get("agentic", {})
    routing_map = raw_yaml.get("routing_map", None)

    engine = AgenticNewGEngine(
        instances=instances,
        llm=llm,
        agentic_config=agentic_config,
        routing_map=routing_map,
    )

    # Phase 3: Query loop
    save_path = await newg_query_loop(
        query_dataset, engine, result_dir, args.eval_limit
    )

    # Phase 4: Evaluate
    await wrapper_evaluation(save_path, dataset_name, result_dir)

    # Phase 5: Print statistics
    if agentic_config.get("use_critic"):
        from Core.Query.CriticModule import get_feedback_stats
        stats = get_feedback_stats()
        total = stats.get("total_rounds", 0)
        if total > 0:
            logger.info("=" * 50)
            logger.info("AgenticNewG Critic Feedback Statistics")
            logger.info(f"  Total critic rounds  : {total}")
            logger.info(f"  pass                 : {stats.get('pass', 0)}")
            logger.info(f"  retrieve_more/revise : {stats.get('retrieve_more', 0) + stats.get('revise', 0)}")
            logger.info(f"  → missing_entity     : {stats.get('missing_entity', 0)}")
            logger.info(f"  → broken_path        : {stats.get('broken_path', 0)}")
            logger.info(f"  → conflict           : {stats.get('conflict', 0)}")
            logger.info("=" * 50)


async def main_legacy(opt, args, result_dir):
    """Legacy per-method Agentic mode."""
    digimon = GraphRAG(config=opt)
    query_dataset = RAGQueryDataset(data_dir=opt.dataset_name)
    corpus = query_dataset.get_corpus()

    await digimon.insert(corpus)

    router = None
    if args.enable_routing:
        from Core.Query.QueryRouter import QueryRouter
        routing_map = getattr(opt.query, 'routing_map', None)
        router = QueryRouter(digimon.llm, routing_map)

    save_path = await wrapper_query(
        query_dataset, digimon, result_dir,
        eval_limit=args.eval_limit, router=router,
    )

    await wrapper_evaluation(save_path, opt.dataset_name, result_dir)

    if getattr(opt.query, 'use_critic', False):
        from Core.Query.CriticModule import get_feedback_stats
        stats = get_feedback_stats()
        total = stats.get("total_rounds", 0)
        if total > 0:
            logger.info("=" * 50)
            logger.info(f"Critic Feedback Statistics")
            logger.info(f"  Total critic rounds  : {total}")
            logger.info(f"  pass                 : {stats.get('pass', 0)}")
            logger.info(f"  retrieve_more/revise : {stats.get('retrieve_more', 0) + stats.get('revise', 0)}")
            logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic GraphRAG Pipeline")
    parser.add_argument("-opt", type=str, required=True, help="Path to method YAML")
    parser.add_argument("-dataset_name", type=str, required=True, help="Dataset name/path")
    parser.add_argument("-eval_limit", "--eval_limit", type=int, default=0,
                        help="Max queries (0=full dataset)")
    parser.add_argument("--enable_routing", action="store_true",
                        help="Enable Innovation 3: Query Router (legacy mode only)")
    args = parser.parse_args()

    opt_path = Path(args.opt)

    if is_agentic_newg(opt_path):
        # ===== AgenticNewG mode =====
        logger.info("=" * 60)
        logger.info("AgenticNewG Mode — Unified Agentic GraphRAG Architecture")
        logger.info("=" * 60)

        raw_yaml = YamlModel.read_yaml(opt_path)

        # Build a minimal Config for check_dirs (needs working_dir, exp_name)
        dataset_name = args.dataset_name
        model_short = raw_yaml.get("llm", {}).get("model", "unknown").split("/")[-1]
        working_dir = os.path.join(
            raw_yaml.get("working_dir", "./output"), dataset_name
        )

        # Create a simple namespace for result dirs
        class _MinimalOpt:
            pass
        min_opt = _MinimalOpt()
        min_opt.working_dir = working_dir
        min_opt.exp_name = f"AgenticNewG_{model_short}"
        min_opt.dataset_name = dataset_name

        result_dir = check_dirs(min_opt, args)

        agentic = raw_yaml.get("agentic", {})
        active = []
        if agentic.get("use_critic"):
            active.append(f"Critic (max {agentic.get('critic_max_rounds', 2)} rounds, "
                          f"cross-method={agentic.get('critic_cross_method_retry', True)})")
        if agentic.get("use_answer_normalizer"):
            active.append("Answer Normalizer")
        active.append("Adaptive Router (always on)")

        instances_available = []
        if "instance_ppr" in raw_yaml:
            instances_available.append("HippoRAG(PPR)")
        if "instance_tog" in raw_yaml:
            instances_available.append("ToG")
        if "instance_raptor" in raw_yaml:
            instances_available.append("RAPTOR")

        logger.info(f"Retriever Bank: {', '.join(instances_available)}")
        logger.info(f"Innovations: {', '.join(active)}")

        asyncio.run(main_newg(args, raw_yaml, result_dir))

    else:
        # ===== Legacy per-method mode =====
        opt = Config.parse(opt_path, dataset_name=args.dataset_name)

        active = []
        if getattr(opt.query, 'use_critic', False):
            active.append(f"Critic (max {opt.query.critic_max_rounds} rounds)")
        if getattr(opt.query, 'use_answer_normalizer', False):
            active.append("Answer Normalizer")
        if args.enable_routing:
            active.append("Query Router")
        if active:
            logger.info(f"Agentic innovations enabled: {', '.join(active)}")
        else:
            logger.info("No agentic innovations enabled (standard mode)")

        result_dir = check_dirs(opt, args)
        asyncio.run(main_legacy(opt, args, result_dir))
