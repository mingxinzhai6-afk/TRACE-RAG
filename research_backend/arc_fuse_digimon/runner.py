"""
ARC-Fuse paper experiment runner for the DIGIMON backend.

Unified agentic Graph + Text RAG framework where the specific graph method
and text method are chosen at runtime via CLI flags (not separate YAMLs).

Usage:
    python arc_fuse_main.py \
        -opt research_backend/configs/arc_fuse.yaml \
        -graph_method hipporag \
        -text_method bm25 \
        -dataset_name datasets/Popqa \
        --eval_limit 100

    graph_method ∈ {hipporag, tog, raptor}
    text_method  ∈ {bm25, vdb}

Six combinations = 3 × 2.
"""

import os
import asyncio
import argparse
from pathlib import Path
from shutil import copyfile

import pandas as pd

try:
    from tqdm import tqdm as _tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm as _logging_redirect_tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    _logging_redirect_tqdm = None

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config, merge_dict
from Data.QueryDataset import RAGQueryDataset
from Core.Utils.YamlModel import YamlModel
from Core.Common.Logger import logger
from arc_fuse_digimon.commendor import get_commendor_stats
from arc_fuse_digimon.critic import get_feedback_stats
from arc_fuse_digimon.engine import ArcFuseEngine, get_route_stats
from research_backend.evaluate import evaluate_path


# Map CLI flag → YAML section name
GRAPH_METHOD_TO_SECTION = {
    "hipporag": "graph_hipporag",
    "tog": "graph_tog",
    "raptor": "graph_raptor",
}

TEXT_METHODS = {"bm25", "vdb"}


# ===========================================================================
# Config builders
# ===========================================================================

def build_graph_config(raw_yaml: dict, graph_method: str, dataset_name: str) -> Config:
    """Compose a full Config for the chosen graph method."""
    from Core.Common.Constants import GRAPHRAG_ROOT, CONFIG_ROOT

    section_name = GRAPH_METHOD_TO_SECTION.get(graph_method)
    if section_name is None:
        raise ValueError(
            f"Unknown graph_method '{graph_method}'. "
            f"Valid: {list(GRAPH_METHOD_TO_SECTION.keys())}"
        )
    if section_name not in raw_yaml:
        raise ValueError(f"YAML is missing section '{section_name}'")

    # Merge: Config2.yaml → global llm/embedding → graph section
    default_paths = [
        GRAPHRAG_ROOT / "Option/Config2.yaml",
        CONFIG_ROOT / "Config2.yaml",
    ]
    dicts = [Config.read_yaml(p) for p in default_paths]

    globals_ = {}
    for k in ("llm", "embedding", "chunk", "llm_model_max_token_size"):
        if k in raw_yaml:
            globals_[k] = raw_yaml[k]
    dicts.append(globals_)

    # Also add top-level flags (use_entities_vdb etc.)
    top_level = {k: v for k, v in raw_yaml.items()
                 if k not in GRAPH_METHOD_TO_SECTION.values()
                 and not k.startswith("text_")
                 and not k.startswith("_")
                 and k not in ("arc_fuse", "llm", "embedding", "chunk")}
    dicts.append(top_level)

    # Graph-section wins
    dicts.append(raw_yaml[section_name])

    final = merge_dict(dicts)
    final["dataset_name"] = dataset_name
    final["working_dir"] = os.path.join(
        final.get("working_dir", "./output"), dataset_name
    )

    model_short = final.get("llm", {}).get("model", "unknown").split("/")[-1]
    text_tag = raw_yaml.get("_text_method_tag", "")
    abl_tag = raw_yaml.get("_abl_tag", "")
    if abl_tag:
        final["exp_name"] = f"ARCFuse_{graph_method}_{text_tag}_{model_short}_{abl_tag}".rstrip("_")
    else:
        final["exp_name"] = f"ARCFuse_{graph_method}_{text_tag}_{model_short}".rstrip("_")

    return Config(**final)


# ===========================================================================
# Text retriever factory
# ===========================================================================

async def build_text_retriever(text_method: str, digimon: GraphRAG, raw_yaml: dict):
    """Return a text retriever object (has async .retrieve(question) -> list[dict])."""
    text_method = text_method.lower()

    if text_method == "bm25":
        from arc_fuse_digimon.bm25 import BM25Retriever
        cfg = raw_yaml.get("text_bm25", {})
        retr = BM25Retriever(
            doc_chunk=digimon.doc_chunk,
            top_k=cfg.get("top_k", 10),
        )
        await retr.build_index()
        return retr

    if text_method == "vdb":
        from Core.Common.Constants import Retriever
        cfg = raw_yaml.get("text_vdb", {})
        top_k = cfg.get("top_k", 10)

        # Wrap DIGIMON's entity-VDB → chunk retrieval into a simple async callable
        mix = digimon._querier._retriever  # MixRetriever

        class _VDBTextRetriever:
            def __init__(self):
                self.top_k = top_k

            async def retrieve(self, query, top_k=None):
                k = top_k or self.top_k
                try:
                    node_datas = await mix.retrieve_relevant_content(
                        type=Retriever.ENTITY, mode="vdb", seed=query
                    )
                    if not node_datas:
                        return []
                    text_units = await mix.retrieve_relevant_content(
                        type=Retriever.CHUNK, mode="entity_occurrence",
                        node_datas=node_datas
                    )
                    if not text_units:
                        return []
                    results = []
                    for i, tu in enumerate(text_units[:k]):
                        content = tu if isinstance(tu, str) else (
                            getattr(tu, "content", None) or str(tu)
                        )
                        results.append({
                            "chunk_id": f"vdb_{i}",
                            "content": content,
                            "score": 1.0 / (i + 1),
                            "rank": i,
                        })
                    return results
                except Exception as e:
                    logger.warning(f"VDB text retrieval failed: {e}")
                    return []

        return _VDBTextRetriever()

    raise ValueError(f"Unknown text_method '{text_method}'. Valid: {TEXT_METHODS}")


# ===========================================================================
# I/O helpers
# ===========================================================================

def check_dirs(working_dir: str, exp_name: str, opt_path: Path) -> str:
    """Create Results/Configs/Metrics dirs and copy the YAML."""
    result_dir = os.path.join(working_dir, exp_name, "Results")
    config_dir = os.path.join(working_dir, exp_name, "Configs")
    metric_dir = os.path.join(working_dir, exp_name, "Metrics")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    copyfile(opt_path, os.path.join(config_dir, opt_path.name))
    basic = opt_path.parent.parent / "Config2.yaml"
    if basic.exists():
        copyfile(basic, os.path.join(config_dir, "Config2.yaml"))
    return result_dir


async def run_query_loop(query_dataset, engine, result_dir: str, eval_limit: int):
    all_res = []
    n = len(query_dataset)
    if eval_limit > 0:
        n = min(n, eval_limit)

    indices = range(n)
    save_path = os.path.join(result_dir, "results.jsonl")
    SAVE_EVERY = 10  # incremental dump every N queries to survive crashes/kills

    def _dump():
        if all_res:
            pd.DataFrame(all_res).to_json(save_path, orient="records", lines=True)

    async def _loop(pbar):
        for i in indices:
            query = query_dataset[i]
            question = query["question"]
            short_q = question[:60].replace("\n", " ")
            if pbar is not None:
                pbar.set_postfix_str(short_q, refresh=True)
            else:
                print(f"[{i+1}/{n}] {short_q}", flush=True)
            logger.info(f"[Q{i+1}/{n}] {question[:80]}")

            result = await engine.query(question)
            query["output"] = result.get("output") or ""
            query["initial_selection"] = result.get("initial_selection", "")
            query["domain"] = result.get("domain", "")
            query["route_source"] = result.get("route_source", "")
            query["initial_entities"] = str(result.get("initial_entities", []))
            query["initial_relations"] = str(result.get("initial_relations", []))
            query["query_understanding_raw"] = result.get("query_understanding_raw", "")
            query["rounds"] = result.get("rounds", 0)
            query["selection_history"] = str(result.get("selection_history", []))
            query["commendor_decisions"] = str(result.get("commendor_decisions", []))
            query["round_details"] = str(result.get("round_details", []))
            all_res.append(query)
            if pbar is not None:
                pbar.update(1)

            if (i + 1) % SAVE_EVERY == 0:
                _dump()
                logger.info(f"[checkpoint] saved {len(all_res)} results to {save_path}")

    if _HAS_TQDM:
        with _tqdm(total=n, desc="ARC-Fuse", unit="q") as pbar:
            with _logging_redirect_tqdm():
                await _loop(pbar)
    else:
        await _loop(None)

    _dump()  # final flush
    return save_path


async def run_evaluation(path: str, dataset_name: str, result_dir: str):
    metrics = evaluate_path(Path(path))
    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2)
    return metrics


def apply_environment_config(raw_yaml: dict) -> None:
    """Inject credentials at runtime so secrets never live in YAML."""
    api_key = os.environ.get("ARC_FUSE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ARC_FUSE_API_KEY is required for the real DIGIMON backend."
        )

    llm = raw_yaml.setdefault("llm", {})
    llm["api_key"] = api_key

    base_url = os.environ.get("ARC_FUSE_BASE_URL", "").strip()
    model = os.environ.get("ARC_FUSE_MODEL", "").strip()
    if base_url:
        llm["base_url"] = base_url
    if model:
        llm["model"] = model


# ===========================================================================
# Main
# ===========================================================================

async def main(args):
    opt_path = Path(args.opt)
    raw_yaml = YamlModel.read_yaml(opt_path)
    apply_environment_config(raw_yaml)

    # Tag the text method into raw_yaml so exp_name includes it
    raw_yaml["_text_method_tag"] = args.text_method

    # Extract ablation tag from YAML filename.
    abl_tag = opt_path.stem.replace("arc_fuse", "").strip("_")
    if abl_tag:
        raw_yaml["_abl_tag"] = abl_tag

    # Build graph config & bring up DIGIMON instance
    graph_cfg = build_graph_config(raw_yaml, args.graph_method, args.dataset_name)

    result_dir = check_dirs(graph_cfg.working_dir, graph_cfg.exp_name, opt_path)

    arc_fuse_cfg = raw_yaml.get("arc_fuse", {})
    logger.info("=" * 60)
    logger.info("ARC-Fuse: Adaptive Routing and Critic-guided Evidence Fusion")
    logger.info("=" * 60)
    logger.info(f"  graph_method  = {args.graph_method}")
    logger.info(f"  text_method   = {args.text_method}")
    logger.info(f"  dataset       = {args.dataset_name}")
    logger.info(f"  max_rounds    = {arc_fuse_cfg.get('max_rounds', 3)}")
    logger.info(f"  modules       = critic={arc_fuse_cfg.get('use_critic')}, "
                f"commendor={arc_fuse_cfg.get('use_commendor')}, "
                f"norm={arc_fuse_cfg.get('use_normalizer')}, "
                f"disambig={arc_fuse_cfg.get('use_disambiguation')}")
    logger.info("=" * 60)

    # Load dataset + insert corpus
    query_dataset = RAGQueryDataset(data_dir=args.dataset_name)
    corpus = query_dataset.get_corpus()

    digimon = GraphRAG(config=graph_cfg)
    await digimon.insert(corpus)

    # Build text retriever
    text_retriever = await build_text_retriever(args.text_method, digimon, raw_yaml)

    # Try to fish out entity VDB for disambiguation
    entities_vdb = None
    try:
        entities_vdb = digimon.retriever_context.entities_vdb
    except Exception:
        pass

    engine = ArcFuseEngine(
        graph_querier=digimon._querier,
        text_retriever=text_retriever,
        entities_vdb=entities_vdb,
        llm=digimon.llm,
        config=arc_fuse_cfg,
    )

    # Run queries
    save_path = await run_query_loop(query_dataset, engine, result_dir, args.eval_limit)

    # Evaluate
    await run_evaluation(save_path, args.dataset_name, result_dir)

    # Print stats
    try:
        cs = get_commendor_stats()
        rs = get_route_stats()
        fs = get_feedback_stats()
        logger.info("=" * 50)
        logger.info("ARC-Fuse Statistics")
        logger.info(f"  Commendor decisions: {cs}")
        logger.info(f"  Selection counts   : {rs}")
        logger.info(f"  Critic verdicts    : {fs}")
        logger.info("=" * 50)
    except Exception as e:
        logger.warning(f"Stats print failed: {e}")


def main_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="ARC-Fuse paper experiments on the DIGIMON backend"
    )
    parser.add_argument("-opt", required=True, help="Path to ARC-Fuse YAML")
    parser.add_argument("-graph_method", required=True,
                        choices=list(GRAPH_METHOD_TO_SECTION.keys()),
                        help="Graph retrieval method")
    parser.add_argument("-text_method", required=True,
                        choices=list(TEXT_METHODS),
                        help="Text retrieval method")
    parser.add_argument("-dataset_name", required=True, help="Dataset path")
    parser.add_argument("-eval_limit", "--eval_limit", type=int, default=0,
                        help="Max queries to evaluate (0 = full dataset)")
    args = parser.parse_args(argv)
    asyncio.run(main(args))


if __name__ == "__main__":
    main_cli()
