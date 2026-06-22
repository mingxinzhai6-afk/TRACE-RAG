"""
Build graph index only (no query), for pre-building before experiments.

Usage:
    python build_graph.py -opt Option/Method/HippoRAG.yaml -dataset_name datasets/MuSiQue
    python build_graph.py -opt Option/Method/ToG.yaml      -dataset_name datasets/MuSiQue
    python build_graph.py -opt Option/Method/RAPTOR.yaml   -dataset_name datasets/MuSiQue
"""

import asyncio
import argparse
from pathlib import Path

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
from Data.QueryDataset import RAGQueryDataset
from Core.Common.Logger import logger


async def main(opt_path, dataset_name):
    opt = Config.parse(Path(opt_path), dataset_name=dataset_name)
    digimon = GraphRAG(config=opt)
    query_dataset = RAGQueryDataset(data_dir=opt.dataset_name)
    corpus = query_dataset.get_corpus()
    logger.info(f"Building graph for {dataset_name} using {opt_path} ...")
    await digimon.insert(corpus)
    logger.info("Graph build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", required=True)
    parser.add_argument("-dataset_name", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.opt, args.dataset_name))
