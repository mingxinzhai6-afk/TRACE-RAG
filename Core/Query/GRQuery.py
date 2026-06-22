from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
from typing import Union
import asyncio
# from torch_geometric.data import Data, InMemoryDataset
from typing import Any, Dict, List, Tuple, no_type_check
from pcst_fast import pcst_fast
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from Core.Common.Utils import truncate_str_by_token_size


class GRQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def initialization(self):
        from Core.Common.Constants import GRAPH_FIELD_SEP
        origin_nodes = await self._retriever.retrieve_relevant_content(type=Retriever.ENTITY,
                                                                 mode="get_all") # list[dict]
        origin_edges = await self._retriever.retrieve_relevant_content(type=Retriever.RELATION,
                                                                 mode="get_all") # list[dict]
        relations = list(map(lambda x: x["relation_name"].split(sep=GRAPH_FIELD_SEP), origin_edges)) # list[list]
        document_graph_triplets = [] # list[tuple]
        for index, edge in enumerate(origin_edges):
            for rel in relations[index]:
                document_graph_triplets.append((edge["src_id"], rel, edge["tgt_id"]))

        raw_nodes: Dict[str, int] = {}
        raw_edges = []

        for node in origin_nodes:
            if node["entity_name"] not in raw_nodes:
                raw_nodes[node["entity_name"]] = len(raw_nodes)

        for tri in document_graph_triplets:
            h, r, t = tri
            if h not in raw_nodes:
                raw_nodes[h] = len(raw_nodes)
            if t not in raw_nodes:
                raw_nodes[t] = len(raw_nodes)
            raw_edges.append({
                "src": raw_nodes[h],
                "edge_attr": r,
                "dst": raw_nodes[t]
            })

        nodes = pd.DataFrame([{ "node_id": v, "node_attr": k} for k, v in raw_nodes.items()],
                             columns=["node_id", "node_attr"])
        edges = pd.DataFrame(raw_edges,
                             columns=["src", "edge_attr", "dst"])

        nodes.node_attr = nodes.node_attr.fillna("")

        edge_index = torch.tensor([
            edges.src.tolist(),
            edges.dst.tolist(),
        ], dtype=torch.long)

        self.edge_index = edge_index # torch[2, -1]
        self.nodes = nodes # pandas: "node_id": int, "node_attr": str
        self.edges = edges # pandas: "src":int,  "edge_attr":str,  "dst": int
        self.raw_nodes = raw_nodes # dict: key: "node_attr": str, "node_id": int
        self.edges_list = relations # list[str]:  "edge_attr":str



    async def retrieval_via_pcst(
            self,
            query: str,
            topk: int = 3,
            topk_e: int = 3,
            cost_e: float = 0.5,
    ):
        c = 0.01
        if len(self.nodes) == 0 or len(self.edges) == 0:
            desc = self.nodes.to_csv(index=False) + "\n" + self.edges.to_csv(
                index=False,
                columns=["src", "edge_attr", "dst"],
            )
            return desc

        root = -1
        num_clusters = 1
        pruning = 'gw'
        verbosity_level = 0
        if topk > 0:
            topk = min(topk, len(self.nodes))
            retrieve_entity = await self._retriever.retrieve_relevant_content(type=Retriever.ENTITY, mode="vdb", seed=query) # list[dict]
            n_prizes = torch.zeros(len(self.nodes))
            if retrieve_entity:
                retrieve_entity = [e for e in retrieve_entity if e is not None and e["entity_name"] in self.raw_nodes]
                n_ent = min(len(retrieve_entity), topk)
                if n_ent > 0:
                    retrieve_entity_id = torch.tensor(
                        list(map(lambda x: self.raw_nodes[x["entity_name"]], retrieve_entity[:n_ent]))
                    )
                    n_prizes[retrieve_entity_id] = torch.arange(n_ent, 0, -1).float()
        else:
            n_prizes = torch.zeros(len(self.nodes))

        if topk_e > 0:
            topk_e = min(topk_e, len(self.edges))
            retrieve_relations, topk_e_values = await self._retriever.retrieve_relevant_content(type=Retriever.RELATION, mode="vdb", seed=query,
                                                                need_score=True, need_context=False)
            e_prizes = torch.zeros(len(self.edges))
            n_scores = len(topk_e_values) if topk_e_values is not None else 0
            if retrieve_relations and n_scores > 0:
                n_pairs = min(len(retrieve_relations), n_scores)
                for i in range(n_pairs):
                    if retrieve_relations[i] is None:
                        continue
                    index = self.edges[
                        (self.edges['src'] == retrieve_relations[i]["src_id"]) &
                        (self.edges['edge_attr'] == retrieve_relations[i]["relation_name"]) &
                        (self.edges['dst'] == retrieve_relations[i]['tgt_id'])
                    ].index
                    e_prizes[index] = topk_e_values[i]

                # topk_e may exceed len(topk_e_values) when the retriever returns fewer hits
                topk_e_loop = min(topk_e, n_scores)
                last_topk_e_value = topk_e
                for k in range(topk_e_loop):
                    indices = e_prizes == topk_e_values[k]
                    num_match = int(indices.sum().item())
                    if num_match == 0:
                        continue
                    value = min((topk_e - k) / num_match, last_topk_e_value - c)
                    e_prizes[indices] = value
                    last_topk_e_value = value * (1 - c)
            # reduce the cost of the edges such that at least one edge is selected
            if e_prizes.max().item() > 0:
                cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
        else:
            e_prizes = torch.zeros(len(self.edges))

        costs = []
        edges = []
        virtual_n_prizes = []
        virtual_edges = []
        virtual_costs = []
        mapping_n = {}
        mapping_e = {}
        for i, (src, dst) in enumerate(self.edge_index.t().numpy()):
            prize_e = e_prizes[i]
            if prize_e <= cost_e:
                mapping_e[len(edges)] = i
                edges.append((src, dst))
                costs.append(cost_e - prize_e)
            else:
                virtual_node_id = len(self.nodes) + len(virtual_n_prizes)
                mapping_n[virtual_node_id] = i
                virtual_edges.append((src, virtual_node_id))
                virtual_edges.append((virtual_node_id, dst))
                virtual_costs.append(0)
                virtual_costs.append(0)
                virtual_n_prizes.append(prize_e - cost_e)

        prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
        num_edges = len(edges)
        if len(virtual_costs) > 0:
            costs = np.array(costs + virtual_costs)
            edges = np.array(edges + virtual_edges)

        vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters,
                                    pruning, verbosity_level)

        selected_nodes = vertices[vertices < len(self.nodes)]
        selected_edges = [mapping_e[e] for e in edges if e < num_edges]
        virtual_vertices = vertices[vertices >= len(self.nodes)]
        if len(virtual_vertices) > 0:
            virtual_vertices = vertices[vertices >= len(self.nodes)]
            virtual_edges = [mapping_n[i] for i in virtual_vertices]
            selected_edges = np.array(selected_edges + virtual_edges)

        edge_index = self.edge_index[:, selected_edges]
        selected_nodes = np.unique(
            np.concatenate(
                [selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

        n = self.nodes.iloc[selected_nodes]
        e = self.edges.iloc[selected_edges]
        # Map integer node_id to node_attr so LLM sees readable triplets
        id_to_name = dict(zip(n['node_id'], n['node_attr']))
        # Build readable triplets: "subject - relation - object"
        triplets = []
        for _, row in e.iterrows():
            src_name = id_to_name.get(row['src'], str(row['src']))
            dst_name = id_to_name.get(row['dst'], str(row['dst']))
            triplets.append(f"{src_name} - {row['edge_attr']} - {dst_name}")
        # Entity list + triplets
        entities = list(n['node_attr'].unique())
        desc = "Entities: " + ", ".join(entities) + "\n"
        desc += "Relations:\n" + "\n".join(triplets)

        return desc


    async def _retrieve_relevant_contexts(self, query: str):
        """Return only the subgraph desc (string), so Critic loop gets a clean context."""
        formatted_query = f"Question: {query}\nAnswer: "
        desc = await self.retrieval_via_pcst(
            query=formatted_query,
            topk=self.config.top_k,
            topk_e=self.config.topk_e,
            cost_e=self.config.cost_e,
        )
        desc = truncate_str_by_token_size(input_str=desc, max_token_size=self.config.max_txt_len)
        return desc

    async def query(self, query):
        await self.initialization()

        context = await self._retrieve_relevant_contexts(query)
        logger.debug(f"GR subgraph context:\n{context}")
        formatted_query = f"Question: {query}\nAnswer: "
        response = await self.generation_qa(formatted_query, context)

        # Agentic innovations (same as BaseQuery.query)
        if getattr(self.config, 'use_critic', False) and response:
            response, context = await self._critic_loop(query, context, response)
        if getattr(self.config, 'use_answer_normalizer', False) and response:
            response = await self._normalize_answer(query, response)

        return response

    async def generation_qa(self, query: str, context: str):
        system = (
            "You are given a knowledge subgraph with entities and relations as triplets "
            "(subject - relation - object). Use these triplets to answer the question.\n"
            "Rules:\n"
            "- Extract the answer from the RELATION or the OBJECT of the relevant triplet.\n"
            "- Do NOT output the person's name as the answer — output the PROPERTY being asked about "
            "(e.g., occupation, nationality, birthplace).\n"
            "- After 'Answer: ', write ONLY the short factual answer (1-3 words, no explanation).\n"
            "- If the subgraph lacks a direct answer, make your best guess based on the available triplets and general knowledge. Never output 'unknown'."
        )
        response = await self.llm.aask(
            msg=context + query,
            system_msgs=[system],
        )
        return response

    async def generation_summary(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE