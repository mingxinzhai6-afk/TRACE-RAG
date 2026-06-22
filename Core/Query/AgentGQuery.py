"""
Agent-G: An Agentic Framework for Graph Retrieval Augmented Generation
(ICLR 2025, Paper 3154)

Implements Algorithm 1 from the paper:
  for t = 1..T:
      a_t = Agent(q, f_{t-1})           # extract entities/relations, select module
      X_t = RetrieverBank(q, a_t, D)    # text or graph retrieval
      y_t = Generator(q, X_t)           # CoT answer generation
      if Validator(q, y_t, X_t):        # accept?
          return y_t
      f_t = Commentor(q, a_t)           # corrective feedback
  return y_T

Adapted to run on the DIGIMON framework with existing er_graph.
"""

import re
import asyncio
from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever, GRAPH_FIELD_SEP
from Core.Prompt import QueryPrompt
from Core.Prompt.AgentGPrompt import (
    AGENT_EXTRACT_ACTION_PROMPT,
    AGENT_REFLECT_PROMPT,
    GENERATOR_SHORT_FORM_PROMPT,
    GENERATOR_COT_PROMPT,
    VALIDATOR_PROMPT,
    COMMENTOR_PROMPT,
)


class AgentGQuery(BaseQuery):
    """
    Agent-G query implementation within the DIGIMON framework.

    Components:
      - Agent: LLM extracts topic entities, useful relations, selects retriever
      - Retriever Bank: graph retrieval (ego-graph + reasoning paths) OR
                        text retrieval (VDB similarity search)
      - Generator: CoT-based answer generation
      - Critic Module: Validator (binary yes/no) + Commentor (corrective feedback)
    """

    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)
        # Will be populated during initialization
        self._graph_initialized = False
        self._raw_nodes = {}       # entity_name -> node_id
        self._node_edges = {}      # entity_name -> list of (src, rel, tgt)

    # ------------------------------------------------------------------
    # Graph initialization (load all nodes/edges once for ego-graph extraction)
    # ------------------------------------------------------------------
    async def _init_graph(self):
        """Load graph structure for ego-graph extraction."""
        if self._graph_initialized:
            return

        origin_nodes = await self._retriever.retrieve_relevant_content(
            type=Retriever.ENTITY, mode="get_all"
        )
        if origin_nodes:
            for node in origin_nodes:
                name = node.get("entity_name", "")
                if name:
                    self._raw_nodes[name] = node

        self._graph_initialized = True
        logger.info(f"Agent-G: graph initialized with {len(self._raw_nodes)} nodes")

    # ------------------------------------------------------------------
    # Agent: extract action (entities, relations, selection)
    # ------------------------------------------------------------------
    async def _agent_extract_action(self, question: str, feedback: str = "") -> dict:
        """
        Agent determines action a_t = {entities, relations, selection}.
        If feedback is provided (iteration > 1), uses reflection prompt.
        """
        if feedback and hasattr(self, '_prev_action'):
            prompt = AGENT_REFLECT_PROMPT.format(
                prev_entities=self._prev_action.get("entities_str", ""),
                prev_relations=self._prev_action.get("relations_str", ""),
                prev_selection=self._prev_action.get("selection", ""),
                feedback=feedback,
                question=question,
            )
        else:
            prompt = AGENT_EXTRACT_ACTION_PROMPT.format(question=question)

        response = await self.llm.aask(msg=prompt)
        action = self._parse_agent_action(response)
        self._prev_action = action
        logger.info(
            f"Agent-G action: entities={action['entities']}, "
            f"relations={action['relations']}, selection={action['selection']}"
        )
        return action

    @staticmethod
    def _parse_agent_action(response: str) -> dict:
        """Parse agent LLM output into structured action dict."""
        entities_str = ""
        relations_str = ""
        selection = "knowledge graph"

        for line in response.strip().split("\n"):
            line_lower = line.strip().lower()
            if line_lower.startswith("topic entities:"):
                entities_str = line.split(":", 1)[1].strip()
            elif line_lower.startswith("useful relations:"):
                relations_str = line.split(":", 1)[1].strip()
            elif line_lower.startswith("selection:"):
                sel = line.split(":", 1)[1].strip().lower()
                if "text" in sel or "document" in sel:
                    selection = "text documents"
                else:
                    selection = "knowledge graph"

        # Extract entity names (strip category annotations like "(person)")
        entity_names = []
        if entities_str and entities_str.lower() != "none":
            # Split by comma, then strip "(category)" part
            for ent in entities_str.split(","):
                ent = ent.strip()
                # Remove "(category)" suffix
                ent_clean = re.sub(r'\s*\([^)]*\)\s*$', '', ent).strip()
                if ent_clean:
                    entity_names.append(ent_clean)

        # Extract relation names
        relation_names = []
        if relations_str and relations_str.lower() != "none":
            for rel in relations_str.split(","):
                rel = rel.strip()
                if rel:
                    relation_names.append(rel)

        return {
            "entities": entity_names,
            "relations": relation_names,
            "selection": selection,
            "entities_str": entities_str,
            "relations_str": relations_str,
        }

    # ------------------------------------------------------------------
    # Retriever Bank
    # ------------------------------------------------------------------
    async def _retriever_bank(self, question: str, action: dict) -> tuple:
        """
        Retriever bank: select text or graph retrieval based on agent action.
        Returns (reference_text, reference_source).
        """
        if action["selection"] == "text documents":
            ref = await self._text_retrieval(question, action)
            return ref, "text documents"
        else:
            ref = await self._graph_retrieval(question, action)
            return ref, "knowledge graph"

    async def _graph_retrieval(self, question: str, action: dict) -> str:
        """
        Graph retrieval module:
        1. Find topic entities in the graph via VDB
        2. Extract ego-graph (1-2 hop neighborhood)
        3. Verbalize as reasoning paths
        """
        topic_entities = action["entities"]
        if not topic_entities:
            # Fallback: use VDB entity search with the question
            return await self._fallback_vdb_retrieval(question)

        # Step 1: Link topic entities to graph entities via VDB
        linked_entities = []
        for ent in topic_entities:
            try:
                nodes = await self._retriever.retrieve_relevant_content(
                    type=Retriever.ENTITY, mode="vdb", seed=ent, top_k=2
                )
                if nodes:
                    for n in nodes:
                        if n and n.get("entity_name"):
                            linked_entities.append(n["entity_name"])
            except Exception as e:
                logger.warning(f"Agent-G: entity linking failed for '{ent}': {e}")

        if not linked_entities:
            return await self._fallback_vdb_retrieval(question)

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for e in linked_entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)
        linked_entities = unique_entities[:6]  # limit to avoid too large ego-graphs

        # Step 2: Extract ego-graph (radius <= 2)
        reasoning_paths = []
        visited_edges = set()

        for entity_name in linked_entities:
            try:
                edges = await self._retriever.retrieve_relevant_content(
                    type=Retriever.RELATION, mode="from_entity",
                    seed=[{"entity_name": entity_name}]
                )
                if not edges:
                    continue

                for edge in edges[:8]:  # limit edges per entity
                    src = edge.get("src_tgt", ("", ""))[0] if "src_tgt" in edge else edge.get("src_id", "")
                    tgt = edge.get("src_tgt", ("", ""))[1] if "src_tgt" in edge else edge.get("tgt_id", "")
                    desc = edge.get("description", "")
                    rel_name = edge.get("relation_name", desc)

                    edge_key = (src, tgt)
                    if edge_key in visited_edges:
                        continue
                    visited_edges.add(edge_key)
                    visited_edges.add((tgt, src))

                    if src and tgt:
                        path = f"{src} -> {rel_name} -> {tgt}"
                        reasoning_paths.append(path)

            except Exception as e:
                logger.warning(f"Agent-G: ego-graph extraction failed for '{entity_name}': {e}")

        if not reasoning_paths:
            return await self._fallback_vdb_retrieval(question)

        # Step 3: Find intersection if multiple topic entities
        # (Paper: if multiple ego-graphs, extract their intersection)
        # For simplicity, we keep all paths but rank by relevance
        reference = "Knowledge Graph Reasoning Paths:\n"
        reference += "\n".join(reasoning_paths[:20])  # limit total paths

        # Also add the linked entity descriptions if available
        entity_descs = []
        for ent_name in linked_entities[:4]:
            node_data = self._raw_nodes.get(ent_name, {})
            desc = node_data.get("description", "") if isinstance(node_data, dict) else ""
            if desc:
                entity_descs.append(f"- {ent_name}: {desc}")

        if entity_descs:
            reference += "\n\nEntity Descriptions:\n" + "\n".join(entity_descs)

        return reference

    async def _text_retrieval(self, question: str, action: dict) -> str:
        """
        Text retrieval module: use VDB similarity search on entities/chunks.
        """
        try:
            # Try entity VDB search
            node_datas = await self._retriever.retrieve_relevant_content(
                type=Retriever.ENTITY, mode="vdb", seed=question
            )
            if not node_datas:
                return "No relevant text documents found."

            # Get associated text units
            text_units = await self._retriever.retrieve_relevant_content(
                type=Retriever.CHUNK, mode="entity_occurrence",
                node_datas=node_datas
            )

            if text_units:
                passages = text_units[:5]
                return "Retrieved Text Documents:\n" + "\n\n".join(
                    str(p) for p in passages
                )
            else:
                # Fallback: use entity descriptions
                descs = []
                for n in node_datas[:5]:
                    name = n.get("entity_name", "")
                    desc = n.get("description", "")
                    if name:
                        descs.append(f"- {name}: {desc}")
                return "Retrieved Entity Information:\n" + "\n".join(descs)

        except Exception as e:
            logger.warning(f"Agent-G text retrieval failed: {e}")
            return "No relevant text documents found."

    async def _fallback_vdb_retrieval(self, question: str) -> str:
        """Fallback: direct VDB search when entity linking fails."""
        try:
            node_datas = await self._retriever.retrieve_relevant_content(
                type=Retriever.ENTITY, mode="vdb", seed=question
            )
            if not node_datas:
                return "No relevant information found."

            # Get relations from found entities
            edges = await self._retriever.retrieve_relevant_content(
                type=Retriever.RELATION, mode="from_entity", seed=node_datas[:4]
            )
            paths = []
            if edges:
                for edge in edges[:10]:
                    src = edge.get("src_tgt", ("", ""))[0]
                    tgt = edge.get("src_tgt", ("", ""))[1]
                    desc = edge.get("description", "")
                    if src and tgt:
                        paths.append(f"{src} -> {desc} -> {tgt}")

            if paths:
                return "Knowledge Graph Reasoning Paths:\n" + "\n".join(paths)
            else:
                descs = []
                for n in node_datas[:5]:
                    name = n.get("entity_name", "")
                    desc = n.get("description", "")
                    descs.append(f"- {name}: {desc}")
                return "Retrieved Entity Information:\n" + "\n".join(descs)

        except Exception as e:
            logger.warning(f"Agent-G fallback retrieval failed: {e}")
            return "No relevant information found."

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------
    async def _generate_answer(self, question: str, reference: str, source: str) -> str:
        """Generator with CoT prompting."""
        response_type = getattr(self.config, 'response_type', 'short-form')

        if response_type.lower() in ("short-form", "short_form", "short", "factual"):
            prompt = GENERATOR_SHORT_FORM_PROMPT.format(
                reference=reference,
                reference_source=source,
                question=question,
            )
        else:
            prompt = GENERATOR_COT_PROMPT.format(
                reference=reference,
                reference_source=source,
                question=question,
            )

        response = await self.llm.aask(msg=prompt)
        return response.strip()

    # ------------------------------------------------------------------
    # Validator
    # ------------------------------------------------------------------
    async def _validate(self, question: str, answer: str, reference: str) -> bool:
        """
        Validator LLM: binary classification — is the answer correct?
        Uses reference as validation context.
        """
        prompt = VALIDATOR_PROMPT.format(
            reference=reference[:3000],
            answer=answer,
            question=question,
        )
        response = await self.llm.aask(msg=prompt)
        result = response.strip().lower()
        is_valid = result.startswith("yes")
        logger.info(f"Agent-G Validator: {'accepted' if is_valid else 'rejected'} (raw: {result})")
        return is_valid

    # ------------------------------------------------------------------
    # Commentor
    # ------------------------------------------------------------------
    async def _comment(self, question: str, action: dict) -> str:
        """
        Commentor LLM: provide corrective feedback on the agent's action.
        Uses ICL with pre-collected examples.
        """
        prompt = COMMENTOR_PROMPT.format(
            question=question,
            entities=action.get("entities_str", ""),
            relations=action.get("relations_str", ""),
            selection=action.get("selection", ""),
        )
        response = await self.llm.aask(msg=prompt)
        logger.info(f"Agent-G Commentor feedback: {response[:200]}")
        return response.strip()

    # ------------------------------------------------------------------
    # Main query loop (Algorithm 1)
    # ------------------------------------------------------------------
    async def query(self, query: str):
        """
        Main Agent-G loop implementing Algorithm 1.
        """
        await self._init_graph()

        max_iterations = getattr(self.config, 'agent_g_max_iterations', 4)
        feedback = ""
        answer = ""
        reference = ""

        for t in range(1, max_iterations + 1):
            logger.info(f"Agent-G iteration {t}/{max_iterations}")

            # Step 1: Agent determines action
            action = await self._agent_extract_action(query, feedback)

            # Step 2: Retriever Bank retrieves reference
            reference, source = await self._retriever_bank(query, action)
            logger.info(f"Agent-G retrieved {len(reference)} chars from {source}")

            # Step 3: Generator produces answer
            answer = await self._generate_answer(query, reference, source)
            logger.info(f"Agent-G answer (iter {t}): {answer[:100]}")

            # Step 4: Validator checks answer
            if t < max_iterations:
                is_valid = await self._validate(query, answer, reference)
                if is_valid:
                    logger.info(f"Agent-G accepted answer at iteration {t}")
                    break

                # Step 5: Commentor provides feedback for next iteration
                feedback = await self._comment(query, action)
            # else: last iteration, return answer as-is

        # Extract final short answer if response contains CoT reasoning
        answer = self._extract_final_answer(answer)

        return answer

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        """Extract the concise final answer from a CoT response."""
        # Try to find "Final Answer:" or "Answer:" pattern
        for prefix in ["Final Answer:", "final answer:", "Answer:", "answer:"]:
            if prefix in response:
                ans = response.split(prefix)[-1].strip()
                # Take only the first line
                ans = ans.split("\n")[0].strip()
                # Remove trailing punctuation
                ans = ans.rstrip(".")
                if ans:
                    return ans
        # If no pattern found, return as-is (already short)
        return response.strip()

    # ------------------------------------------------------------------
    # Required abstract method implementations (not used in Agent-G flow)
    # ------------------------------------------------------------------
    async def _retrieve_relevant_contexts(self, query: str):
        """Not used directly — Agent-G has its own retrieval pipeline."""
        return await self._fallback_vdb_retrieval(query)

    async def generation_qa(self, query, context):
        """Not used directly — Agent-G has its own generation pipeline."""
        return await self._generate_answer(query, str(context), "knowledge graph")

    async def generation_summary(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        return await self._generate_answer(query, str(context), "knowledge graph")
