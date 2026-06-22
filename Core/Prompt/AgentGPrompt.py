"""
Agent-G Prompts (ICLR 2025)

Implements the prompt templates for the Agent-G framework:
- Agent: entity/relation extraction + retrieval module selection
- Agent (reflection): refine action based on critic feedback
- Generator: CoT-based answer generation
- Validator: binary check if answer is correct
- Commentor: corrective feedback on agent's action
"""

# ============================================================
# Agent Prompts
# ============================================================

AGENT_EXTRACT_ACTION_PROMPT = """You are a helpful, pattern-following assistant. Given the following question, extract the information from the question as requested.

Rules:
1. Extract topic entities that are central to answering the question.
2. Extract useful relations that connect these entities.
3. Each entity should have a category in parentheses (e.g., "Albert Einstein (person)").
4. Based on the extracted entities and relations, decide whether a knowledge graph or text documents would be more helpful to answer the question.

Examples:
Question: What is the occupation of Albert Einstein?
Topic Entities: Albert Einstein (person)
Useful Relations: person has occupation
Selection: knowledge graph

Question: What genre of music does Taylor Swift perform?
Topic Entities: Taylor Swift (person)
Useful Relations: person performs genre
Selection: knowledge graph

Question: What is nanofluid heat transfer?
Topic Entities: nanofluid heat transfer (concept)
Useful Relations: none
Selection: text documents

Question: Who wrote the novel "1984"?
Topic Entities: 1984 (literary work)
Useful Relations: person writes literary work
Selection: knowledge graph

Question: What country is the Eiffel Tower located in?
Topic Entities: Eiffel Tower (landmark)
Useful Relations: landmark located in country
Selection: knowledge graph

Now extract the information for the following question.

Question: {question}

Output your answer in the following format exactly:
Topic Entities: <comma-separated list of entities with categories>
Useful Relations: <comma-separated list of relations, or "none">
Selection: <"knowledge graph" or "text documents">"""


AGENT_REFLECT_PROMPT = """The retrieved reference was not sufficient to correctly answer the question.

Previous action:
- Topic Entities: {prev_entities}
- Useful Relations: {prev_relations}
- Selection: {prev_selection}

Feedback from critic: {feedback}

Question: {question}

Based on this feedback, please re-extract the topic entities and useful relations, and decide whether to use knowledge graph or text documents. You may need to:
- Remove or substitute incorrect entities/relations
- Add missing entities that were overlooked
- Switch the retrieval module if the current one is not helpful

Output your answer in the following format exactly:
Topic Entities: <comma-separated list of entities with categories>
Useful Relations: <comma-separated list of relations, or "none">
Selection: <"knowledge graph" or "text documents">"""


# ============================================================
# Generator Prompt
# ============================================================

GENERATOR_COT_PROMPT = """You are a helpful, pattern-following assistant.

### Reference: {reference}
### Reference Source: {reference_source}
### Question: {question}

### Task: First, list the key facts from the reference that are relevant to answering the question. Then, based on these facts, provide a concise final answer.

Think step by step, then conclude with:
Final Answer: <your concise answer in 1-5 words>"""


GENERATOR_SHORT_FORM_PROMPT = """You are a precise question-answering assistant.

### Reference: {reference}
### Reference Source: {reference_source}
### Question: {question}

### Task: Answer the question based on the reference. Output ONLY the single most concise factual answer (1-5 words maximum). No explanations, no full sentences.

If the reference does not contain the answer, output: unknown

Answer:"""


# ============================================================
# Validator Prompt
# ============================================================

VALIDATOR_PROMPT = """You are a helpful, pattern-following assistant.

### Reference: {reference}
### Answer: {answer}
### Question: {question}

### Task: Based on the reference, does the answer correctly and sufficiently address the question? Consider:
1. Is the answer factually supported by the reference?
2. Does the answer directly address what the question is asking?
3. Is the answer complete (not missing key information)?

Reply with only "yes" or "no"."""


# ============================================================
# Commentor Prompt
# ============================================================

COMMENTOR_PROMPT = """You are a helpful, pattern-following assistant. Your job is to identify errors in the agent's extracted entities and relations, and provide corrective feedback.

Examples of action and feedback pairs:

Example 1:
Question: What papers have been published by researchers at MIT about neural networks?
Topic Entities: MIT (institution), neural networks (field of study)
Useful Relations: researcher affiliated with institution, researcher writes paper, paper has topic field of study
Feedback: The entities and relations look correct. No changes needed.

Example 2:
Question: What is the occupation of the person who wrote "Harry Potter"?
Topic Entities: Harry Potter (literary work), occupation (attribute)
Useful Relations: person writes literary work
Feedback: Entity "occupation (attribute)" is incorrect — it is not a topic entity but rather the property being asked about. Please remove it and keep only "Harry Potter (literary work)". Also, the relation should include "person has occupation".

Example 3:
Question: Who directed the movie "Inception"?
Topic Entities: Inception (movie), directed (relation)
Useful Relations: person directs movie
Feedback: Entity "directed (relation)" is incorrect — "directed" is a relation, not an entity. Please remove it. The entity should be "Inception (movie)" only.

Example 4:
Question: What genre does the band Radiohead play?
Topic Entities: Radiohead (band)
Useful Relations: band plays genre
Selection: text documents
Feedback: The selection "text documents" may not be optimal. Since "Radiohead" is an entity with structured relations in the knowledge graph, "knowledge graph" would be more helpful to narrow down the search space.

Example 5:
Question: What is the capital of France?
Topic Entities: France (country)
Useful Relations: country has capital
Feedback: The entities and relations look correct. No changes needed.

Now analyze the following:

Question: {question}
Topic Entities: {entities}
Useful Relations: {relations}
Selection: {selection}
The answer generated was incorrect.

Please point out what is wrong with the extracted entities, relations, or the selection of retrieval module. Provide specific corrective feedback."""
