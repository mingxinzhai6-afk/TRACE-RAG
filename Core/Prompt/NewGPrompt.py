"""
NewG Prompts — unified agentic Graph+Text RAG framework.

Prompts for each module:
  - QueryUnderstanding  : extract action a_t = {d_t, E_t, R_t, s_t}
  - AdaptiveRouter      : map action to retriever selection
  - EntityDisambiguation: alias resolution via LLM (optional)
  - EvidenceFusion      : build candidate answer from fused evidence
  - ReGenerationAgent   : CoT generation with retry guidance
  - Commendor           : 3-way decision (Wrong Retriever / Insufficient / Poor Gen)
"""


# ============================================================
# Query Understanding: extract {domain, entities, relations, selection}
# ============================================================

QUERY_UNDERSTANDING_PROMPT = """You are a query understanding assistant.

Analyze the question once and output a routing triplet for NewG.

Routing policy:
- graph: the question mainly asks for entity attributes or explicit relations between named entities.
- text: the question mainly needs paragraph-level background, explanation, or descriptive context.
- hybrid: both relation reasoning and paragraph/background evidence are needed.

Guidelines:
1. Domain should be one short word such as biography, history, geography, movie, music, science, finance, sports, general.
2. Entities should be the key entities or concepts, annotated with a category in parentheses.
3. Relations are optional but helpful. Use concise relation phrases. If no clear relation exists, return an empty list.
4. Prefer hybrid for compositional questions such as:
   - "the X of the country where Y ..."
   - "the same state as ..."
   - "named after ..."
   - questions that require both identifying an entity and then reading background facts about it.

Examples:

Question: What is Henry Feilden's occupation?
Output:
{{"domain":"biography","entities":["Henry Feilden (person)"],"relations":["person has occupation"],"selection":"graph"}}

Question: What is nanofluid heat transfer?
Output:
{{"domain":"science","entities":["nanofluid heat transfer (concept)"],"relations":[],"selection":"text"}}

Question: What papers on deep learning has Geoffrey Hinton published?
Output:
{{"domain":"science","entities":["Geoffrey Hinton (person)","deep learning (field)"],"relations":["person wrote paper","paper has topic"],"selection":"hybrid"}}

Question: In what county of the state where the majority of sweet corn is grown is Shady Grove?
Output:
{{"domain":"geography","entities":["sweet corn (crop)","Shady Grove (place)"],"relations":["crop mainly grown in state","place located in county"],"selection":"hybrid"}}

Question: {question}

Return ONLY valid JSON in this schema:
{{"domain":"<word>","entities":["<entity (type)>"],"relations":["<relation>"],"selection":"graph|text|hybrid"}}"""


QUERY_UNDERSTANDING_REFLECT_PROMPT = """Your previous routing action did not yield a correct answer. Refine it based on feedback.

Question: {question}

Previous action:
- Domain: {prev_domain}
- Topic Entities: {prev_entities}
- Useful Relations: {prev_relations}
- Selection: {prev_selection}

Feedback from critic: {feedback}

Re-extract the action, applying the feedback. You may:
- Remove or substitute incorrect entities/relations
- Add overlooked entities
- Switch the selection if the current retriever is not helpful

Return ONLY valid JSON in this schema:
{{"domain":"<word>","entities":["<entity (type)>"],"relations":["<relation>"],"selection":"graph|text|hybrid"}}"""


# ============================================================
# Entity Disambiguation (optional LLM-based alias resolution)
# ============================================================

ENTITY_DISAMBIGUATION_PROMPT = """You are an entity disambiguation assistant.

Given a query entity and a list of candidate entities from the knowledge base,
select the candidates that most likely refer to the SAME real-world entity as
the query. Consider aliases, abbreviations, and spelling variants.

Query entity: {query_entity}
Candidates:
{candidates}

Output a comma-separated list of indices (0-based) of the matching candidates,
or "none" if no candidate matches. Examples:
- "0" (first candidate matches)
- "0, 2" (first and third match)
- "none"

Indices:"""


# ============================================================
# Evidence Fusion & Candidate Generator
# ============================================================

EVIDENCE_FUSION_CANDIDATE_PROMPT = """You are a question-answering assistant.

Based on the retrieved evidence below, generate an initial candidate answer.
Be concise and factual. If the evidence is insufficient, reply exactly: unknown

### Question: {question}

### Evidence:
{evidence}

### Instructions:
1. Identify key facts in the evidence relevant to the question.
2. Produce a candidate answer in 1-5 words (short-form factoid).
3. If not answerable from the evidence, reply: unknown

Candidate Answer:"""


# ============================================================
# Re-Generation Agent (CoT with retry)
# ============================================================

REGENERATION_COT_PROMPT = """You are a precise question-answering assistant. Use chain-of-thought reasoning.

### Question: {question}

### Evidence:
{evidence}

### Previous candidate answer: {candidate}

### Task:
1. List the key facts from evidence relevant to the question.
2. Verify whether the candidate answer is correct.
3. If correct, confirm. If not, derive a better answer.
4. Conclude with the final answer in 1-5 words.

Reasoning, then:
Final Answer: <1-5 word factoid>"""


REGENERATION_RETRY_PROMPT = """Your previous answer was rejected. Re-generate with the feedback below.

### Question: {question}

### Evidence:
{evidence}

### Previous answer: {prev_answer}

### Feedback: {feedback}

Re-generate a better answer. Think step by step, then conclude:
Final Answer: <1-5 word factoid>"""


# ============================================================
# Commendor (3-way decision)
# ============================================================

COMMENDOR_PROMPT = """You are a diagnostic assistant that categorizes the cause of an incorrect answer.

Given the question, the action taken (retrieval selection + extracted entities/relations),
the retrieved evidence, and the generated answer, determine WHY the answer is unsatisfactory.

Categorize the error into EXACTLY ONE of:
1. wrong_retriever: The selected retrieval module (graph/text/hybrid) is inappropriate for this question.
   Signs: evidence is irrelevant in content/form; selection does not match question type.
2. insufficient_evidence: The retriever is correct, but the retrieved evidence does not contain
   enough information to answer. More/broader retrieval is needed (missing entities, incomplete paths).
3. poor_generation: The evidence clearly contains the answer OR the answer contradicts the evidence,
   but the generator mis-used, hallucinated, or failed to extract it. A re-generation with the SAME evidence should fix it.
   IMPORTANT: If the evidence mentions the correct entity/answer but the generated answer is different or wrong, prioritize poor_generation over insufficient_evidence.
   Signs: evidence contains keywords from the question/answer; evidence is relevant but answer mismatches.
4. pass: The answer is actually correct — do not retry.

PRIORITY RULE: If the evidence is clearly relevant and contains information related to the question,
prefer "poor_generation" over "insufficient_evidence". Only choose insufficient_evidence when the evidence is truly sparse or off-topic.

### Question: {question}

### Action taken:
- Selection: {selection}
- Topic Entities: {entities}
- Useful Relations: {relations}

### Retrieved Evidence (first 2000 chars):
{evidence}

### Generated Answer: {answer}

### Critic's prior evaluation: {critic_feedback}

Output JSON exactly in this form:
{{
  "decision": "wrong_retriever" | "insufficient_evidence" | "poor_generation" | "pass",
  "confidence": 0.0 to 1.0,
  "reason": "<one-sentence rationale>",
  "hint": "<short guidance for the next step, e.g., 'switch to text retriever' or 'look up entity X's birth year'>"
}}"""


# ============================================================
# Multi-agent Judge Framework prompts
# ============================================================

JUDGE_PROMPT = """You are Judge {judge_id} in a multi-agent QA evaluation system.

Given the question and retrieved evidence, independently evaluate the evidence quality and generate a candidate answer.

### Question: {question}

### Evidence:
{evidence}

### Your task:
Score (0-10): How well does the evidence support answering this question?
  0-3: Evidence is irrelevant or missing key information
  4-6: Evidence partially supports the answer
  7-10: Evidence clearly contains the answer
Reason: What specific information is present or missing?
Candidate: Your best factoid answer (1-5 words), or "unknown" if evidence is insufficient.

Output JSON exactly:
{{
  "score": <0-10>,
  "reason": "<one sentence about evidence quality>",
  "candidate": "<1-5 word answer or unknown>"
}}"""


VOTING_PROMPT = """You are Voting Agent {voter_id} in a multi-agent QA system.

Multiple judges have independently evaluated the evidence. Review all their assessments and vote for the best answer.

### Question: {question}

### Evidence:
{evidence}

### Judge Assessments:
{judge_outputs}

### Aggregated Evidence Score: {aggregated_score:.2f}/10
### Merged Reasoning: {merged_reason}

Vote for the most accurate answer based on the judges' evaluations and the evidence.

Output JSON exactly:
{{
  "vote": "<the answer you vote for, 1-5 words>",
  "reason": "<one sentence explaining your vote>"
}}"""


INFERENCE_COT_PROMPT = """You are the final Inference Agent in a multi-agent QA system.

Multiple judges evaluated the evidence and voters reached a consensus. Perform final chain-of-thought reasoning to output the definitive answer.

### Question: {question}

### Evidence:
{evidence}

### Evidence Quality (aggregated score: {aggregated_score:.2f}/10):
{merged_reason}

### Voting Results:
{vote_summary}

### Preliminary winning answer: {winning_answer}

Review the evidence and voting consensus, then reason step by step.

Final Answer: <1-5 word factoid>"""


# ============================================================
# Adaptive Router (lightweight mapping, mostly pass-through)
# ============================================================

ADAPTIVE_ROUTER_PROMPT = """Given a question and its extracted structure, choose the best retrieval module.

Question: {question}
Domain: {domain}
Topic Entities: {entities}
Useful Relations: {relations}
Prior selection: {prev_selection}

Rules:
- If entities are clear AND relations exist → "graph"
- If entities are vague OR question is descriptive → "text"
- If BOTH relational and textual aspects present → "hybrid"
- If prior was "graph" and failed → prefer "text" or "hybrid"
- If prior was "text" and failed → prefer "graph" or "hybrid"

Output one word only: graph | text | hybrid"""
