GLOBAL_MAP_RAG_POINTS = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1...", "score": score_value}},
        {{"description": "Description of point 2...", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}
"""

FAIL_RESPONSE = "Sorry, I'm not able to provide an answer to that question."

# Used for short-form factoid QA datasets (PopQA, HotpotQA, etc.)
# response_type: "short-form"
SHORT_FORM_QA_RESPONSE = """---Role---
You are a precise question-answering assistant.

---Goal---
Answer the question using ONLY the provided context. Follow the rules below strictly.

Rules:
- If the question starts with Does/Do/Is/Are/Was/Were/Did/Has/Have/Had → answer ONLY "Yes" or "No"
- If the question starts with Who/Which/What/Where/When → answer with the specific name or entity (e.g., "Sam Bankman-Fried", "OpenAI", "Google"), NOT a generic category like "entrepreneur" or "politician"
- If the context does not contain enough information to answer → output exactly: Insufficient information.
- Output ONLY the answer, no explanations, no punctuation (except for "Insufficient information.")
- Answer ONLY based on the provided context — do NOT use external knowledge

---Context---
{report_data}
"""

# Override the generic short-form prompt with a PopQA-compatible occupation rule
# while keeping entity/date/location behavior for other factoid datasets.
SHORT_FORM_QA_RESPONSE = """---Role---
You are a precise question-answering assistant.

---Goal---
Answer the question using ONLY the provided context. Follow the rules below strictly.

Rules:
- If the question asks for an occupation, profession, or job, output ONLY one generic occupation label, 1-3 words maximum.
  Examples: output "politician" not "Icelandic politician"; output "journalist" not "Honduran journalist and television reporter"; output "actor" not "Norwegian actor".
  If multiple occupations are listed, choose the most central or best-supported occupation label from the context.
  Do not include nationality, adjectives, explanations, or "and" combinations for occupation questions.
- If the question starts with Does/Do/Is/Are/Was/Were/Did/Has/Have/Had, answer ONLY "Yes" or "No".
- For other Who/Which/What/Where/When questions, answer with the specific name, entity, date, number, or location requested.
- If the context contains any clear clue for an occupation/profession/job, make the best short occupation-label guess instead of abstaining.
- If the context does not contain enough information to answer other question types, output exactly: Insufficient information.
- Output ONLY the answer, no explanations, no punctuation (except for "Insufficient information.")
- Answer ONLY based on the provided context; do NOT use external knowledge.

---Context---
{report_data}
"""

GLOBAL_REDUCE_RAG_RESPONSE = """---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


LOCAL_RAG_RESPONSE = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

KEYWORDS_EXTRACTION = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Retrieval: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Retrieval: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Retrieval: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Retrieval: {query}
######################
Output:

"""

#used for lightrag
RAG_RESPONSE = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{report_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""



IRCOT_REASON_INSTRUCTION = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'




def prompt_qac_wiki(context):
    context = '\n'.join(f'{i}: {c}' for i, c in enumerate(context, start=1))

    start = "Given the following question and contexts, create a final answer to the question."

    return start + '\n=========\n' + 'QUESTION: {question}' + '\n=========\n' + 'CONTEXT:\n' + context + '\n=========\n' + 'QUESTION: {question}' + '\n=========\n' + 'ANSWER: please answer less than 6 words.'

KGP_REASON_PROMPT = """
What evidence do we need to answer the question given the current evidence?"
Question: {question}
Evidence: {context}
"""

KGP_QUERY_PROMPT = """
"Given the following guestion and contexts, create a final answer to the question.
Question: {question}
Contexts: {context}
"""

# Used for FastGraphRAG 
GENERATE_RESPONSE_QUERY_WITH_REFERENCE = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

# INPUT DATA
{context}

# USER QUERY
{query}

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. While you write the response, you must include inline references to the all the sources you are using by appending `[<source_id>]` at the end of each sentence, where `source_id` is the corresponding source ID from the "Sources" list.
5. Write the response to the user query - which must include the inline references - based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

Answer:
"""


# ============================================================
# Agentic GraphRAG Prompts (Innovation Modules)
# ============================================================

# ---- 创新2: Answer Normalizer ----
ANSWER_NORMALIZER_PROMPT = """You are a factoid answer normalizer. Given a question and a raw (possibly verbose) answer, extract ONLY the single most precise factual answer that would match a benchmark ground truth label.

Strict Rules:
- If the raw answer is "Yes" or "No" (or a clear yes/no), output it unchanged
- If the raw answer indicates missing/unavailable information (e.g., "Insufficient information", "not enough context", "cannot be determined"), output exactly: Insufficient information.
- Output a concise answer, but preserve complete dates, age ranges, numeric ranges, ordinal rankings, and qualified place/organization names when they are required by the question
- Pick the term that MOST DIRECTLY and SPECIFICALLY answers the question (e.g., for "What is X's occupation?", prefer "composer" over the broader "musician"; prefer "novelist" over "writer")
- If the raw answer is a comma-separated list, choose the MOST SPECIFIC term that best matches what the question asks — do NOT default to the first term
- Do NOT reduce "July 1, 1984" to "1984", "18 - 20" to "18", "Camden County" to "Camden", or "United States dollar" to "dollar"
- NO articles (a/an/the), NO unnecessary adjectives, NO nationality prefixes unless they are part of the entity name
- NO "and" combinations — pick ONE primary label
- NO explanations, sentences, or punctuation (except "Insufficient information.")
- If the raw answer contains reasoning (e.g., "Thought: ... Answer: X"), extract only the final answer X
- If no useful information is present at all, output: Insufficient information.

Examples:
Question: What is Bruce McDaniel's occupation?
Raw Answer: musician, composer, producer, recording engineer
Normalized Answer: composer

Question: What is J.K. Rowling's occupation?
Raw Answer: She is a well-known author and novelist famous for Harry Potter
Normalized Answer: novelist

Question: What is Barack Obama's occupation?
Raw Answer: politician, lawyer, author, 44th president of the United States
Normalized Answer: politician

Question: {question}
Raw Answer: {raw_answer}

Normalized Answer:"""

# ---- 创新1: Critic Module ----
CRITIC_EVALUATE_PROMPT = """You are a Critic module for a Graph-based RAG system. Your job is to evaluate whether the current answer adequately addresses the question, given the retrieved evidence.

Question: {question}
Retrieved Context (abbreviated): {context}
Current Answer: {answer}

Analyze carefully and output a JSON object:
{{
  "verdict": "pass" or "retrieve_more" or "revise",
  "confidence": <float 0.0-1.0>,
  "feedback": {{
    "missing_entities": [<list of entity names that should be in the context but are missing>],
    "broken_paths": [<list of incomplete reasoning chains, e.g. "A->?->B">],
    "conflicts": [<list of contradictory evidence pairs, e.g. "X vs Y">],
    "suggestion": "<one-sentence suggestion for what to retrieve next>"
  }},
  "refined_query": "<a refined query string that specifically targets the identified gaps>"
}}

Guidelines:
- "pass": The answer is well-supported and directly addresses the question. If the answer is partially correct and the evidence contains enough relevant information, DO NOT default to retrieve_more — set verdict="pass" with confidence >= 0.5.
- "retrieve_more": ONLY if key entities or paths are clearly missing and the evidence is largely irrelevant — the system should retrieve more evidence using the refined_query.
- "revise": The answer has conflicts or errors but the evidence is basically relevant — the system should re-generate using existing evidence.
- Keep refined_query focused: mention specific missing entities or relationships.
- Do not choose "pass" for a merely plausible answer. Choose "pass" only when no follow-up retrieval or correction is needed.

Additional strict rules:
- For multi-hop questions, DO NOT pass if the answer only completes the first hop. Examples: returning the country when asked who leads that country, returning a university when asked where it is located, or returning a continent when asked for its ranking.
- If feedback.suggestion says to retrieve, confirm, expand, or get more specific evidence, verdict MUST be "retrieve_more" or "revise", not "pass".
- If verdict is "pass", leave missing_entities, broken_paths, conflicts, refined_query, and retrieval suggestions empty.

IMPORTANT: Output ONLY a valid JSON object. No markdown fences, no extra text, no commentary before or after the JSON.
"""

# ---- 创新3: Query Complexity-Aware Routing ----
QUERY_ROUTER_PROMPT = """You are a query complexity classifier for a Graph-based RAG system. Classify the question into exactly one category that determines which retrieval strategy to use.

Categories:
- "factoid": Simple single-hop factual question (Who is X? What is Y's occupation? When did Z happen?)
- "multi_hop": Requires connecting 2+ pieces of information across different entities (What is the capital of the country where X was born?)
- "comparative": Comparing attributes of 2+ entities (Who is older, X or Y? Which city is larger?)
- "aggregation": Requires collecting information about multiple entities (List all X that Y)
- "reasoning": Requires logical inference or chain-of-thought beyond simple retrieval

Question: {question}

IMPORTANT: Output ONLY a valid JSON object. No markdown fences, no extra text, no commentary.
{{"category": "<one of: factoid, multi_hop, comparative, aggregation, reasoning>", "confidence": <float 0.0-1.0>}}"""

# ============================================================
# Original Prompts
# ============================================================

COT_SYSTEM_DOC = ('As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
                          'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                          'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.')
COT_SYSTEM_NO_DOC  = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.')

DALK_RERANK_PROMPT = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {graph}
    \n\n
    ##Question: {question}
    \n\n
    Please rerank the knowledge graph and output at most 5 important and relevant triples for solving the given question. Output the reranked knowledge in the following format:
    Reranked Triple1: xxx ——> xxx
    Reranked Triple2: xxx ——> xxx
    Reranked Triple3: xxx ——> xxx
    Answer:
"""

DALK_CONVERT_PROMPT = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {graph}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
"""

DALK_STEP_PROMPT = """
    You are an excellent AI assistant to answering the following question,
    Question: 
    \n\n
    {question}
    \n\n
    You have some knowledge information in the following:\n\n
    ### {paths}
    \n\n
    ### {neis},
    Answer: Let's think step by step:
"""

DALK_CONTEXT_PROMPT = """
    You have some knowledge information in the following:\n\n
    ### {paths}
    \n\n
    ### {neis},
    Answer: Let's think step by step: {step}
    \n\n
"""

DALK_QUERY_PROMPT = """
    You are an excellent AI assistant to answering the following question,
    Question: 
    \n\n
    {question}
    \n\n
    {context}
    The final answer is:
"""
