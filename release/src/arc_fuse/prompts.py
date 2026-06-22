ROUTE_PROMPT = """<task:route>
Analyze the question and return JSON only.

Routing policy:
- graph: named entity attributes or explicit relationships;
- text: descriptive or paragraph-level context;
- hybrid: both relational and textual evidence are useful.

Question: {question}

Schema:
{{"domain":"general","entities":[],"relations":[],"selection":"graph|text|hybrid"}}
"""

ROUTE_REFLECT_PROMPT = """<task:route>
Refine the previous routing action using Critic feedback. Return JSON only.

Question: {question}
Previous action: {previous_action}
Feedback: {feedback}

Schema:
{{"domain":"general","entities":[],"relations":[],"selection":"graph|text|hybrid"}}
"""

CANDIDATE_PROMPT = """<task:candidate>
Answer the question from the evidence. Return a 1-5 word factual span, or
exactly "unknown" when the evidence is insufficient.

Question: {question}
Evidence:
{evidence}

Candidate Answer:
"""

JUDGE_PROMPT = """<task:judge>
You are judge {judge_id}. Evaluate the evidence and return JSON only.

Question: {question}
Evidence:
{evidence}

Schema:
{{"score":0,"reason":"one sentence","candidate":"1-5 words or unknown"}}
"""

VOTE_PROMPT = """<task:vote>
You are voter {voter_id}. Select the best candidate and return JSON only.

Question: {question}
Evidence:
{evidence}
Judge outputs:
{judge_outputs}

Schema:
{{"vote":"1-5 words or unknown","reason":"one sentence"}}
"""

INFERENCE_PROMPT = """<task:infer>
Produce the definitive short answer from the evidence and votes.

Question: {question}
Evidence:
{evidence}
Aggregated score: {aggregated_score}
Votes: {votes}
Winning answer: {winning_answer}

Final Answer:
"""

REVISE_PROMPT = """<task:infer>
Revise the previous answer using the feedback.

Question: {question}
Evidence:
{evidence}
Previous answer: {previous_answer}
Feedback: {feedback}

Final Answer:
"""

CRITIC_PROMPT = """<task:critic>
Evaluate whether the answer fully addresses the question using the evidence.
For multi-hop questions, do not pass an answer that only completes one hop.
Return JSON only.

Question: {question}
Evidence:
{evidence}
Current Answer: {answer}

Schema:
{{
  "verdict":"pass|retrieve_more|revise",
  "confidence":0.0,
  "feedback":{{
    "missing_entities":[],
    "broken_paths":[],
    "conflicts":[],
    "suggestion":""
  }},
  "refined_query":""
}}
"""

COMMENDOR_PROMPT = """<task:commendor>
Diagnose a failed answer. Return JSON only.

Allowed decisions:
- wrong_retriever
- insufficient_evidence
- poor_generation
- pass

Question: {question}
Selection: {selection}
Evidence:
{evidence}
Current Answer: {answer}
Critic feedback: {critic_feedback}

Schema:
{{"decision":"pass","confidence":0.0,"reason":"","hint":""}}
"""

NORMALIZE_PROMPT = """<task:normalize>
Extract only the most precise short factual answer. Preserve complete dates,
ranges, rankings, and qualified place names.

Question: {question}
Raw Answer: {answer}

Normalized Answer:
"""
