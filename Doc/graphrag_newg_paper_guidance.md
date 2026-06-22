# GraphRAG / NewG Paper Guidance

Updated: 2026-06-17 CST.

This document preserves the current project direction and can be used as the
working guide when continuing the paper-writing and code-publication task.

## Project Goal

The project is a GraphRAG-based research framework that now includes an
agentic, closed-loop extension. The intended paper direction is to write a
manuscript in the style of **AGENT-G: An Agentic Framework for Graph
Retrieval-Augmented Generation**, using the current codebase and experiment
results as support.

The core story is not "we built another retriever". The story is:

1. queries differ in structure and evidence needs;
2. static retrieval is insufficient;
3. the system should route queries adaptively;
4. retrieved evidence should be fused and regenerated into a candidate answer;
5. a critic should decide whether to pass, revise, or retrieve more;
6. a diagnostic module should explain the failure mode;
7. the final output should be normalized.

## Architecture Summary

The current design, as reflected in the user's updated diagram, is:

`Query -> Query Understanding -> Adaptive Router -> Retriever Bank ->
Entity Disambiguation / Alias Resolution -> Evidence Fusion & Candidate
Generation -> Re-Generation Agent -> Critic Module -> Commendor ->
Answer Normalizer -> Final Answer`

### What each block means

- `Query Understanding`
  - extracts query intent, domain, entities, relations, and selection hints.
- `Adaptive Router`
  - chooses graph, text, or hybrid retrieval based on the query.
- `Retriever Bank`
  - includes graph retrievers, text retrievers, and hybrid fusion logic.
- `Entity Disambiguation / Alias Resolution`
  - resolves ambiguous mentions against the knowledge base.
- `Evidence Fusion & Candidate Generation`
  - merges retrieved evidence into a candidate answer context.
- `Re-Generation Agent`
  - revises or regenerates the answer from fused evidence.
- `Critic Module`
  - evaluates answer quality and decides whether to stop or iterate.
- `Commendor`
  - diagnoses whether the problem is the retriever, evidence, or generation.
- `Answer Normalizer`
  - standardizes the final answer for cleaner evaluation and output.

## Paper Framing

The manuscript should emphasize the following contributions:

1. **Adaptive routing**
   - the system does not rely on a fixed retriever for every query.
2. **Closed-loop generation**
   - retrieval and generation are connected by a critic-driven iteration loop.
3. **Failure diagnosis**
   - Commendor helps explain why an answer failed.
4. **Output normalization**
   - final answer formatting is explicitly controlled.
5. **Unified graph-text framework**
   - graph, text, and hybrid retrieval are handled in one system.

## Writing Strategy

Use a paper structure close to AGENT-G:

1. Introduction
2. Related Work
3. Method
4. Experiments
5. Results
6. Ablations
7. Case Study
8. Limitations
9. Conclusion

### Method section recommendation

The method section should mirror the diagram one block at a time. Do not write
it as a vague system overview. Instead, define the pipeline in a modular way:

- query understanding;
- routing policy;
- retriever bank;
- evidence fusion;
- regeneration;
- critic and commendor;
- normalization.

### Results section recommendation

Structure the experiments in three layers:

1. Main comparison against baselines.
2. Ablation study for each module.
3. Case-based analysis showing how the loop corrects errors.

## Codebase Usage

The paper should be grounded in the current repository. Relevant areas are:

- `Core/Query/NewGQuery.py`
- `Core/Query/QueryUnderstanding.py`
- `Core/Query/EvidenceFusion.py`
- `Core/Query/ReGenerationAgent.py`
- `Core/Query/CriticModule.py`
- `Core/Query/Commendor.py`
- `Core/Query/AnswerNormalizer.py`
- `Option/Method/NewG.yaml`
- `agentic_main.py`

The repository should eventually be prepared as a public project support
package, with:

- reproducible configs;
- run scripts;
- result tables;
- ablation scripts;
- figure generation scripts;
- a short usage guide.

## Practical Next Steps

If the work continues from this document, the next tasks should be:

1. Write the paper outline and contribution statements.
2. Map architecture blocks to code modules and config sections.
3. Organize experiment results into tables.
4. Prepare the GitHub repository structure.
5. Draft the results narrative and ablation interpretation.

## Rules For Continuation

- Do not repeat secrets such as passwords or API keys.
- Prefer concrete artifacts over abstract discussion.
- When the user says "continue previous task" or "继续上次任务", use this
  document and the existing memory file as the starting point.

