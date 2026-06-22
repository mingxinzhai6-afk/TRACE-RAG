# 引用清单与引言引用点

下面是这篇 TRACE-RAG 论文里最建议保留的外部引用，以及它们适合出现的位置。

## 1. 建议引用的工作

| 主题 | 建议文献 | 链接 | 适合引用的位置 |
| --- | --- | --- | --- |
| RAG 基础范式 | Lewis et al., 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | https://arxiv.org/abs/2005.11401 | 引言第 1 段，说明 RAG 的基本动机 |
| 稀疏检索 | Robertson and Zaragoza, 2009, *The Probabilistic Relevance Framework: BM25 and Beyond* | https://nowpublishers.com/article/Details/INR-019 | 引言第 1 段，说明 BM25 仍是经典稀疏检索基线 |
| 稠密检索 | Karpukhin et al., 2020, *Dense Passage Retrieval for Open-Domain Question Answering* | https://arxiv.org/abs/2004.04906 | 引言第 1 段，说明 dense retrieval 的代表性路线 |
| 图检索 | Jiménez Gutiérrez et al., 2024, *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models* | https://arxiv.org/abs/2405.14831 | 引言第 2 段，说明图结构对多跳 QA 的价值 |
| 图增强 RAG | Hu et al., 2024, *GRAG: Graph Retrieval-Augmented Generation* | https://arxiv.org/abs/2405.16506 | 引言第 2 段，说明图 RAG 的基础问题 |
| 图增强代理检索 | Shen et al., 2024, *GeAR: Graph-enhanced Agent for Retrieval-augmented Generation* | https://arxiv.org/abs/2412.18431 | 引言第 2 段，说明图增强 + agent 的趋势 |
| 图路径推理 | Yao et al., 2023, *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* | https://arxiv.org/abs/2305.10601 | 引言第 2 段，可用来类比“搜索/扩展式推理” |
| 层级图检索 | Sarthi et al., 2024, *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* | https://arxiv.org/abs/2401.18059 | 引言第 2 段，说明树状层级检索路线 |
| 自我反思式 RAG | Asai et al., 2023, *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* | https://arxiv.org/abs/2310.11511 | 引言第 3 段，说明“检索-生成-批判”闭环 |
| 主动检索 | Jiang et al., 2023, *Active Retrieval Augmented Generation* / FLARE | https://arxiv.org/abs/2305.06983 | 引言第 3 段，说明“证据不足时继续检索” |

## 2. 引言三段式建议

### 第 1 段：问题背景

建议写成：

1. 先说 LLM 有幻觉与知识时效性问题。
2. 再说 RAG 是主流缓解方式。
3. 顺带提 BM25 和 DPR，说明经典检索仍然重要。

建议引用：

- `rag2020`
- `robertson2009probabilistic`
- `karpukhin2020dpr`

### 第 2 段：为什么图 RAG 还不够

建议写成：

1. 图结构对多跳推理有帮助。
2. 但构图质量、覆盖率和长尾实体问题仍然存在。
3. 引出图增强和图代理检索，但说明它们仍缺少统一的闭环控制。

建议引用：

- `jimenez2024hipporag`
- `hu2024grag`
- `shen2024gear`
- `yao2023tot`
- `sarthi2024raptor`

### 第 3 段：为什么需要 TRACE-RAG

建议写成：

1. 现有 RAG 需要主动判断是否继续检索。
2. 还需要对答案是否可接受进行显式批判。
3. 还需要输出规范化，否则评测会被格式噪声影响。

建议引用：

- `asai2023selfrag`
- `jiang2023flare`

## 3. 论文里不建议强行引用的内容

- 不建议把你自己的模块都硬套成已有论文的直接延伸。
- 不建议把当前项目里的内部设计写成“已有 SOTA 证明”。
- 不建议在 PopQA 上写过度强势的全局领先表述，当前结果更适合写成“竞争力”和“稳健性”。

## 4. 项目内资料

这些不是外部文献，但适合在论文写作和实现说明中引用为“项目支持材料”：

- `paper_cn_draft_rewrite.md`
- `paper_results_narrative.md`
- `paper_results_latex.tex`
- `results_summary.tsv`
- `figures/positive_showcase/README_positive_showcase.md`
- `Doc/graphrag_newg_paper_guidance.md`


