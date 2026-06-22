# TRACE-RAG：面向复杂问答的自适应图文融合检索增强生成框架

## 摘要

检索增强生成（Retrieval-Augmented Generation, RAG）已经成为缓解大语言模型幻觉、提升事实问答能力的重要方法。然而，在复杂问答场景中，现有系统仍普遍存在三个问题：其一，检索策略往往固定，难以针对不同问题类型在图检索、文本检索与混合检索之间自适应切换；其二，检索到的证据通常被一次性送入生成模块，缺少对答案质量的显式验证与迭代修正；其三，生成结果的表达形式与自动评测标准之间存在偏差，影响精确匹配与指标稳定性。针对这些问题，本文提出 TRACE-RAG，一种统一的图文融合 agentic RAG 框架。TRACE-RAG 将查询理解、自适应路由、证据融合、多智能体重生成、Critic 反馈以及答案规范化整合为闭环流程，使系统能够根据问题复杂度动态选择检索路径，并在答案不可置信时继续补充证据和修正输出。

我们在 PopQA 与 MuSiQue 两个公开基准上对 TRACE-RAG 进行了系统评测，并与 BM25、VDB、HippoRAG、RAPTOR、ToG、AgentG 等方法比较。结果表明，TRACE-RAG 在 MuSiQue 上表现出最稳定且最显著的收益：在三种模型骨干下，最优配置平均提升 4.33 个 Accuracy、4.33 个 EM 和 5.84 个 F1。PopQA 上的增益相对保守，说明 TRACE-RAG 的优势主要体现在多跳、跨证据组合与迭代修正更强的任务上，而不是在所有简单事实问答场景中都呈现绝对领先。进一步的 paired-retriever 分析显示，TRACE-RAG 在 36 组对照中有 25 组优于其所依附的图检索基线，平均胜率达到 69.4%，平均提升 4.78 F1；leave-one-out 消融进一步说明 Critic 与 AnswerNormalizer 是当前版本中最稳定、最通用的两个关键模块。

## 1. 引言

### 1.1 研究动机

大语言模型显著提升了自然语言理解与生成能力，但其内部参数记忆无法保证知识的时效性、可追溯性与事实一致性。检索增强生成通过引入外部知识，在一定程度上缓解了幻觉问题，因此成为开放域问答、知识密集型任务和企业检索场景中的主流范式。然而，传统 RAG 默认所有问题都适合同一种检索方式，或者将“检索”和“生成”拆成两个弱耦合阶段，这使得系统在复杂问答中暴露出明显瓶颈。

### 1.2 研究背景与现状

首先，不同问题对证据形态的需求并不相同。实体属性类问题通常更依赖实体之间的显式关系，因此图检索更具优势；而多跳推理、背景依赖强或跨文档组合的任务则更依赖文本检索补充上下文。若系统始终采用固定检索路径，往往会在某一类问题上稳定吃亏。

其次，检索结果并不天然等价于可用答案。很多 RAG 系统拿到证据后直接生成最终回答，但并没有显式判断该答案是否被证据充分支持，也没有在证据不足时继续补检索。这类系统在“看起来合理但实际上错误”的场景中容易产生高置信度误答。

再次，生成模型倾向于输出解释性文本，而自动评测往往要求短、直接、可字符串匹配的答案。若缺少规范化模块，大量本应判对的样本会因为格式、别名、大小写或冗余解释而在 EM/Precision 上被误判为错误。

### 1.3 研究问题

围绕现有方法的局限，本文关注三个研究问题：第一，能否在图检索、文本检索与混合检索之间根据问题特征动态切换；第二，能否把证据充分性、生成质量与检索错误显式分离，从而构建可迭代修正的闭环；第三，能否通过答案规范化降低评测时的格式波动，提升指标稳定性。

### 1.4 主要贡献

基于 DIGIMON 的统一图文 RAG 基础，本文提出 TRACE-RAG，并围绕自适应路由、多智能体式重生成、Critic 驱动迭代和答案规范化形成完整闭环。本文的主要贡献如下：

1. 提出 TRACE-RAG，一个面向复杂问答的统一图文 agentic RAG 框架。
2. 设计 ReGenerationAgent、Critic、Commendor 和 AnswerNormalizer 的协同流程，其中 ReGenerationAgent 仅在 poor generation 情况下触发。
3. 在 PopQA 与 MuSiQue 上进行系统评测，并提供 paired-retriever 和 leave-one-out 消融分析。

图1给出了 TRACE-RAG 的主架构。图中左侧的 Retriever Bank 负责召回图证据、文本证据与混合证据；中间的 Query Understanding 与 Adaptive Router 负责判定该使用何种检索路径；下方的 Evidence Fusion 与 Re-Generation Agent 负责将证据整合并生成候选答案；右侧的 Critic Module、Commendor 与 AnswerNormalizer 则负责显式验证、失败归因和输出规范化。该图强调的不是模块数量，而是闭环控制逻辑：TRACE-RAG 试图把复杂问答分解为可诊断、可修复、可比较的多个环节。

### 1.5 文章结构

本文余下部分组织如下：第 2 节回顾相关工作；第 3 节介绍 TRACE-RAG 的方法设计；第 4 节给出实验设置；第 5 节报告主要结果与消融分析；第 6 节展示典型案例；第 7 节讨论局限性；第 8 节总结全文。

## 2. 相关工作

### 2.1 文本检索增强生成

传统 RAG 通常依赖 BM25、DPR 或其他文本检索器召回相关段落，再将证据拼接给生成模型完成回答。这类方法实现简单、适用范围广，但在复杂问答中容易受到检索粒度与证据覆盖率的限制。尤其当问题需要跨句、跨段甚至跨文档推理时，纯文本检索往往难以保证关键证据完整召回。

### 2.2 图检索增强生成

图检索方法尝试把实体、关系与路径结构显式编码到知识图或关系图中，从而支持多跳推理和结构化证据定位。HippoRAG、ToG、RAPTOR、GRAG、GeAR 等工作表明，图结构对于复杂问题具有天然优势，因为它保留了“谁和谁有关”“事件如何连接”这类显式关系。Zhou 等人还从统一框架视角系统比较了图检索 RAG 方法，强调了不同图方法在统一设置下的行为差异 \cite{zhou2025graphragunified}。然而，图方法也面临图覆盖率、实体消歧、长尾实体稀疏和噪声传播等问题，因此难以单独覆盖所有问答类型。

### 2.3 Agentic 与迭代式 RAG

Self-RAG、FLARE 以及 AGENT-G 等工作进一步说明：系统不应默认一次检索就足够，而应在生成过程中主动判断是否需要补检索、重生成或自我修正。这类方法的共同点是把“是否继续查证”显式化，并通过迭代缓解一次性生成的偶然误差 \cite{asai2023selfrag,jiang2023flare,agentg2025}。TRACE-RAG 继承了这一思路，但进一步把路由、重生成和失败归因拆分为多个独立模块，增强了系统的可解释性与可诊断性。

### 2.4 统一评测框架

DIGIMON 提供了统一的图文 RAG 评测基础，使不同检索器、不同图方法与不同问答策略能够在相同接口下对比。TRACE-RAG 建立在这一思路之上，进一步关注三个更具体的问题：检索策略如何自适应、证据如何被反复利用、答案如何对齐评测标准。也就是说，TRACE-RAG 不是单一检索器，而是一个围绕复杂问答设计的闭环系统。

## 3. 方法

### 3.1 总体架构

TRACE-RAG 的整体流程可以概括为：

$$
q \rightarrow \text{QueryUnderstanding} \rightarrow \text{Retriever Bank} \rightarrow
\text{Evidence Fusion} \rightarrow \text{ReGenerationAgent} \rightarrow
\text{Critic} \rightarrow \text{AnswerNormalizer} \rightarrow a^*
$$

其中，$q$ 是输入问题，$a^*$ 是最终输出答案。若 Critic 判断当前答案仍不可靠，则系统会在最多 $R$ 轮内继续触发补检索与再生成。当前实现中，$R$ 默认为 3。

为便于形式化描述，记第 $t$ 轮检索得到的证据为 $E_t$，重生成输出为 $\hat{a}_t$，Critic 评分为 $s_t$，答案规范化函数为 $\mathcal{N}(\cdot)$，则 TRACE-RAG 的一轮闭环更新可写为：

$$
\begin{aligned}
E_t &= \mathcal{R}(q, t), \\
\hat{a}_t &= \mathcal{G}(q, E_t), \\
s_t &= \mathcal{C}(q, E_t, \hat{a}_t), \\
a_t &= \mathcal{N}(\hat{a}_t), \\
a^* &= a_t \quad \text{if } s_t \ge \tau \text{ or } t = R .
\end{aligned}
$$

其中，$\mathcal{R}$ 表示检索模块，$\mathcal{G}$ 表示重生成模块，$\mathcal{C}$ 表示 Critic，$\tau$ 为停止阈值。

如图1所示，TRACE-RAG 不是简单串联多个模块，而是把系统拆成三层：第一层是路由层，由 Query Understanding 与 Adaptive Router 负责判断问题类型和检索偏好；第二层是证据层，由 Retriever Bank、Entity Disambiguation / Alias Resolution 与 Evidence Fusion 负责把异构召回结果整理成统一上下文；第三层是反馈层，由 Re-Generation Agent、Critic、Commendor 与 AnswerNormalizer 负责答案生成、失败归因与输出规范化。

这样的设计有两个直接好处。第一，它把“错误发生在哪里”显式化，便于分析系统瓶颈是来自路由、召回、重生成还是规范化；第二，它让不同模块的职责边界清晰，便于做 leave-one-out 消融并比较其边际贡献。

### 3.2 查询理解与自适应路由

不同问题对证据形态的需求并不相同。实体属性类问题通常更适合图检索，因为答案往往与实体关系直接相关；多跳、背景依赖强的问题则更需要文本检索补充上下文。如果系统对所有问题都采用同一条检索路径，就会在某一类问题上稳定吃亏。

TRACE-RAG 的 QueryUnderstanding 模块对输入问题进行一次 LLM 推断，输出问题的粗粒度结构信息，例如问题域、关键实体、关系线索、时间线索以及检索偏好。Adaptive Router 再根据这些结构信息，在 graph、text、hybrid 三类模式之间选择后续检索分支。这里的“自适应”不是无限制自由路由，而是在当前图方法与文本方法配置下，动态决定更偏向哪一类证据。

从实现角度看，路由器输出的不是最终答案，而是后续检索策略。它回答的是“先从哪里找”，而不是“答案是什么”。这样做的好处是把路由错误与生成错误分离开来，便于后续分析和调试。

### 3.3 检索器集合与证据融合

TRACE-RAG 的 Retriever Bank 兼容三类图方法与两类文本方法：

- 图检索：HippoRAG 风格的实体关系检索、ToG 风格的图遍历检索、RAPTOR 风格的层级摘要检索；
- 文本检索：BM25 与 VDB；
- 混合检索：通过融合策略合并图证据与文本证据。

BM25 偏重词项匹配，适合关键词明确的事实问答；VDB 偏重语义相似，适合表达方式变化较大但语义关联明显的问题。混合模式下，TRACE-RAG 通过融合图与文两路召回结果，尽量在结构约束与语义补充之间取得平衡。

证据融合模块会将检索到的候选片段整理为统一上下文，并构造面向生成模块的输入。其目标不是简单拼接文本，而是尽可能保留支持答案判断的关键信息，减少噪声证据对生成的干扰。结合图1可以更直观地理解这一点：Graph Retriever 输出结构化关系路径，Text Retriever 输出段落级证据，Hybrid Retriever 则通过 RRF 等策略把两类信号合并。随后 Entity Disambiguation / Alias Resolution 把同名实体、别名实体和歧义实体统一到同一语义空间，避免后续生成阶段围绕错误实体展开。

从论文角度看，证据融合不只是“拼接证据”，而是把异构召回结果压缩成适合生成的上下文，并保留潜在冲突证据供后续 Critic 判断。换言之，TRACE-RAG 在检索阶段并不急于抹平所有分歧，而是优先保证证据结构完整，然后再由反馈模块做验证与筛选。

### 3.4 ReGenerationAgent

单次生成在证据不足或证据冲突时容易输出不稳定答案。为降低这一风险，TRACE-RAG 将答案生成拆解为 judge-vote-infer 三个阶段；其中 ReGenerationAgent 只在 Critic 判定为 poor generation 时触发，用来修正证据已经足够但答案表达不稳定的情况：

- Judge 阶段：多个 judge 独立阅读证据并给出候选答案与置信度；
- Vote 阶段：vote 模块综合多个 judge 的结果，选出更一致的候选；
- Infer 阶段：最终 inference agent 基于证据与投票结果输出简洁答案。

这一设计的核心目的不是让模型“思考更久”，而是通过并行协作降低单次生成的偶然误差。当前实现默认使用较小规模的 judge/voter 组合，以控制推理成本，因此 ReGenerationAgent 的作用更偏向稳定化，而不是无成本地提升模型能力。

从功能上看，ReGenerationAgent 可以理解为“证据条件下的答案再建构器”。它不负责决定证据是否足够，也不负责判断答案能否接受，而是把已经收集到的证据尽可能转化为稳定、简洁、可评测的候选答案。

### 3.5 Critic 驱动的迭代检索

Critic 是 TRACE-RAG 闭环的核心。它读取当前候选答案与证据集，对答案是否被证据充分支持进行结构化判断，输出三类决策之一：

- `pass`：当前答案可接受，直接结束；
- `revise`：答案可以小幅修正，但不必重新检索；
- `retrieve_more`：证据不足，需要继续补检索。

与此同时，Commendor 会把失败原因进一步归类为 `insufficient_evidence`、`wrong_retriever`、`poor_generation` 或 `pass`。这里的价值在于把“为什么失败”拆得更细：`wrong_retriever` 表示路由或检索器选型不对，`insufficient_evidence` 表示召回不够，`poor_generation` 表示证据已经足够但表达质量仍不足。

实验中我们观察到，TRACE-RAG 的大多数迭代来自证据不足，而不是纯生成失败。这说明当前系统瓶颈仍主要在检索阶段，而不是单纯的语言表达阶段。换言之，TRACE-RAG 的闭环设计不是为了让模型永远输出更长的解释，而是为了在证据不足时主动补检索，在答案质量不足时主动重生成。

一旦 Critic 判断需要补检索，系统就会依据反馈调整检索问题、重选检索分支，并重新进入证据融合与生成流程，直到达到最大轮数或者获得可接受答案。

### 3.6 答案规范化

评测任务通常要求短、直接、可字符串匹配的答案，而生成模型天然倾向于输出解释性文本。若不加规范化，很多实际上答对的样本会因为格式差异而被 EM 判错。为此，TRACE-RAG 在最终输出前引入 AnswerNormalizer，对候选答案做抽取、压缩与格式统一，使结果更接近评测标准。

AnswerNormalizer 的目标不是越短越好，而是保留语义核心并去掉无关解释。它是一个对评测友好的后处理模块，不应被误解为简单的字符串截断。更准确地说，它承担的是“评测接口层”的作用，直接影响 EM、Precision 和字符串一致性。

综上，TRACE-RAG 的方法设计可以概括为一句话：它不是把 RAG 的各个模块简单串联，而是把检索、融合、生成、反馈与规范化组织成一个可迭代控制系统。这个控制系统的意义在于，它允许复杂问答的失败模式被显式识别、局部修复并重新进入流程。

## 4. 实验设置

### 4.1 数据集

本文在两个公开基准上进行实验：

- PopQA：以实体属性和事实问答为主，适合测试图检索与事实提取能力；
- MuSiQue：以多跳推理为主，适合测试复杂问答、证据组合与迭代修正能力。

每个数据集均抽取 200 条样本进行统一评测，便于跨方法、跨模型比较。这个规模约占 PopQA 全集 1399 条的 14.3\%，占 MuSiQue 全集 3000 条的 6.7\%。整体上，PopQA 更适合观察系统在简单事实问答上的边界，而 MuSiQue 更适合体现 TRACE-RAG 的闭环优势。

### 4.2 对比方法

我们比较的方法包括：

- BM25
- VDB
- HippoRAG
- RAPTOR
- ToG
- AgentG
- TRACE-RAG 的多种 graph/text 组合版本

这些方法覆盖纯文本检索、图结构检索以及图文混合检索三个层次。TRACE-RAG 并不是与所有方法在完全相同实现细节下比较，而是在可复现配置上尽量统一检索策略、重生成策略和反馈策略，从而更贴近真实系统对比。

### 4.3 模型骨干与评测指标

实验使用三种模型骨干：

- deepseek-v3.2
- gpt-4o-mini
- gemini-2.5-flash-lite

评测指标包括：

- Accuracy
- Exact Match (EM)
- Precision
- Recall
- F1

其中，Accuracy 用于给出总体正确率，EM 用于衡量是否与标准答案字符串完全一致，Precision / Recall / F1 则用于更细致地反映答案覆盖与误召回情况。由于生成式答案常常存在格式波动，因此本文尤其重视 AnswerNormalizer 对 EM 与 Precision 的影响。本文的核心实验设置如下：LLM 为 gemini-2.5-flash-lite，Embedding 为 BAAI/bge-small-en-v1.5，Judge 数为 3，Voter 数为 3，最大迭代轮数为 3，Judge 分数阈值为 3.0，Critic 输入使用规范化答案，AnswerNormalizer 与 Disambiguation 均启用，BM25 与 VDB 的 top-k 均为 10。

### 4.4 实现细节

TRACE-RAG 在当前代码中以固定的图方法与文本方法为基础运行，例如 `hipporag + bm25`、`hipporag + vdb`、`tog + bm25`、`tog + vdb` 等组合。对于 ablation，我们构造了 leave-one-out 版本，分别关闭 Router、ReGeneration、Critic、Commendor、AnswerNormalizer 等模块，以观察各组件对最终效果的边际贡献。所有实验统一采用相同的评测样本、相同的答案后处理规则和相同的停止条件，从而保证不同配置之间的差异主要来自方法本身。

本文的实现设置可分为三类。检索侧，HippoRAG 采用 `er_graph_colbert` 索引与 FAISS 向量库，启用实体向量索引与 entity link chunk，使用 PPR 检索，其中 `top_k_entity_for_ppr=8`、`damping=0.1`、`top_k=5`；ToG 采用 `er_graph` 索引与向量库，同时启用实体与关系向量索引，`query_type=tog`、`width=3`、`depth=3`、`top_k=3`；RAPTOR 采用 `tree_graph_balanced` 索引与 FAISS 向量库，`query_type=basic`、`top_k=5`，并使用 `selection_mode=top_k`、`start_layer=5`、`num_layers=5`、`threshold=0.1`、`max_iter=10`、`size_of_clusters=10`。文本检索分支中，BM25 与 VDB 的 `top_k` 均设为 10。生成侧，`n_judges=3`、`n_voters=3`、`max_rounds=3`、`judge_score_threshold=3.0`，并启用 Router、ReGeneration、Critic、Commendor、AnswerNormalizer 和 Disambiguation；`use_last_resort_guess` 关闭。评测侧，MuSiQue 使用 validation split，过滤 `answerable=False` 样本并对 corpus title 去重；两个数据集均按固定的 200 条样本进行评测，并统一采用答案规范化后的打分流程。

## 5. 实验结果

### 5.1 主结果

总体上，TRACE-RAG 在 MuSiQue 上的优势最为稳定。在三种模型骨干下，最优 TRACE-RAG 配置均优于对应最强基线：DeepSeek 骨干下，TRACE-RAG `tog+bm25` 的 F1 从 30.06 提升到 31.44；GPT-4o-mini 骨干下，TRACE-RAG `hippo+bm25` 的 F1 从 24.90 提升到 31.77；Gemini 骨干下，TRACE-RAG `tog+bm25` 的 F1 从 20.99 提升到 30.26。

这组结果说明，TRACE-RAG 的收益并不依赖单一模型骨干，而是能够在不同模型上起到补强作用。特别是在 MuSiQue 这类多跳问答场景中，检索路由、证据融合与迭代修正的组合效应非常明显。换句话说，TRACE-RAG 的价值主要不是把某个单项指标推到极限，而是在复杂任务中提供稳定的系统增益。

PopQA 的结果更为克制。DeepSeek 和 GPT-4o-mini 在最优 TRACE-RAG 配置下分别获得 +0.92 和 +0.20 的 F1 提升，而 Gemini 的最优配置略低于最强基线（-0.35 F1）。这说明 TRACE-RAG 并不是在所有场景下都能大幅超越传统方法；对于更简单、更偏单实体事实的任务，强基线本身已经较强，TRACE-RAG 的主要作用更多体现为稳健性而不是绝对领先。

因此，论文叙述上应当把 MuSiQue 作为主战场，把 PopQA 定位为边界条件与稳健性测试。这样的表述更符合实际结果，也更能体现方法适用范围。

### 5.2 Paired-Retriever 分析

为了进一步分析 TRACE-RAG 是否真的优于其所依附的图检索基线，我们做了 paired-retriever 分析。结果显示，在 36 组对照中，TRACE-RAG 有 25 组优于对应基线，胜率达到 69.4%，平均 F1 变化为 +4.78；在所有增益样本中，平均提升达到 +7.67 F1。

这一结果的重要性在于，它表明 TRACE-RAG 的收益并不只是来自“选到了更强的底座”，而是来自图文融合、重生成和反馈闭环共同带来的额外增益。特别是在图基线较弱或不稳定时，TRACE-RAG 往往能够补上缺失证据，从而显著改善最终结果。

### 5.3 消融分析

leave-one-out 消融表明，Critic 与 AnswerNormalizer 是当前最稳定、最可解释的两个关键模块。以 Gemini + `hippo+bm25` 为例：

- 去掉 Critic 后，MuSiQue 的 F1 下降 5.33，PopQA 的 F1 下降 1.04；
- 去掉 AnswerNormalizer 后，MuSiQue 的 EM 下降 6.50，PopQA 的 EM 下降 38.50；
- AnswerNormalizer 同时对 Precision 也有明显帮助，说明它并非只是格式修饰，而是在统一输出边界、降低字符串噪声。

相比之下，Router、ReGenerator、Commendor 的收益更加依赖具体数据集和问题类型。换句话说，TRACE-RAG 的模块贡献不是“每个模块都同样重要”，而是呈现出明显的任务依赖性：Critic 和规范化更像通用收益模块，而路由和重生成更像场景敏感模块。

这一结论与图1结构是一致的。图中位于中间的路由与检索链路负责把证据带进来，但真正决定系统能否稳定输出的，往往是右侧反馈链路是否足够敏感、是否能及时识别失败类型。因此，消融结果不是在证明某个模块“无用”，而是在说明 TRACE-RAG 的性能由主检索链路和控制反馈链路共同决定。

## 6. 案例分析

为了避免只报告平均分而掩盖真实问题，本节给出两类更具代表性的失败样本。现有 case report 中的样本都属于 Critic 放行了错误答案的情形，这说明系统当前仍存在高置信度误判风险。

### 6.1 案例概览

为了避免只报告平均分而掩盖系统的真实行为，本节选取服务器上真实运行日志中的两个成功样本和一个失败样本。三者分别对应多轮修正、路由切换和图检索偏置三类典型情形，能够较好地说明 TRACE-RAG 的优势与边界。下表对三条样本的关键信息进行了概括。

| 数据集 | ID | 问题 | 结果 | 关键信号 |
| --- | ---: | --- | --- | --- |
| MuSiQue | 121 | What county houses the community of Robinson, in the state where Tom Harkin was from? | Delaware County | 3 轮 graph 推理，前两轮 Critic 连续返回 `retrieve_more`，第三轮 `pass`，说明系统在证据补全后能够稳定收敛。 |
| PopQA | 183 | In what city was Robert Zawada born? | Jedlnia-Letnisko | `graph` -> `hybrid` 路由切换，第一轮 `revise`，第二轮 `retrieve_more`，第三轮 `pass`，体现出较强的自适应纠正能力。 |
| MuSiQue | 0 | Which network subsidiary broadcasts the weeknight evening news show in part named after the network that aired Crowd Rules? | CNBC / CNBC Asia | 3 轮均偏向 graph 路由，Critic 多次要求补证，但最终答案被压缩为更泛化的上位实体，暴露出粒度不足问题。 |

### 6.2 成功样本一：多轮图检索后完成纠正的地理问答

该样本来自 MuSiQue，样本编号为 121，问题为“*What county houses the community of Robinson, in the state where Tom Harkin was from?*”，标准答案为 *Delaware County*。这个问题需要先解析 Tom Harkin 的出生州，再在州内完成社区到县的定位，属于典型的两跳地理问答。实际运行中，TRACE-RAG 连续进行了 3 轮 graph 推理，前两轮已定位到 Iowa 与 *Delaware County, Iowa*，但 Critic 仍返回 `retrieve_more`；直到第三轮在补充完整图证据后，系统才最终 `pass` 并稳定收敛到 *Delaware County*。

### 6.3 成功样本二：路由切换后修正成功的跨跳生物问答

该样本来自 PopQA，样本编号为 183，问题为“*In what city was Robert Zawada born?*”，标准答案为 *Jedlnia-Letnisko*。这一样本在 3 轮推理中呈现出更明显的自适应路由特征：第一轮使用 graph 路由，模型已给出 *Jedlnia-Letnisko*，但 Critic 认为论证不够完整并要求 `revise`；第二轮路由切换为 hybrid，系统进一步细化第二跳检索，但 Critic 仍认为信息不足并返回 `retrieve_more`；第三轮继续补充证据后，答案稳定收敛到 *Jedlnia-Letnisko*，Critic 最终放行。

### 6.4 失败样本：图检索偏置导致的近似答案

该样本来自 MuSiQue，样本编号为 0，问题为“*Which network subsidiary broadcasts the weeknight evening news show in part named after the network that aired Crowd Rules?*”，标准答案为 *CNBC Asia*，而系统最终输出为 *CNBC*。在 3 轮运行中，系统始终更偏向 graph 路由，Critic 也多次要求 `retrieve_more`；然而由于检索到的证据更接近品牌层面的上位实体，而不是具体的地区子公司，最终答案被收敛成更泛化的 *CNBC*。换言之，系统没有完全失效，而是把一个需要精确下钻的答案压缩成了看似相关但粒度不足的近似实体。

综合这三个样本可以看到，TRACE-RAG 的强项主要体现在两点：一是能够在证据不足时多轮纠正并最终收敛，二是能够根据证据形态在 graph 与 hybrid 之间切换；而其主要风险则集中在答案粒度不足和过泛化实体上。相比只看平均指标，这类案例更能揭示系统在真实推理中的行为模式，也更符合 SCI 论文中“代表性案例 + 机制解释”的写法。

## 7. 局限性

尽管 TRACE-RAG 已经在复杂问答上取得较稳定收益，但当前版本仍有以下局限：

1. 图覆盖率仍然有限。对长尾实体、冷门概念或构图不完整的样本，图检索仍可能缺少关键节点，导致后续重生成也无法补救。
2. Critic 仍会出现误放行。当前样本中仍存在错误答案被放行的情况，说明 Critic 的校准与问题类型敏感性还需要加强。
3. 推理成本较高。由于每题可能涉及多轮 judge、vote、inference 以及补检索，整体调用成本明显高于单次生成式方法。
4. PopQA 上的收益有限。对于更简单、更偏事实型的问题，强基线本身已经很强，TRACE-RAG 的优势更多体现在稳健性而不是绝对领先。
5. 当前案例分析样本仍偏少，尚不足以覆盖所有失败模式，因此更适合作为方向性证据，而不是完整统计结论。

此外，当前版本的推理链路较长，这意味着在高并发或低延迟场景中必须在效果与成本之间做取舍。若未来将其应用到在线系统，可能需要减少 judge/voter 数量、限制最大迭代轮数，或者仅对低置信度问题开启完整闭环。

## 8. 结论

本文提出 TRACE-RAG，一种面向复杂问答的自适应图文融合 agentic RAG 框架。与固定检索路径的传统方法不同，TRACE-RAG 将查询理解、检索路由、证据融合、多智能体重生成、Critic 反馈和答案规范化整合到一个闭环流程中，试图同时解决“检索选不对、证据用不好、答案不稳定、格式不对齐”四类问题。

实验结果表明，TRACE-RAG 在 MuSiQue 上表现出清晰且稳定的优势，在多种模型骨干上均能带来一致增益；在 PopQA 上则主要体现为稳健性提升，而非无条件的大幅领先。paired-retriever 与消融分析进一步说明，Critic 与 AnswerNormalizer 是当前版本中最通用、最稳定的模块，而路由与重生成更依赖具体问题和数据集特征。

总体而言，TRACE-RAG 的价值不在于再造一个单点更强的检索器，而在于把图检索、文本检索、生成验证与输出规范组织成一个可迭代、可诊断、可比较的系统。对于复杂问答而言，这种系统化设计比单一模块的局部增强更接近真实应用需求。

