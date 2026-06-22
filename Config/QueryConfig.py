from Core.Utils.YamlModel import YamlModel
from dataclasses import field
from typing import Optional, Dict, Any


class QueryConfig(YamlModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    query_type: str = "qa"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    nei_k: int = 3
    num_doc: int = 5  # Default parameter for the HippoRAG
    # naive search
    naive_max_token_for_text_unit: int = 12000
    use_keywords: bool = False
    use_communiy_info: bool = False
    # local search
    
    enable_local: bool = False
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33

    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False
    community_information: bool = False  # Open for MS-GraphRAG based method
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    max_token_for_global_context: int = 4000
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    use_global_query: bool = False # For LightRAG and GraphRAG
    use_community: bool = False # True for LGraphRAG and GGraphRAG
    enable_hybrid_query: bool = False # For LightRAG 
    # For IR-COT
    max_ir_steps: int = 2

    # For Hipporag
    augmentation_ppr: bool = False
    entities_max_tokens: int = 2000
    relationships_max_tokens: int = 2000

    # For RAPTOR
    tree_search: bool = False

    # For TOG
    depth: int = 3
    width: int = 3

    # For G-Retriever (GR)
    max_txt_len: int = 512
    topk_e: int = 3
    cost_e: float = 0.5

    # For Medical GraphRAG
    topk_entity: int = 10
    k_hop: int = 2

    # ---- Agentic GraphRAG Innovations ----
    # Innovation 2: Answer Normalizer
    use_answer_normalizer: bool = False  # Post-process answers for benchmark alignment

    # Innovation 1: Critic-driven Directed Subgraph Expansion
    use_critic: bool = False             # Enable Critic evaluation loop
    critic_max_rounds: int = 2           # Max Critic re-retrieval iterations
    # Ablation: "directed" = structured feedback → targeted action (our method)
    #           "blind"    = Critic validates only, re-retrieval uses original query (Agent-G style)
    critic_mode: str = "directed"

    # Innovation 3: Query Complexity-Aware Routing
    use_query_router: bool = False       # Enable query routing (used at pipeline level)
    routing_map: Optional[Dict[str, Any]] = None  # Custom category→method mapping (optional)

    # Agent-G (ICLR 2025 baseline)
    agent_g_max_iterations: int = 4      # T in Algorithm 1
