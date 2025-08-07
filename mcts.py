import math
import random
import json
import os
import asyncio
import logging
import re
import time
from openai import AsyncOpenAI
from collections import defaultdict # For more advanced UCT and statistics

# --- Setup for LLM Calling ---
# --- LLM 调用设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LLM_MODEL = "gpt-4.1-mini"
# client = AsyncOpenAI(api_key="xxx", base_url="xxx")

def extract_json_from_text(input_text: str) -> str:
    """Attempts to extract a JSON string from a larger text, handling common LLM output issues."""
    # 尝试从较大文本中提取 JSON 字符串，处理常见的 LLM 输出问题。
    start_time = time.time()
    try:
        start_index = input_text.find('{')
        end_index = input_text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_string = input_text[start_index : end_index + 1]
            logger.debug(f"JSON extracted successfully in {time.time() - start_time:.4f} seconds.")
            return json_string
        else:
            logger.warning("No valid JSON object boundaries found using simple string find. Returning empty JSON.")
            return "{}"
    except Exception as e:
        logger.error(f"Error during JSON extraction: {e}. Returning empty JSON.", exc_info=True)
        return "{}"

async def call_llm(prompt: str, model: str = LLM_MODEL, temperature: float = 0.0, response_format_type: str = "text"):
    """
    Asynchronously calls the LLM with a given prompt.
    Includes retry logic and optional JSON response handling.
    """
    # 异步调用 LLM。包含重试逻辑和可选的 JSON 响应处理。
    start_time = time.time()
    retries = 3
    while retries:
        try:
            retries -= 1
            logger.debug(f"Calling LLM ({model}, attempt {3-retries}). Prompt (truncated): {prompt[:200]}...")

            response_format = {"type": "json_object"} if 'json' in response_format_type else {"type": "text"}

            response = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=1000,
                            response_format=response_format
                        )
            end_time = time.time()
            content = response.choices[0].message.content.strip()
            logger.debug(f"LLM call successful in {end_time - start_time:.2f} seconds. Response (truncated): {content[:200]}...")

            if 'json' in response_format_type:
                try:
                    parsed_content = json.loads(content)
                    return parsed_content
                except json.JSONDecodeError:
                    logger.error("LLM did not return valid JSON despite response_format. Attempting regex extraction.")
                    json_str_extracted = extract_json_from_text(content)
                    try:
                        return json.loads(json_str_extracted)
                    except json.JSONDecodeError:
                        logger.error(f"Extracted string is also not valid JSON: {json_str_extracted[:100]}...")
                        return {}
            return content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}. Retries left: {retries}", exc_info=True)
            if retries == 0:
                break
            await asyncio.sleep(2)
    logger.error("LLM call failed after multiple retries.")
    return ""

class LLMRewardModel:
    """
    A reward model that uses an LLM to evaluate the quality of a generated answer
    based on a query and a set of evidence.
    """
    # 一个奖励模型，使用 LLM 根据查询和一组证据评估生成答案的质量。
    def __init__(self, llm_caller_func):
        self.llm_caller_func = llm_caller_func
        # Cache for LLM responses to avoid redundant calls
        # LLM 响应缓存，避免冗余调用
        self.llm_response_cache = {} 

    async def _check_answer_evidence_consistency(self, sub_query: str, derived_answer: str, evidence_text: str) -> bool:
        """
        Uses LLM to check if the evidence text directly supports the derived answer for the given sub-query.
        """
        # 使用 LLM 检查证据文本是否直接支持给定子查询的派生答案。
        # Cache key for consistency check
        # 一致性检查的缓存键
        cache_key = ("consistency_check", sub_query, derived_answer, evidence_text)
        if cache_key in self.llm_response_cache:
            logger.debug(f"Cache hit for consistency check: {cache_key[0][:50]}...")
            return self.llm_response_cache[cache_key]

        prompt = f"""
        You are an expert fact-checker. Your task is to determine if the 'Provided Evidence' directly and explicitly supports the 'Derived Answer' for the 'Sub-query'.

        Sub-query: "{sub_query}"
        Derived Answer: "{derived_answer}"
        Provided Evidence: "{evidence_text}"

        Instructions:
        Evaluate strictly whether the 'Derived Answer' can be directly inferred or found within the 'Provided Evidence' in response to the 'Sub-query'.
        Ignore outside knowledge.

        Output your answer as a JSON object with a single key "is_consistent", which is a boolean (true/false).
        Example: {{"is_consistent": true}} or {{"is_consistent": false}}
        """
        llm_response_dict = await self.llm_caller_func(prompt, temperature=0.0, response_format_type="json")
        is_consistent = llm_response_dict.get("is_consistent", False)
        
        self.llm_response_cache[cache_key] = is_consistent # Cache the result
        return is_consistent

    async def evaluate_evidence_impact(self,
                                original_question: str,
                                current_reasoning_path_details: list,
                                new_evidence_text: str,
                                solved_sub_queries_ref: set,
                                needed_sub_queries_ref: set) -> dict:
        
        # Create a cache key for the current evaluation
        # 创建当前评估的缓存键
        cache_key = (original_question, tuple(current_reasoning_path_details), new_evidence_text, 
                     frozenset(solved_sub_queries_ref), frozenset(needed_sub_queries_ref))
        if cache_key in self.llm_response_cache:
            logger.debug(f"Cache hit for evidence impact evaluation: {cache_key[:50]}...")
            # Deep copy to prevent accidental modification of cached sets
            # 深度复制以防止意外修改缓存集
            cached_result = self.llm_response_cache[cache_key]
            # Ensure the passed by reference sets are updated from cached data
            # 确保通过引用传递的集合从缓存数据中更新
            solved_sub_queries_ref.clear()
            solved_sub_queries_ref.update(cached_result['updated_solved_sub_queries'])
            needed_sub_queries_ref.clear()
            needed_sub_queries_ref.update(cached_result['updated_needed_sub_queries'])
            return cached_result['evaluation_result']


        reasoning_context_parts = []
        if current_reasoning_path_details:
            for i, (sq, sa, et_id, et) in enumerate(current_reasoning_path_details):
                reasoning_context_parts.append(f"Step {i+1}:")
                if sq and sa:
                    reasoning_context_parts.append(f"  Sub-query: {sq}")
                    reasoning_context_parts.append(f"  Answer: {sa}")
                reasoning_context_parts.append(f"  Evidence used (ID {et_id}): {et[:150]}...")
            context = "\n".join(reasoning_context_parts)
        else:
            context = "No previous reasoning steps have been taken."
        
        # --- Optimized Prompt for more granular rewards and explicit instructions ---
        # --- 优化后的提示，用于更细粒度的奖励和明确的指令 ---
        prompt = f"""
        You are an expert reasoning engine designed to evaluate the utility of new evidence in solving a complex, multi-hop question.
        Your task is to determine how a 'New Evidence' snippet advances or hinders the progress toward answering the 'Original Question', given the 'Current Reasoning Path'.

        Original Question: {original_question}

        ---
        Current Reasoning Context:
        {context}

        Sub-queries Addressed So Far: {', '.join(solved_sub_queries_ref) if solved_sub_queries_ref else 'None'}
        Sub-queries Still Needed to Answer the Original Question: {', '.join(needed_sub_queries_ref) if needed_sub_queries_ref else 'None'}
        ---

        New Evidence to Evaluate:
        "{new_evidence_text}"

        ---
        Instructions:
        Carefully analyze the 'New Evidence' in relation to the 'Original Question' and the 'Current Reasoning Context'.
        Provide your evaluation as a JSON object with the following fields:

        1.  `reward`: A float representing the impact of this 'New Evidence' (from -1.0 to 1.0):
            - `1.0`: **Direct Solution/Major Advance.** The evidence directly and clearly helps solve one or more *unsolved* sub-queries listed in 'Sub-queries Still Needed'. This is the primary goal.            - `0.3`: **Supporting/Redundant.** The evidence:
                - Confirms or provides additional support for an already 'Sub-query Addressed So Far'.
                - Is relevant to the 'Original Question' but does not directly solve or significantly advance any *unsolved* sub-queries.
            - `0.0`: **Irrelevant/No Value.** The evidence is not relevant, provides no new useful information, or does not fit any other category.
            - `-1.0`: **Minor Contradiction/Misleading.** The evidence subtly contradicts or misleads, but isn't a direct negation of established facts.

        2.  `newly_addressed_sub_query`: A list of strings. If `reward` is `1.0`, state the specific sub-query from 'Sub-queries Still Needed' that this evidence helps solve or significantly advance. If it helps multiple, choose the most significant one. Otherwise, an empty string `""`.

        3.  `derived_answer_for_sub_query`: A list of strings. If `newly_addressed_sub_query` is not empty, provide a concise answer to that sub-query based *only* on the 'New Evidence' and relevant 'Current Reasoning Context'. Otherwise, an empty string `""`.

        4.  `is_contradictory`: A boolean (`true` or `false`). Set to `true` if the `New Evidence` contradicts prior information (even minor contradiction), `false` otherwise. If `true`, the `reward` must be negative.
        
        5.  `updated_solved_sub_queries`: A list of strings representing all sub-queries considered solved *after* evaluating this new evidence. This list should be the union of `solved_sub_queries_ref` and any `newly_addressed_sub_query` that is now considered fully solved.
        6.  `updated_needed_sub_queries`: A list of strings representing all sub-queries still *needed* to be solved *after* evaluating this new evidence. This list should be `needed_sub_queries_ref` minus any `newly_addressed_sub_query` that is now considered fully solved.

        Ensure your output is a single, valid JSON object.
        """
        
        logger.debug(f"evaluate_evidence_impact: Incoming evidence snippet (truncated): {new_evidence_text[:150]}...")
        llm_response_dict = await self.llm_caller_func(prompt, temperature=0.0, response_format_type="json")
        logger.debug(f"evaluate_evidence_impact: LLM response dictionary: {llm_response_dict}")

        result = {
            "reward": 0.0, # Changed to float
            "newly_addressed_sub_query": "",
            "derived_answer_for_sub_query": "",
            "is_contradictory": False
        }
        
        # Initialize lists for caching
        # 初始化用于缓存的列表
        updated_solved_for_cache = []
        updated_needed_for_cache = []

        if isinstance(llm_response_dict, dict):
            # Ensure reward is parsed as float
            # 确保奖励解析为浮点数
            result["reward"] = float(llm_response_dict.get("reward", 0.0))
            result["newly_addressed_sub_query"] = str(llm_response_dict.get("newly_addressed_sub_query", ""))
            result["derived_answer_for_sub_query"] = str(llm_response_dict.get("derived_answer_for_sub_query", ""))
            result["is_contradictory"] = llm_response_dict.get("is_contradictory", False)

            updated_solved = llm_response_dict.get("updated_solved_sub_queries", [])
            updated_needed = llm_response_dict.get("updated_needed_sub_queries", [])

            if isinstance(updated_solved, list) and isinstance(updated_needed, list):
                # Update the passed-by-reference sets
                # 更新通过引用传递的集合
                solved_sub_queries_ref.clear()
                solved_sub_queries_ref.update(updated_solved)
                needed_sub_queries_ref.clear()
                needed_sub_queries_ref.update(updated_needed)
                logger.info(f"  (In-place update) LLM-provided: Solved: {solved_sub_queries_ref}, Needed: {needed_sub_queries_ref}")
                
                # Store for cache
                # 存储以备缓存
                updated_solved_for_cache = list(solved_sub_queries_ref)
                updated_needed_for_cache = list(needed_sub_queries_ref)
            else:
                logger.warning(f"  (In-place update) LLM did not provide valid lists for updated_solved_sub_queries or updated_needed_sub_queries. Sets remain unchanged.")
                # Store current state for cache if LLM output was invalid
                # 如果 LLM 输出无效，则存储当前状态以备缓存
                updated_solved_for_cache = list(solved_sub_queries_ref)
                updated_needed_for_cache = list(needed_sub_queries_ref)

        else:
            logger.error(f"LLM did not return a dictionary for reward evaluation: {llm_response_dict}")
            # Store current state for cache if LLM output was invalid
            # 如果 LLM 输出无效，则存储当前状态以备缓存
            updated_solved_for_cache = list(solved_sub_queries_ref)
            updated_needed_for_cache = list(needed_sub_queries_ref)

        # New consistency check logic: If sub-query and derived answer are present, verify consistency
        # 新的一致性检查逻辑：如果子查询和派生答案存在，则验证一致性
        consistency_multiplier = 1.0 # Default to 1.0 (no change)

        if result["newly_addressed_sub_query"] and result["derived_answer_for_sub_query"]:
            logger.info("Performing consistency check for newly addressed sub-query and derived answer.")
            is_consistent = await self._check_answer_evidence_consistency(
                result["newly_addressed_sub_query"],
                result["derived_answer_for_sub_query"],
                new_evidence_text
            )
            if not is_consistent:
                consistency_multiplier = 0.0 # If not consistent, make reward 0
                logger.warning(f"Consistency check failed! Sub-query: '{result['newly_addressed_sub_query']}', Derived Answer: '{result['derived_answer_for_sub_query'][:50]}...', Evidence: '{new_evidence_text[:50]}...'. Setting consistency_multiplier to 0.0.")
            else:
                logger.info("Consistency check passed. Consistency_multiplier remains 1.0.")
        
        # Apply the consistency multiplier to the LLM's initial reward
        # 将一致性乘数应用于 LLM 的初始奖励
        result["reward"] *= consistency_multiplier

        # Enforce negative reward for contradiction based on is_contradictory flag
        # 根据 is_contradictory 标志强制执行负奖励
        if result["is_contradictory"] and result["reward"] > -0.5:
            result["reward"] = -1.0 # Ensure severe penalty for direct contradiction
            logger.warning(f"Contradiction detected, enforcing reward to {result['reward']:.2f}.")
        
        # Cache the result before returning (must be done AFTER all modifications to result)
        # 在返回之前缓存结果（必须在所有修改结果之后完成）
        # Note: the cache key for evaluate_evidence_impact will now have an updated result due to consistency_multiplier.
        # This is fine, as subsequent identical calls will get the final calculated reward.
        # 注意：evaluate_evidence_impact 的缓存键现在将因 consistency_multiplier 而更新结果。
        # 这没关系，因为后续相同的调用将获得最终计算出的奖励。
        self.llm_response_cache[cache_key] = {
            'evaluation_result': result,
            'updated_solved_sub_queries': updated_solved_for_cache,
            'updated_needed_sub_queries': updated_needed_for_cache
        }

        return result

    async def synthesize_answer(self, query: str, evidence_list: list[dict]) -> str:
        """
        Synthesizes an answer to the query based on the provided evidence.
        """
        # 根据提供的证据合成查询的答案。
        # Cache for synthesis to avoid redundant calls
        # 合成的缓存，避免冗余调用
        cache_key = (query, tuple(sorted(ev['content'] for ev in evidence_list)))
        if cache_key in self.llm_response_cache:
            logger.debug(f"Cache hit for synthesis: {cache_key[0][:50]}...")
            return self.llm_response_cache[cache_key]

        evidence_texts = "\n".join([f"Evidence {i+1}: {ev['content']}" for i, ev in enumerate(evidence_list)])
        prompt = f"""
        You are an expert in synthesizing information. Your task is to generate a concise and accurate answer to the 'Original Query' using ONLY the 'Supporting Evidence' provided. Do not use outside knowledge. If the evidence is insufficient to answer the query, state that clearly.

        Original Query: "{query}"

        Supporting Evidence:
        {evidence_texts}

        Generated Answer:
        """
        answer = await self.llm_caller_func(prompt, temperature=0.0)
        self.llm_response_cache[cache_key] = answer.strip() # Cache the answer
        return answer.strip()

    async def get_initial_sub_queries(self, question: str) -> set:
        cache_key = ("initial_sub_queries", question)
        if cache_key in self.llm_response_cache:
            logger.debug(f"Cache hit for initial sub-queries: {cache_key[1][:50]}...")
            return self.llm_response_cache[cache_key]

        prompt = f"""
        Decompose the following complex question into a list of atomic, factual sub-questions that need to be answered to fully resolve the original question.
        
        Original Question: "{question}"
        
        Provide the sub-questions as a JSON object with a single key "sub_queries" which maps to a list of strings.
        Example: {{"sub_queries": ["Who directed the movie X?", "When was director X born?", "Which of these birth dates is later?"]}}
        """
        llm_response_dict = await self.llm_caller_func(prompt, temperature=0.7, response_format_type="json")

        sub_queries_set = set()
        if isinstance(llm_response_dict, dict) and "sub_queries" in llm_response_dict:
            sub_queries_set = set(str(sq) for sq in llm_response_dict["sub_queries"] if isinstance(sq, str))
        else:
            logger.warning(f"LLM failed to decompose question or returned unexpected format: {llm_response_dict}. Falling back to heuristic.")
            # Heuristic fallback for testing/demonstration if LLM fails
            # 如果 LLM 失败，则启发式回退用于测试/演示
            if "Life Hits" in question and "It's In The Air" in question:
                sub_queries_set = {"Who is the director of Life Hits?", "When was the director of Life Hits born?",
                        "Who is the director of It's In The Air?", "When was the director of It's In The Air born?"}
            elif "country that borders France" in question and "cheese" in question and "capital" in question:
                sub_queries_set = {"Which country borders France?", "Which country is known for its cheese?", "What is the capital of that country?"}
            else: # Generic fallback for broader cases
                # 适用于更广泛情况的通用回退
                sub_queries_set = {"Identify key entities in the question.", "Identify key relationships in the question."} # More general fallbacks if domain-specific ones don't apply

        self.llm_response_cache[cache_key] = sub_queries_set # Cache the result
        return sub_queries_set


# --- MCTS Classes ---
# --- MCTS 类 ---

class MCTSNode:
    """Represents a node in the MCTS tree."""
    # 表示 MCTS 树中的一个节点。
    def __init__(self,
                 evidence_id: int,
                 parent=None,
                 initial_question: str = None,
                 reasoning_path_details: list = None,
                 solved_sub_queries: set = None,
                 needed_sub_queries: set = None):
        self.evidence_id = evidence_id
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.is_terminal = False

        self.initial_question = initial_question if initial_question else (parent.initial_question if parent else "")
        self.reasoning_path_details = list(reasoning_path_details) if reasoning_path_details else (list(parent.reasoning_path_details) if parent else [])

        self.solved_sub_queries = set(solved_sub_queries) if solved_sub_queries else (set(parent.solved_sub_queries) if parent else set())
        self.needed_sub_queries = set(needed_sub_queries) if needed_sub_queries else (set(parent.needed_sub_queries) if parent else set())

        self.final_answer = ""
    
    # Improved UCT score calculation with dynamic exploration weight
    # 改进的 UCT 分数计算，具有动态探索权重
    def uct_score(self, exploration_weight: float, parent_visits: int):
        if self.visits == 0:
            return float('inf') # Prioritize unvisited nodes
        
        # Dynamic exploration weight: decreases as parent visits increase, encouraging exploitation later
        # Or you could use a fixed C for UCB1: C = exploration_weight
        # c_t = exploration_weight * math.sqrt(math.log(parent_visits + 1e-6) / (self.visits + 1e-6))
        
        # Using a fixed exploration_weight (C in UCB1) as provided in __init__
        # 使用 __init__ 中提供的固定探索权重 (UCB1 中的 C)
        exploitation_term = self.total_reward / self.visits
        exploration_term = exploration_weight * math.sqrt(math.log(parent_visits + 1e-6) / (self.visits + 1e-6))
        
        return exploitation_term + exploration_term


    def add_child(self, child_node):
        self.children.append(child_node)

class EvidenceSelectionMCTS:
    """
    Monte Carlo Tree Search for evidence selection in QA.
    """
    # 用于 QA 中证据选择的蒙特卡洛树搜索。
    def __init__(self,
                 evidence_file_path: str,
                 llm_reward_model: LLMRewardModel,
                 exploration_weight: float = 0.2,
                 max_simulations: int = 50,
                 max_path_length: int = 6,
                 top_n_evidence_selection: int = 5,
                 pruning_threshold_visits: int = 5): # New parameter for pruning
        self.evidence_file_path = evidence_file_path
        self.all_evidence_snippets, self.original_question, self.evidence_utility_scores = self._load_evidence_from_file() 
        
        self.llm_reward_model = llm_reward_model
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.max_path_length = max_path_length
        self.top_n_evidence_selection = top_n_evidence_selection 
        self.pruning_threshold_visits = pruning_threshold_visits # Store pruning threshold
        # 存储修剪阈值

        self.root = None
        self.initial_needed_sub_queries_global = set()
        self.collected_evidence_library = [] # New: Initialize the evidence library
        # 新增：初始化证据库
        

    def _load_evidence_from_file(self) -> (dict, str, dict):
        """
        Loads evidence snippets, the original question, and initializes evidence utility scores
        from the specified JSON file.
        Assumes the JSON structure contains an "original_query" key for the question
        and a "retrieved_evidence" key, which is a list of dictionaries,
        each with a "content" and "score" field.
        """
        # 从指定的 JSON 文件加载证据片段、原始问题，并初始化证据效用分数。
        # 假设 JSON 结构包含一个用于问题的 "original_query" 键
        # 和一个 "retrieved_evidence" 键，它是一个字典列表，
        # 每个字典都包含 "content" 和 "score" 字段。
        if not os.path.exists(self.evidence_file_path):
            raise FileNotFoundError(f"Evidence file not found: {self.evidence_file_path}")

        logger.info(f"Loading evidence and original question from: {self.evidence_file_path}")
        evidence_data = {}
        original_question = ""
        evidence_utility_scores_init = {}

        try:
            with open(self.evidence_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and "original_query" in data and isinstance(data["original_query"], str):
                    original_question = data["original_query"]
                    logger.info(f"Found original_query: {original_question}")
                else:
                    logger.warning("Could not find 'original_query' key or it's not a string in the evidence file.")

                if isinstance(data, dict) and "retrieved_evidence" in data:
                    retrieved_evidence_list = data["retrieved_evidence"]
                    if isinstance(retrieved_evidence_list, list):
                        current_id = 0
                        for item in retrieved_evidence_list:
                            if isinstance(item, dict) and "content" in item:
                                # 强制从0开始依次编号，忽略原始数据中的'id'字段
                                # Force sequential numbering starting from 0, ignoring the 'id' field in original data
                                evidence_id = current_id 
                                
                                evidence_data[evidence_id] = item["content"]
                                
                                # 加载证据的score，并确保它是浮点数
                                # Load the evidence score and ensure it's a float
                                initial_score = item.get("score", 0.0) 
                                if not isinstance(initial_score, (int, float)):
                                    logger.warning(f"Invalid 'score' for evidence ID {evidence_id}. Defaulting to 0.0. Score: {initial_score}")
                                    initial_score = 0.0
                                evidence_utility_scores_init[evidence_id] = initial_score

                                current_id += 1 # Increment for next item
                            else:
                                logger.warning(f"Skipping malformed evidence item: {item}. Missing 'content' or not a dictionary.")
                    else:
                        logger.warning("Expected 'retrieved_evidence' to be a list, but found a different type.")
               
                                    
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file: {self.evidence_file_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the evidence file: {e}")

        if not evidence_data:
            logger.warning("No evidence snippets were loaded from the file.")
        logger.info(f"Loaded {len(evidence_data)} evidence snippets and their initial utility scores.")
        
        if not original_question:
            logger.error("Original question could not be loaded from the evidence file. Using a default empty string.")
            original_question = "No question loaded."

        return evidence_data, original_question, evidence_utility_scores_init


    async def search(self):
        self.initial_needed_sub_queries_global = await self.llm_reward_model.get_initial_sub_queries(self.original_question)
        self.root = MCTSNode(
            evidence_id=None,
            initial_question=self.original_question,
            reasoning_path_details=[],
            solved_sub_queries=set(),
            needed_sub_queries=set(self.initial_needed_sub_queries_global)
        )
        logger.info(f"Initial needed sub-queries: {self.root.needed_sub_queries}")

        solved_path_found_overall = False
        total_simulations_run = 0

        for simulation in range(self.max_simulations):
            total_simulations_run += 1
            logger.info(f"\n--- Simulation {simulation + 1}/{self.max_simulations} ---")
            current_node = self.root
            path_to_leaf = [current_node]

            # Selection Phase
            # 选择阶段
            logger.debug(f"Selection: Starting from root node. Visits: {current_node.visits}")
            while current_node.children and not current_node.is_terminal:
                # Use parent.visits for UCT calculation
                # 使用父节点访问量进行 UCT 计算
                next_node = max(current_node.children, key=lambda child: child.uct_score(self.exploration_weight, current_node.visits))
                current_node = next_node
                path_to_leaf.append(current_node)
                logger.debug(f"Selection: Moving to child node (Evidence ID: {current_node.evidence_id}). UCT: {current_node.uct_score(self.exploration_weight, current_node.parent.visits if current_node.parent else 1):.4f}, Visits: {current_node.visits}") # Added parent check for root

            current_path_evidence_ids = [detail[2] for detail in current_node.reasoning_path_details if len(detail) > 2]
            logger.info(f"Selection: Reached leaf node (Evidence IDs: {current_path_evidence_ids}).")
            logger.info(f"  Current Node Solved Sub-queries: {current_node.solved_sub_queries}")
            logger.info(f"  Current Node Needed Sub-queries: {current_node.needed_sub_queries}")

            # Early Stopping Check (after selection, before expansion)
            # 提前停止检查（在选择之后，扩展之前）
            all_initial_sub_queries_solved_current_node = (current_node.needed_sub_queries == set() and 
                                                           current_node.solved_sub_queries.issuperset(self.initial_needed_sub_queries_global))
            
            if all_initial_sub_queries_solved_current_node:
                logger.info(f"  Current path has solved all initial sub-queries. Marking as terminal for this simulation.")
                current_node.is_terminal = True
                solved_path_found_overall = True
                total_path_reward = 10.0 # High reward for solving, now float
                current_node.final_answer = "Path indicates all sub-queries solved. Final answer to be synthesized."
                
                logger.debug(f"Backpropagation (Early Exit Solved Path): Propagating reward {total_path_reward} up the tree.")
                # 反向传播（提前退出已解决路径）：将奖励 {total_path_reward} 传播到树的顶部。
                for node in reversed(path_to_leaf):
                    node.visits += 1
                    node.total_reward += total_path_reward
                    if node.parent is not None:
                        total_path_reward *= 0.9 # Decay factor for rewards up the tree
                break

            # Expansion Phase (only if not terminal and max_path_length not reached)
            # 扩展阶段（仅当未终止且未达到最大路径长度时）
            if not current_node.is_terminal and len(current_node.reasoning_path_details) < self.max_path_length:
                logger.info(f"Expansion: Attempting to expand node (Path Length: {len(current_node.reasoning_path_details)}/{self.max_path_length}).")
                available_evidence_ids = list(self.all_evidence_snippets.keys())
                
                # Exclude evidence already in the current path
                # 排除当前路径中已有的证据
                candidate_evidence_for_expansion = [
                    eid for eid in available_evidence_ids if eid not in current_path_evidence_ids
                ]

                if not candidate_evidence_for_expansion:
                    logger.info("Expansion: No new evidence to expand this path. Marking as terminal.")
                    current_node.is_terminal = True
                    step_reward = -0.5
                else:
                    # Select top-N highest scoring evidence snippets (based on initial utility + MCTS updates)
                    # 根据初始效用和 MCTS 更新选择得分最高的 Top-N 证据片段
                    sorted_candidates = sorted(candidate_evidence_for_expansion, 
                                            key=lambda eid: self.evidence_utility_scores.get(eid, 0.0), # 基于当前效用分数（包含初始分数和MCTS累积奖励）排序
                                            reverse=True)
                    
                    top_n_candidates = sorted_candidates[:self.top_n_evidence_selection]
                    
                    if not top_n_candidates: 
                        top_n_candidates = candidate_evidence_for_expansion # Fallback if top_n leads to empty
                        
                    # Prioritize evidence that helps solve 'needed_sub_queries'
                    # 优先选择有助于解决“所需子查询”的证据
                    # You could add another LLM call here to re-rank top_n_candidates based on specific sub-query relevance
                    
                    # new_evidence_id = random.choice(top_n_candidates) # 从Top-N中随机选择一个
                    new_evidence_id = top_n_candidates[0]
                    new_evidence_text = self.all_evidence_snippets[new_evidence_id]
                    logger.info(f"Expansion: Selected new evidence ID: {new_evidence_id} (Text: {new_evidence_text[:50]}...) from top-{len(top_n_candidates)} candidates.")

                    llm_eval_result = await self.llm_reward_model.evaluate_evidence_impact(
                        original_question=self.original_question,
                        current_reasoning_path_details=current_node.reasoning_path_details,
                        new_evidence_text=new_evidence_text,
                        solved_sub_queries_ref=current_node.solved_sub_queries,
                        needed_sub_queries_ref=current_node.needed_sub_queries
                    )
                    
                    reward = llm_eval_result["reward"]
                    is_contradictory = llm_eval_result["is_contradictory"]
                    newly_addressed_sub_query = llm_eval_result["newly_addressed_sub_query"]
                    derived_answer_for_sub_query = llm_eval_result["derived_answer_for_sub_query"]

                    logger.info(f"  LLM Eval Result: Reward={reward:.2f}, Newly Addressed Sub-query='{newly_addressed_sub_query}', Answer='{derived_answer_for_sub_query[:50]}...', Contradictory={is_contradictory}")

                    # NEW: Collect evidence for the evidence library
                    # 新增：为证据库收集证据
                    if newly_addressed_sub_query and derived_answer_for_sub_query:
                        self.collected_evidence_library.append({
                            "sub_query": newly_addressed_sub_query,
                            "sub_answer": derived_answer_for_sub_query,
                            "evidence_id": new_evidence_id,
                            "evidence_content": new_evidence_text
                        })
                        logger.info(f"  Added to collected evidence library: Sub-query='{newly_addressed_sub_query}', Answer='{derived_answer_for_sub_query[:50]}...'")

                    # Update evidence utility score based on positive rewards
                    # 根据正奖励更新证据效用分数
                    if reward > 0: # Only increment for positive impact
                        # self.evidence_utility_scores[new_evidence_id] += reward # Add the reward value to utility
                        logger.info(f"  Evidence ID {new_evidence_id} utility score incremented to {self.evidence_utility_scores[new_evidence_id]:.2f}")


                    new_reasoning_step = None
                    if newly_addressed_sub_query:
                        new_reasoning_step = (newly_addressed_sub_query, derived_answer_for_sub_query, new_evidence_id, new_evidence_text)
                    else:
                        new_reasoning_step = ("", "", new_evidence_id, new_evidence_text)

                    if is_contradictory:
                        step_reward = reward # Use the negative reward directly
                        current_node.is_terminal = True
                        logger.warning(f"  Contradictory evidence found. Marking node as terminal with reward {step_reward:.2f}.")
                        for node_to_propagate in reversed(path_to_leaf):
                            node_to_propagate.visits += 1
                            node_to_propagate.total_reward += step_reward
                            step_reward *= 0.9 # Decay factor
                        continue

                    new_reasoning_path_details = list(current_node.reasoning_path_details)
                    if new_reasoning_step:
                        new_reasoning_path_details.append(new_reasoning_step)

                    new_node = MCTSNode(
                        evidence_id=new_evidence_id,
                        parent=current_node,
                        reasoning_path_details=new_reasoning_path_details,
                        solved_sub_queries=current_node.solved_sub_queries,
                        needed_sub_queries=current_node.needed_sub_queries
                    )
                    current_node.add_child(new_node)
                    current_node = new_node
                    path_to_leaf.append(current_node)
                    
                    step_reward = reward

            else:
                logger.info(f"Expansion: Node is terminal or max path length reached ({len(current_node.reasoning_path_details)}/{self.max_path_length}).")
                step_reward = -0.5
                current_node.is_terminal = True

            # Backpropagation Phase
            # 反向传播阶段
            if not solved_path_found_overall:
                total_path_reward = step_reward
                if not current_node.is_terminal and len(current_node.reasoning_path_details) >= self.max_path_length:
                    total_path_reward -= 1.0 # Additional penalty for incomplete paths at max length, float
                    # 对达到最大长度但路径不完整的额外惩罚

                logger.info(f"Backpropagation: Propagating reward {total_path_reward:.2f} up the tree from node (Evidence IDs: {[detail[2] for detail in current_node.reasoning_path_details if len(detail) > 2]}).")

                for node_to_propagate in reversed(path_to_leaf):
                    node_to_propagate.visits += 1
                    node_to_propagate.total_reward += total_path_reward
                    if node_to_propagate.parent is not None:
                        total_path_reward *= 0.9 # Decay factor
            
            # Pruning (After backpropagation in each simulation)
            # 剪枝（在每次模拟的反向传播之后）
            if simulation % 10 == 0 and simulation > 0: # Prune every 10 simulations, after some initial exploration
                # 每 10 次模拟进行一次剪枝，在一些初始探索之后
                self._prune_tree()

            if solved_path_found_overall:
                logger.info(f"MCTS Search terminated early because a solved path was found.")
                break

        # Deduplicate the collected evidence library
        # 对收集到的证据库进行去重
        deduplicated_evidence_library_map = {}
        for entry in self.collected_evidence_library:
            key = (entry["sub_query"], entry["sub_answer"])
            if key not in deduplicated_evidence_library_map:
                deduplicated_evidence_library_map[key] = entry
            # You could add logic here to prioritize which duplicate to keep if needed,
            # e.g., based on some quality score of the evidence_content.
            # For simplicity, keeping the first encountered.
            # 您可以在此处添加逻辑，以在需要时优先保留哪个重复项，
            # 例如，基于证据内容的某些质量分数。
            # 为简单起见，保留第一个遇到的。
        self.collected_evidence_library = list(deduplicated_evidence_library_map.values())
        logger.info(f"Deduplicated evidence library contains {len(self.collected_evidence_library)} unique entries.")

        best_path_details, final_answer_synthesis = await self._get_best_path_and_synthesize()
        
        self.root.final_answer = final_answer_synthesis

        logger.info(f"\nTotal MCTS Simulations Run: {total_simulations_run}")
        
        top_k_paths = self._get_top_k_paths(self.root, k=5)
        
        # Return the collected evidence library as well
        # 也返回收集到的证据库
        return best_path_details, top_k_paths, self.collected_evidence_library

    def _prune_tree(self):
        """
        Prunes nodes with very few visits, especially those that are not on promising paths.
        This helps manage memory and focuses search on more relevant areas.
        """
        # 剪枝访问次数很少的节点，特别是那些不在有前景路径上的节点。
        # 这有助于管理内存并将搜索集中在更相关的区域。
        logger.info(f"Starting tree pruning. Threshold visits: {self.pruning_threshold_visits}")
        nodes_to_check = [self.root]
        nodes_to_prune_children_of = defaultdict(list) # parent_node -> [child_node_to_prune]
        # 父节点 -> [要剪枝的子节点]

        while nodes_to_check:
            node = nodes_to_check.pop(0)
            
            # Collect children to potentially prune
            # 收集可能要剪枝的子节点
            children_to_keep = []
            for child in node.children:
                if child.visits < self.pruning_threshold_visits and child.total_reward <= 0:
                    nodes_to_prune_children_of[node].append(child)
                    logger.debug(f"Marking child node (ID: {child.evidence_id}) for pruning. Visits: {child.visits}, Reward: {child.total_reward:.2f}")
                else:
                    children_to_keep.append(child)
                    nodes_to_check.append(child) # Continue BFS for promising children
                    # 继续对有前景的子节点进行 BFS

            # Actually prune (remove from parent's children list)
            # 实际剪枝（从父节点的子节点列表中删除）
            if node in nodes_to_prune_children_of:
                node.children = children_to_keep
                logger.info(f"Pruned {len(nodes_to_prune_children_of[node])} children from node (ID: {node.evidence_id if node.evidence_id is not None else 'Root'}).")
        
        # After pruning, check for isolated nodes (not strictly necessary as MCTS doesn't usually keep them)
        # But good for sanity check if tree structure gets complex.
        # 剪枝后，检查孤立节点（MCTS 通常不会保留它们，因此不是严格必需的）
        # 但如果树结构变得复杂，则有助于进行健全性检查。
        
        logger.info("Tree pruning complete.")


    async def _get_best_path_and_synthesize(self) -> (list, str):
        best_node = None
        
        queue = [self.root]
        all_visited_nodes = [] 

        while queue:
            node = queue.pop(0)
            if node.visits > 0: 
                all_visited_nodes.append(node)
            for child in node.children:
                queue.append(child)
        
        if not all_visited_nodes:
            logger.warning("No nodes were visited during MCTS search.")
            return [], "No paths were explored."

        solved_all_initial_sub_queries_nodes = [
            n for n in all_visited_nodes 
            if n.needed_sub_queries == set() and n.solved_sub_queries.issuperset(self.initial_needed_sub_queries_global)
        ]
        
        if solved_all_initial_sub_queries_nodes:
            # If there are solved nodes, pick the one with the highest average reward
            # 如果存在已解决的节点，选择平均奖励最高的节点
            best_node = max(solved_all_initial_sub_queries_nodes, key=lambda n: n.total_reward / n.visits if n.visits > 0 else -float('inf'))
            logger.info(f"Found {len(solved_all_initial_sub_queries_nodes)} nodes that solved all initial sub-queries. Selecting best among them.")
        else:
            # If no node solved all, pick the one with the best average reward regardless of solved status
            # 如果没有节点解决所有问题，则选择平均奖励最高的节点，无论是否解决
            best_node = max(all_visited_nodes, key=lambda n: n.total_reward / n.visits if n.visits > 0 else -float('inf'))
            logger.info("No path solved all initial sub-queries. Selecting path with highest average reward.")

        if best_node is None:
            logger.error("Could not determine a best node after search.")
            return [], "Could not determine a best path."

        path_details_reversed = []
        current = best_node
        while current.parent is not None:
            if current.reasoning_path_details:
                # 获取当前步骤的子查询和答案
                # Get sub-query and answer for the current step
                step = current.reasoning_path_details[-1]
                sq, sa, eid, et = step
                # 只有当子查询不为空时才添加到路径中
                # Only add to path if sub-query is not empty
                if sq: 
                    path_details_reversed.append(step)
            current = current.parent
        path_details = path_details_reversed[::-1]

        final_context_for_synthesis = []
        for sq, sa, eid, et in path_details:
            if sq and sa:
                final_context_for_synthesis.append(f"Sub-query: {sq}\nAnswer: {sa}\nEvidence (ID {eid}): {et}")
            else:
                # This else block should ideally not be hit if we filter above, but keeping for robustness
                # 理论上，如果我们上面进行了过滤，这个 else 块不应该被触及，但为了鲁棒性保留。
                final_context_for_synthesis.append(f"Evidence (ID {eid}): {et}")

        synthesize_prompt = f"""
        Given the following original question and a series of reasoning steps with evidence,
        synthesize a comprehensive and concise final answer to the original question.
        
        Original Question: "{self.original_question}"
        
        ---
        Reasoning Steps and Evidence:
        {".\n".join(final_context_for_synthesis)}
        ---
        
        Based on the above, provide the final answer. If the question cannot be fully answered, state why.
        """
        
        final_answer = await self.llm_reward_model.llm_caller_func(synthesize_prompt, temperature=0.1, response_format_type="text")
        
        if not final_answer or "cannot be fully answered" in final_answer.lower() or "not fully answered" in final_answer.lower():
            final_answer = "Answer could not be fully synthesized or all sub-queries not sufficiently addressed."

        return path_details, final_answer

    def _get_top_k_paths(self, root_node: MCTSNode, k: int = 5) -> list[dict]:
        """
        Collects all paths (nodes) and returns the top K based on average reward.
        """
        # 收集所有路径（节点），并根据平均奖励返回前 K 个。
        all_paths = []
        q = [root_node]
        while q:
            node = q.pop(0)
            if node.visits > 0: 
                avg_reward = node.total_reward / node.visits
                all_paths.append({
                    "node": node,
                    "score": avg_reward,
                    "details": node.reasoning_path_details
                })
            for child in node.children:
                q.append(child)
        
        all_paths.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Collected {len(all_paths)} total paths. Returning top {min(k, len(all_paths))}.")
        return all_paths[:k]

# --- Example Usage ---
# --- 示例用法 ---
async def main():
    evidence_file = "./tmp/temp_evidence_for_mcts.json"
    
    dummy_data = {
        "original_query": "What is the capital of France and what is its population?",
        "retrieved_evidence": [
            {"content": "Paris is the capital of France.", "score": 0.9, "display_id": "ev_paris_cap", "id": 1},
            {"content": "The population of Paris is approximately 2.1 million.", "score": 0.85, "display_id": "ev_paris_pop", "id": 2},
            {"content": "The Eiffel Tower is in Paris.", "score": 0.6, "display_id": "ev_eiffel", "id": 3},
            {"content": "Berlin is the capital of Germany.", "score": 0.3, "display_id": "ev_berlin_cap", "id": 4},
            {"content": "The Louvre Museum is a famous landmark in Paris.", "score": 0.7, "display_id": "ev_louvre", "id": 5},
            {"content": "The current French president is Emmanuel Macron.", "score": 0.2, "display_id": "ev_macron", "id": 6},
            {"content": "France is known for its exquisite cuisine.", "score": 0.1, "display_id": "ev_cuisine", "id": 7},
            {"content": "The river Seine flows through Paris.", "score": 0.5, "display_id": "ev_seine", "id": 8},
        ]
    }
    os.makedirs(os.path.dirname(evidence_file), exist_ok=True)
    with open(evidence_file, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Dummy evidence file created at: {evidence_file}")


    llm_reward_model = LLMRewardModel(llm_caller_func=call_llm)

    mcts = EvidenceSelectionMCTS(
        evidence_file_path=evidence_file,
        llm_reward_model=llm_reward_model,
        max_simulations=50, # Increased simulations to allow more exploration
        # 增加模拟次数以允许更多探索
        max_path_length=7,
        top_n_evidence_selection=5,
        pruning_threshold_visits=3 # Prune nodes with <=3 visits and non-positive reward
        # 剪枝访问次数 <=3 且奖励非正的节点
    )

    # 修正：现在 search() 返回 3 个值
    best_path_details, top_k_paths, collected_evidence_library = await mcts.search()

    logger.info("\n--- MCTS Search Complete ---")
    logger.info(f"Original Question: {mcts.original_question}")
    logger.info(f"Best Evidence Path (Details): {best_path_details}")
    logger.info(f"Synthesized Final Answer (from root): {mcts.root.final_answer}")
    logger.info(f"\n--- Collected Evidence Library ---")
    # --- 收集到的证据库 ---
    for entry in collected_evidence_library:
        logger.info(f"  Sub-query: {entry['sub_query']}")
        logger.info(f"  Sub-answer: {entry['sub_answer']}")
        logger.info(f"  Evidence ID: {entry['evidence_id']}")
        logger.info(f"  Evidence Content: {entry['evidence_content'][:100]}...")
        logger.info("-" * 20)
    
    if top_k_paths:
        logger.info("\nTop 5 Alternative Paths (by average reward):")
        # --- Top 5 替代路径（按平均奖励）---
        for path_idx, path in enumerate(top_k_paths):
            logger.info(f"  Path {path_idx + 1} (Score: {path['score']:.4f}):")
            for i, (sub_q, sub_a, ev_id, ev_text) in enumerate(path['details']):
                logger.info(f"    Step {i+1}: Sub-query='{sub_q}', Answer='{sub_a}', Evidence ID='{ev_id}', Content='{ev_text[:100]}...'")

    if os.path.exists(evidence_file):
        os.remove(evidence_file)
        logger.info(f"Cleaned up dummy evidence file: {os.path.abspath(evidence_file)}")
        # 清理虚拟证据文件

if __name__ == "__main__":
    asyncio.run(main())

