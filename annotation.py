import gradio as gr
import re
import random
import hashlib
import json
import torch
import logging
import numpy as np
import asyncio
import os
import glob
import time
import tempfile # For temporary file management

from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import MCTS related classes and functions from mcts.py
# 导入 MCTS 相关的类和函数。
from mcts import LLMRewardModel, EvidenceSelectionMCTS, call_llm as mcts_call_llm # Alias to avoid conflict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d')
logger = logging.getLogger(__name__)

# --- Configuration ---
# --- 配置 ---
LLM_MODEL = "gpt-4o-mini"
TOP_K_RETRIEVED_CANDIDATES = 20
# Make sure this path is correct or accessible. Replace with your actual path.
# 确保此路径正确或可访问。请替换为您的实际路径。
BGE_MODEL_NAME = "/apdcephfs_cq8/share_1324356/xxx/pytorch_models/BAAI/bge-large-en-v1.5"
# Replace with your actual API key and base URL. Ensure secure management.
# IMPORTANT: For production, use environment variables or a secure configuration management system
# to store API keys, NOT hardcoding them directly in the code.
# 请替换为您的实际 API 密钥和基本 URL。确保安全管理。
# 重要提示：对于生产环境，请使用环境变量或安全的配置管理系统来存储 API 密钥，不要直接硬编码在代码中。
# client = AsyncOpenAI(api_key="xxx", base_url="xxx")

# --- Model Loading ---
# --- 模型加载 ---
bge_model = None
try:
    logger.info(f"Attempting to load BGE model from {BGE_MODEL_NAME}")
    bge_model = SentenceTransformer(BGE_MODEL_NAME)
    if torch.cuda.is_available():
        bge_model.to("cuda")
        logger.info("BGE model moved to CUDA successfully.")
    else:
        logger.info("CUDA not available, BGE model running on CPU.")
except Exception as e:
    logger.error(f"Error loading BGE model: {e}. BGE retrieval will be skipped.", exc_info=True)
    bge_model = None

# --- Global State ---
# --- 全局状态 ---
current_dataset = []
current_dataset_index = 0
output_save_dir = "./saved_annotations"
os.makedirs(output_save_dir, exist_ok=True)
logger.info(f"Output save directory created/verified: {output_save_dir}")

# Define a temporary directory for evidence files
# 定义一个用于证据文件的临时目录。
TMP_DIR = "./tmp"
os.makedirs(TMP_DIR, exist_ok=True)
logger.info(f"Temporary directory created/verified: {TMP_DIR}")


# --- Utility Functions ---
# --- 工具函数 ---

def load_dataset(dataset_name: str):
    """Loads a dataset from a JSONL file."""
    # 从 JSONL 文件加载数据集。
    global current_dataset
    dataset = []
    dataset_path = f'./data/{dataset_name}.jsonl' # Assuming datasets are in a 'data' folder
    logger.info(f"Attempting to load dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        current_dataset = []
        return []

    try:
        start_time = time.time()
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for item in f.readlines():
                dataset.append(json.loads(item))
        end_time = time.time()
        logger.info(f"Loaded {len(dataset)} items from {dataset_name}.jsonl in {end_time - start_time:.2f} seconds.")
        current_dataset = dataset
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}.jsonl: {e}", exc_info=True)
        current_dataset = []
        return []

def generate_content_hash_id(evidence_content: str) -> str:
    """Generates a short MD5 hash for a given content string."""
    # 为给定内容字符串生成一个短的 MD5 哈希。
    return hashlib.md5(evidence_content.encode('utf-8')).hexdigest()

def split_document_into_chunks(text: str, chunk_size_words: int = 256, overlap_percentage: float = 0.20):
    """Splits a document into overlapping chunks based on word count."""
    # 根据字数将文档分割成重叠的块。
    start_time = time.time()
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)
    chunks = []

    if chunk_size_words <= 0:
        logger.warning("Chunk size must be positive. Returning empty list.")
        return []

    overlap_words = max(1, int(chunk_size_words * overlap_percentage))
    step = chunk_size_words - overlap_words
    if step <= 0: # Ensure step is at least 1 to avoid infinite loops if overlap is too high
        step = 1

    i = 0
    while i < total_words:
        end_index = min(i + chunk_size_words, total_words)

        chunk_words = words[i:end_index]
        chunk = " ".join(chunk_words)

        if chunk:
            chunks.append(chunk)

        # Move to the next chunk's start
        i += step

        # Handle the very last chunk to ensure all content is covered, especially if it's smaller than `step`
        # 处理最后一个块，以确保所有内容都被覆盖，特别是当它小于 `step` 时。
        if i >= total_words and end_index < total_words:
            remaining_words = words[end_index:]
            if remaining_words:
                last_chunk_content = " ".join(remaining_words)
                # Only add if it's new content and not just a duplicate of the previous chunk
                # 仅当它是新内容且不是前一个块的重复时才添加。
                if not chunks or (last_chunk_content.strip() and last_chunk_content.strip() != chunks[-1].strip()):
                    chunks.append(last_chunk_content)
            break
        elif i >= total_words:
            break # All words processed

    end_time = time.time()
    logger.info(f"Document (total words {total_words}) split into {len(chunks)} chunks (avg {chunk_size_words} words) in {end_time - start_time:.4f} seconds.")
    return chunks

def extract_json_from_text(input_text: str) -> str:
    """Attempts to extract a JSON string from a larger text, handling common LLM output issues."""
    # 尝试从较大文本中提取 JSON 字符串，处理常见的 LLM 输出问题。
    start_time = time.time()
    try:
        # Find the first opening brace and last closing brace
        # 找到第一个左大括号和最后一个右大括号。
        start_index = input_text.find('{')
        end_index = input_text.rfind('}')

        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_string = input_text[start_index : end_index + 1]
            logger.debug(f"JSON extracted successfully in {time.time() - start_time:.4f} seconds.")
            return json_string
        else:
            logger.warning("No valid JSON object boundaries found using simple string find. Returning empty JSON.")
            return "{}" # Return an empty JSON object string if not found
    except Exception as e:
        logger.error(f"Error during JSON extraction: {e}. Returning empty JSON.", exc_info=True)
        return "{}" # Return an empty JSON object string on error

# --- LLM Interactions ---
# --- LLM 交互 ---

async def call_llm(prompt: str, model: str = LLM_MODEL, temperature: float = 0.0, response_format_type: str = "text"):
    """
    Asynchronously calls the LLM with a given prompt.
    Includes retry logic and optional JSON response handling.
    """
    # 使用给定提示异步调用 LLM。
    # 包括重试逻辑和可选的 JSON 响应处理。
    start_time = time.time()
    retries = 3
    while retries:
        try:
            retries -= 1
            logger.info(f"Calling LLM ({model}, attempt {3-retries}). Prompt (truncated): {prompt[:200]}...")

            # OpenAI API allows specifying response_format directly for JSON
            # OpenAI API 允许直接为 JSON 指定 response_format。
            response_format = {"type": "json_object"} if 'json' in response_format_type else {"type": "text"}

            response = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                            max_tokens=1000, # Increased max_tokens for potentially longer JSON responses
                            response_format=response_format # Use response_format parameter
                        )
            end_time = time.time()
            content = response.choices[0].message.content.strip()
            logger.info(f"LLM call successful in {end_time - start_time:.2f} seconds. Response (truncated): {content[:200]}...")

            if 'json' in response_format_type:
                try:
                    # If response_format="json_object" was used, the content should already be valid JSON
                    # 如果使用了 response_format="json_object"，则内容应该已经是有效的 JSON。
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    logger.error("LLM did not return valid JSON despite response_format. Attempting regex extraction.")
                    return extract_json_from_text(content) # Fallback to regex extraction
            return content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}. Retries left: {retries}", exc_info=True)
            if retries == 0:
                break
            await asyncio.sleep(2) # Wait longer before retrying
    logger.error("LLM call failed after multiple retries.")
    return ""

async def expand_query_with_llm(original_query: str) -> dict:
    """Expands an original query into keywords and sub-questions using an LLM."""
    # 使用 LLM 将原始查询扩展为关键字和子问题。
    logger.info(f"Starting LLM-based query expansion for: '{original_query}'")
    prompt = f"""
    You are an expert Query Expansion specialist. Your task is to precisely expand a given original query into multiple types to maximize information retrieval recall.

    **YOUR OUTPUT MUST BE A STRICT JSON OBJECT.** Do not include any other text, explanations, or formatting outside of the JSON.
    The JSON object must contain the following keys, and their values must strictly adhere to these constraints:

    - **'keywords'**:
        * A JSON array (list) of strings.
        * Each string must be an independent, important, and searchable **word or phrase** (e.g., noun phrases, technical terms, proper nouns).
        * These keywords should be **directly from the original query or highly relevant to it**.
        * They should NOT contain full sentences or interrogative phrases (questions).
        * Example: ["World Car of the Year", "2020 World Car of the Year Award", "car height", "inches"]

    - **'decomposition'**:
        * A JSON array (list) of strings.
        * Each string must be a **simpler, more atomic sub-question**.
        * Each sub-question **must be a complete interrogative sentence** that can be answered independently.
        * These sub-questions collectively cover the **entire intent** of the original query.
        * Example: ["Which car won the World Car of the Year in 2020?", "What was the height of the 2020 World Car of the Year winner in inches?"]
    Ensure your output format is exactly as shown in the example below:
    Query: What was the height of the 2020~2022 World Car of the Year winner?
    Output:
    ```json
    {{
      "keywords": ["World Car Awards", "2020 World Car of the Year", "2021 World Car of the Year", "2022 World Car of the Year", "car height", "inches"],
      "decomposition": [
        "Which cars won the World Car of the Year from 2020 to 2022?",
        "What was the height of the 2020 World Car of the Year winner?",
        "What was the height of the 2021 World Car of the Year winner?",
        "What was the height of the 2022 World Car of the Year winner?"
      ]
    }}
    ```
    Query: Who is Sobe (Sister Of Saint Anne)'s grandchild?
    Output:
    ```json

    {{
      "keywords": ["the son of Sobe", "who is the son of Sobe's son", "son", "father"],
      "decomposition": [
        "Who is the son of Sobe?",
        "Who is the son of Sobe'son?",
        "Sobe is the father of whom?",
        "Sobe's father is the father of whom?"
      ]
    }}
    ```
    Query: Who is Helmichis's father-in-law?
    Output:
    ```json
    {{
      "keywords": ["Helmichis's father-in-law", "Helmichis's wife", "Helmichis's wife's father", "father", "wife"],
      "decomposition": [
        "Who is Helmichis's wife?",
        "Who is the father of Helmichis's wife?",
      ]
    }}
    ```
    Now, please expand the following query:
    Query: "{original_query}"
    Output:
    """
    start_time = time.time()
    expanded_json_str = await call_llm(prompt, temperature=0.0, response_format_type="json")
    end_time = time.time()
    logger.info(f"LLM query expansion completed in {end_time - start_time:.2f} seconds.")

    logger.debug(f"Expanded JSON string from LLM: {expanded_json_str}")

    default_expansion = {
        "keywords": [original_query],
        "decomposition": [original_query],
    }

    try:
        expanded_data = json.loads(expanded_json_str)
        for key in ["keywords", "decomposition"]:
            if key not in expanded_data or not isinstance(expanded_data[key], list):
                logger.warning(f"Key '{key}' missing or not a list in LLM output. Using default.")
                expanded_data[key] = default_expansion[key]
            # Ensure each item in the list is a non-empty string after stripping whitespace
            # 确保列表中的每个项目都是非空字符串，去除空白字符后。
            expanded_data[key] = [item.strip() for item in expanded_data[key] if isinstance(item, str) and item.strip()]

            # Always ensure the original query is present in both lists
            # 始终确保原始查询存在于两个列表中。
            if original_query not in expanded_data[key]:
                expanded_data[key].insert(0, original_query)
        logger.info(f"Query expansion result: Keywords={len(expanded_data['keywords'])}, Decomposition={len(expanded_data['decomposition'])}")
        return expanded_data
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError during query expansion: {e}. Raw LLM output: {expanded_json_str}", exc_info=True)
        return default_expansion
    except Exception as e:
        logger.error(f"An unexpected error occurred during query expansion: {e}. Raw LLM output: {expanded_json_str}", exc_info=True)
        return default_expansion

async def get_all_expanded_queries(user_query: str) -> list[str]:
    """Combines original query with LLM-expanded keywords and sub-questions."""
    # 将原始查询与 LLM 扩展的关键字和子问题结合起来。
    start_time = time.time()
    expanded_data = await expand_query_with_llm(user_query)

    all_expanded = []
    all_expanded.extend(expanded_data.get("keywords", []))
    all_expanded.extend(expanded_data.get("decomposition", []))

    # Deduplicate and ensure original query is first
    # 去重并确保原始查询在首位。
    unique_expanded = list(dict.fromkeys(all_expanded)) # Preserves order while deduplicating
    if user_query not in unique_expanded:
        unique_expanded.insert(0, user_query)
    else:
        # Move original query to the front if it's not already there
        # 如果原始查询不在首位，则将其移到首位。
        unique_expanded.remove(user_query)
        unique_expanded.insert(0, user_query)

    end_time = time.time()
    logger.info(f"Consolidated {len(unique_expanded)} unique expanded queries in {end_time - start_time:.4f} seconds.")
    return unique_expanded

# --- Retrieval Functions ---
# --- 检索函数 ---

def bm25_retrieval(queries: list[str], document_chunks: list[str], bm25_model_instance: BM25Okapi) -> list[dict]:
    """
    Performs BM25 retrieval for a list of queries against document chunks.
    Optimized to aggregate scores across all queries before final selection.
    """
    # 对文档块执行 BM25 检索。
    # 优化以在最终选择之前聚合所有查询的分数。
    start_time = time.time()

    if not document_chunks:
        logger.warning("BM25 Retrieval: No document chunks provided.")
        return []

    if not queries:
        logger.warning("BM25 Retrieval: No queries provided.")
        return []

    # Dictionary to store the maximum score for each unique chunk, and its content
    # 用于存储每个唯一块的最大分数及其内容的字典。
    retrieved_evidences = []

    for query in queries:
        tokenized_bm25_query = re.findall(r'\b\w+\b', query.lower())
        if not tokenized_bm25_query:
            logger.debug(f"Skipping BM25 for empty or un-tokenizable query: '{query}'")
            continue

        # Get scores for the current query across all document chunks
        # 获取当前查询在所有文档块中的分数。
        scores = bm25_model_instance.get_scores(tokenized_bm25_query)

        if scores.size == 0:
            logger.debug(f"No scores generated for query: '{query}'")
            continue

        min_score = scores.min()
        max_score = scores.max()

        # Iterate through chunks and add to retrieved_evidences
        # 遍历块并添加到检索到的证据中。
        for idx, score in enumerate(scores):
            if idx < len(document_chunks):
                chunk_content = document_chunks[idx]

                # Normalize score to 0-1, or use 0.5 if all scores are identical
                # 将分数标准化到 0-1，如果所有分数都相同则使用 0.5。
                normalized_score = 0.5 if max_score == min_score else (score - min_score) / (max_score - min_score)

                retrieved_evidences.append({
                    "content": chunk_content, # The chunk content itself is the retrieved evidence
                    "source": "BM25",
                    "score": float(normalized_score),
                    "original_query_for_retrieval": query # Keep track of which expanded query found it
                })

    # Sort and take top K before returning, as per common retrieval practice
    # This acts as a preliminary filter before the final, more sophisticated ranking
    # 在返回之前进行排序并取前 K 个，这是常见的检索实践。
    # 这作为最终更复杂的排名之前的初步过滤器。
    retrieved_evidences.sort(key=lambda x: x['score'], reverse=True)
    final_retrieved_evidence = retrieved_evidences[:TOP_K_RETRIEVED_CANDIDATES * 2] # Retrieve more than TOP_K for later clustering

    end_time = time.time()
    logger.info(f"BM25 Retrieval processed for {len(queries)} queries, found {len(final_retrieved_evidence)} initial candidates in {end_time - start_time:.4f} seconds.")
    return final_retrieved_evidence

def bge_retrieval(queries: list[str], document_chunks: list[str], chunk_embeddings: np.ndarray) -> list[dict]:
    """Performs BGE (semantic) retrieval for a list of queries against document chunks."""
    # 对文档块执行 BGE（语义）检索。
    start_time = time.time()
    if bge_model is None:
        logger.error("BGE model not loaded, skipping BGE retrieval.")
        return []

    if not document_chunks or chunk_embeddings.size == 0:
        logger.warning("BGE Retrieval: document_chunks or chunk_embeddings are empty. Returning empty list.")
        return []

    logger.info(f"BGE Retrieval: Encoding {len(queries)} queries.")
    query_encoding_start = time.time()

    non_empty_queries = [q for q in queries if q.strip()]
    if not non_empty_queries:
        logger.warning("BGE Retrieval: All queries are empty after filtering. Returning empty list.")
        return []

    all_query_embeddings = bge_model.encode(non_empty_queries, convert_to_numpy=True, show_progress_bar=False)
    logger.info(f"BGE query encoding completed in {time.time() - query_encoding_start:.4f} seconds.")

    if all_query_embeddings.size == 0:
        logger.warning("BGE Retrieval: Query embeddings are empty after encoding. Returning empty list.")
        return []

    similarity_start = time.time()
    all_similarities_matrix = cosine_similarity(all_query_embeddings, chunk_embeddings)

    retrieved_evidences = []

    # Iterate through each query's similarities to chunks
    # 遍历每个查询与块的相似度。
    for q_idx, query_sims in enumerate(all_similarities_matrix):
        query_text = non_empty_queries[q_idx]
        for c_idx, score in enumerate(query_sims):
            if c_idx < len(document_chunks):
                chunk_content = document_chunks[c_idx]
                retrieved_evidences.append({
                    "content": chunk_content, # The chunk content itself
                    "source": "BGE",
                    "score": float(score),
                    "original_query_for_retrieval": query_text
                })

    # Sort and take top K before returning
    # 排序并取前 K 个返回。
    retrieved_evidences.sort(key=lambda x: x['score'], reverse=True)
    final_retrieved_evidence = retrieved_evidences[:TOP_K_RETRIEVED_CANDIDATES] # Retrieve more than TOP_K for later clustering

    end_time = time.time()
    logger.info(f"BGE Retrieval processed for {len(queries)} queries, found {len(final_retrieved_evidence)} initial candidates in {end_time - start_time:.4f} seconds.")
    return final_retrieved_evidence

async def llm_retrieval(queries: list[str], llm_document_chunks : list, full_document_content: str) -> list[dict]:
    """
    Performs LLM-guided extraction from document chunks.
    It simulates reading through the document, updating a summary, and extracting relevant info.
    This version processes chunks in one pass, without iterative refinement for follow-up queries.
    """
    # 从文档块执行 LLM 引导的提取。
    # 它模拟阅读文档，更新摘要，并提取相关信息。
    # 此版本一次性处理块，无需迭代细化以进行后续查询。
    start_time = time.time()
    collected_llm_evidence = []

    # Define how much context to expand extracted sentences by
    # 定义扩展提取句子的上下文量。
    CONTEXT_EXPANSION_WORDS = 100

    # Use the primary query for the main LLM instruction
    # 使用主要查询作为 LLM 的主要指令。
    single_query_for_llm = queries[0] if queries else ""
    # Use other queries as additional context for LLM, joined by tabs
    # 使用其他查询作为 LLM 的附加上下文，用制表符连接。
    other_query_for_llm = "\t".join(queries[1:])
    if len(queries) > 1:
        logger.info(f"Using primary query '{single_query_for_llm}' and additional queries '{other_query_for_llm[:50]}...' for LLM retrieval.")
    else:
        logger.info(f"Using single query '{single_query_for_llm}' for LLM retrieval.")

    if not llm_document_chunks: # Corrected from document_chunks to llm_document_chunks
        logger.warning("LLM Retrieval: No document chunks provided. Returning empty list.")
        return []

    # Pre-process full document for context expansion
    # 预处理整个文档以进行上下文扩展。
    full_doc_words = re.findall(r'\b\w+\b', full_document_content.lower())
    sentence_splitter = re.compile(r'(?<=[.?!])\s+') # Simple sentence splitter
    full_doc_sentences = sentence_splitter.split(full_document_content)

    # Create a BM25 model on sentences for quick lookup of extracted evidence within the full document
    # 在句子上创建 BM25 模型，以便快速查找完整文档中提取的证据。
    tokenized_full_doc_sentences = [re.findall(r'\b\w+\b', s.lower()) for s in full_doc_sentences]
    bm25_full_doc_sentences_model = BM25Okapi(tokenized_full_doc_sentences)


    logger.info(f"LLM Retrieval: Starting chunk processing for {len(llm_document_chunks)} chunks.")
    for i, chunk in enumerate(llm_document_chunks):
        chunk_start_time = time.time()

        # Craft the prompt for the LLM to assess the current chunk
        # 制作 LLM 评估当前块的提示。
        prompt_template = f"""
        You are an expert information extractor. Your task is to carefully read the provided 'Current Document Chunk' and extract all sentences or phrases that are directly and highly relevant to the 'Original Query'.

        Original Query: "{single_query_for_llm}"
        Additional Search Context (from expanded queries): "{other_query_for_llm}"

        Current Document Chunk (Chunk {i+1} of {len(llm_document_chunks)}):
        "{chunk}"

        Based on the 'Original Query' and 'Additional Search Context':

        1.  **Does this 'Current Document Chunk' contain any new, direct information relevant to the 'Original Query'?:** (Yes/No)
        2.  If 'Yes', what are the **exact, most relevant sentences or phrases from *this current chunk* that directly answer or contribute to answering the query?** Extract them verbatim as a list of strings.
        3.  For each piece of 'extracted_evidence', assign a **relevance score from 0.0 to 1.0**, where 1.0 means extremely useful and directly answers the query, and 0.0 means completely irrelevant. If multiple pieces of evidence are extracted, provide a score for each, maintaining their order.

        Your output MUST be a strict JSON object with the following keys:
        - "is_relevant": (boolean) True if the current chunk contains relevant info, False otherwise.
        - "extracted_evidence": (list of strings) A list of relevant sentences/phrases extracted from this chunk. Empty list if not relevant.
        - "relevance_scores": (list of floats) A list of scores (0.0 to 1.0) corresponding to each item in 'extracted_evidence'. Empty list if no evidence.

        Example JSON Output 1 (Relevant info in current chunk):
        ```json
        {{
          "is_relevant": true,
          "extracted_evidence": ["2020 winner was Kia Telluride at 68.9 inches", "2021: Volkswagen ID.4. This electric crossover stands at about 65.5 inches tall."],
          "relevance_scores": [0.98, 0.95]
        }}
        ```
        Example JSON Output 2 (Current chunk not relevant):
        ```json
        {{
          "is_relevant": false,
          "extracted_evidence": [],
          "relevance_scores": []
        }}
        ```
        Output:
        """
        llm_response_json_str = await call_llm(prompt_template, temperature=0.7, response_format_type="json")

        try:
            llm_output = json.loads(llm_response_json_str)
            is_relevant = llm_output.get("is_relevant", False)
            extracted_evidence = llm_output.get("extracted_evidence", [])
            relevance_scores = llm_output.get("relevance_scores", [])

            if is_relevant and extracted_evidence:
                if len(extracted_evidence) != len(relevance_scores):
                    logger.warning(f"Warning: Extracted evidence count mismatch with scores for chunk {i+1}. Defaulting scores to 0.0.")
                    relevance_scores = [0.0] * len(extracted_evidence)

                for k, ev_text in enumerate(extracted_evidence):
                    score = relevance_scores[k] if k < len(relevance_scores) else 0.0

                    expanded_content = ev_text # Start with the extracted text

                    # Attempt to find the exact sentence in the full document and expand its context
                    # 尝试在完整文档中找到确切的句子并扩展其上下文。
                    tokenized_ev_text = re.findall(r'\b\w+\b', ev_text.lower())

                    if tokenized_ev_text and full_doc_sentences:
                        # Find the best matching sentence for the extracted snippet
                        # 找到提取片段的最佳匹配句子。
                        bm25_scores_for_snippet = bm25_full_doc_sentences_model.get_scores(tokenized_ev_text)

                        if bm25_scores_for_snippet.size > 0 and np.max(bm25_scores_for_snippet) > 0:
                            best_match_idx = np.argmax(bm25_scores_for_snippet)
                            best_match_sentence = full_doc_sentences[best_match_idx]

                            # Find the character start index of the best matching sentence in the full document
                            # 在完整文档中找到最佳匹配句子的字符起始索引。
                            char_start_idx_of_sentence = full_document_content.find(best_match_sentence)

                            if char_start_idx_of_sentence != -1:
                                # Count words before the sentence to get its starting word index
                                # 计算句子之前的单词以获取其起始单词索引。
                                words_before_sentence = re.findall(r'\b\w+\b', full_document_content[:char_start_idx_of_sentence])
                                start_word_idx_of_sentence = len(words_before_sentence)

                                # Count words in the sentence to get its ending word index
                                # 计算句子中的单词以获取其结束单词索引。
                                words_in_best_sentence = re.findall(r'\b\w+\b', best_match_sentence)
                                end_word_idx_of_sentence = start_word_idx_of_sentence + len(words_in_best_sentence)

                                # Define expansion boundaries
                                # 定义扩展边界。
                                expand_left_word_idx = max(0, start_word_idx_of_sentence - CONTEXT_EXPANSION_WORDS)
                                expand_right_word_idx = min(len(full_doc_words), end_word_idx_of_sentence + CONTEXT_EXPANSION_WORDS)

                                # Extract the expanded context
                                # 提取扩展的上下文。
                                expanded_content_words = full_doc_words[expand_left_word_idx:expand_right_word_idx]
                                expanded_content = " ".join(expanded_content_words)
                                logger.debug(f"Expanded evidence from '{ev_text[:50]}...' (original words {len(re.findall(r'\b\w+\b', ev_text))}) to {len(expanded_content_words)} words based on BM25 sentence match.")
                            else:
                                logger.warning(f"Best matching sentence '{best_match_sentence[:50]}...' not found in full document. Cannot expand context using character index.")
                        else:
                            logger.warning(f"BM25 found no strong match for extracted evidence '{ev_text[:50]}...' in full document sentences (max score 0). Cannot expand context.")
                    else:
                        logger.warning(f"No tokenized evidence text or full document sentences available. Cannot expand context for '{ev_text[:50]}...'.")

                    collected_llm_evidence.append({
                        "content": expanded_content.strip(),
                        "source": f"LLM_Extracted_Chunk_{i+1}",
                        "score": float(score),
                        "original_query_for_retrieval": single_query_for_llm # Store the primary query used for LLM
                    })

            logger.debug(f"Chunk {i+1} processed in {time.time() - chunk_start_time:.2f}s. Relevant: {is_relevant}, Extracted: {len(extracted_evidence)}.")

        except json.JSONDecodeError as e:
            logger.error(f"ERROR: LLM did not return valid JSON for chunk {i+1}: {e}. Raw LLM output: {llm_response_json_str}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM processing of chunk {i+1}: {e}", exc_info=True)

    end_time = time.time()
    logger.info(f"LLM Retrieval processed for {len(queries)} queries, found {len(collected_llm_evidence)} raw candidates in {end_time - start_time:.2f} seconds.")
    return collected_llm_evidence

def rank_evidence(evidence_pool: list[dict], document_chunks: list[str], bm25_canonical_chunks_model: BM25Okapi, top_n_results: int = None) -> list[dict]:
    """
    Ranks and deduplicates evidence by mapping them to canonical document chunks
    and aggregating their scores. This version now calculates the average score
    from all contributing evidence within a cluster.
    """
    # 通过将证据映射到规范文档块并聚合其分数来对证据进行排名和去重。
    # 此版本现在计算簇内所有贡献证据的平均分数。
    start_time = time.time()

    if not evidence_pool:
        logger.info("Evidence pool is empty, skipping ranking.")
        return []

    if not document_chunks:
        logger.warning("Canonical document chunks are empty, cannot map evidence. Returning empty list.")
        return []

    # Dictionary to hold aggregated results, keyed by the hash of the canonical chunk content
    # Now storing total_score and count for calculating average
    # 用于存储聚合结果的字典，以规范块内容的哈希值为键。
    # 现在存储总分和计数以计算平均值。
    clustered_evidence = {} # {chunk_hash: {content: chunk_text, total_score: N, count: M, contributing_sources: {source: count}, original_retrievals: []}}

    logger.info(f"Starting to map and cluster {len(evidence_pool)} raw retrieved evidence items.")

    for i, ev in enumerate(evidence_pool):
        ev_content = ev['content'].strip()
        ev_score = ev.get('score', 0.0)
        ev_source = ev.get('source', 'Unknown')
        original_query_for_retrieval = ev.get('original_query_for_retrieval', 'N/A')

        if not ev_content:
            logger.debug(f"Skipping empty evidence item: {ev}")
            continue

        tokenized_ev_content = re.findall(r'\b\w+\b', ev_content.lower())
        if not tokenized_ev_content:
            logger.debug(f"Skipping un-tokenizable evidence item: {ev_content[:50]}...")
            continue

        # Use BM25 to find the best matching canonical chunk for this evidence item
        # 使用 BM25 查找此证据项的最佳匹配规范块。
        scores_against_canonical_chunks = bm25_canonical_chunks_model.get_scores(tokenized_ev_content)

        if scores_against_canonical_chunks.size == 0 or np.max(scores_against_canonical_chunks) == 0:
            logger.debug(f"Evidence '{ev_content[:50]}...' did not map significantly to any canonical chunk (max BM25 score 0). Skipping.")
            continue

        best_chunk_idx = np.argmax(scores_against_canonical_chunks)
        mapped_chunk_content = document_chunks[best_chunk_idx]
        mapped_chunk_hash = generate_content_hash_id(mapped_chunk_content)

        # Initialize cluster if new
        # 如果是新的，则初始化簇。
        if mapped_chunk_hash not in clustered_evidence:
            clustered_evidence[mapped_chunk_hash] = {
                "content": mapped_chunk_content,
                "total_score": 0.0, # Initialize total score
                "count": 0,         # Initialize count
                "contributing_sources": {}, # {source_type: count}
                "original_retrievals": [] # Store original retrieved items for debugging/provenance
            }

        # Accumulate score and count for the average calculation
        # 累积分数和计数以进行平均计算。
        clustered_evidence[mapped_chunk_hash]["total_score"] += ev_score
        clustered_evidence[mapped_chunk_hash]["count"] += 1

        # Track contributing sources (e.g., how many BM25 hits vs BGE hits contributed)
        # 跟踪贡献来源（例如，BM25 命中数与 BGE 命中数的贡献）。
        clustered_evidence[mapped_chunk_hash]["contributing_sources"].setdefault(ev_source, 0)
        clustered_evidence[mapped_chunk_hash]["contributing_sources"][ev_source] += 1

        # Store original retrieved item (optional, for detailed provenance if needed)
        # 存储原始检索到的项目（可选，如果需要详细出处）。
        clustered_evidence[mapped_chunk_hash]["original_retrievals"].append(ev)
        logger.debug(f"Mapped evidence '{ev_content[:50]}...' (Score: {ev_score:.2f}, Source: {ev_source}) to canonical chunk '{mapped_chunk_content[:50]}...'. Current total score for chunk: {clustered_evidence[mapped_chunk_hash]['total_score']:.2f}, count: {clustered_evidence[mapped_chunk_hash]['count']}")

    final_ranked_evidence = []
    for chunk_hash, cluster_data in clustered_evidence.items():
        average_score = 0.0
        if cluster_data["count"] > 0: # Avoid division by zero
            average_score = cluster_data["total_score"] / cluster_data["count"]

        final_ranked_evidence.append({
            "content": cluster_data["content"],
            "source": f"Cluster ({', '.join(cluster_data['contributing_sources'].keys())})",
            "score": average_score, # Use the average score as the representative score
            "contributing_details": cluster_data["contributing_sources"],
            "original_retrievals": cluster_data["original_retrievals"]
        })

    logger.info(f"Clustering resulted in {len(final_ranked_evidence)} unique canonical chunks.")

    # Sort the clustered evidence by their average score
    # 按平均分数对聚类证据进行排序。
    final_ranked_evidence.sort(key=lambda x: x['score'], reverse=True)

    # Assign unique display IDs and truncate to top_n_results
    # 分配唯一的显示 ID 并截断到 top_n_results。
    for i, item in enumerate(final_ranked_evidence):
        item['display_id'] = f"ID_{i}_{generate_content_hash_id(item['content'])}"

    end_time = time.time()
    logger.info(f"rank_evidence: Finished ranking and clustering. {len(final_ranked_evidence)} final evidence items in {end_time - start_time:.4f} seconds.")

    if top_n_results is not None:
        return final_ranked_evidence[:top_n_results]
    return final_ranked_evidence

async def retrieve_evidence(current_query: str, document_chunks: list, llm_document_chunks : list, chunk_embeddings: np.ndarray, bm25_model_instance: BM25Okapi, full_document_content: str) -> list[dict]:
    """
    Orchestrates multi-source evidence retrieval (BM25, BGE, LLM) in parallel.
    """
    # 并行协调多源证据检索（BM25、BGE、LLM）。
    start_time = time.time()
    expanded_queries_list = await get_all_expanded_queries(current_query)
    logger.info(f"Expanded Query List ({len(expanded_queries_list)} items): {expanded_queries_list}")

    logger.info("Creating parallel retrieval tasks (BM25, BGE, LLM).")

    # Run BM25 retrieval
    # 运行 BM25 检索。
    bm25_task = asyncio.create_task(asyncio.to_thread(bm25_retrieval, expanded_queries_list, document_chunks, bm25_model_instance))

    # Run BGE retrieval (only if model is loaded and embeddings are available)
    # 运行 BGE 检索（仅当模型已加载且嵌入可用时）。
    if bge_model and chunk_embeddings.size > 0:
        bge_task = asyncio.create_task(asyncio.to_thread(bge_retrieval, expanded_queries_list, document_chunks, chunk_embeddings))
    else:
        logger.warning("BGE model not available or chunk embeddings empty. Skipping BGE retrieval task.")
        bge_task = asyncio.create_task(asyncio.sleep(0.01, result=[])).result # Create a dummy task that returns an empty list

    # Run LLM-guided retrieval
    # 运行 LLM 引导的检索。
    llm_task = asyncio.create_task(llm_retrieval(expanded_queries_list, llm_document_chunks, full_document_content))

    bm25_results, bge_results, llm_guided_results = await asyncio.gather(bm25_task, bge_task, llm_task)
    logger.info("All parallel retrieval tasks completed.")

    retrieved_evidence_pool = []
    retrieved_evidence_pool.extend(bm25_results)
    retrieved_evidence_pool.extend(bge_results)
    retrieved_evidence_pool.extend(llm_guided_results)

    end_time = time.time()
    logger.info(f"Total raw candidates from all sources: {len(retrieved_evidence_pool)} in {end_time - start_time:.4f} seconds.")
    return retrieved_evidence_pool

async def process_single_query_for_retrieval(query: str, document_content: str):
    """
    Main function to perform single-round multi-source retrieval and ranking
    for a single query and document.
    """
    # 主函数，用于对单个查询和文档执行单轮多源检索和排名。
    overall_start_time = time.time()
    logger.info(f"Starting single-round evidence retrieval for original query: '{query}'")

    document_chunks_start = time.time()
    # Chunks for BM25/BGE (smaller, more granular, and now *canonical* chunks)
    # 用于 BM25/BGE 的块（更小、更细粒度，现在是*规范*块）。
    document_chunks = split_document_into_chunks(document_content, chunk_size_words=256, overlap_percentage=0.1)
    # Chunks for LLM (larger, to fit more context in single API calls for extraction)
    # 用于 LLM 的块（更大，以适应单个 API 调用中更多的上下文进行提取）。
    llm_document_chunks = split_document_into_chunks(document_content, chunk_size_words=4096, overlap_percentage=0.1)
    logger.info(f"Document chunking completed in {time.time() - document_chunks_start:.4f} seconds. {len(document_chunks)} canonical chunks, {len(llm_document_chunks)} for LLM.")

    # Log if document chunks are empty
    # 如果文档块为空，则记录日志。
    if not document_chunks:
        logger.warning("process_single_query_for_retrieval: No document chunks generated. Retrieval will likely be empty.")

    bm25_init_start = time.time()
    tokenized_chunks_for_bm25 = [re.findall(r'\b\w+\b', chunk.lower()) for chunk in document_chunks]
    bm25_model_instance = BM25Okapi(tokenized_chunks_for_bm25)
    # Also create a BM25 model on the canonical chunks for the rank_evidence function
    # 还在规范块上创建 BM25 模型，用于 rank_evidence 函数。
    bm25_canonical_chunks_model = BM25Okapi(tokenized_chunks_for_bm25)
    logger.info(f"BM25 models initialized for current document in {time.time() - bm25_init_start:.4f} seconds.")

    bge_embedding_start = time.time()
    chunk_embeddings = np.array([])
    if document_chunks and bge_model is not None:
        try:
            bge_model.eval() # Ensure model is in eval mode
            with torch.no_grad(): # Disable gradient calculations
                chunk_embeddings = bge_model.encode(document_chunks, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error encoding document chunks with BGE model: {e}. BGE retrieval will be skipped.", exc_info=True)
            chunk_embeddings = np.array([])
    else:
        logger.warning("No document chunks or BGE model not loaded, cannot compute chunk embeddings. BGE retrieval will be skipped.")
    logger.info(f"BGE chunk embeddings computed in {time.time() - bge_embedding_start:.4f} seconds. Shape: {chunk_embeddings.shape}")

    current_query_for_retrieval = query

    logger.info("\n======== Performing Single-Round Multi-Source Retrieval ========")

    raw_retrieved_candidates = await retrieve_evidence(
        current_query=current_query_for_retrieval,
        document_chunks=document_chunks,
        llm_document_chunks=llm_document_chunks, # Still passing these for the single LLM retrieval call
        chunk_embeddings=chunk_embeddings,
        bm25_model_instance=bm25_model_instance,
        full_document_content=document_content
    )
    logger.info(f"Raw retrieved candidates from all sources: {len(raw_retrieved_candidates)}.")

    ranking_start_time = time.time()
    # Pass document_chunks and bm25_canonical_chunks_model to rank_evidence for proper mapping and aggregation
    # 将 document_chunks 和 bm25_canonical_chunks_model 传递给 rank_evidence 以进行适当的映射和聚合。
    final_ranked_evidence = rank_evidence(
        evidence_pool=raw_retrieved_candidates,
        document_chunks=document_chunks, # Pass the canonical chunks
        bm25_canonical_chunks_model=bm25_canonical_chunks_model, # Pass the BM25 model for mapping
        top_n_results=TOP_K_RETRIEVED_CANDIDATES
    )
    logger.info(f"Top {len(final_ranked_evidence)} candidates after final ranking and deduplication in {time.time() - ranking_start_time:.4f} seconds.")
    # Log the content of final_ranked_evidence
    # 记录 final_ranked_evidence 的内容。
    if final_ranked_evidence:
        logger.info(f"process_single_query_for_retrieval: Final ranked evidence (first 3 items):")
        for i, ev in enumerate(final_ranked_evidence[:3]):
            logger.info(f"  Item {i+1}: ID={ev.get('display_id', 'N/A')}, Score={ev.get('score', 'N/A'):.2f}, Content='{ev.get('content', '')[:100]}...'")
    else:
        logger.warning("process_single_query_for_retrieval: Final ranked evidence list is empty.")


    logger.info(f"Overall retrieval process for query '{query}' completed in {time.time() - overall_start_time:.2f} seconds.")

    logger.info(f"Returning {len(final_ranked_evidence)} final evidence items to Gradio state.")
    return final_ranked_evidence

# --- Gradio Interface Logic ---
# --- Gradio 界面逻辑 ---

def add_gold_evidence(current_evidence_list: list, new_evidence_text: str):
    """Adds a new gold evidence entry to the list."""
    # 将新的黄金证据条目添加到列表中。
    new_evidence_text = new_evidence_text.strip()
    # Ensure current_evidence_list is indeed a list
    if not isinstance(current_evidence_list, list):
        logger.error(f"add_gold_evidence received current_evidence_list of type {type(current_evidence_list)}. Converting to list.")
        current_evidence_list = [] # Reset to empty list if it's not a list

    if new_evidence_text and new_evidence_text not in current_evidence_list:
        current_evidence_list.append(new_evidence_text)
    return current_evidence_list, "" # Return updated list and clear the input textbox

def clear_gold_evidence():
    """Clears all gold evidence entries."""
    # 清除所有黄金证据条目。
    return [], "" # Return empty list and clear display

def update_gold_evidence_display(gold_evidence_list: list) -> str:
    """Formats the list of gold evidence for display."""
    # 格式化黄金证据列表以供显示。
    if not gold_evidence_list:
        return "No gold evidence added yet."
    return "\n".join([f"- {ev}" for ev in gold_evidence_list])

def _prepare_retrieved_evidence_checkbox_data(all_retrieved_evidence_data: list, gold_evidence_list: list):
    """
    Helper function to prepare choices and values for a retrieved evidence checkbox group.
    Args:
        all_retrieved_evidence_data (list): List of all retrieved evidence dictionaries.
        gold_evidence_list (list): List of gold evidence strings (usually content).
    Returns:
        tuple: (choices_list, selected_values_list)
    """
    choices = []
    selected_values = []
    
    # Create a set of gold evidence content for efficient lookup
    # 创建一个用于高效查找的黄金证据内容集合。
    gold_evidence_content_set = {ev.strip() for ev in gold_evidence_list if ev.strip()}

    for i, ev in enumerate(all_retrieved_evidence_data):
        display_id = ev.get('display_id', f"ID_{i}_{generate_content_hash_id(ev['content'])}")
        ev['display_id'] = display_id # Ensure display_id is present on the object
        score = ev.get('score', 'N/A')
        source = ev.get('source', 'N/A')
        truncated_content = ev['content'].replace('\n', ' ').strip()
        label = f"[{display_id}] (Score: {score:.4f}, Src: {source}): {truncated_content}"
        choices.append(label)

    # NEW: Use BM25 for fuzzy matching between gold evidence and retrieved evidence for pre-selection
    # 新增：使用 BM25 进行黄金证据和检索证据之间的模糊匹配，以进行预选。
    tokenized_retrieved_contents = [
        re.findall(r'\b\w+\b', ev['content'].lower())
        for ev in all_retrieved_evidence_data
    ]

    # Check if there are any valid tokens to build the BM25 model
    # 检查是否有任何有效令牌来构建 BM25 模型。
    if not tokenized_retrieved_contents or all(not tokens for tokens in tokenized_retrieved_contents):
        logger.warning("No valid tokenized content for BM25 model. Skipping BM25 pre-selection.")
        return choices, selected_values # Return choices and empty selected_values
        
    bm25_retrieved_model = BM25Okapi(tokenized_retrieved_contents)

    # Use a set to keep track of already selected display_ids to avoid duplicates
    # 使用一个集合来跟踪已选择的 display_id，以避免重复。
    already_selected_display_ids = set()
    PRE_SELECTION_BM25_THRESHOLD = 0.01 # Adjust this threshold as needed

    for gold_ev_content in gold_evidence_list: # This is the *pure* content from saved gold evidence
        if not gold_ev_content.strip():
            continue

        tokenized_gold_ev = re.findall(r'\b\w+\b', gold_ev_content.lower())
        if not tokenized_gold_ev:
            continue

        scores = bm25_retrieved_model.get_scores(tokenized_gold_ev)
        
        # Find the best matching retrieved evidence item
        if scores.size > 0 and np.max(scores) > 0:
            best_match_idx = np.argmax(scores)
            best_match_score = scores[best_match_idx]

            if best_match_score > PRE_SELECTION_BM25_THRESHOLD:
                matched_retrieved_ev = all_retrieved_evidence_data[best_match_idx]
                matched_display_id = matched_retrieved_ev.get('display_id')

                if matched_display_id and matched_display_id not in already_selected_display_ids:
                    # Construct the label for the matched item exactly as it appears in 'choices'
                    score = matched_retrieved_ev.get('score', 'N/A')
                    source = matched_retrieved_ev.get('source', 'N/A')
                    truncated_content = matched_retrieved_ev['content'].replace('\n', ' ').strip()
                    label_to_select = f"[{matched_display_id}] (Score: {score:.4f}, Src: {source}): {truncated_content}"
                    
                    selected_values.append(label_to_select)
                    already_selected_display_ids.add(matched_display_id)
                    logger.debug(f"Pre-selected checkbox for gold evidence '{gold_ev_content[:50]}...' via BM25 match (Score: {best_match_score:.4f}) to ID: {matched_display_id}")
            else:
                logger.debug(f"Gold evidence '{gold_ev_content[:50]}...' did not meet BM25 threshold for pre-selection (Max Score: {best_match_score:.4f}).")
    
    return choices, selected_values


# This function now returns a single gr.update object for the CheckboxGroup
# 此函数现在为 CheckboxGroup 返回单个 gr.update 对象。
def display_temp_retrieved_evidence_file_content(file_path: str, gold_evidence_list: list):
    """
    Reads and returns the JSON content from the specified temporary file,
    formatted for a CheckboxGroup, with gold evidence pre-selected.
    """
    # 从指定的临时文件读取并返回 JSON 内容，
    # 格式化为 CheckboxGroup，并预选黄金证据。
    if not file_path or not os.path.exists(file_path):
        logger.warning(f"No temp file path provided or file not found: {file_path}")
        return gr.update(choices=[], value=[]) # Return empty update for the CheckboxGroup
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_retrieved_evidence_data = json.load(f)
        logger.info(f"Successfully loaded data from temp file: {file_path}")

        choices, selected_values = _prepare_retrieved_evidence_checkbox_data(all_retrieved_evidence_data, gold_evidence_list)
        
        return gr.update(choices=choices, value=selected_values)
    except Exception as e:
        logger.error(f"Error loading data from temp file {file_path}: {e}", exc_info=True)
        return gr.update(choices=[], value=[])

def get_current_item(dataset_name: str, index: int):
    """
    Retrieves and displays the current item from the dataset.
    If an annotation for the item already exists, it loads the saved data.
    """
    # 从数据集中检索并显示当前项目。
    # 如果项目已存在注释，则加载保存的数据。
    global current_dataset, current_dataset_index

    logger.info(f"Attempting to get item for dataset '{dataset_name}' at index {index}.")

    # Load dataset if not already loaded or if it's empty
    if not current_dataset or not (0 <= index < len(current_dataset)):
        logger.info(f"Dataset not loaded or index {index} out of bounds for current dataset size {len(current_dataset)}. Attempting to (re)load dataset.")
        load_dataset(dataset_name)
        if not current_dataset:
            logger.warning(f"Dataset '{dataset_name}' is empty or failed to load after attempting to load.")
            # Ensure all outputs are consistently returned.
            # 确保始终返回所有输出。
            return (
                0, # current_idx
                "", # query_val
                "", # doc_val
                "", # gold_ans_val
                [], # gold_ev_list_val
                gr.Markdown("No gold evidence added yet."), # gold_ev_display_val
                gr.update(choices=[], value=[]), # checkbox_choices (for retrieved_evidence_checkbox_group)
                [], # full_data_store_initial (for evidence_full_data_store)
                "", # expanded_output_val (for expanded_query_output)
                "", # retrieval_time_val (for retrieval_time_display)
                {}, # mcts_chain_val (for mcts_reasoning_chain_state)
                "",  # temp_all_retrieved_evidence_file_path_state
                0.0, # total_retrieval_time_state
                "", # new_gold_evidence_input initial value
                "", # answer_eval initial value
                "", # evidence_eval initial value (string for Textbox)
                ""  # collected_evidence_library_display initial value
            )

    # Clamp index to valid range
    if len(current_dataset) > 0:
        index = max(0, min(index, len(current_dataset) - 1))
        logger.info(f"Requested index {index} clamped to {index} (max {len(current_dataset)-1}).")
    else:
        index = 0 # No items, set to 0

    current_dataset_index = index

    item = current_dataset[current_dataset_index]
    query = item.get("input", "")
    document_text = item.get("context", "")
    gold_answer = item.get("answers", "")

    # Fix: Ensure gold_evidence_from_dataset is always a list
    gold_evidence_from_dataset_raw = item.get("gold_evidence", "")
    if isinstance(gold_evidence_from_dataset_raw, str):
        if ';' in gold_evidence_from_dataset_raw:
            gold_evidence_from_dataset = [s.strip() for s in gold_evidence_from_dataset_raw.split(';') if s.strip()]
        else:
            gold_evidence_from_dataset = [gold_evidence_from_dataset_raw.strip()] if gold_evidence_from_dataset_raw.strip() else []
    elif isinstance(gold_evidence_from_dataset_raw, list):
        gold_evidence_from_dataset = [s.strip() for s in gold_evidence_from_dataset_raw if isinstance(s, str) and s.strip()]
    else:
        gold_evidence_from_dataset = []

    # --- Check for existing saved annotation file ---
    # --- 检查是否存在已保存的注释文件 ---
    item_id = item.get("id", f"item_{index}_no_id")
    safe_query_snippet = re.sub(r'[^\w\-_\. ]', '', query[:30]).strip()
    file_hash = hashlib.sha256(f"{dataset_name}_{index}_{item_id}_{safe_query_snippet}".encode()).hexdigest()[:10]
    save_path = os.path.join(output_save_dir, f"{dataset_name}_{index}_{file_hash}.json")

    loaded_expanded_query_output = ""
    loaded_retrieved_evidence_choices = []
    loaded_retrieved_evidence_value = []
    loaded_evidence_full_data_store = []
    loaded_mcts_chain = {}
    loaded_temp_all_retrieved_evidence_file_path = ""
    loaded_retrieval_time_display = "N/A"
    loaded_retrieval_duration_seconds = 0.0 # NEW: Initialize numerical value
    loaded_new_gold_evidence_input_value = "" # Initialize for new output
    loaded_answer_eval_value = "" # Initialize for answer_eval
    loaded_evidence_eval_value = "" # NEW: Initialize for evidence_eval
    loaded_collected_evidence_library_display = "" # NEW: Initialize for collected_evidence_library_display

    if os.path.exists(save_path):
        logger.info(f"Found existing annotation file: {save_path}. Loading data from it.")
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)

            # Load Query Expansion
            # 加载查询扩展。
            loaded_expanded_query_output = saved_data.get("expanded_queries", "") # Direct load from saved data
            if not loaded_expanded_query_output: # Fallback if not found in saved data for older files
                original_query_from_saved = saved_data.get("original_query", "")
                loaded_expanded_query_output = "Expanded Queries (from saved file - old format):\n"
                if original_query_from_saved:
                    loaded_expanded_query_output += f"- {original_query_from_saved}"
                else:
                    loaded_expanded_query_output += "N/A"


            # Load Retrieved Evidence (for the primary checkbox group)
            # This group will now always show ALL retrieved evidence, with gold evidence pre-selected.
            loaded_evidence_full_data_store = saved_data.get("retrieved_evidence", [])
            
            # From the saved file, load the evidence explicitly selected by the user.
            # This list will be used to pre-select items in the UI's checkbox group.
            loaded_gold_selected_retrieved_evidence = saved_data.get("gold_selected_retrieved_evidence", [])
            loaded_gold_selected_content_for_checkbox = []
            for ev_item in loaded_gold_selected_retrieved_evidence:
                if isinstance(ev_item, dict) and 'content' in ev_item:
                    content_to_add = ev_item['content'].strip()
                    # Updated regex to robustly match the full metadata prefix including potentially complex Src fields
                    # with internal parentheses, by matching greedily until the final '):'
                    # 匹配 ID、分数、来源等元数据，然后捕获冒号后的实际内容。
                    label_prefix_match = re.match(r"\[(ID_\d+_[0-9a-f]{8})\]\s*\(Score:\s*[\d\.]+,?\s*Src:.*?\):\s*(.*)", content_to_add)
                    if label_prefix_match:
                        # Extract the part after the colon, which should be the actual content
                        # 提取冒号后的部分，这应该是实际内容。
                        content_to_add = label_prefix_match.group(1).strip()
                        logger.debug(f"Extracted content from formatted label: '{content_to_add[:100]}...'")
                    loaded_gold_selected_content_for_checkbox.append(content_to_add)

            loaded_retrieved_evidence_choices, loaded_retrieved_evidence_value = \
                _prepare_retrieved_evidence_checkbox_data(
                    loaded_evidence_full_data_store,
                    loaded_gold_selected_content_for_checkbox # Use the extracted content for pre-selection
                )
            
            # Load MCTS Based Reasoning Chain
            # 加载基于 MCTS 的推理链。
            loaded_mcts_chain = saved_data.get("mcts_based_reasoning_chain", {})
            if loaded_mcts_chain:
                logger.info(f"Loaded MCTS chain for index {index}.")
                # NEW: Format collected_evidence_library for display
                collected_evidence_lib = loaded_mcts_chain.get("collected_evidence_library", [])
                if collected_evidence_lib:
                    formatted_collected_evidence = []
                    for i, entry in enumerate(collected_evidence_lib):
                        formatted_collected_evidence.append(f"  Step {i+1}: Sub-query: {entry.get('sub_query', 'N/A')}\n    Answer: {entry.get('sub_answer', 'N/A')}\n    Evidence ID: {entry.get('evidence_id', 'N/A')}\n    Content: {entry.get('evidence_content', '')[:150]}...")
                    loaded_collected_evidence_library_display = "--- Collected Evidence Library ---\n" + "\n".join(formatted_collected_evidence)
                else:
                    loaded_collected_evidence_library_display = "No collected evidence found."
            else:
                logger.info(f"No MCTS chain found in saved file for index {index}.")
                loaded_collected_evidence_library_display = "No MCTS chain generated yet."


            # Gold evidence in dataset might be a string or list, normalize to list
            # Ensure gold_evidence is loaded as a list
            gold_evidence_from_saved_raw = saved_data.get("gold_evidence", [])
            if isinstance(gold_evidence_from_saved_raw, str):
                if ';' in gold_evidence_from_saved_raw:
                    gold_evidence_from_dataset = [s.strip() for s in gold_evidence_from_saved_raw.split(';') if s.strip()]
                else:
                    gold_evidence_from_dataset = [gold_evidence_from_saved_raw.strip()] if gold_evidence_from_saved_raw.strip() else []
            elif isinstance(gold_evidence_from_saved_raw, list):
                gold_evidence_from_dataset = [s.strip() for s in gold_evidence_from_saved_raw if isinstance(s, str) and s.strip()]
            else:
                gold_evidence_from_dataset = []


            gold_answer = saved_data.get("gold_answer", gold_answer) # Use existing gold_answer if not in saved_data
            loaded_answer_eval_value = saved_data.get("answer_eval", "") # Load answer_eval
            loaded_evidence_eval_value = str(saved_data.get("evidence_eval", "")) # NEW: Load evidence_eval as string

            logger.info(f"Successfully loaded gold_evidence. gold_evidence_from_dataset: {gold_evidence_from_dataset}")
            logger.info(f"Successfully loaded item at index {current_dataset_index}. Query (truncated): {query[:50]}..., Document (truncated): {document_text[:50]}...")

            # Update temp_all_retrieved_evidence_file_path_state for the loaded data
            # 更新 temp_all_retrieved_evidence_file_path_state 以获取加载的数据。
            if loaded_evidence_full_data_store:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix=".json", dir=TMP_DIR) as temp_f:
                    json.dump(loaded_evidence_full_data_store, temp_f, indent=4, ensure_ascii=False)
                    loaded_temp_all_retrieved_evidence_file_path = temp_f.name
                logger.info(f"Loaded retrieved evidence written to new temporary file: {loaded_temp_all_retrieved_evidence_file_path}")
                
            # Load retrieval duration from saved data
            # 从保存的数据中加载检索持续时间。
            loaded_retrieval_duration_seconds = saved_data.get("retrieval_duration_seconds", 0.0)
            if loaded_retrieval_duration_seconds > 0:
                loaded_retrieval_time_display = f"Retrieval Time: {loaded_retrieval_duration_seconds:.2f} seconds (Loaded)"
            else:
                # Fallback for old files or if 0.0 was explicitly saved
                loaded_retrieval_time_display = "Loaded from file (Time N/A)" # Clarify if time wasn't saved or was 0

            # Set new_gold_evidence_input value from loaded data
            # 从加载的数据中设置 new_gold_evidence_input 值。
            # if len(gold_evidence_from_dataset) == 1:
            #     loaded_new_gold_evidence_input_value = gold_evidence_from_dataset[0]
            # elif len(gold_evidence_from_dataset) > 1:
                # logger.warning(f"Multiple gold evidence items found ({len(gold_evidence_from_dataset)}). 'Add New Gold Evidence' input will remain empty.")

        except Exception as e:
            logger.error(f"Error loading annotation from {save_path}: {e}", exc_info=True)
            gr.Warning(f"Error loading saved annotation for index {index}: {e}. Loading raw item instead.")
            # Reset all loaded variables to default if error occurs
            # 如果发生错误，将所有加载的变量重置为默认值。
            loaded_expanded_query_output = ""
            loaded_retrieved_evidence_choices = []
            loaded_retrieved_evidence_value = []
            loaded_evidence_full_data_store = []
            loaded_mcts_chain = {}
            loaded_temp_all_retrieved_evidence_file_path = ""
            loaded_retrieval_time_display = "N/A"
            loaded_retrieval_duration_seconds = 0.0
            loaded_new_gold_evidence_input_value = "" # Ensure it's reset on error
            loaded_answer_eval_value = "" # Ensure it's reset on error
            loaded_evidence_eval_value = "" # NEW: Ensure it's reset on error
            loaded_collected_evidence_library_display = "" # NEW: Ensure it's reset on error
    else:
        logger.info(f"No existing annotation file found for index {index} at {save_path}. Loading raw item.")
        # When loading a raw item, the 'new_gold_evidence_input' should be empty,
        # unless 'gold_evidence_from_dataset' (from the raw item) has exactly one.
        if len(gold_evidence_from_dataset) == 1:
             loaded_new_gold_evidence_input_value = gold_evidence_from_dataset[0] # Access the first element
        elif len(gold_evidence_from_dataset) > 1:
             logger.warning(f"Multiple gold evidence items found in raw item ({len(gold_evidence_from_dataset)}). 'Add New Gold Evidence' input will remain empty.")


    # Return values
    # 返回值。
    return (
        current_dataset_index,
        query,
        document_text,
        gold_answer,
        gold_evidence_from_dataset,
        gr.Markdown(update_gold_evidence_display(gold_evidence_from_dataset)),
        gr.update(choices=loaded_retrieved_evidence_choices, value=loaded_retrieved_evidence_value),
        gr.update(value=loaded_evidence_full_data_store),
        loaded_expanded_query_output,
        loaded_retrieval_time_display,
        loaded_mcts_chain,
        loaded_temp_all_retrieved_evidence_file_path,
        loaded_retrieval_duration_seconds, # Changed to loaded_retrieval_duration_seconds
        loaded_new_gold_evidence_input_value, # NEW: Return this value
        loaded_answer_eval_value, # NEW: Return this value
        loaded_evidence_eval_value, # NEW: Return evidence_eval_value
        loaded_collected_evidence_library_display # NEW: Return the formatted collected_evidence_library_display
    )


# Moved initial_load_and_status and go_to_index here to ensure they are defined before gr.Blocks
# 将 initial_load_and_status 和 go_to_index 移动到此处，以确保它们在 gr.Blocks 之前定义。
def initial_load_and_status(dataset_name: str):
    """
    Initializes the dataset and loads the first item when the Gradio interface loads.
    """
    # 初始化数据集并在 Gradio 界面加载时加载第一个项目。
    logger.info(f"Initial load triggered for dataset: {dataset_name}")
    load_dataset(dataset_name)
    if not current_dataset:
        gr.Warning(f"Dataset '{dataset_name}' could not be loaded or is empty.")
        # Ensure consistent return even if dataset is empty
        # 即使数据集为空，也要确保一致的返回。
        return (0, "", "", "", [], gr.Markdown("No gold evidence added yet."),
                gr.update(choices=[], value=[]), [], "", "N/A", {}, "", 0.0,
                "", # Initial empty value for new_gold_evidence_input
                "", # Initial empty value for answer_eval
                "", # Initial empty value for evidence_eval
                ""  # Initial empty value for collected_evidence_library_display
                )
    # Load the first item
    # 加载第一个项目。
    return get_current_item(dataset_name, 0)

def go_to_index(dataset_name: str, index: int):
    """
    Navigates to a specific item in the dataset by index.
    This function will call get_current_item to handle the loading and display.
    """
    # 根据索引导航到数据集中的特定项目。
    # 此函数将调用 get_current_item 来处理加载和显示。
    logger.info(f"Go to index triggered for dataset '{dataset_name}' at index {index}.")
    return get_current_item(dataset_name, index)


# Modified function to generate MCTS based reasoning chain
# 修改后的函数，用于生成基于 MCTS 的推理链。
async def generate_mcts_reasoning_chain_actual(
    query: str,
    # Removed selected_evidence_labels as MCTS will now use all retrieved evidence
    # all_retrieved_evidence_file_path: str # NEW: Accept file path
    all_retrieved_evidence_file_path: str
) -> (str, dict, str): # Returns formatted chain string, the raw chain dict, and formatted collected_evidence_library
    """
    Generates an MCTS based reasoning chain using the MCTS implementation.
    The MCTS will operate on all retrieved evidence.
    """
    # 使用 MCTS 实现生成基于 MCTS 的推理链。
    # MCTS 将对所有检索到的证据进行操作。
    logger.info("Attempting to generate MCTS reasoning chain...")
    logger.info(f"Query: {query}")
    # logger.info(f"Selected evidence labels: {selected_evidence_labels}") # No longer needed
    logger.info(f"Retrieved evidence file path: {all_retrieved_evidence_file_path}")

    # Removed check for selected_evidence_labels as it's no longer used for filtering
    
    if not all_retrieved_evidence_file_path or not os.path.exists(all_retrieved_evidence_file_path):
        gr.Warning("No retrieved evidence file found. Please run retrieval first.")
        return "No retrieved evidence file found. Please run retrieval first.", {}, "No collected evidence."

    # Load all retrieved evidence from the temporary file
    # 从临时文件加载所有检索到的证据。
    evidence_full_data_store = []
    try:
        with open(all_retrieved_evidence_file_path, 'r', encoding='utf-8') as f:
            evidence_full_data_store = json.load(f)
        logger.info(f"Loaded {len(evidence_full_data_store)} retrieved evidence items from {all_retrieved_evidence_file_path}. evidence num: {len(evidence_full_data_store)}")
    except Exception as e:
        logger.error(f"Error loading retrieved evidence from temporary file {all_retrieved_evidence_file_path}: {e}", exc_info=True)
        gr.Error(f"Error loading retrieved evidence: {e}")
        return "Error loading retrieved evidence.", {}, "Error loading collected evidence."

    # MCTS will now use all retrieved evidence directly
    # MCTS 现在将直接使用所有检索到的证据。
    selected_evidence_for_mcts = evidence_full_data_store

    if not selected_evidence_for_mcts:
        gr.Warning("No retrieved evidence found. Please run retrieval and ensure evidence is present.")
        logger.warning("No retrieved evidence found for MCTS. Returning empty chain.")
        return "No retrieved evidence found for MCTS.", {}, "No collected evidence."

    # Prepare data for MCTS: write to a temporary JSON file
    # 准备 MCTS 数据：写入临时 JSON 文件。
    mcts_input_data = {
        "original_query": query,
        "retrieved_evidence": selected_evidence_for_mcts # MCTS expects this format
    }

    temp_file_name = ""
    mcts_chain_raw = {}
    formatted_chain_str = ""
    formatted_collected_evidence_lib_str = "" # Initialize new output string
    try:
        # Create a temporary file to store the final_evidence
        # 创建一个临时文件来存储 final_evidence。
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix=".json", dir=TMP_DIR) as temp_f:
            json.dump(mcts_input_data, temp_f, indent=2, ensure_ascii=False)
            temp_file_name = temp_f.name
        logger.info(f"MCTS input data written to temporary file: {temp_file_name}")

        # Initialize LLM Reward Model (using the aliased call_llm from mcts.py)
        # 初始化 LLM 奖励模型（使用 mcts.py 中别名化的 call_llm）。
        llm_reward_model = LLMRewardModel(llm_caller_func=mcts_call_llm)

        # Initialize MCTS
        # max_simulations and max_path_length can be tuned.
        # top_n_evidence_selection should be based on the number of selected evidence.
        # 初始化 MCTS。
        # 可以调整 max_simulations 和 max_path_length。
        # top_n_evidence_selection 应该基于选定证据的数量。
        mcts_instance = EvidenceSelectionMCTS(
            evidence_file_path=temp_file_name,
            llm_reward_model=llm_reward_model,
            exploration_weight=0.2,
            max_simulations=50, # Adjust based on desired computation time vs quality
            max_path_length=20, # Max number of evidence pieces in a chain
            top_n_evidence_selection=min(5, len(selected_evidence_for_mcts)) # Select from top 5 or fewer if less are selected
        )

        gr.Info("Generating MCTS reasoning chain... This may take a moment.")
        logger.info("Starting MCTS search...")
        best_path_details, top_k_paths, collected_evidence_library = await mcts_instance.search()
        logger.info("MCTS search completed.")

        mcts_chain_raw["original_query"] = query

        if best_path_details:
            logger.info(f"MCTS found a best path with {len(best_path_details)} steps.")
            formatted_chain_str += "--- MCTS Best Reasoning Path ---\n"
            # mcts_chain_raw["mcts_evidence_pool"] = [] # Renamed to 'mcts_evidence_pool' as requested
            mcts_chain_raw["mcts_best_path_details"] = [] # Store full details of best path

            for i, (sub_q, sub_a, ev_id, ev_text) in enumerate(best_path_details):
                formatted_chain_str += f"Step {i+1}:\n"
                step_data = {
                    "step": i+1,
                    "evidence_id": ev_id, # Numeric ID as requested
                }
                if sub_q: # Only include sub_query and sub_answer if sub_q is not empty
                    formatted_chain_str += f"  Sub-query: {sub_q}\n"
                    formatted_chain_str += f"  Answer: {sub_a}\n"
                    step_data["sub_query"] = sub_q
                    step_data["sub_answer"] = sub_a
                formatted_chain_str += f"  Evidence Used [ID: {ev_id}]: {ev_text[:150]}...\n"
                step_data["evidence_content_snippet"] = ev_text[:200] + "..." if len(ev_text) > 200 else ev_text
                mcts_chain_raw["mcts_best_path_details"].append(step_data)

            if mcts_instance.root.final_answer: # Access root's final answer if set
                 formatted_chain_str += f"Final Answer from MCTS: {mcts_instance.root.final_answer}\n"
                 mcts_chain_raw["final_answer_mcts"] = mcts_instance.root.final_answer
            else:
                 formatted_chain_str += "MCTS did not synthesize a final answer, but found a path.\n"
                 mcts_chain_raw["final_answer_mcts"] = "Not synthesized"
        else: # If no best path found, indicate so
            formatted_chain_str = "MCTS did not find any useful reasoning paths."
            mcts_chain_raw = {}

        # Store the collected_evidence_library in the raw MCTS chain data
        # 将 collected_evidence_library 存储在原始 MCTS 链数据中
        mcts_chain_raw["collected_evidence_library"] = collected_evidence_library

        # Format collected_evidence_library for display in its own textbox
        # 格式化 collected_evidence_library 以便在自己的文本框中显示
        if collected_evidence_library:
            formatted_collected_evidence = []
            for i, entry in enumerate(collected_evidence_library):
                formatted_collected_evidence.append(f"  Step {i+1}:\n    Sub-query: {entry.get('sub_query', 'N/A')}\n    Answer: {entry.get('sub_answer', 'N/A')}\n    Evidence ID: {entry.get('evidence_id', 'N/A')}\n    Content: {entry.get('evidence_content', '')[:200]}...")
            formatted_collected_evidence_lib_str = "--- Collected Evidence Library ---\n" + "\n".join(formatted_collected_evidence)
        else:
            formatted_collected_evidence_lib_str = "No collected evidence found during MCTS search."


        gr.Info("MCTS reasoning chain generated!")

    except FileNotFoundError as e:
        logger.error(f"Error: Temporary file not found for MCTS. {e}", exc_info=True)
        formatted_chain_str = f"Error: Required data file missing for MCTS. {e}"
        mcts_chain_raw = {"error": f"Required data file missing for MCTS: {e}"}
        formatted_collected_evidence_lib_str = "Error: Collected evidence data unavailable."
        gr.Error(formatted_chain_str)
    except ValueError as e:
        logger.error(f"Error: Data format issue in MCTS input file. {e}", exc_info=True)
        formatted_chain_str = f"Error: Data format issue for MCTS. {e}"
        mcts_chain_raw = {"error": f"Data format issue for MCTS: {e}"}
        formatted_collected_evidence_lib_str = "Error: Collected evidence data unavailable."
        gr.Error(formatted_chain_str)
    except RuntimeError as e:
        logger.error(f"Error during MCTS execution: {e}", exc_info=True)
        formatted_chain_str = f"Runtime Error during MCTS: {e}"
        mcts_chain_raw = {"error": f"Runtime Error during MCTS: {e}"}
        formatted_collected_evidence_lib_str = "Error: Collected evidence data unavailable."
        gr.Error(formatted_chain_str)
    except Exception as e:
        logger.error(f"An unexpected error occurred during MCTS processing: {e}", exc_info=True)
        formatted_chain_str = f"An unexpected error occurred: {e}"
        mcts_chain_raw = {"error": f"An unexpected error occurred: {e}"}
        formatted_collected_evidence_lib_str = "Error: Collected evidence data unavailable."
        gr.Error(formatted_chain_str)
    finally:
        # Clean up the temporary file
        # 清理临时文件。
        # if temp_file_name and os.path.exists(temp_file_name):
        #     os.remove(temp_file_name)
        logger.info(f"Cleaned up temporary file: {temp_file_name}")

    return formatted_chain_str, mcts_chain_raw, formatted_collected_evidence_lib_str


async def run_retrieval_and_display(query: str, document_content: str, gold_evidence_list_current: list): # Added gold_evidence_list_current
    """
    Triggers the evidence retrieval process and updates the Gradio UI with results.
    """
    # 触发证据检索过程并使用结果更新 Gradio UI。
    retrieval_start_time_overall = time.time() # Start timing here

    logger.info(f"Retrieval initiated for query: '{query[:100]}...'")
    if not query or not document_content:
        gr.Warning("Please provide both a query and document content.")
        # Return empty updates for CheckboxGroup and State on error, and clear expanded_output and retrieval_time_display and mcts_chain
        # 发生错误时，返回 CheckboxGroup 和 State 的空更新，并清除 expanded_output、retrieval_time_display 和 mcts_chain。
        return gr.Textbox(value="Please provide both a query and document content.", interactive=False), \
               gr.update(choices=[], value=[]), \
               gr.update(value=[]), \
               gr.Textbox(value="N/A", interactive=False), \
               {}, \
               "", \
               0.0 # Also return 0.0 for retrieval duration state

    temp_file_path_for_all_evidence = "" # Initialize here
    try:
        gr.Info("Performing query expansion and multi-source retrieval... This may take a moment.")

        # Capture query expansion details for display
        # 捕获查询扩展详细信息以供显示。
        query_expansion_start_time = time.time()
        expanded_data = await expand_query_with_llm(query)
        expanded_queries_str = "Expanded Keywords:\n- " + "\n- ".join(expanded_data.get("keywords", [])) + \
                               "\n\nDecomposition:\n- " + "\n- ".join(expanded_data.get("decomposition", []))
        logger.info(f"Query expansion phase completed in {time.time() - query_expansion_start_time:.2f} seconds.")

        retrieval_func_start_time = time.time() # Timing the core retrieval function
        final_evidence = await process_single_query_for_retrieval(query, document_content)
        retrieval_func_end_time = time.time()
        logger.info(f"Multi-source retrieval and ranking phase completed in {retrieval_func_end_time - retrieval_func_start_time:.2f} seconds.")

        # Prepare choices and values for the main retrieved_evidence_checkbox_group
        # 使用 gold_evidence_list_current 来预选。
        choices_for_retrieval_checkbox, values_for_retrieval_checkbox = \
            _prepare_retrieved_evidence_checkbox_data(final_evidence, gold_evidence_list_current)


        gr.Info("Retrieval complete! Please select relevant evidence.")

        retrieval_end_time_overall = time.time()
        total_retrieval_time = retrieval_end_time_overall - retrieval_start_time_overall
        time_display_str = f"Retrieval Time: {total_retrieval_time:.2f} seconds"

        # **IMPORTANT LOG**: Log the content of final_evidence right before it's passed to the state
        # **重要日志**：在将 final_evidence 的内容传递给状态之前立即记录。
        logger.info(f"run_retrieval_and_display: Preparing to pass {len(final_evidence)} items to evidence_full_data_store. First item content (truncated): {final_evidence[0]['content'][:100] if final_evidence else 'N/A'}")

        # Save final_evidence to a temporary file
        # 将 final_evidence 保存到临时文件。
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix=".json", dir=TMP_DIR) as temp_f:
            json.dump(final_evidence, temp_f, indent=4, ensure_ascii=False)
            temp_file_path_for_all_evidence = temp_f.name
        logger.info(f"All retrieved evidence saved to temporary file: {temp_file_path_for_all_evidence}")

        # Initially, the MCTS chain is empty. It will be generated when the button is clicked.
        # 最初，MCTS 链是空的。它将在点击按钮时生成。
        mcts_chain = {}


        # Return expanded queries string (for its own Textbox),
        # gr.update for CheckboxGroup choices, gr.update for CheckboxGroup value (empty list),
        # the full evidence data for the State variable, the retrieval time string, and the temp file path, and MCTS chain
        # 返回扩展的查询字符串（用于其自己的文本框）、
        # CheckboxGroup 选项的 gr.update、CheckboxGroup 值的 gr.update（空列表）、
        # State 变量的完整证据数据、检索时间字符串以及临时文件路径和 MCTS 链。
        return gr.Textbox(value=expanded_queries_str, interactive=False), \
               gr.update(choices=choices_for_retrieval_checkbox, value=values_for_retrieval_checkbox), \
               gr.update(value=final_evidence), \
               gr.Textbox(value=time_display_str, interactive=False), \
               mcts_chain, \
               temp_file_path_for_all_evidence, \
               total_retrieval_time # NEW output
    except Exception as e:
        logger.error(f"Error in Gradio interface processing during retrieval: {e}", exc_info=True)
        gr.Warning(f"An unexpected error occurred during query expansion or retrieval: {e}")
        # Return empty updates for CheckboxGroup and State on error, and clear expanded_output and retrieval_time_display and mcts_chain
        # 发生错误时，返回 CheckboxGroup 和 State 的空更新，并清除 expanded_output、retrieval_time_display 和 mcts_chain。
        return gr.Textbox(value=f"An error occurred: {e}", interactive=False), \
               gr.update(choices=[], value=[]), \
               gr.update(value=[]), \
               gr.Textbox(value="Error", interactive=False), \
               {}, \
               "", \
               0.0 # Also return 0.0 for retrieval duration state on error
    finally:
        # Clean up the temporary file for all retrieved evidence
        # 清理所有检索到的证据的临时文件。
        logger.info(f"Cleaned up temporary file for all retrieved evidence: {temp_file_path_for_all_evidence}")


def save_annotation(query: str, document_content: str, selected_evidence_labels: list, gold_answer: str, gold_evidence_list: list, dataset_name: str, idx: int, all_retrieved_evidence_file_path: str, mcts_chain: dict, expanded_queries_str: str, retrieval_duration_seconds: float, answer_eval: str, evidence_eval: str): # Added retrieval_duration_seconds, answer_eval, evidence_eval
    """Saves the annotated data to a JSON file."""
    # 将注释数据保存到 JSON 文件。
    global current_dataset
    logger.info(f"Saving annotation for item index {idx} from dataset '{dataset_name}'.")
    if not current_dataset:
        return "Error: Dataset not loaded. Cannot save annotation."

    if not (0 <= idx < len(current_dataset)):
        return f"Error: Index {idx} is out of bounds for the current dataset (size {len(current_dataset)})."

    item_id = current_dataset[idx].get("id", f"item_{idx}_no_id") # Use a default if 'id' key is missing

    # Load all retrieved evidence from the temporary file for saving
    # 从临时文件加载所有检索到的证据以进行保存。
    all_retrieved_evidence_data = []
    if all_retrieved_evidence_file_path and os.path.exists(all_retrieved_evidence_file_path):
        try:
            with open(all_retrieved_evidence_file_path, 'r', encoding='utf-8') as f:
                all_retrieved_evidence_data = json.load(f)
            logger.info(f"Loaded {len(all_retrieved_evidence_data)} all retrieved evidence items from {all_retrieved_evidence_file_path} for saving.")
        except Exception as e:
            logger.error(f"Error loading all retrieved evidence from temporary file {all_retrieved_evidence_file_path} for saving: {e}", exc_info=True)
            # Continue with empty data if loading fails, or handle as appropriate
    else:
        logger.warning("No all retrieved evidence file path provided or file not found for saving. Saving with empty retrieved_evidence.")


    # 2. `gold_selected_retrieved_evidence`: Evidences explicitly selected by the user
    # 2. `gold_selected_retrieved_evidence`：用户明确选择的证据。
    gold_selected_retrieved_evidence = []

    # Create a map from the display_id to the full evidence object for quick lookup
    # 创建从 display_id 到完整证据对象的映射以进行快速查找。
    label_id_to_evidence_map = {ev.get('display_id'): ev for ev in all_retrieved_evidence_data if ev.get('display_id')}

    for selected_label in selected_evidence_labels:
        # Extract the display_id from the selected label string, e.g., "[ID_0_abcdefgh]"
        # 从选定的标签字符串中提取 display_id，例如 "[ID_0_abcdefgh]"。
        match = re.match(r"\[(ID_\d+_[0-9a-f]{8})\]", selected_label)
        if match:
            display_id_from_label = match.group(1)
            if display_id_from_label in label_id_to_evidence_map:
                gold_selected_retrieved_evidence.append(label_id_to_evidence_map[display_id_from_label])
            else:
                logger.warning(f"Selected label ID '{display_id_from_label}' not found in all retrieved evidence data. Saving label content as 'content'.")
                # Fallback: if ID is not mapped, save the full label string as 'content'
                # Note: No 'utility_score' here, just the raw score from retrieval.
                # 回退：如果 ID 未映射，则将完整标签字符串保存为“content”。
                # 注意：此处没有“utility_score”，只有来自检索的原始分数。
                gold_selected_retrieved_evidence.append({"content": selected_label, "source": "User Selected (ID Mismatch)"})
        else:
            logger.warning(f"Selected label '{selected_label[:50]}...' did not match expected ID format. Saving label content as 'content'.")
            # Fallback: if label doesn't match ID format, save the full label string as 'content'
            # Note: No 'utility_score' here, just the raw score from retrieval.
            # 注意：此处没有“utility_score”，只有来自检索的原始分数。
            gold_selected_retrieved_evidence.append({"content": selected_label, "source": "User Selected (Format Mismatch)"})


    annotation_data = {
        "original_query": query,
        "document_content": document_content,
        "retrieved_evidence": all_retrieved_evidence_data,          # All retrieved evidences presented to the user
        "gold_selected_retrieved_evidence": gold_selected_retrieved_evidence, # User-selected evidence
        "gold_answer": gold_answer,
        "gold_evidence": gold_evidence_list, # Now saved as a list, directly from the state
        "dataset_name": dataset_name,
        "index_in_dataset": idx,
        "item_id": item_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), # This is the save time
        "mcts_based_reasoning_chain": mcts_chain, # Include the MCTS chain
        "expanded_queries": expanded_queries_str, # Include expanded queries string
        "retrieval_duration_seconds": retrieval_duration_seconds, # NEW: Include retrieval duration
        "answer_eval": answer_eval, # NEW: Include answer_eval
        "evidence_eval": evidence_eval # NEW: Include evidence_eval
    }

    # Use a more robust filename including hash of item_id and query for uniqueness
    # 使用包含 item_id 和查询哈希的更健壮的文件名以确保唯一性。
    safe_query_snippet = re.sub(r'[^\w\-_\. ]', '', query[:30]).strip()
    file_hash = hashlib.sha256(f"{dataset_name}_{idx}_{item_id}_{safe_query_snippet}".encode()).hexdigest()[:10]
    save_path = os.path.join(output_save_dir, f"{dataset_name}_{idx}_{file_hash}.json")

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Annotation saved successfully to {save_path}")
        return f"Annotation saved successfully to {save_path}"
    except Exception as e:
        logger.error(f"Error saving annotation to {save_path}: {e}", exc_info=True)
        return f"Error saving annotation: {e}"


# --- Gradio UI Definition ---
# --- Gradio UI 定义 ---
with gr.Blocks(theme=gr.themes.Soft(), css="""
    footer {visibility: hidden} /* Hide Gradio footer */
    .gradio-container {max-width: 1400px; margin: auto; padding: 20px; box-shadow: var(--shadow-xl);} /* Max width and shadow for better layout */
    .gr-box {border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg); padding: var(--spacing-lg); margin-bottom: var(--spacing-lg); background-color: var(--background-fill-primary);} /* General box styling */
    .gr-checkbox-group .gr-check-label {font-size: 0.9em; line-height: 1.3em;} /* Smaller font for checkboxes */
    #gold_data_accordion {margin-top: var(--spacing-lg); border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg);} /* Spacing above gold data */
    #retrieved_evidence_checkbox_group { /* Apply to this checkbox group only */
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid var(--border-color-primary);
        border-radius: var(--radius-md);
        padding: var(--spacing-sm);
        margin-top: var(--spacing-md);
        background-color: var(--background-fill-secondary);
    } /* Scrollable checkbox group */
    #retrieval_section_heading {margin-top: 0; padding-bottom: 0; color: var(--text-color-subdued);} /* Align heading */
    .gr-textbox.gr-box {min-height: 80px;} /* Ensure textboxes have a decent minimum height */
    .gr-button {padding: var(--spacing-md) var(--spacing-lg); border-radius: var(--radius-lg); font-weight: bold;} /* Larger button padding and rounded corners */
    h2 {color: var(--text-color-body); margin-top: 2em; margin-bottom: 0.5em;}
    h3 {color: var(--text-color-body); margin-top: 1.5em; margin-bottom: 0.5em;}
    .section-header {
        font-size: 1.5em;
        font-weight: bold;
        color: var(--primary-500); /* Use a theme color */
        margin-bottom: 1em;
        border-bottom: 2px solid var(--border-color-primary);
        padding-bottom: 0.5em;
    }
    #mcts_chain_display, #mcts_chain_raw_json, #collected_evidence_library_display { /* Apply to MCTS Reasoning Chain Textbox and Raw JSON and new collected evidence textbox */
        max-height: 500px; /* Adjust as needed */
        overflow-y: auto;
        border: 1px solid var(--border-color-primary);
        border-radius: var(--radius-md);
        padding: var(--spacing-sm);
        margin-top: var(--spacing-md);
        background-color: var(--background-fill-secondary);
    }
""") as demo:
    # State variables to hold data across interactions
    # 用于在交互中保存数据的状态变量。
    evidence_full_data_store = gr.State(value=[]) # Stores the full list of retrieved evidence with all its details
    mcts_reasoning_chain_state = gr.State(value={}) # State to hold the MCTS reasoning chain
    temp_all_retrieved_evidence_file_path_state = gr.State(value="") # NEW: Stores path to temp file with all retrieved evidence
    total_retrieval_time_state = gr.State(value=0.0) # NEW: Stores the numerical retrieval time

    gr.Markdown("<h1 style='text-align: center; color: var(--primary-500);'>Long-Text QA Annotation System</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div class='section-header'>Data Navigation</div>")
            dataset_name_input = gr.Textbox(
                label="Dataset Name (e.g., 2wikimqa)",
                value="musique",
                interactive=True,
                scale=2
            )
            current_index_input = gr.Number(label="Current Item Index", value=0, interactive=True, scale=1)
            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous")
                go_to_index_btn = gr.Button("🎯 Go to Index")
                next_btn = gr.Button("➡️ Next")
            status_output = gr.Textbox(label="Status", value="Ready", interactive=False, lines=1)

        with gr.Column(scale=2):
            gr.Markdown("<div class='section-header'>Query & Document</div>")
            query_input = gr.Textbox(label="Query", interactive=True, lines=3, placeholder="Enter the query here...")
            document_text_input = gr.Textbox(label="Document Content", interactive=True, lines=15, placeholder="Paste the document content here...")
            with gr.Row():
                run_retrieval_btn = gr.Button("🚀 Run Multi-Source Retrieval")
                retrieval_time_display = gr.Textbox(label="Retrieval Performance", interactive=False, lines=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div class='section-header'>Query Expansion & Retrieved Evidence</div>")
            expanded_query_output = gr.Textbox(label="Expanded Queries (LLM Generated)", interactive=False, lines=6)
            gr.Markdown("### <span id='retrieval_section_heading'>Retrieved Evidence (Select ALL Relevant)</span>", show_label=False)
            retrieved_evidence_checkbox_group = gr.CheckboxGroup(
                label="Select ALL evidence chunks relevant to the query:",
                choices=[], # Populated dynamically
                value=[],
                interactive=True,
                elem_id="retrieved_evidence_checkbox_group"
            )
            # Removed the raw_all_retrieved_json_from_file checkbox group and its accordion
            with gr.Accordion("Raw Retrieved Evidence Details (For Debugging)", open=False):
                raw_retrieved_json = gr.JSON(label="Full Raw Retrieved Evidence JSON (from state)", value=[])


        with gr.Column(scale=1):
            gr.Markdown("<div class='section-header'>MCTS Based Reasoning Chain</div>")
            generate_mcts_chain_btn = gr.Button("Generate MCTS Reasoning Chain")
            # Changed to Textbox for formatted output
            # 更改为文本框以用于格式化输出。
            mcts_chain_display = gr.Textbox(label="MCTS Based Reasoning Chain", lines=15, interactive=False, elem_id="mcts_chain_display")
            # Raw JSON for debugging, always visible
            # 用于调试的原始 JSON，始终可见。
            mcts_chain_raw_json = gr.JSON(label="MCTS Chain (Raw JSON for Debugging)", visible=True, elem_id="mcts_chain_raw_json") 
            # NEW: Add Textbox for collected_evidence_library
            collected_evidence_library_display = gr.Textbox(label="Collected Evidence Library (from MCTS)", lines=10, interactive=False, elem_id="collected_evidence_library_display")

            gr.Markdown("<div class='section-header'>Annotation & Gold Data</div>")
            with gr.Accordion("Gold Answer & Gold Evidence", open=True, elem_id="gold_data_accordion"):
                gold_answer_input = gr.Textbox(label="Gold Answer", interactive=True, lines=5, placeholder="Enter the gold answer here...")

                gold_evidence_list_state = gr.State(value=[]) # Stores the actual list of gold evidence strings
                gold_evidence_display = gr.Markdown("No gold evidence added yet.")

                new_gold_evidence_input = gr.Textbox(label="Add New Gold Evidence (sentence/phrase)", placeholder="Type a sentence or phrase and click 'Add Gold Evidence'")
                with gr.Row():
                    add_gold_evidence_btn = gr.Button("➕ Add Gold Evidence")
                    clear_gold_evidence_btn = gr.Button("🗑️ Clear All Gold Evidence")
            
            # New evaluation inputs
            # 新的评估输入。
            answer_eval_input = gr.Textbox(label="Answer Evaluation (0/1)", placeholder="Enter 0 for incorrect, 1 for correct")
            # NEW: evidence_eval input (changed to Textbox as requested)
            evidence_eval_input = gr.Textbox(label="Evidence Evaluation (0-1)", placeholder="输入 0 到 1 之间的分数 (例如, 0.85)", interactive=True)

            save_btn = gr.Button("💾 Save Annotation", size="lg")
            save_status_output = gr.Textbox(label="Save Status", interactive=False, lines=1)

    # --- Event Handlers ---
    # --- 事件处理程序 ---
    demo.load(
        initial_load_and_status,
        inputs=[dataset_name_input],
        outputs=[current_index_input, query_input, document_text_input, gold_answer_input,
                 gold_evidence_list_state, gold_evidence_display,
                 retrieved_evidence_checkbox_group, evidence_full_data_store,
                 expanded_query_output, retrieval_time_display, mcts_reasoning_chain_state,
                 temp_all_retrieved_evidence_file_path_state,
                 total_retrieval_time_state,
                 new_gold_evidence_input,
                 answer_eval_input,
                 evidence_eval_input,
                 collected_evidence_library_display] # NEW output
    )

    # Navigation buttons
    # 导航按钮。
    prev_btn.click(
        lambda idx: max(0, idx - 1),
        inputs=[current_index_input],
        outputs=[current_index_input]
    ).then(
        go_to_index, # Changed to go_to_index for consistency
        inputs=[dataset_name_input, current_index_input],
        outputs=[current_index_input, query_input, document_text_input, gold_answer_input,
                 gold_evidence_list_state, gold_evidence_display,
                 retrieved_evidence_checkbox_group, evidence_full_data_store,
                 expanded_query_output, retrieval_time_display, mcts_reasoning_chain_state,
                 temp_all_retrieved_evidence_file_path_state,
                 total_retrieval_time_state,
                 new_gold_evidence_input,
                 answer_eval_input,
                 evidence_eval_input,
                 collected_evidence_library_display] # NEW output
    )

    next_btn.click(
        lambda idx, total: min(total - 1, idx + 1) if total > 0 else 0,
        inputs=[current_index_input, gr.State(lambda: len(current_dataset))], # Pass dataset size
        outputs=[current_index_input]
    ).then(
        go_to_index, # Changed to go_to_index for consistency
        inputs=[dataset_name_input, current_index_input],
        outputs=[current_index_input, query_input, document_text_input, gold_answer_input,
                 gold_evidence_list_state, gold_evidence_display,
                 retrieved_evidence_checkbox_group, evidence_full_data_store,
                 expanded_query_output, retrieval_time_display, mcts_reasoning_chain_state,
                 temp_all_retrieved_evidence_file_path_state,
                 total_retrieval_time_state,
                 new_gold_evidence_input,
                 answer_eval_input,
                 evidence_eval_input,
                 collected_evidence_library_display] # NEW output
    )

    go_to_index_btn.click(
        go_to_index,
        inputs=[dataset_name_input, current_index_input],
        outputs=[current_index_input, query_input, document_text_input, gold_answer_input,
                 gold_evidence_list_state, gold_evidence_display,
                 retrieved_evidence_checkbox_group, evidence_full_data_store,
                 expanded_query_output, retrieval_time_display, mcts_reasoning_chain_state,
                 temp_all_retrieved_evidence_file_path_state,
                 total_retrieval_time_state,
                 new_gold_evidence_input,
                 answer_eval_input,
                 evidence_eval_input,
                 collected_evidence_library_display] # NEW output
    )

    # Retrieval button
    # 检索按钮。
    run_retrieval_btn.click(
        run_retrieval_and_display,
        inputs=[query_input, document_text_input, gold_evidence_list_state], # Added gold_evidence_list_state as input
        outputs=[expanded_query_output, retrieved_evidence_checkbox_group,
                 evidence_full_data_store, retrieval_time_display, mcts_reasoning_chain_state,
                 temp_all_retrieved_evidence_file_path_state,
                 total_retrieval_time_state] # NEW output
    )
    # Removed the .then chained to run_retrieval_btn

    # NEW: MCTS Chain Generation
    # 新增：MCTS 链生成。
    generate_mcts_chain_btn.click(
        generate_mcts_reasoning_chain_actual,
        inputs=[query_input, temp_all_retrieved_evidence_file_path_state], # Removed retrieved_evidence_checkbox_group
        outputs=[mcts_chain_display, mcts_reasoning_chain_state, collected_evidence_library_display] # Now outputs formatted chain string, raw JSON, and collected_evidence_library_display
    )

    # Gold evidence management
    # 黄金证据管理。
    add_gold_evidence_btn.click(
        add_gold_evidence,
        inputs=[gold_evidence_list_state, new_gold_evidence_input],
        outputs=[gold_evidence_list_state, new_gold_evidence_input]
    ).then(
        update_gold_evidence_display,
        inputs=[gold_evidence_list_state],
        outputs=[gold_evidence_display]
    ).then( # Update the main retrieved_evidence_checkbox_group when gold evidence changes
        display_temp_retrieved_evidence_file_content,
        inputs=[temp_all_retrieved_evidence_file_path_state, gold_evidence_list_state],
        outputs=[retrieved_evidence_checkbox_group] # Changed output to main checkbox group
    )

    clear_gold_evidence_btn.click(
        clear_gold_evidence,
        inputs=[],
        outputs=[gold_evidence_list_state, new_gold_evidence_input]
    ).then(
        update_gold_evidence_display,
        inputs=[gold_evidence_list_state],
        outputs=[gold_evidence_display]
    ).then( # Update the main retrieved_evidence_checkbox_group when gold evidence changes
        display_temp_retrieved_evidence_file_content,
        inputs=[temp_all_retrieved_evidence_file_path_state, gold_evidence_list_state],
        outputs=[retrieved_evidence_checkbox_group] # Changed output to main checkbox group
    )

    # Save Annotation
    # 保存注释。
    save_btn.click(
        save_annotation,
        inputs=[
            query_input,
            document_text_input,
            retrieved_evidence_checkbox_group, # These are the *labels* from the checkbox group
            gold_answer_input,
            gold_evidence_list_state,
            dataset_name_input,
            current_index_input,
            temp_all_retrieved_evidence_file_path_state, # NEW input
            mcts_reasoning_chain_state,
            expanded_query_output, # Added expanded_query_output as input
            total_retrieval_time_state, # NEW: Added total_retrieval_time_state as input
            answer_eval_input, # NEW: Added answer_eval_input as input
            evidence_eval_input # NEW: Added evidence_eval_input as input
        ],
        outputs=[save_status_output]
    )

    # Update raw JSON display whenever evidence_full_data_store changes
    # 每当 evidence_full_data_store 更改时更新原始 JSON 显示。
    evidence_full_data_store.change(
        lambda data: gr.JSON(value=data),
        inputs=[evidence_full_data_store],
        outputs=[raw_retrieved_json]
    )

    # Link the MCTS chain state to the raw JSON display (now always visible)
    # 将 MCTS 链状态链接到原始 JSON 显示（现在始终可见）。
    mcts_reasoning_chain_state.change(
        lambda data: gr.JSON(value=data),
        inputs=[mcts_reasoning_chain_state],
        outputs=[mcts_chain_raw_json]
    )

    # Removed the temp_all_retrieved_evidence_file_path_state.change event listener as its functionality is now handled by other events.
    # The main checkbox group is now responsible for displaying all retrieved evidence with proper selection.


if __name__ == "__main__":
    logger.info("Starting Gradio UI...")
    demo.launch()
