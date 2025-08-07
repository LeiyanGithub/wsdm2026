# LongAF: Beyond Length Limitations: An Evidence-Centric Long-Document Question Answer Annotation Framework

### Overview of **LongAF** 🚀

**LongAF** is an innovative framework designed to address the inherent challenges in long-document Question Answering (QA) data annotation, which include low human efficiency, high costs, and issues like the "lost in the middle" and "length extrapolation" problems of large language models (LLMs). The framework significantly improves the accuracy and efficiency of annotation by effectively compressing lengthy documents into concise reasoning chains.

🔹 **Key Features & Innovations:**

* **Efficient Long Document Compression**: Compresses ultra-long documents (over 10,000 tokens) into concise reasoning chains, reducing the reading burden by up to 40 times.
* **Multi-Path Evidence Retrieval**: Employs a combination of document chunking, query expansion, and sparse (BM25), dense (BGE), and LLM-enhanced retrieval to ensure comprehensive evidence recall.
* **MCTS-Based Reasoning Chain Generation**: Utilizes a Monte Carlo Tree Search (MCTS) algorithm to construct structured reasoning paths from retrieved evidence candidates, effectively solving the combinatorial explosion problem of evidence combinations.
* **Intuitive Human Verification Interface**: Provides an intuitive interface that allows annotators to manually verify, refine, and supplement the generated reasoning chains, ensuring annotation quality while reducing time-consuming document reading.

🔬 **Why LongAF?**

The LongAF framework achieves significant breakthroughs in both efficiency and accuracy. Compared to purely manual annotation methods, it reduces annotation time to approximately **1/8** and cost to about **1/4**, all while maintaining near-human-level quality. This demonstrates the superior performance of LongAF's human-LLM collaborative model in handling long-document annotation tasks, providing a viable and efficient solution for creating large-scale, high-quality real-world datasets.


## Datasets

We conducted a comprehensive evaluation of our framework on several long-document QA datasets from the LongBench benchmark.
* HotpotQA
* MuSiQue
* NarrativeQA
* Qasper
* 2WikimQA
* FanoutQA
* LongBench_v2 (covering various domains such as academic, legal, financial, and government reports)

## Requirements

Based on the models and tools mentioned in the paper, you can infer the following potential dependencies:
```
openai
sentence_transformers
```

## Step-to-step Instructions on How to Run

The paper does not provide a directly executable script, but we offer an overview of the framework's running process based on its methodology.

1.  Write your OpenAI API key and your python environment path in the `mcts.py` and `annotation.py` file.
2.  Annotators use an intuitive interface to verify and refine the generated reasoning chains to ensure the accuracy of the final answer and the completeness of the evidenc, 
```bash
python annotation.py
```

