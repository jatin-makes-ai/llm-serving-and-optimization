# LLM Quality & Accuracy Evaluation Framework

Evaluating an LLM for production selection requires balancing **Performance (Latency & Throughput)** against **Accuracy & Reliability**.

---

## The 4 Quality Evaluation Pillars

### 1. Mathematical & Logical Reasoning (Exact Match)
* **Goal:** Test whether the model can perform multi-step deduction without arithmetic hallucinations.
* **Metric:** **Exact Match (EM)** of the final numeric answer.
* **Scoring Logic:** Normalize output, extract the final numerical token (or `#### <answer>`), and compare against ground truth.

### 2. Factual Knowledge (Token-level F1 Score)
* **Goal:** Check factual recall while allowing natural conversational phrasing.
* **Metric:** **Token F1 Score** (harmonic mean of Precision and Recall):
  $$\text{Precision} = \frac{|\text{Tokens}_{\text{pred}} \cap \text{Tokens}_{\text{gold}}|}{|\text{Tokens}_{\text{pred}}|}, \quad \text{Recall} = \frac{|\text{Tokens}_{\text{pred}} \cap \text{Tokens}_{\text{gold}}|}{|\text{Tokens}_{\text{gold}}|}$$
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 3. Instruction & Format Adherence (IFEval)
* **Goal:** Test whether the model obeys programmatic constraints required for software integration.
* **Sub-checks:**
  * **JSON Syntactic Validity:** Does `json.loads(response)` succeed without markdown code blocks (` ```json `) or extraneous text?
  * **Negative Constraints:** Did it avoid prohibited words/tokens?
  * **Length Limits:** Did it obey word/character ceilings?

### 4. Context Faithfulness & Extraction
* **Goal:** Verify that in a RAG or document context, the model extracts the exact facts provided rather than hallucinating external training data.

---

## Combined Model Selection Score (Performance vs. Quality)

When comparing candidate models, we plot a **Quality vs. Throughput Frontier**:

$$\text{Selection Index} = \frac{\text{Composite Accuracy (\%)}}{\text{TPOT (ms)}}$$

* Higher Quality + Lower TPOT = The optimal production model.
