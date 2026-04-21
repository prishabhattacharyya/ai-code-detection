# Multi-Lingual AI Code Detection

## Overview

With the growing use of AI tools like ChatGPT, identifying whether code is human-written or machine-generated has become critical for software quality, security, and academic integrity. This project tackles that challenge by fine-tuning transformer-based models on 500,000+ code samples spanning ten major LLM families. Beyond standard detection, it extends to more realistic scenarios such as hybrid (human + AI) and adversarial code, reflecting how modern AI-assisted development actually occurs. This task was done as a part of SemEval-2026 Task 13, an international NLP shared task focused on detecting and attributing AI-generated code across languages, generators, and domains.

---
## Problem
Modern LLMs generate highly realistic code, making authorship detection increasingly difficult. Existing methods struggle to generalize across languages and domains, fail on unseen LLMs, and often perform near random (~50% accuracy) in code detection settings. This project aims to build a system that can distinguish human from machine-generated code, generalize across multiple languages and LLM families, and extend to real-world scenarios like hybrid and adversarial code.

---
## Approach

### Subtask A: Binary Machine-Generated Code Detection
- **Goal:** Classify code as human-written or machine-generated  
- **Model:** CodeBERT with binary classification head  
- **Setup:**
  - Sequence length: 256 tokens  
  - Learning rate: 2e-5  
  - Batch size: 8 (train), 16 (eval)  
- **Key Insight:** Local stylistic patterns (naming, structure) are highly discriminative.  

---

### Subtask B: Multi-Class LLM Attribution
- **Goal:** Identify the source of code across 11 classes consisting of human-written or one of 10 LLM families (DeepSeek, Qwen, OpenAI, Meta-LLaMA, and others).
- **Model:** CodeBERT with multi-class classification head (11 classes)  
- **Setup:**
  - Sequence length: 512 tokens  
  - Mixed precision training (fp16)  
  - AdamW optimizer with learning rate scheduling  
- **Key Insight:**
  - Severe class imbalance (88.4% human samples) combined with high stylistic similarity across LLM outputs, was a challenge. 
---

### Subtask C: Hybrid & Adversarial Code Detection
- **Goal:** Handle more realistic scenarios by classifying code into four classes including human, machine-generated, hybrid, and adversarial
- **Model:** Qwen2.5-Coder-7B with 4-class classification head
- **Setup:**
  - Training method: QLoRA (4-bit quantization + LoRA adapters)
  - Loss computed on response tokens only (train_on_response_only)
  - Best checkpoint selected by validation Macro F1
- **Key Insight:** Real-world AI-assisted development produces hybrid and adversarial code that standard binary detection completely misses, making this the most practically relevant formulation of the problem

---
## Results

| Subtask | Model | Accuracy | Macro F1 |
|---------|-------|----------|----------|
| A — Binary Detection | CodeBERT | 99.3% | 0.993 |
| B — Multi-class Attribution | CodeBERT | 88.4% | 0.086 |
| C — Hybrid & Adversarial | Qwen2.5-Coder-7B + QLoRA | In progress | — |

The gap between accuracy and Macro F1 on Subtask B reflects the class imbalance, since a naive baseline that always predicts "human" would also achieve 88.4% accuracy. Macro F1 is therefore the more meaningful metric.

---

## Key Insights
- Pretrained code models capture meaningful stylistic signals like naming conventions, control flow, and library usage, which are sufficient for strong binary detection
- Fine-grained attribution across LLM families is significantly harder due to class imbalance and stylistic similarity between generators
- Real-world detection requires handling hybrid and adversarial scenarios, which standard benchmarks do not address
---

## Tech Stack
- Python  
- PyTorch  
- HuggingFace Transformers  
- CodeBERT  
- Qwen2.5-Coder  
- QLoRA (LoRA + 4-bit quantization)
- scikit-learn

---
## How to Run

Each subtask has its own notebook. Open them directly in Google Colab or clone the repo and run them in Jupyter.

```bash
git clone https://github.com/prishabhattacharyya/ai-code-detection.git
```

| Notebook | Subtask |
|----------|---------|
| `SemEval_A.ipynb` | Binary Detection |
| `SemEval_B.ipynb` | Multi-Class Attribution |
| `SemEval_C.ipynb` | Hybrid & Adversarial Detection |

**Notes:**
- GPU is recommended (T4 or better) for all notebooks
- Subtask A loads data via Kaggle: you will be prompted to run `kagglehub.login()` at the start
- Subtask B loads data directly from HuggingFace (no Kaggle account needed)
- Subtask A saves checkpoints to Google Drive: you will be prompted to authenticate when running in Colab

## Contributions
This project was completed as part of a team. I independently designed and implemented the Subtask C pipeline, including model selection, QLoRA fine-tuning setup, and the 4-class task formulation for hybrid and adversarial code detection. I also co-authored the system description paper.

## References
Based on SemEval-2026 Task 13 on machine-generated code detection. See the [accompanying paper](AI_Code_Detection_SemEval2026.pdf) for full experimental details.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


