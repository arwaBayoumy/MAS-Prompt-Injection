# Arabic LLMs Prompt Injection Attacks – A Case Study on GPT, Jais, and Multi-Agent Defense System

This repository implements a Multi-Agent System (MAS) designed to defend Arabic Large Language Models (LLMs) against prompt injection attacks. Our framework is built to detect adversarial behavior across various Arabic dialects including Egyptian, Levantine, and Tunisian and uses a layered agent-based approach to assess both prompt structure and model response before returning output to the user.

---

## Project Overview

Prompt injection attacks are a major threat to LLM safety, especially in low-resource languages like Arabic. This system was developed to:
- Evaluate GPT-o1 and Jais 30b against adversarial prompts
- Understand vulnerabilities across Arabic dialects
- Propose a MAS-based defense against unsafe generation

The MAS performs multi-stage analysis using specialized agents for dialect detection, manipulation detection, ethical behavior monitoring, and final decision-making.

---

## Dataset Creation

Due to the lack of adversarial Arabic datasets, we curated our own:

###  Source & Categories
- We selected 15 prompts from the **AdvBench** dataset (Zou et al., 2023) and expanded them into 45 categorized prompts across:
  1. **Instruction Manipulation**
  2. **Obfuscation (Syntactic Variation)**
  3. **User-driven Injections (Indirect)**

### Dialect Translation
Each of the 45 prompts was translated into:
- **Egyptian Arabic**
- **Levantine Arabic**
- **Tunisian Arabic**

This resulted in a total of **135 dialect-specific adversarial prompts**.

---

##  LLM Evaluation & Annotation

- We tested GPT-o1 (automated via API) and Jais 30b (manually via chat interface) using our 135-prompt dataset.
- Responses were labeled manually using an adapted version of the evaluation framework from Al Ghanim et al. (2024).
- Labels were organized under:
  - **Main categories**: Refusal / Non-Refusal
  - **Subcategories**: Direct, Advice, Unsafe, Misunderstanding

All results were stored in annotated CSV files for training and evaluation.

---

## MAS System Overview


| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **Agent 1** | Dialectal Analysis | Prompt | Detected dialect + confidence |
| **Agent 2** | Context Manipulation Detection | Prompt | Severity score (0–5) + Action |
| **Agent 3** | Response Monitoring | Prompt + Output | Subcategory, Main category, confidence |
| **Agent 4** | Decision Fusion | All agent results | Final system decision (Safe / Rephrase / Block) |

---

## Flask-Based UI

Our system includes a simple UI for researchers and testers:
- Submit prompts in Arabic (or Arabizi)
- See dialect and manipulation scores
- Evaluate output behavior
- View final MAS decision

---

## Techniques Used

- Fine-tuned `camelbert` (from CAMeL Lab) for dialect classification - Agent 1
- Heuristic Decision-Making + Bellman equation for risk scoring - Agent 2
- TF-IDF vectorization + Naive Bayes classifier for response safety classification - Agent 3
- Flask backend for interactive testing

---



