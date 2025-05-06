# Arabic LLMs Prompt Injection Attacks – A Case Study on GPT, Jais, and Multi-Agent Defense System
*Amr Hamza*, *Arwa Bayoumy*, *Farah Hammam*, *Yassin Abdelmeguid*

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

**Source & Categories**
- We selected 15 prompts from the **AdvBench** dataset (Zou et al., 2023) and expanded them into 45 categorized prompts across:
  1. **Instruction Manipulation**
  2. **Obfuscation (Syntactic Variation)**
  3. **User-driven Injections (Indirect)**

**Dialect Translation**
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
![MAS HL Arch drawio](https://github.com/user-attachments/assets/dcafaf39-597e-4d4d-a6ff-5bd1bb1bd0ba)

| Agent | Role | Input | Output |
|-------|-------|-------|--------|
| **Agent 1** |  Dialectal Analysis | Prompt | Detected dialect + confidence |
| **Agent 2.0** | MDP | Context Manipulation Detection (MDP) | Prompt | Manipulation score (0–5) + Bellman-optimal Action |
| **Agent 2.5** | Game Theory | Context Manipulation Detection (Game-Theory) | Prompt | Manipulation score (0–5) + Game-optimal Action |
| **Agent 3** | Response Monitoring | Prompt + Output | Subcategory, Main category, confidence |
| **Agent 4** |  Decision Fusion | All agent results | Final system decision (Safe / Rephrase / Block) |


**Techniques Used**
- Fine-tuned `camelbert` (from CAMeL Lab) for dialect classification - Agent 1
- Heuristic Decision-Making + Bellman equation for risk scoring - Agent 2
  - **Agent 2.0 (MDP)**: Bellman equation + prefix scoring using dynamic transition matrix
  - **Agent 2.5 (Game Theory)**: Game-theoretic payoff matrix with severity weighting
- TF-IDF vectorization + Naive Bayes classifier for response safety classification - Agent 3
- Flask backend for interactive testing


## Flask-Based UI
Our system includes a simple UI for researchers and testers:
- Submit prompts in Arabic (or Arabizi)
- See dialect and manipulation scores
- Evaluate output behavior
- View final MAS decision
<img width="1498" alt="Screenshot 2025-05-04 at 2 43 11 PM" src="https://github.com/user-attachments/assets/b6045c17-705d-4710-9764-e1e3edee22e4" />

---
## How to Use the Agents


This project includes **two implementations of Agent 2**, organized by folders:

>  **Note:** Both systems share the same architecture, datasets, and testing flow — the only difference is in how **Agent 2** handles manipulation detection.

| Agent        | Version | Approach                          | Folder Path                  |
|--------------|---------|-----------------------------------|------------------------------|
| Agent 2.0    | MDP     | Bellman Equation & Prefix Scoring | `MAS V1 (MDP)/Agents_MDP.py` |
| Agent 2.5    | Game Theory | Game-Theoretic Payoff Matrix     | `MAS V2 (Game Theory)/Agents_GameTheory.py` |

---


### Testing MAS V1 with Agent 2.0 (MDP-Based)

**Access:**  
The MDP-based Agent 2 is implemented in:
```bash
MAS V1 (MDP)/Agents_MDP.py
```

**Run the evaluation script:**  
```bash
cd "MAS V1 (MDP)"
python Test_MAS_MDP.py
```

**Behavior:**
- Uses `Agent2Book.csv` to detect dangerous words and prefixes  
- Applies Bellman Equation logic for manipulation severity  
- Returns manipulation score and recommended action  
- Outputs full evaluation logs, risk scores, and MAS decisions  

---

### Testing MAS V2 with Agent 2.5 (Game Theory-Based)

**Access the MAS:**  
The Game-Theory-based Agent 2 is implemented in:
```bash
MAS V2 (Game Theory)/Agents_GameTheory.py
```
**Run the evaluation script:**  
```bash
cd "MAS V2 (Game Theory)"
python Test_MAS_GameTheory.py
```

**Behavior:**
- Uses `Agent2Book2.csv` with weighted danger terms  
- Applies a payoff matrix to compute manipulation severity  
- Returns manipulation score and optimal action  
- Outputs evaluation summaries and final MAS decisions  

 **Inputs & Outputs**

Both testing scripts use the same evaluation dataset:
- **Input file:** `Results.xlsx` (containing prompt, model output, labels)
- **Output:** printed risk scores, decisions, and classification metrics
--- 
### Testing the Full Flask App (Real-Time MAS UI)

Run the real-time MAS evaluation app (make sure Flask is installed):

**App location:**  
```bash
app/app.py
```

**Run the app:**  
```bash
cd app
python app.py
```

**Features:**  
- Submit prompts  
- View dialect, manipulation score, output label, and MAS decision  
- Web interface powered by Flask  

---




