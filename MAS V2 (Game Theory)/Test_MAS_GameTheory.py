import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from Agents_GameTheory import (
    DialectalAnalysisAgent,
    ContextManipulationDetectionAgent,
    ResponseMonitoringAgent,
    DecisionFusionAgent
)

# 1) Load & label
df = pd.read_excel("Results.xlsx")
df["main_cat"]        = df["Result"].str.split(", ").str[0]
df["actual_response"] = df["main_cat"].map({
    "Refusal":      "Safe",
    "Non-Refusal":  "Unsafe"
})
df["actual_prompt"]   = "Prompt Injection Detected"  # all prompts are malicious

# 2) Split 80/20 stratified on output safety
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["actual_response"]
)
print(f"Train: {len(train_df)} rows; Test: {len(test_df)} rows")

# 3) Persist train split as Excel so Agent 3 can read it
train_df.to_excel("train_split.xlsx", index=False)

# 4) Instantiate your agents
agent1 = DialectalAnalysisAgent()
agent2 = ContextManipulationDetectionAgent(csv_file_path="Agent2Book2.csv")
agent3 = ResponseMonitoringAgent("train_split.xlsx")  # positional only
agent4 = DecisionFusionAgent()                        # uses default weights

# 5) Run the full pipeline on test_df
records = []
for _, row in test_df.iterrows():
    prompt      = row["Prompt"]
    llm_out     = row["Model Output"]
    actual_o    = row["actual_response"]

    # Agent 1: dialect confidence
    _, dialect_conf = agent1.predict_dialect(prompt)

    # Agent 2: raw context manipulation score
    _, raw_ctx_score, _ = agent2.context_manipulation_detection(prompt, dialect_conf)
    norm_ctx_score = min(raw_ctx_score / 4.0, 1.0)




    # Agent 3: response monitoring
    out3              = agent3.classify_llm_output(prompt, llm_out)
    response_score    = out3["Conf"]
    predicted_response = "Unsafe" if out3["Main Category"] == "Non-Refusal" else "Safe"

    # Agent 4: fused final risk & decision
    final_risk, decision = agent4.decision_fusion(
        dialect_conf, raw_ctx_score, response_score
    )

    records.append({
        "prompt":            prompt,
        "actual_response":   actual_o,
        "predicted_response":predicted_response,
        "dialect_conf":      round(dialect_conf, 3),
        "raw_ctx_score":     (raw_ctx_score),
        "norm_ctx_score":  norm_ctx_score,
        "response_score":    round(response_score, 3),
        "final_risk":        round(final_risk, 3),
        "prediction":        decision
    })

results_df = pd.DataFrame(records)

# 6) Agent 3 metrics (Unsafe vs Safe on model outputs)
print("=== Agent 3: Model-Output Safety ===\n")
cm3 = confusion_matrix(
    results_df["actual_response"],
    results_df["predicted_response"],
    labels=["Unsafe","Safe"]
)
print(pd.DataFrame(
    cm3,
    index=["Actual Unsafe","Actual Safe"],
    columns=["Pred Unsafe","Pred Safe"]
))
print("\n" + classification_report(
    results_df["actual_response"],
    results_df["predicted_response"],
    labels=["Unsafe","Safe"],
    target_names=["Unsafe","Safe"]
))

# 7) Summary stats for all signals
print("\nSummary statistics for all signals:")
print(results_df[[
    "dialect_conf",
    "norm_ctx_score",
    "response_score",
    "final_risk"
]].describe().T)

# 8) Agent 4 metrics (Prompt-Injection Detection)
total     = len(results_df)
strict    = (results_df["prediction"] == "Prompt Injection Detected").sum()
potential = (results_df["prediction"] == 
             "Potential Prompt Injection Detected, Please Rephrase"
            ).sum()

print("\n=== Agent 4: Prompt-Injection Detection ===\n")
print(f"Average final risk score: {results_df['final_risk'].mean():.3f}")
print(f"Prompt Injection Detected: {strict}/{total} ({strict/total:.1%})")
print(f"Potential Prompt Injection Detected, Please Rephrase: {strict+potential}/{total} ({(strict+potential)/total:.1%})")
