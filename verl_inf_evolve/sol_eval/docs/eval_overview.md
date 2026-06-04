## Existing Scores

The scores for non-coding benchmark is documented under
https://exp.evolverealty.uk/reports/sol_eval_qwen3_8b_seed456_ckpt95


math500, mmlu_pro, olympiadbench-math partition, supergpqa, aime2024, aime2025, bbeh all have scores close to SPICE, RZero's reported baseline for Qwen3 8b model. 

We find our score is systematically lower than their reported baseline by 2~4 points. 

The likely reason is that we use non-zero temperature for evaluation. 

## Prompt Comparison

For prompt, we all consistently use the same prompt across all non coding benchmark. 

Ours (defined in `verl_inf_evolve/utils/prompts.py`, shared between training and sol_eval):

### MCQ (multiple choice) benchmarks

**System prompt:**
```
You are a knowledgeable assistant that solves multiple choice questions step by step. Always show your reasoning and put your final answer letter in \boxed{}.
```

**User prompt:**
```
Solve the following multiple choice question step by step.

{question}

Think through this problem carefully. Show your reasoning process, then provide your final answer.

IMPORTANT: Your final answer MUST be enclosed in \boxed{} using ONLY the letter of the correct choice (A, B, C, D, etc.).

Example format for your final answer:
\boxed{B}

Now solve the problem:
```

### Free-form (open-ended) benchmarks

**System prompt:**
```
You are a knowledgeable assistant that solves questions step by step. Always show your reasoning and put your final answer in \boxed{}.
```

**User prompt:**
```
Solve the following question step by step.

{question}

Think through this problem carefully. Show your reasoning process, then provide your final answer.

IMPORTANT: Your final answer MUST be enclosed in \boxed{<answer>}

Now solve the problem:
```

### R-Zero baseline prompts

Defined in `baselines/r-zero/`. Much simpler — a single-sentence system prompt with no reasoning scaffold or format examples.

**System prompt (all benchmarks):**
```
Please reason step by step, and put your final answer within \boxed{}.
```

**User prompt (free-form):** the raw question text, no wrapper.

**User prompt (MCQ — MMLU-Pro, SuperGPQA):**
```
{question}
Options are:
(A): {option_A}
(B): {option_B}
...

Please reason step by step, and put your final answer option within \boxed{}. Only put the option letter in the box, e.g. \boxed{A}. There is only one correct answer.
```

**User prompt (MCQ — BBEH):**
```
{question}

Please reason step by step, and put your final answer option within \boxed{}.
```

## Remaining Scoring issue

### GPQA-Diamond
Sol_eval scores 38%, which matches the non-thinking mode for Qwen3-8b model in https://arxiv.org/pdf/2505.09388

### Coding Score
I'm refactoring sol_eval to support LCB and humaneval+
