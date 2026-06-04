"""Prompt templates used by the verl_inf_evolve trainer.

This is a local copy of the subset currently used by
`verl_inf_evolve.trainer.self_evolution_trainer`.

Layout
------
1. Shared ICL canonical shots + render helpers
2. MCQ answer generation         (system / user / ICL system / few-shot / ICL user)
3. Free-form answer generation   (system / user / ICL system / few-shot / ICL user / builder)
4. MCQ question generation       (document-conditioned)
5. Free-form question generation (document-conditioned, math-verifier-compatible)
6. MCQ question generation       (seeded, no document)
"""

from typing import Optional


# =============================================================================
# Shared ICL canonical shots + render helpers
# NOTE: The ICL / few-shot path is NOT invoked by the current trainer pipeline.
#       Everything in this section only feeds the *_ICL_* / *_FEW_SHOT_EXAMPLES
#       constants below, which are likewise unused at runtime. Kept for now in
#       case the few-shot path is re-enabled.
# =============================================================================

_CANONICAL_ANSWER_GENERATION_ICL_SHOTS = [
    {
        "question_text": (
            "A refracting telescope consists of two converging lenses separated by 100 cm. "
            "If the eyepiece has a focal length of 20 cm, what is the angular magnification of the telescope?"
        ),
        "choices": ["10", "6", "4", "25"],
        "reasoning": (
            "The total length of a refracting telescope equals the sum of the focal lengths: "
            "f_objective + f_eyepiece = 100 cm, so f_objective = 80 cm. "
            "Angular magnification is f_objective / f_eyepiece = 80/20 = 4."
        ),
        "correct_letter": "C",
        "correct_answer_text": "4",
    },
    {
        "question_text": "What is the hybridization of the central carbon atom in formaldehyde (CH2O)?",
        "choices": ["sp", "sp2", "sp3", "sp3d"],
        "reasoning": (
            "In formaldehyde, the carbon forms two C-H single bonds and one C=O double bond, "
            "giving three regions of electron density. Three electron regions correspond to sp2 hybridization."
        ),
        "correct_letter": "B",
        "correct_answer_text": "sp2",
    },
    {
        "question_text": "During which phase of meiosis does crossing over between homologous chromosomes occur?",
        "choices": ["Prophase I", "Metaphase I", "Prophase II", "Anaphase I"],
        "reasoning": (
            "Crossing over occurs when homologous chromosomes pair up and exchange genetic material. "
            "This synapsis and recombination happens during prophase I of meiosis."
        ),
        "correct_letter": "A",
        "correct_answer_text": "Prophase I",
    },
    {
        "question_text": (
            "A microwave oven is connected to a 120 V outlet and draws a current of 2 A. "
            "What is the rate at which energy is used by the microwave?"
        ),
        "choices": ["240 W", "120 W", "480 W", "60 W"],
        "reasoning": "Power equals voltage times current: P = V * I = 120 V * 2 A = 240 W.",
        "correct_letter": "A",
        "correct_answer_text": "240 W",
    },
    {
        "question_text": (
            "The colors seen in a soap bubble are most directly a result of which optical phenomenon?"
        ),
        "choices": ["Dispersion", "Refraction", "Interference", "Diffraction"],
        "reasoning": (
            "The colorful patterns in soap bubbles arise from thin-film interference, "
            "where light reflecting off the inner and outer surfaces of the film "
            "interferes constructively or destructively depending on thickness and wavelength."
        ),
        "correct_letter": "C",
        "correct_answer_text": "Interference",
    },
]


def _render_mcq_icl_user_turn(question_text: str, choices: list[str]) -> str:
    lines = [f"Question: {question_text}"]
    for idx, choice in enumerate(choices):
        lines.append(f"{chr(ord('A') + idx)}) {choice}")
    return "\n".join(lines) + "\n\nAnswer: Let's think step by step."


def _render_free_form_icl_user_turn(question_text: str) -> str:
    return f"Question: {question_text}\n\nAnswer: Let's think step by step."


def _render_icl_assistant_turn(reasoning: str, boxed_answer: str) -> str:
    return f"{reasoning} The answer is \\boxed{{{boxed_answer}}}"


# =============================================================================
# MCQ answer generation
# =============================================================================

MCQ_ANSWER_GENERATION_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that solves multiple choice questions step by step. "
    "Always show your reasoning and put your final answer letter in \\boxed{{}}."
)


MCQ_ANSWER_GENERATION_PROMPT = """Solve the following multiple choice question step by step.

{question}

Think through this problem carefully. Show your reasoning process, then provide your final answer.

IMPORTANT: Your final answer MUST be enclosed in \\boxed{{<letter choice>}} using ONLY the letter of the correct choice (A, B, C, D, E, F, G, H, I, J, etc.).

Now solve the problem:"""


# UNUSED: ICL/few-shot path is not invoked in the current pipeline.
MCQ_ANSWER_GENERATION_ICL_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that solves multiple choice questions step by step. "
    "Think step by step and then finish your answer with \"The answer is \\boxed{X}\" where X is the correct letter choice. "
    "Do not repeat the question or instructions."
)


# UNUSED: ICL/few-shot path is not invoked in the current pipeline.
MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES = [
    (
        _render_mcq_icl_user_turn(shot["question_text"], shot["choices"]),
        _render_icl_assistant_turn(shot["reasoning"], shot["correct_letter"]),
    )
    for shot in _CANONICAL_ANSWER_GENERATION_ICL_SHOTS
]


# UNUSED: ICL/few-shot path is not invoked in the current pipeline.
MCQ_ANSWER_GENERATION_ICL_PROMPT = """{question}

Answer: Let's think step by step."""


# =============================================================================
# Free-form answer generation
# =============================================================================

FREE_FORM_ANSWER_GENERATION_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that solves questions step by step. "
    "Always show your reasoning and put your final answer in \\boxed{{}}."
)


FREE_FORM_ANSWER_GENERATION_PROMPT = """Solve the following question step by step.

{question}

Think through this problem carefully. Show your reasoning process, then provide your final answer.

IMPORTANT: Your final answer MUST be enclosed in \\boxed{{<answer>}}

Now solve the problem:"""


# UNUSED: ICL/few-shot path is not invoked in the current pipeline.
FREE_FORM_ANSWER_GENERATION_ICL_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that solves questions step by step. "
    "Think step by step and then finish your answer with \"The answer is \\boxed{X}\" where X is the correct final answer. "
    "Do not repeat the question or instructions."
)


# UNUSED: ICL/few-shot path is not invoked in the current pipeline.
FREE_FORM_ANSWER_GENERATION_FEW_SHOT_EXAMPLES = [
    (
        _render_free_form_icl_user_turn(shot["question_text"]),
        _render_icl_assistant_turn(shot["reasoning"], shot["correct_answer_text"]),
    )
    for shot in _CANONICAL_ANSWER_GENERATION_ICL_SHOTS
]


# UNUSED: ICL/few-shot path is not invoked in the current pipeline (only reached
# when build_free_form_messages is called with use_few_shot_icl=True, which the
# trainer never does).
FREE_FORM_ANSWER_GENERATION_ICL_PROMPT = """Question: {question}

Answer: Let's think step by step."""


def build_free_form_messages(
    question_text: str,
    use_few_shot_icl: bool = False,
    system_prompt: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build chat messages for open-ended/free-form answer generation."""
    if system_prompt is not None:
        effective_system_prompt = system_prompt
    elif use_few_shot_icl:
        effective_system_prompt = FREE_FORM_ANSWER_GENERATION_ICL_SYSTEM_PROMPT
    else:
        effective_system_prompt = FREE_FORM_ANSWER_GENERATION_SYSTEM_PROMPT
    normalized_question = str(question_text or "")

    if not use_few_shot_icl:
        return [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": normalized_question},
        ]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": effective_system_prompt},
    ]
    for example_user, example_assistant in FREE_FORM_ANSWER_GENERATION_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": example_assistant})
    messages.append(
        {
            "role": "user",
            "content": FREE_FORM_ANSWER_GENERATION_ICL_PROMPT.format(
                question=normalized_question
            ),
        }
    )
    return messages


# =============================================================================
# MCQ question generation (document-conditioned)
# =============================================================================

MCQ_QUESTION_GENERATION_SYSTEM_PROMPT = (
    "You are an expert educator creating challenging multiple-choice questions. "
    "Always output valid JSON with the exact structure requested."
)


MCQ_QUESTION_GENERATION_PROMPT = """Your task is to create CHALLENGING exam questions from a document by identifying complex relationships and multi-step reasoning paths.

## Document
[BEGINNING OF THE DOCUMENT]
{text}
[END OF THE DOCUMENT]

## Instructions

### Step 1: Complex Information Extraction for MCQ Design

**PRIORITY: Focus on information that enables multiple plausible interpretations and requires synthesis.**

Scan the text and identify information that naturally creates opportunities for sophisticated multiple-choice questions:

**Ideal MCQ content requires:**
* **Synthesis opportunities**: Relationships between 3+ concepts spanning different sections, implicit conclusions requiring combination of multiple facts, systems where changing one parameter affects others
* **Multi-step reasoning paths**: Processes with intermediate steps (each step = potential distractor), calculations with sequential dependencies, procedures requiring decision points about when/how to apply methods
* **Rich comparison spaces**: Comparative analyses revealing subtle distinctions ("however," "but," "except," "unlike"), trade-offs between approaches, overlapping categories or edge cases, prerequisites or conditional relationships
* **Application complexity**: Principles applied to novel scenarios, cause-and-effect chains with intermediate stages, mechanisms where partial understanding yields plausible-but-incomplete explanations
* **Domain-specific depth**: Multi-variable calculations (unit conversions, stoichiometry, equilibrium perturbations), classification problems with boundary conditions, experimental design with multiple controlling factors, predictions integrating multiple scientific laws

**AVOID** (these create poor MCQ questions):
* Single, directly stated facts that allow simple lookup
* Simple definitions that stands alone
* Values or numbers mentioned in isolation
* Information that requires no synthesis
* Lists without relationships between items
* Trivial categorizations
* Information where all wrong answers would be obviously implausible

### Step 2: Difficulty Enhancement Process

**EXPLICITLY STATE YOUR HARDENING PROCESS** Before generating the question, describe your strategy to make it harder:
1. What simple version would you avoid?
2. What complexity layers will you add?
3. Are there any seamless traps for common misconceptions to exploit for distractors?
4. How can you leverage subtle, non-obvious interactions between different content elements to create more engaging and intellectually demanding questions?
5. What common shortcuts will you block?
6. How will you ensure multi-step reasoning is required?

### Step 3: Advanced Question Generation

Generate ONE high-quality MCQ question that:
* Requires applying multiple concepts from different parts of the document
* Tests understanding of relationships, not just recall of facts
* Forces reasoning through multiple steps to reach the answer
* May require comparing or contrasting different scenarios
* Could involve "what if" scenarios based on principles in the text
* Tests ability to apply concepts to slightly modified situations

**CRITICAL - Self-Contained Requirements**:
* Questions must be 100% self-contained and standalone
* NEVER use: "according to the document", "in the document", "as mentioned", "the passage states", "based on the analysis", etc.
* Write as if for a formal exam with no reference material
* Include all necessary context within the question itself, but don't reveal any intermediate reasoning steps or key insights that would make the question easy
* Define any specialized terms if needed for clarity

### Step 4: Difficulty-Driven Design

**TARGET: Generate HARD/EXTRA HARD questions by design**
* HARD: Synthesize 4+ concepts; multi-step problem solving; pattern recognition
* EXTRA HARD: Complex system analysis; counter-intuitive applications; edge cases

Design questions that CANNOT be answered by:
* Looking up a single fact
* Finding one sentence with the answer
* Simple keyword matching

### Step 5: Knowledge Integration Requirements

Document the reasoning path that shows why this is a difficult question:
* List 3+ distinct pieces of information needed from different parts
* Show the logical connections required between these pieces
* Explain why simple lookup won't work
* Include intermediate reasoning steps

### Step 6: Multiple Choice Design Guidelines

Create between 4 and 8 answer choices following these STRICT rules:

**Length Balance**: All options must be approximately equal length (+/-20%)
**Unit Consistency**: All numerical answers must use identical units and formatting
**Tone Neutrality**: Avoid overly certain language ("definitely", "always", "never") unless justified
**Plausibility**: All distractors must be genuinely plausible based on partial understanding
**More choices increase difficulty**: Use 6-8 choices for complex questions

**Distractor Design**:
* Common calculation errors from the multi-step process
* Results from applying only partial reasoning
* Mixing up related concepts from the document
* Reasonable approximations that miss key factors

### Step 7: Self-Testing Filter (AFTER MCQ Creation)

**SOLVE YOUR OWN MCQ AS A STUDENT WOULD** Now test the complete multiple choice question:
1. What's the quickest path a student might try with these options?
2. Can you eliminate 2+ options without full understanding? If yes, redesign distractors
3. Does seeing the options make the answer obvious? If yes, improve distractors
4. Count the reasoning steps required even with options visible - if less than 3, REJECT
5. Time estimate: Would this MCQ take <30 seconds? If yes, make it harder
6. Could a student guess correctly by pattern matching the options? If yes, rebalance

### Step 8: Final Complexity Verification

Before finalizing, verify your question is NOT Easy by checking:
* Can it be answered by finding one sentence? If yes, redesign
* Does it require connecting multiple document sections? If no, add complexity
* Would someone need to understand relationships, not just facts? If no, refocus
* Are all MCQ options balanced and using consistent formatting? If no, revise
* Did your self-test of the MCQ take more than 1 minute? If no, increase difficulty

## Output Format

You MUST output ONLY a valid JSON object with this exact structure:

{{
    "question_text": "Your complete, self-contained question here?",
    "choices": [
        "First choice text (without letter prefix)",
        "Second choice text (without letter prefix)",
        "Third choice text (without letter prefix)",
        "Fourth choice text (without letter prefix)",
        "Fifth choice text (optional)",
        "Sixth choice text (optional)",
        "Seventh choice text (optional)",
        "Eighth choice text (optional)"
    ],
    "ground_truth": "The exact text of the correct choice (must match one of the choices exactly)",
    "difficulty": "hard",
    "answer_quote": [
        "Relevant quote 1 from the document showing key information",
        "Relevant quote 2 from the document showing different piece needed",
        "Relevant quote 3 (include multiple quotes showing different pieces needed)"
    ],
    "hardening_process": "Your explicit strategy for making this question difficult (from Step 2)",
    "knowledge_and_reasoning_steps": "Detailed reasoning path showing why this is Hard/Extra Hard difficulty",
    "self_test_solution": "Your step-by-step solution of the MCQ showing the difficulty (from Step 7)"
}}

Field descriptions:
- "question_text": A challenging, self-contained question requiring synthesis. Return empty string if document lacks sufficient complexity.
- "choices": Array of 4-8 answer options without letter prefixes
- "ground_truth": The exact text of the correct choice (MUST match one of the choices exactly)
- "difficulty": Target difficulty level (hard or extra_hard)
- "answer_quote": Multiple verbatim quotes from the document showing the different pieces needed (not just one quote)
- "hardening_process": Your explicit strategy for making this question difficult (from Step 2)
- "knowledge_and_reasoning_steps": Detailed reasoning path showing why this is Hard/Extra Hard difficulty
- "self_test_solution": Your step-by-step solution of the MCQ showing the difficulty (from Step 7)

CRITICAL RULES:
1. Your final answer must contain exactly one JSON object matching the requested schema. Put this JSON object last.
2. The "choices" array must have at least 4 items and at most 8 items
3. Do NOT include letter prefixes (A), B), etc.) in the choices
4. "ground_truth" must be the exact text of one of the choices
5. The question must be answerable, with the key component or solution step within the document content
6. If the document lacks sufficient complexity, return empty strings for all fields

Schema illustration only (do not copy this content):
{{"question_text": "<new self-contained question derived from the document>", "choices": ["<choice 1>", "<choice 2>", "<choice 3>", "<choice 4>"], "ground_truth": "<exact text of one choice>", "difficulty": "hard", "answer_quote": ["<document quote 1>", "<document quote 2>", "<document quote 3>"], "hardening_process": "<difficulty strategy>", "knowledge_and_reasoning_steps": "<reasoning path>", "self_test_solution": "<step-by-step solution>"}}.

Now, generate the question:"""


# =============================================================================
# Free-form question generation (document-conditioned, math-verifier-compatible)
# =============================================================================

FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT = (
    "You are an expert educator creating challenging free-form math questions. "
    "d"
)


FREE_FORM_QUESTION_GENERATION_PROMPT = """Your task is to create ONE CHALLENGING free-form math question from a document by identifying relationships that require multi-step reasoning.

The generated question will be evaluated with the repo's symbolic math verifier (`data_source="math"`), which extracts a boxed final answer and compares it with `math_verify` plus a custom symbolic fallback. Therefore the ground truth must be a concise, machine-verifiable mathematical answer, not prose.

## Document
[BEGINNING OF THE DOCUMENT]
{text}
[END OF THE DOCUMENT]

## Instructions

### Step 1: Select a Verifiable Reasoning Target

First identify a compact mathematical quantity that the document determines indirectly. Good targets are values, formulas, or equations that emerge only after combining several stated relationships.

Prioritize targets with these properties:
* The answer is short enough to compare symbolically, such as a number, fraction, radical expression, polynomial, ratio, or equation.
* At least three document facts are needed, and they are not all adjacent in the document.
* One or more intermediate quantities must be computed or inferred before the final answer is available.
* The problem can be made self-contained by restating only the necessary setup, not by pasting a full document excerpt.
* A partially correct solution would lead to a plausible but different mathematical answer.

Reject targets with these properties:
* The answer is a sentence, explanation, proof, name, definition, or subjective judgment.
* The answer depends on units, formatting conventions, or unstated assumptions.
* The answer is directly quoted or obtained by substituting into a single obvious formula.
* The document supports several reasonable interpretations of the requested quantity.
* The final answer would require a list, interval, or multiple alternatives unless there is no cleaner single-answer target.

### Step 2: Difficulty Enhancement Process

Document your strategy in "hardening_process":
1. What simple version would be too easy?
2. Which 3+ document facts or relationships must be synthesized?
3. What intermediate calculations or logical steps are required?
4. What common shortcuts or edge cases will the question block?
5. Why is the final answer uniquely determined and machine-verifiable?

### Step 3: Advanced Free-Form Question Generation

Generate ONE high-quality free-form question that:
* Requires connecting multiple concepts from different parts of the document
* Cannot be answered by simple lookup or keyword matching
* Forces multi-step reasoning to reach a precise mathematical answer
* Is self-contained and can be solved without seeing the document
* Clearly asks for a single final answer
* Uses constraints that make the answer unique

**CRITICAL - Self-Contained Requirements**:
* Questions must be 100% self-contained and standalone
* NEVER use: "according to the document", "in the document", "as mentioned", "the passage states", "based on the analysis", etc.
* Write as if for a formal math contest with no reference material
* Include all necessary context, definitions, units, and constraints
* Do not reveal intermediate reasoning steps or key insights in the question itself

### Step 4: Math-Verifier-Compatible Answer Requirements

The "ground_truth" must be compatible with the symbolic math verifier:
* It MUST be a single concise mathematical answer string.
* Prefer exact forms over decimal approximations: use "3/7", "\\frac{{3}}{{7}}", "2\\sqrt{{5}}", "x^2+3x+2", "x=4", "\\frac{{1}}{{3}}", "\\int_a^b x^2\\,dx", "\\int_0^T v(t)\\,dt", "\\sum_{{k=1}}^{{n}} k", or "\\binom{{n}}{{2}}" when appropriate.
* Use decimal notation only when the problem explicitly asks for a rounded decimal; state the required precision in the question.
* Use no units, prose, explanations, lists, multiple answers, or surrounding "\\boxed{{}}" in "ground_truth".
* Definite integrals with numeric bounds should normally be evaluated exactly. Definite integrals with symbolic bounds are acceptable when the unevaluated integral is the natural final form; make the variable of integration and bounds unambiguous.
* Indefinite-integral answers are acceptable only when the question explicitly specifies the arbitrary constant symbol used in "ground_truth" (for example, "use C for the constant of integration"). Do not use an unevaluated indefinite integral as "ground_truth".
* Avoid intervals unless the question is explicitly about a set of values; these are less robust in the generic math path.
* Avoid answers that are only equivalent under unstated conventions, such as unsimplified expressions with ambiguous variables.

Valid ground_truth examples: "42", "-2/3", "\\frac{{5}}{{12}}", "2\\sqrt{{3}}", "x=7", "n^2+n", "\\frac{{1}}{{3}}", "\\int_a^b x^2\\,dx", "\\int_0^T v(t)\\,dt", "\\sum_{{k=1}}^{{n}} k"
Invalid ground_truth examples: "\\boxed{{42}}", "42 meters", "about 3.14", "x is 7", "41 or 42", "\\int x^2\\,dx", ["42"]

### Step 5: Solution Verification Process

Document the complete solution in "self_test_solution":
* Identify all required information pieces
* Show each intermediate calculation or reasoning step
* Handle edge cases or special conditions
* Derive the final mathematical answer
* Verify the final answer is unique
* Verify the formatted "ground_truth" exactly matches the math-verifier-compatible format

### Step 6: Alternative Interpretations Check

Before finalizing, verify:
* The question has exactly one defensible answer
* All constraints are stated clearly
* Different valid approaches yield the same mathematical answer
* The answer does not depend on unstated assumptions
* The problem cannot be solved by copying one sentence from the document

## Output Format

Your final answer must contain exactly one JSON object with this exact structure. Put this JSON object last.

{{
    "question_text": "Your complete, self-contained free-form question here?",
    "ground_truth": "The exact concise mathematical answer",
    "difficulty": "hard",
    "answer_type": "numerical",
    "benchmark_type": "qa_open",
    "data_source": "math",
    "answer_quote": [
        "Relevant quote 1 from the document showing key information",
        "Relevant quote 2 from the document showing different piece needed",
        "Relevant quote 3 (include multiple quotes showing different pieces needed)"
    ],
    "hardening_process": "Your explicit strategy for making this question difficult",
    "knowledge_and_reasoning_steps": "Detailed reasoning path showing why this is Hard/Extra Hard difficulty",
    "self_test_solution": "Your step-by-step solution ending with the exact ground_truth answer"
}}

Field descriptions:
- "question_text": A challenging, self-contained free-form question requiring synthesis. Return empty string if document lacks sufficient complexity.
- "ground_truth": The exact math-verifier-compatible answer string.
- "difficulty": Target difficulty level (hard or extra_hard).
- "answer_type": One of "numerical", "expression", or "equation".
- "benchmark_type": Must be exactly "qa_open".
- "data_source": Must be exactly "math".
- "answer_quote": Multiple verbatim quotes from the document showing the different pieces needed.
- "hardening_process": Your explicit strategy for making this question difficult.
- "knowledge_and_reasoning_steps": Detailed reasoning path showing why this is Hard/Extra Hard difficulty.
- "self_test_solution": Your step-by-step solution of the question.

CRITICAL RULES:
1. Your final answer must contain exactly one JSON object matching the requested schema. Put this JSON object last.
2. Do not include a "choices" field for free-form questions.
3. "ground_truth" must be a single concise mathematical answer string.
4. "ground_truth" must not contain units, prose, explanations, lists, multiple alternatives, or surrounding "\\boxed{{}}".
5. The question must ask for a single final mathematical answer and have exactly one defensible answer.
6. The question must be answerable, with the key component or solution step supported by the document content.
7. If the document lacks sufficient complexity for a machine-verifiable math question, return empty strings for string fields.
8. Do NOT copy, paraphrase, or lightly modify any schema illustration as the generated question content.

Schema illustration only (do not copy this content):
{{"question_text": "<new self-contained math question derived from the document>", "ground_truth": "<exact concise mathematical answer>", "difficulty": "hard", "answer_type": "<numerical|expression|equation>", "benchmark_type": "qa_open", "data_source": "math", "answer_quote": ["<document quote 1>", "<document quote 2>", "<document quote 3>"], "hardening_process": "<difficulty strategy>", "knowledge_and_reasoning_steps": "<reasoning path>", "self_test_solution": "<step-by-step solution ending with the ground_truth answer>"}}

Now, generate the question:"""


# =============================================================================
# MCQ question generation (seeded, no document)
# =============================================================================

SEEDED_MCQ_QUESTION_GENERATION_SYSTEM_PROMPT = (
    "You are an expert educator creating challenging multiple-choice questions. "
    "Always output valid JSON with the exact structure requested."
)


SEEDED_MCQ_QUESTION_GENERATION_PROMPT = """\
Below are example multiple-choice questions for reference. Study their domain, difficulty level, and style.

{seed_examples}

Now create ONE new, original multiple-choice question that:
- Uses the examples ONLY to infer broad domain and difficulty
- Tests a specific concept, scenario, quantities, entities, and answer that are NOT present in any example
- Requires multi-step reasoning or synthesis of multiple concepts
- Is completely self-contained (no references to external documents or passages)
- Is NOT a copy, paraphrase, template-fill, number swap, entity swap, or minor variant of any example above
- Has 4-8 plausible answer choices of approximately equal length

Think step-by-step before writing the question:
1. What domain/topic area do the examples cover?
2. What specific concept can I test that is NOT already covered by the examples?
3. How will I ensure multi-step reasoning is required?
4. What plausible distractors arise from partial understanding?
5. Compare the draft against every example; if it shares the same stem structure, surface scenario, variables, named entities, or answer set, discard it and create a different question.

You MUST output ONLY a valid JSON object with this exact structure:

{{"question_text": "Your complete, self-contained question here?", "choices": ["First choice", "Second choice", "Third choice", "Fourth choice"], "ground_truth": "The exact text of the correct choice", "difficulty": "hard"}}

CRITICAL RULES:
1. Your final answer must contain exactly one JSON object matching the requested schema. Put this JSON object last.
2. The "choices" array must have 4 to 8 items
3. Do NOT include letter prefixes (A), B), etc.) in the choices
4. "ground_truth" must exactly match one of the choices
5. Do NOT copy or paraphrase any example question
6. Do NOT reuse any example's exact question text, answer choices, ground truth, scenario, named entities, or numeric values"""


def format_seed_examples(seed_questions: list[dict]) -> str:
    """Format a list of seed question dicts into a human-readable string for
    inclusion in the seeded MCQ generation prompt.

    Args:
        seed_questions: List of dicts, each with at least ``question_text``,
            ``choices``, and ``ground_truth``.

    Returns:
        Formatted string with numbered examples.
    """
    parts = []
    for i, q in enumerate(seed_questions, 1):
        lines = [f"Example {i}:"]
        lines.append(f"Question: {q['question_text']}")
        lines.append("Choices:")
        for j, choice in enumerate(q["choices"]):
            lines.append(f"  {chr(ord('A') + j)}) {choice}")
        lines.append(f"Answer: {q['ground_truth']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


__all__ = [
    # MCQ answer generation
    "MCQ_ANSWER_GENERATION_SYSTEM_PROMPT",
    "MCQ_ANSWER_GENERATION_PROMPT",
    "MCQ_ANSWER_GENERATION_ICL_SYSTEM_PROMPT",
    "MCQ_ANSWER_GENERATION_FEW_SHOT_EXAMPLES",
    "MCQ_ANSWER_GENERATION_ICL_PROMPT",
    # Free-form answer generation
    "FREE_FORM_ANSWER_GENERATION_SYSTEM_PROMPT",
    "FREE_FORM_ANSWER_GENERATION_PROMPT",
    "FREE_FORM_ANSWER_GENERATION_ICL_SYSTEM_PROMPT",
    "FREE_FORM_ANSWER_GENERATION_FEW_SHOT_EXAMPLES",
    "FREE_FORM_ANSWER_GENERATION_ICL_PROMPT",
    "build_free_form_messages",
    # MCQ question generation (document-conditioned)
    "MCQ_QUESTION_GENERATION_SYSTEM_PROMPT",
    "MCQ_QUESTION_GENERATION_PROMPT",
    # Free-form question generation (document-conditioned, AIME-compatible)
    "FREE_FORM_QUESTION_GENERATION_SYSTEM_PROMPT",
    "FREE_FORM_QUESTION_GENERATION_PROMPT",
    # MCQ question generation (seeded)
    "SEEDED_MCQ_QUESTION_GENERATION_SYSTEM_PROMPT",
    "SEEDED_MCQ_QUESTION_GENERATION_PROMPT",
    "format_seed_examples",
]
