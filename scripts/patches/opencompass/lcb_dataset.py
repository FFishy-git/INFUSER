"""Patched LCB dataset and evaluator that matches official LiveCodeBench prompts.

Official LCB uses different prompts for base vs chat models:
  - Base (GenericBase): 1-shot example + question (no system message)
  - Chat (GenericChat): system message + 0-shot question with format instructions

The upstream OC config uses the GenericChat prompt for both — no few-shot
example for base models, no system message distinction.

This module provides:
  - LCBCodeGenerationDatasetOfficial: dataset with official prompt formatting
  - LCBCodeGenerationEvaluatorOfficial: evaluator with v2 extraction and model_type
"""

import json
import os
from pathlib import Path
from types import ModuleType

from datasets import DatasetDict, load_dataset


# ── Fix pyext / RuntimeModule for Python 3.12+ ─────────────────────────────
# OC's testing_util.py imports pyext.RuntimeModule for code execution, but
# pyext is incompatible with Python 3.12 (inspect.getargspec removed).
# Provide a minimal drop-in replacement.
class _RuntimeModule:
    @staticmethod
    def from_string(module_name, extra, code):
        mod = ModuleType(module_name)
        exec(compile(code, "<string>", "exec"), mod.__dict__)
        return mod


def _patch_runtime_module():
    """Monkey-patch RuntimeModule into OC's testing_util if missing."""
    try:
        from opencompass.datasets.livecodebench import testing_util
        if testing_util.RuntimeModule is None:
            testing_util.RuntimeModule = _RuntimeModule
    except ImportError:
        pass


_patch_runtime_module()

from opencompass.datasets.livecodebench.livecodebench import (
    LCBCodeGenerationDataset,
)
from opencompass.datasets.livecodebench.evaluator import (
    LCBCodeGenerationEvaluator,
    codegen_metrics,
)
from opencompass.datasets.livecodebench.extract_utils import (
    extract_code_generation_v2,
)
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path


# ── Load few-shot examples ──────────────────────────────────────────────────
_PATCH_DIR = Path(__file__).parent / "lcb_few_shot"


def _load_few_shot():
    with open(_PATCH_DIR / "func.json") as f:
        func_examples = json.load(f)
    with open(_PATCH_DIR / "stdin.json") as f:
        stdin_examples = json.load(f)
    return func_examples, stdin_examples


# ── Official prompt builders ────────────────────────────────────────────────

# Constants from official LCB
_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
_FMT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
_FMT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def _build_base_prompt(question_content: str, starter_code: str) -> str:
    """Build official GenericBase prompt: 1-shot example + actual question."""
    func_examples, stdin_examples = _load_few_shot()

    has_starter = bool(starter_code)
    examples = func_examples if has_starter else stdin_examples

    def _example_block(example, is_actual=False):
        p = "### Question\n"
        p += example["question"]
        p += "\n\n"
        if has_starter:
            p += "### Starter Code\n"
            p += example.get("sample_code", starter_code)
            p += "\n\n"
        p += "### Answer\n\n"
        p += example.get("answer", "")
        if example.get("answer", ""):
            p += "\n\n"
        return p

    prompt = ""
    # 1-shot example
    prompt += _example_block(examples[0])
    # Actual question (empty answer for model to fill)
    prompt += _example_block({
        "question": question_content,
        "sample_code": starter_code,
        "answer": "",
    }, is_actual=True)
    return prompt


def _build_chat_prompt(question_content: str, starter_code: str) -> str:
    """Build official GenericChat prompt: question with format instructions."""
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {_FMT_WITH_STARTER}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {_FMT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


# ── Patched Dataset ─────────────────────────────────────────────────────────

@LOAD_DATASET.register_module()
class LCBCodeGenerationDatasetOfficial(LCBCodeGenerationDataset):
    """LCB dataset with official prompt formatting based on model_type."""

    @staticmethod
    def load(path: str = "opencompass/code_generation_lite",
             local_mode: bool = False,
             release_version: str = "release_v5",
             start_date: str = None,
             end_date: str = None,
             model_type: str = "base"):

        from datetime import datetime

        path = get_data_path(path, local_mode=local_mode)
        dataset = load_dataset(
            path,
            split="test",
            version_tag=release_version,
            trust_remote_code=True,
        )

        def transform(item):
            question_content = item["question_content"]
            starter_code = item.get("starter_code", "") or ""

            if model_type == "base":
                # Official GenericBase: 1-shot prompt, no format_prompt needed
                item["full_prompt"] = _build_base_prompt(
                    question_content, starter_code
                )
                # For OC template compatibility: put entire prompt in
                # question_content and leave format_prompt empty
                item["question_content"] = item["full_prompt"]
                item["format_prompt"] = ""
            else:
                # Official GenericChat: 0-shot with format instructions
                item["full_prompt"] = _build_chat_prompt(
                    question_content, starter_code
                )
                # Build format_prompt the same way as upstream
                if starter_code:
                    item["format_prompt"] = (
                        f"### Format: {_FMT_WITH_STARTER}\n"
                        f"```python\n{starter_code}\n```\n\n"
                    )
                else:
                    item["format_prompt"] = (
                        f"### Format: {_FMT_WITHOUT_STARTER}\n"
                        "```python\n# YOUR CODE HERE\n```\n\n"
                    )

            # Load test cases for evaluation
            import pickle, zlib, base64
            public_test_cases = json.loads(item["public_test_cases"])
            private_test_cases = item["private_test_cases"]
            try:
                private_test_cases = json.loads(private_test_cases)
            except Exception:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(
                                private_test_cases.encode("utf-8")
                            )
                        )
                    )
                )

            metadata = json.loads(item["metadata"])
            item["evaluation_sample"] = json.dumps({
                "inputs": [
                    t["input"] for t in public_test_cases + private_test_cases
                ],
                "outputs": [
                    t["output"] for t in public_test_cases + private_test_cases
                ],
                "fn_name": metadata.get("func_name", None),
            })

            return item

        dataset = dataset.map(transform)

        if start_date is not None:
            p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
            dataset = dataset.filter(
                lambda e: p_start_date
                <= datetime.fromisoformat(e["contest_date"])
            )
        if end_date is not None:
            p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
            dataset = dataset.filter(
                lambda e: datetime.fromisoformat(e["contest_date"])
                <= p_end_date
            )

        return DatasetDict({"test": dataset, "train": dataset})


# ── Patched Evaluator ───────────────────────────────────────────────────────

@ICL_EVALUATORS.register_module()
class LCBCodeGenerationEvaluatorOfficial(LCBCodeGenerationEvaluator):
    """LCB evaluator with v2 extraction (last code block).

    For base models, official LCB uses model_output.strip() (GenericBase).
    For chat models, official LCB uses last ``` block.
    Both are handled by extract_code_generation_v2 with appropriate model_type.
    """

    def __init__(self, *args, model_type="base", k_list=None, **kwargs):
        # Force extractor_version to v2
        kwargs["extractor_version"] = "v2"
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.k_list = list(k_list or [1])

    @staticmethod
    def _extract_base(output: str) -> str:
        """Extract code from base model output.

        Base models with 1-shot prompts generate the answer then continue
        hallucinating more question-answer pairs.  Truncate at the first
        ``### Question`` or ``### Answer`` boundary after the code starts.
        """
        # Find the first continuation marker
        for marker in ("### Question", "### Answer"):
            idx = output.find(marker)
            if idx > 0:
                output = output[:idx]
                break
        return output.strip()

    def score(self, predictions, references):
        from opencompass.datasets.livecodebench.extract_utils import (
            extract_code_generation_v2,
        )

        if len(predictions) != len(references):
            return {
                "error":
                "predictions and references have different "
                f"length. len(predictions): {len(predictions)}, "
                f"len(references): {len(references)}"
            }

        if self.model_type == "base":
            # Base models: truncate at first continuation marker, then strip
            predictions = [
                [self._extract_base(item)]
                for item in predictions
            ]
        else:
            # Chat models: extract last code block
            predictions = [
                [extract_code_generation_v2(item, model_type="chat")]
                for item in predictions
            ]

        evaluation_samples = dict()
        for idx in range(len(self.dataset)):
            evaluation_samples[
                self.dataset[idx]["question_id"]
            ] = self.dataset[idx]["evaluation_sample"]

        filtered_predictions = []
        filtered_references = []
        for idx, item in enumerate(references):
            if item in self.dataset["question_id"]:
                filtered_predictions.append(predictions[idx])
                filtered_references.append(item)

        filtered_references = [
            evaluation_samples[item] for item in filtered_references
        ]
        filtered_references = [
            {"input_output": item} for item in filtered_references
        ]

        extracted_predictions = {}
        for idx, content in enumerate(filtered_predictions):
            extracted_predictions[idx] = content

        metrics, eval_results, final_metadata = codegen_metrics(
            filtered_references,
            filtered_predictions,
            k_list=self.k_list,
            num_process_evaluate=self.num_process_evaluate,
            timeout=self.timeout,
        )

        return self._build_results(
            extracted_predictions, metrics, eval_results, final_metadata
        )
