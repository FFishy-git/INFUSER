"""Patched HumanEvalPlusEvaluator with AZR-aligned config.

Changes from OC's default HumanEvalPlusEvaluator:
1. Time limits: min=10s, gt_factor=8x (OC: 0.2s, 4x)
2. Tree-sitter sanitization: uses evalplus's sanitize() to extract valid
   Python from prompt+prediction, fixing indentation issues with OC's
   humaneval_postprocess_v2 which strips leading whitespace via lstrip().
"""

import json
import os.path as osp
import tempfile
from typing import Dict, List, Union

import numpy as np
from datasets import Dataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def _estimate_pass_at_k(
    num_samples: np.ndarray,
    num_correct: np.ndarray,
    k: int,
) -> np.ndarray:
    """Estimates pass@k using the unbiased estimator from the Codex paper."""

    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)]
    )


@ICL_EVALUATORS.register_module()
class HumanEvalPlusEvaluatorAZR(BaseEvaluator):
    """HumanEval+ evaluator with AZR-aligned time limits and tree-sitter sanitization."""

    def __init__(
        self,
        k: List[int] = [1, 10, 100],
        min_time_limit: float = 10.0,
        gt_time_limit_factor: float = 8.0,
    ) -> None:
        try:
            import evalplus  # noqa: F401
        except ImportError:
            raise ImportError(
                'Please install evalplus: pip install evalplus'
            )
        self.k = k
        self.min_time_limit = min_time_limit
        self.gt_time_limit_factor = gt_time_limit_factor
        super().__init__()

    def evaluate(
        self,
        k: Union[int, List[int]],
        n: int,
        original_dataset: Dataset,
        **score_kwargs,
    ):
        """Override base evaluate to bypass per-replica splitting.

        Our score() handles pass@k internally via evalplus which groups by
        task_id. The base evaluator's replica logic (split → score → group)
        doesn't work with our custom evaluator.

        We pass ALL n*M predictions flat to score() alongside M references
        and M test_set entries. score() iterates predictions in chunks of n
        per task_id, and evalplus groups by task_id for pass@k computation.
        """
        real_size = len(original_dataset) // max(n, 1)
        predictions = score_kwargs.get('predictions', [])
        references = score_kwargs.get('references', [])[:real_size]
        test_set = original_dataset.select(range(real_size))

        # Reshape predictions: [run0_task0..run0_taskM, run1_task0.., ..]
        # → list of lists: [[task0_run0, task0_run1, ..], [task1_run0, ..], ..]
        grouped_preds = [[] for _ in range(real_size)]
        for i, pred in enumerate(predictions):
            task_idx = i % real_size
            grouped_preds[task_idx].append(pred)

        results = self.score(
            predictions=grouped_preds,
            references=references,
            test_set=test_set,
        )
        return results

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        from evalplus.data import get_human_eval_plus, write_jsonl
        from evalplus.evaluate import evaluate

        try:
            from evalplus.sanitize import sanitize
            _has_sanitize = True
        except ImportError:
            _has_sanitize = False

        # Load problem metadata for entry_point lookup.
        problems = get_human_eval_plus()

        prompts = [item['prompt'] for item in test_set]
        humaneval_preds = []
        for preds, refer, prompt in zip(predictions, references, prompts):
            if not isinstance(preds, list):
                preds = [preds]
            entry_point = problems[refer]['entry_point'] if refer in problems else None
            for pred in preds:
                solution = prompt + pred
                # Tree-sitter sanitization: extracts the target function and
                # its dependencies from the full solution text.  Handles both
                # markdown code blocks and raw function-body completions.
                if _has_sanitize:
                    try:
                        sanitized = sanitize(solution, entry_point)
                        if sanitized.strip():
                            solution = sanitized
                    except Exception:
                        pass  # fall back to raw prompt+pred on sanitizer failure
                humaneval_preds.append({'task_id': refer, 'solution': solution})
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = osp.join(tmp_dir, 'human_eval.jsonl')
            write_jsonl(out_dir, humaneval_preds)
            # evalplus evaluate() writes results next to the samples file.
            # PyPI evalplus: <samples>.replace('.jsonl', '_eval_results.json')
            # AZR fork: supports explicit output_file kwarg.
            # Try output_file first (AZR), fall back to default path (PyPI).
            import inspect as _inspect
            _eval_sig = _inspect.signature(evaluate)
            _eval_kwargs = dict(
                dataset='humaneval',
                samples=out_dir,
                base_only=False,
                parallel=None,
                i_just_wanna_run=False,
                test_details=False,
                min_time_limit=self.min_time_limit,
                gt_time_limit_factor=self.gt_time_limit_factor,
                mini=False,
            )
            results_path = osp.join(tmp_dir, 'eval_results.json')
            if 'output_file' in _eval_sig.parameters:
                _eval_kwargs['output_file'] = results_path
            evaluate(**_eval_kwargs)
            # Find result file: try explicit path, then PyPI default location
            if not osp.isfile(results_path):
                results_path = out_dir.replace('.jsonl', '_eval_results.json')
            with open(results_path, 'r') as f:
                results = json.load(f)

            # Compute pass@k from eval results
            PASS = "pass"
            total = np.array([len(r) for r in results['eval'].values()])
            base_correct = np.array([
                sum(1 for entry in r if entry['base_status'] == PASS)
                for r in results['eval'].values()
            ])
            plus_correct = np.array([
                sum(1 for entry in r if entry['base_status'] == PASS and entry['plus_status'] == PASS)
                for r in results['eval'].values()
            ])

            pass_at_k = {}
            for k in self.k:
                if total.min() >= k:
                    pass_at_k[f'pass@{k}'] = float(_estimate_pass_at_k(total, plus_correct, k).mean())

            details = {}
            for index in range(len(predictions)):
                task_id = references[index]
                if task_id in results['eval']:
                    r = results['eval'][task_id]
                    # Emit per-sample base_result / plus_result / is_correct as
                    # lists when n > 1 so downstream sol_eval normalizers
                    # (verl_inf_evolve.sol_eval.external_benchmarks) can
                    # recover the full per-sample correctness vector and
                    # compute pass@k for every power of 2. Upstream OC
                    # collapsed these to ``r[0]`` only, which dropped 99% of
                    # the data on n=128 runs.
                    base_statuses = [entry['base_status'] for entry in r]
                    plus_statuses = [entry['plus_status'] for entry in r]
                    is_correct_list = [
                        b == PASS and p == PASS
                        for b, p in zip(base_statuses, plus_statuses)
                    ]
                    details[str(index)] = {
                        'prompt': prompts[index],
                        'prediction': predictions[index],
                        'reference': task_id,
                        'base_result': base_statuses if len(r) > 1 else base_statuses[0],
                        'plus_result': plus_statuses if len(r) > 1 else plus_statuses[0],
                        'is_correct': is_correct_list if len(r) > 1 else is_correct_list[0],
                    }
                    if len(r) > 1:
                        details[str(index)]['n_samples'] = len(r)

        out = {f'humaneval_plus_{k}': v * 100 for k, v in pass_at_k.items()}
        out['details'] = details
        return out
