"""Patched LCBCodeGenerationEvaluator that supports model_type='base'.

The upstream OC evaluator hardcodes extract_code_generation(item) without
model_type, defaulting to 'chat' which requires ```python code blocks.
Base models often output raw code without backticks, scoring 0%.

This subclass adds a model_type parameter and passes it through to the
extraction function.
"""

from opencompass.datasets.livecodebench.evaluator import (
    LCBCodeGenerationEvaluator,
    codegen_metrics,
)
from opencompass.datasets.livecodebench.extract_utils import (
    extract_code_generation,
    extract_code_generation_v2,
)
from opencompass.registry import ICL_EVALUATORS


@ICL_EVALUATORS.register_module()
class LCBCodeGenerationEvaluatorBase(LCBCodeGenerationEvaluator):
    """LCB code generation evaluator that extracts raw output for base models."""

    def __init__(self, *args, model_type="base", k_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.k_list = list(k_list or [1])

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                "error":
                "predictions and references have different "
                f"length. len(predictions): {len(predictions)}, "
                f"len(references): {len(references)}"
            }

        if self.extractor_version == "v1":
            predictions = [
                [extract_code_generation(item, model_type=self.model_type)]
                for item in predictions
            ]
        elif self.extractor_version == "v2":
            predictions = [
                [extract_code_generation_v2(item, model_type=self.model_type)]
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
