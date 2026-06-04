"""BBEH open-form verifier.

Implements the core normalization + fuzzy matching behavior used by
google-deepmind/bbeh's ``evaluate.py`` for open-form answers.
"""

from __future__ import annotations

from typing import Optional

from verl_inf_evolve.utils.benchmarks.verifiers import register


def _strip_latex(response: str) -> str:
    """Strip simple latex wrappers similar to BBEH's reference evaluator."""
    text = response.strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1]
    if "boxed{" in text and text.endswith("}"):
        text = text[:-1].split("boxed{")[-1]
    if "text{" in text and text.endswith("}"):
        text = text[:-1].split("text{")[-1]
    if "texttt{" in text and text.endswith("}"):
        text = text[:-1].split("texttt{")[-1]
    return text


def _preprocess_prediction(prediction: str) -> str:
    pred = _strip_latex(prediction.strip()).lower()
    pred = pred.replace(", ", ",").replace("**", "")
    pred = pred.split("\n")[0]
    if pred.endswith("."):
        pred = pred[:-1]
    return pred


def _preprocess_reference(reference: str) -> str:
    ref = reference.strip().lower()
    ref = ref.replace(", ", ",")
    return ref


def _fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True

    # "(a)" vs "a"
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numeric equality
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # Quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Trailing question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


@register("bbeh_open")
def verify(
    predicted: Optional[str],
    ground_truth: str,
    metadata: Optional[dict] = None,
) -> Optional[bool]:
    """Verify BBEH open-form answer via normalization + fuzzy matching."""
    if predicted is None:
        return None
    if not predicted.strip():
        return None

    prediction = _preprocess_prediction(predicted)
    if not prediction:
        return None
    reference = _preprocess_reference(ground_truth)
    return _fuzzy_match(prediction, reference)

