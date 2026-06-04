"""LaTeX normalization helpers used by the official PHYBench EED code."""

from __future__ import annotations

import re

from latex2sympy2_extended import latex2sympy


def brackets_balanced(text: str) -> bool:
    stack = []
    bracket_pairs = {")": "(", "]": "[", "}": "{"}
    for char in text:
        if char in bracket_pairs.values():
            stack.append(char)
        elif char in bracket_pairs:
            if not stack or stack[-1] != bracket_pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0


def extract_bracket_content(text: str, bracket_position: int) -> tuple[str | None, int]:
    content = []
    escaped = False
    brace_start = bracket_position + 1
    brace_depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if escaped:
            content.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            content.append(char)
            continue
        if char == "{":
            brace_depth += 1
            content.append(char)
        elif char == "}":
            if brace_depth == 0:
                return "".join(content), index
            brace_depth -= 1
            content.append(char)
        else:
            content.append(char)
    return None, -1


def find_first_unescaped_brace(text: str) -> int:
    escaped = False
    for index, char in enumerate(text):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "{" and not escaped:
            return index
        escaped = False
    return -1


def extract_command(text: str, brace_pos: int) -> str | None:
    i = brace_pos - 1
    parameter_mode = False
    while i >= 0:
        if not parameter_mode and text[i] in ("^", "_"):
            return text[i]
        if not parameter_mode and text[i] not in (" ", "\t", "]", "["):
            break
        if text[i] == "]":
            parameter_mode = True
        if text[i] == "[" and parameter_mode:
            parameter_mode = False
        i -= 1

    if i < 0 or text[i] == "\\":
        return None

    command_end = i
    i -= 1
    while i >= 0 and text[i].isalpha():
        i -= 1
    if i < -1 or text[i] != "\\":
        return None
    return text[i + 1 : command_end + 1]


def remove_command(text: str, command: str, keep_inside: bool = False) -> str:
    pos = text.find(command)
    if pos < 0:
        return text
    end_index = pos + len(command)
    level = 0

    if end_index < len(text) and text[end_index] == "{":
        while end_index < len(text):
            if text[end_index] == "{":
                level += 1
            elif text[end_index] == "}":
                level -= 1
                if level == 0:
                    break
            end_index += 1
    else:
        reduced = "".join([text[0:pos], text[end_index:]])
        return reduced if command not in reduced else remove_command(reduced, command, keep_inside)

    if keep_inside:
        reduced = "".join(
            [text[0:pos], text[pos + len(command) + 1 : end_index], text[end_index + 1 :]]
        )
    else:
        reduced = "".join([text[0:pos], text[end_index + 1 :]])

    return reduced if command not in reduced else remove_command(reduced, command, keep_inside)


def convert_latex_fractions(latex_str: str) -> str:
    pattern = r"\\frac((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))"

    def replacer(match):
        numerator, denominator = match.group(1), match.group(2)
        wrapped_num = (
            f"{{{numerator}}}"
            if not (numerator.startswith("{") and numerator.endswith("}"))
            else numerator
        )
        wrapped_den = (
            f"{{{denominator}}}"
            if not (denominator.startswith("{") and denominator.endswith("}"))
            else denominator
        )
        return fr"\frac{wrapped_num}{wrapped_den}"

    return re.sub(pattern, replacer, latex_str)


def get_first_brace_command(text: str) -> str | None:
    brace_pos = find_first_unescaped_brace(text)
    if brace_pos == -1:
        return None
    return extract_command(text, brace_pos)


def remove_overall_brace(text: str) -> tuple[str, int]:
    pos = find_first_unescaped_brace(text)
    if pos == -1:
        return text, 0
    command = get_first_brace_command(text)
    if not command:
        content, final = extract_bracket_content(text, pos)
        if final == len(text) or "}" not in text[final + 1 :]:
            return content or "", 1
    return text, 0


def exp_frac(text: str) -> str:
    def exp_frac_single(inner: str) -> str:
        position = inner.find("^\\frac") + 1
        if position == 0:
            return inner
        level = 0
        count = 0
        idx = position
        while idx < len(inner):
            if inner[idx] == "{":
                count += 1
            elif inner[idx] == "}":
                count -= 1
                if count == 0:
                    level += 1
                    if level == 2:
                        break
            idx += 1
        return "".join([inner[0:position], "{", inner[position:idx], "}", inner[idx:]])

    updated = exp_frac_single(text)
    count = 0
    while updated != text and count < 100:
        count += 1
        text = updated
        updated = exp_frac_single(text)
    return updated


def find_all(text: str, sub_str: str, allow_overlap: bool = True):
    indexes = []
    start = 0
    step = 1 if allow_overlap else len(sub_str)
    count = 0
    while True and count < 100:
        pos = text.find(sub_str, start)
        if pos == -1:
            break
        indexes.append(pos)
        start = pos + step
        count += 1
    return indexes


def bar_inside_vec(text: str) -> str:
    indices = find_all(text, "\\vec{")
    if not indices:
        return text
    for position in indices:
        idx = position + 4
        idx2 = idx
        level = 0
        while idx2 < len(text):
            if text[idx2] == "{":
                level += 1
            if text[idx2] == "}":
                level -= 1
                if level == 0:
                    break
            idx2 += 1

        inner = text[idx + 1 : idx2]
        inner = remove_command(inner, "\\bar", keep_inside=True)
        text = "".join([text[0 : idx + 1], inner, text[idx2:]])
    return text


def vec_lower_idx(input_str: str) -> str:
    pattern = r"\\vec\{([^{}]+)_{([^{}]+)}\}"
    replacement = r"\\vec{\1}_{\2}"
    return re.sub(pattern, replacement, input_str)


def convert_vec_syntax(text: str) -> str:
    pattern = r"\\vec(\s*)(\\?[a-zA-Zα-ωΑ-Ω]+)"
    replacement = r"\\vec{\2}"
    return re.sub(pattern, replacement, text)


def extract_last_equal_content(text: str, strip_whitespace: bool = True) -> str:
    comparison_operators = ("=", "\\approx", "\\ge", "\\le", "\\geq", "\\leq", "<", ">")
    content = text
    for sign in comparison_operators:
        if sign in text:
            rfind_index = text.rfind(sign)
            if rfind_index != -1:
                content = text[rfind_index + 1 :]
    return content.strip() if strip_whitespace else content


def first_pre_process(text: str, extrac_box: bool = True) -> str:
    text = text.replace("\\{", "(")
    text = text.replace("\\}", ")")
    if not brackets_balanced(text):
        return text
    boxed_content = remove_command(text, "\\boxed", keep_inside=True) if extrac_box else text

    exist_overall_brace = True
    count = 0
    while exist_overall_brace and count < 10:
        boxed_content, exist_overall_brace = remove_overall_brace(boxed_content)
        count += 1

    if "\\quad" in boxed_content:
        boxed_content = boxed_content.split("\\quad")[0]

    last_equal_content = extract_last_equal_content(boxed_content)

    exist_overall_brace = True
    count = 0
    while exist_overall_brace and count < 10:
        last_equal_content, exist_overall_brace = remove_overall_brace(last_equal_content)
        count += 1
    return last_equal_content


def second_pre_process(text: str) -> str:
    kill_commands = ["\\begin", "\\end"]
    remove_commands = [
        "\\text",
        "\\mathbf",
        "\\mathrm",
        "\\pmb",
        "\\hat",
        "\\overline",
        "\\boldsymbol",
    ]
    remove_content = [
        "\\,",
        "$",
        ",",
        "`",
        "latex",
        "\\left",
        "\\right",
        "\\text",
        "\\mathrm",
        "\\Bigr",
        "\\Bigl",
        "\n",
        "\\]",
        "\\[",
        "\\Big",
        "\\bigl",
        "\\bigr",
        "\\biggl",
        "\\biggr",
        "\\displaystyle",
        "\\boldsymbol",
        "\\infty",
    ]
    replace_content = [
        ("\\operatorname{asin}", "\\asin"),
        ("\\operatorname{sech}", "\\sech"),
        ("\\operatorname{acos}", "\\acos"),
        ("\\operatorname{sinh}", "\\sinh"),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\Exp", "\\exp"),
        ("\\times", "\\bar{times}"),
        ("\\partial", "\\bar{partial}"),
        ("\\perp", "\\bar{perp}"),
        ("\\epsilon", "\\varepsilon"),
        ("\\varOmega", "\\Omega"),
        ("I", "\\bar{I}"),
        ("_e", "_{e}"),
        ("e_", "\\bar{e}_"),
        ("E_", "\\bar{E}_"),
        ("\\pm", "+"),
        ("\\mp", "-"),
        ("{+}", "{p}"),
        ("{-}", "{m}"),
        ("_+", "_p"),
        ("_-", "_m"),
    ]

    for command in kill_commands:
        text = remove_command(text, command, keep_inside=False)
    for command in remove_commands:
        text = remove_command(text, command, keep_inside=True)
    for content in remove_content:
        text = text.replace(content, "")
    for source, target in replace_content:
        text = text.replace(source, target)
    text = convert_latex_fractions(text)
    text = bar_inside_vec(text)
    text = vec_lower_idx(text)
    text = convert_vec_syntax(text)
    text = exp_frac(text)
    if text and text[-1] == ".":
        return text[:-1]
    return text


class MyConfig:
    interpret_as_mixed_fractions: bool = False
    interpret_simple_eq_as_assignment: bool = False
    interpret_contains_as_eq: bool = True
    lowercase_symbols: bool = False


class MyNormalization:
    basic_latex: bool = True
    units: bool = False
    malformed_operators: bool = True
    nits: bool = True
    boxed = "all"
    equations: bool = False


def master_convert(text):
    preprocessed_stage1 = first_pre_process(text)
    preprocessed_stage2 = second_pre_process(preprocessed_stage1)
    return latex2sympy(
        preprocessed_stage2,
        normalization_config=MyNormalization(),
        conversion_config=MyConfig(),
    )
