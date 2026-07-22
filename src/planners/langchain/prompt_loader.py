import re
from pathlib import Path
from typing import FrozenSet

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt_text(name: str, expected_variables: FrozenSet[str] = frozenset()) -> str:
    """Loads a static system-prompt text file (editable via CVaLab) and
    checks its set of {template_variable} placeholders against exactly what
    the call site will later fill in via .invoke(...). LangChain accepts any
    syntactically valid {name} as a new required input variable without
    complaint, so a typo'd literal brace meant as prose (e.g. "the {sex}
    field") or an accidentally-deleted real placeholder both parse fine and
    either crash mid-request or silently drop instructions from the prompt,
    instead of failing at import time. This check turns both into a loud
    Action-startup failure instead. Literal braces meant as prose (e.g.
    showing JSON syntax to the model) must be escaped as doubled {{ }},
    matching LangChain's own escaping convention.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    text = path.read_text(encoding="utf-8")
    found = set(re.findall(r"\{([^{}]*)\}", text.replace("{{", "").replace("}}", "")))
    if found != expected_variables:
        problems = []
        missing = expected_variables - found
        if missing:
            problems.append(f"missing required placeholder(s) {sorted(missing)}")
        unexpected = found - expected_variables
        if unexpected:
            problems.append(
                f"unexpected placeholder(s) {sorted(unexpected)} -- escape literal braces as {{{{ }}}} if not intentional"
            )
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return text
