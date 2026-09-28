"""
Core extraction engine.

Uses a hosted LLM (any OpenAI-compatible provider -- see _provider_config) to extract structured metadata fields
from raw contract text. Deliberately NOT regex/rule-based: the model
reads the document the way a human paralegal would and reasons about
which spans of text correspond to which field, which is what lets a
single implementation generalize across templates it has never seen.

On top of the base extraction, this module adds two things a naive
"call the LLM once" solution skips:

Self-consistency check -- the same document is extracted twice
(independent sampled calls). Fields where both runs agree are kept
as-is; disagreements are resolved by a tie-breaking third call. This
materially reduces one-off hallucinations without needing any
training data.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI

from .schema import FIELDS, FIELD_DESCRIPTIONS



@dataclass
class FieldResult:
    value: str | None


@dataclass
class ExtractionResult:
    fields: dict[str, FieldResult] = field(default_factory=dict)
    raw_runs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {name: {"value": fr.value} for name, fr in self.fields.items()}

    def flat_values(self) -> dict[str, str | None]:
        """Just the values -- convenient for CSV export and for
        computing the recall metric against ground truth."""
        return {name: fr.value for name, fr in self.fields.items()}


def _build_prompt(document_text: str) -> str:
    field_lines = "\n".join(
        f'- "{name}": {FIELD_DESCRIPTIONS[name]}' for name in FIELDS
    )
    schema_example = ", ".join(f'"{name}": "..."' for name in FIELDS)

    return f"""You are an expert legal document analyst. You will be given the raw
text of a rental/lease agreement. The document may follow ANY layout,
wording style, or template -- do not rely on fixed phrasing or position.
Read it the way a careful human would and reason about meaning.

Extract the following fields:
{field_lines}

Rules:
- If a field is genuinely not present in the document, use null for its value.
- Do not guess values that are not supported by the text.
- Return dates as DD.MM.YYYY exactly as written in the source if possible.
- Return only valid JSON, no explanation, no markdown fences.

Respond with exactly this JSON shape:
{{{schema_example}}}

Document text:
---
{document_text}
---
"""


class LLMError(RuntimeError):
    """Raised with a human-readable explanation of why the LLM call failed."""


def _provider_config() -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for the selected provider.

    Every provider below speaks the OpenAI-compatible chat API, so the same
    code path works for all of them. Pick one with LLM_PROVIDER in .env.
    """
    provider = os.environ.get("LLM_PROVIDER", "nvidia").lower()  # nvidia | gemini | groq | anthropic | openai | openrouter
    presets = {
        "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
        "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"),
        "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "moonshotai/kimi-k2.6"),
        "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-4o-mini"),
    }
    if provider not in presets:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of {sorted(presets)}.")
    base_url, key_env, default_model = presets[provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise LLMError(f"{key_env} is not set. Add it to your .env file.")
    model = os.environ.get("LLM_MODEL") or default_model
    return base_url, api_key, model


def _parse_json(content: str) -> dict:
    """Parse model output as JSON, tolerating markdown fences / extra prose
    by slicing from the first '{' to the last '}' (plain string ops, no regex)."""
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        lo, hi = content.find("{"), content.rfind("}")
        if lo != -1 and hi > lo:
            try:
                return json.loads(content[lo : hi + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Model did not return valid JSON. Got: {content[:200]!r}")


def _call_llm(client, model: str, prompt: str, temperature: float, retries: int = 4) -> dict:
    """One extraction call with retry/backoff and use of JSON mode when supported."""
    messages = [
        {"role": "system", "content": "You extract structured data from legal documents and respond only in JSON."},
        {"role": "user", "content": prompt},
    ]
    use_json_mode = True
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs = dict(model=model, messages=messages, temperature=temperature)
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            return _parse_json(response.choices[0].message.content)
        except LLMError as exc:          # bad JSON -> retry once more
            last_exc = exc
        except Exception as exc:         # noqa: BLE001
            last_exc = exc
            status = getattr(exc, "status_code", None)
            text = str(exc)
            if status in (401, 403):
                raise LLMError(f"Authentication failed ({status}). Check your API key.") from exc
            if status == 404 or "model_not_found" in text or "decommissioned" in text:
                raise LLMError(
                    f"Model '{model}' is not available on this provider (404). "
                    "Set LLM_MODEL in .env to a current model."
                ) from exc
            if status in (400, 422) and use_json_mode:
                use_json_mode = False    # provider rejected JSON mode; retry without it
                continue
            if status not in (429, 500, 502, 503, 504, None):
                raise LLMError(f"LLM request failed ({status}): {text[:300]}") from exc
        time.sleep(min(2 ** attempt * 2, 30))   # 2s, 4s, 8s, 16s
    raise LLMError(f"LLM call failed after {retries} attempts: {last_exc}")


def _normalize(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"null", "none", "n/a", "na"}:
        return None
    return value


def extract_fields(
    document_text: str,
    api_key: str | None = None,
    self_consistency: bool = True,
) -> ExtractionResult:
    """Extract all schema fields from a single document's text.

    When self_consistency=True (default), runs the extraction twice
    at different sampling temperatures. Fields where both runs agree
    are kept. Disagreements trigger one tie-breaking third call; if it
    matches either prior answer that value is kept, otherwise the
    first run's value is kept.
    """
    base_url, key, model = _provider_config()
    client = OpenAI(base_url=base_url, api_key=api_key or key)
    prompt = _build_prompt(document_text)

    run_a = _call_llm(client, model, prompt, temperature=0.0)
    runs = [run_a]

    result = ExtractionResult()

    if not self_consistency:
        for name in FIELDS:
            result.fields[name] = FieldResult(value=_normalize(run_a.get(name)))
        result.raw_runs = runs
        return result

    run_b = _call_llm(client, model, prompt, temperature=0.4)
    runs.append(run_b)

    needs_tiebreak = []
    for name in FIELDS:
        val_a = _normalize(run_a.get(name))
        val_b = _normalize(run_b.get(name))
        if val_a == val_b:
            result.fields[name] = FieldResult(value=val_a)
        else:
            needs_tiebreak.append(name)

    if needs_tiebreak:
        run_c = _call_llm(client, model, prompt, temperature=0.2)
        runs.append(run_c)
        for name in needs_tiebreak:
            val_a = _normalize(run_a.get(name))
            val_b = _normalize(run_b.get(name))
            val_c = _normalize(run_c.get(name))
            chosen = val_c if val_c in (val_a, val_b) else val_a
            result.fields[name] = FieldResult(value=chosen)

    result.raw_runs = runs
    # keep schema order regardless of tie-break ordering
    result.fields = {name: result.fields[name] for name in FIELDS}
    return result
