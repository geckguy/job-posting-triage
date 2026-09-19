"""Standard approach: one chat completion per posting, all questions at once.

Points at any OpenAI-compatible endpoint. The default is a local llama-server
running Qwen2.5-1.5B-Instruct Q4_K_M, so the repo runs for anyone with no keys;
``--base-url``/``--model``/``--api-key-env`` run the identical code path against
a hosted model instead.

Structured output is requested with ``response_format.json_schema`` and the
reply is validated with ``jsonschema``. A reply that does not validate is
retried once with the error appended, and is otherwise recorded with
``invalid=true`` and counted in the reported ``invalid_json_rate`` -- never
silently dropped.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import httpx
import jsonschema

from triage.arms.base import Answer, Arm, Record
from triage.schema import Question

SYSTEM_PREAMBLE = "You label job postings."
FOOTER = "Answer from the posting only."
MAX_SCHEMA_RETRIES = 1
MAX_TRANSPORT_ATTEMPTS = 3
#: Output budget per completion. A recipe for empty replies: a reasoning model
#: spends its output budget on reasoning first, so a cap sized for the JSON alone
#: (about 150 tokens) leaves `content` empty and every row invalid. The budget has
#: to cover reasoning plus the answer, and it is only a ceiling, so a model that
#: finishes early is not made slower or more expensive by it. Override with the
#: runner's --max-tokens.
MAX_TOKENS = 4096

CONFIDENCE_KEY = "confidence"

#: How the reply is constrained. ``json_schema`` is the planned default and the
#: only mode where the server guarantees the shape. ``prompt`` sends no
#: ``response_format`` at all and relies on the instruction plus jsonschema
#: validation, which is the only mode in which a small local model was measured
#: to actually condition on the posting (see README "constrained decoding").
RESPONSE_FORMATS = ("json_schema", "json_object", "prompt")


def response_schema(questions: Iterable[Question]) -> dict[str, Any]:
    """The exact object the model must return: one enum per question + confidence."""
    questions = tuple(questions)
    properties: dict[str, Any] = {
        question.id: {"type": "string", "enum": list(question.labels)}
        for question in questions
    }
    properties[CONFIDENCE_KEY] = {
        "type": "object",
        "properties": {
            question.id: {"type": "number", "minimum": 0, "maximum": 1}
            for question in questions
        },
        "required": [question.id for question in questions],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [question.id for question in questions] + [CONFIDENCE_KEY],
        "additionalProperties": False,
    }


def system_prompt(questions: Iterable[Question]) -> str:
    lines = [SYSTEM_PREAMBLE]
    for question in questions:
        lines.append(f"{question.id}: {question.instructions}")
    lines.append(
        "Return one label per question, and for each question a confidence in "
        "[0,1] saying how likely your label is right."
    )
    lines.append(FOOTER)
    return "\n".join(lines)


def shape_instruction(questions: Iterable[Question]) -> str:
    """The JSON shape spelled out in words, for the grammar-free mode."""
    questions = tuple(questions)
    fields = ", ".join(
        f'"{q.id}": one of [{", ".join(q.labels)}]' for q in questions
    )
    confidences = ", ".join(f'"{q.id}": a number in [0,1]' for q in questions)
    return (
        "Reply with ONLY a JSON object and nothing else: no prose, no code fences. "
        f"Fields: {fields}, plus \"{CONFIDENCE_KEY}\": {{{confidences}}}."
    )


def _response_format(mode: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    """The ``response_format`` for a mode, or ``None`` to send none at all."""
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "triage", "strict": True, "schema": schema},
        }
    if mode == "json_object":
        return {"type": "json_object"}
    return None


class LlmArm(Arm):
    def __init__(
        self,
        name: str = "llm_local",
        base_url: str = "http://127.0.0.1:8080",
        model: str = "local",
        api_key: str | None = None,
        concurrency: int = 1,
        timeout: float = 600.0,
        response_format: str = "json_schema",
        max_tokens: int = MAX_TOKENS,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if response_format not in RESPONSE_FORMATS:
            raise ValueError(
                f"response_format must be one of {RESPONSE_FORMATS}, got {response_format!r}"
            )
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.concurrency = max(1, concurrency)
        self.response_format = response_format
        self.max_tokens = max_tokens
        self.extra_headers = dict(extra_headers or {})
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def _headers(self) -> dict[str, str]:
        # The key is never logged or written into any artifact.
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def available(self) -> tuple[bool, str]:
        try:
            response = self._client.get(
                f"{self.base_url}/models", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            return False, f"{self.base_url} unreachable: {type(exc).__name__}: {exc}"
        if response.status_code != 200:
            return False, f"{self.base_url}/models returned HTTP {response.status_code}"
        return True, ""

    # -- one record -------------------------------------------------------

    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        response_format = _response_format(self.response_format, schema)
        if response_format is not None:
            payload["response_format"] = response_format
        last_error = ""
        for attempt in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
            try:
                response = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2.0 * attempt)
                continue
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            time.sleep(2.0 * attempt)
        raise RuntimeError(last_error)

    def _messages(self, record: Record, questions: tuple[Question, ...]) -> list[dict[str, str]]:
        """The system and user turns, with the JSON shape spelled out when the
        server is not enforcing it."""
        system = system_prompt(questions)
        if self.response_format != "json_schema":
            system = f"{system}\n{shape_instruction(questions)}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": record["state"]},
        ]

    def _one(
        self, record: Record, questions: tuple[Question, ...], schema: dict[str, Any]
    ) -> dict[str, Answer]:
        messages = self._messages(record, questions)
        started = time.perf_counter()
        invalid = False
        tokens_in = tokens_out = None
        content = ""
        finish_reason = None
        try:
            body = self._complete(messages, schema)
            usage = body.get("usage") or {}
            tokens_in = usage.get("prompt_tokens")
            tokens_out = usage.get("completion_tokens")
            choice = body["choices"][0]
            finish_reason = choice.get("finish_reason")
            content = (choice["message"].get("content") or "").strip()
            parsed = json.loads(content)
        except (RuntimeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            # An empty reply with finish_reason "length" means the output budget
            # ran out before any answer was emitted, which is what a reasoning
            # model does when the cap only covers the JSON.
            note = str(exc)
            if finish_reason == "length":
                note = (
                    f"output budget of {self.max_tokens} tokens exhausted before an "
                    f"answer (finish_reason=length); raise --max-tokens"
                )
            return {
                question.id: Answer(
                    value="",
                    probabilities=None,
                    confidence=None,
                    invalid=True,
                    raw=content or str(exc),
                    meta={
                        "error": note[:200],
                        "finish_reason": finish_reason,
                        "max_tokens": self.max_tokens,
                        "model": self.model,
                    },
                )
                for question in questions
            }

        error = _validate(parsed, schema)
        if error is not None:
            for _ in range(MAX_SCHEMA_RETRIES):
                messages = messages + [
                    {"role": "assistant", "content": content or "{}"},
                    {
                        "role": "user",
                        "content": f"That reply was invalid: {error}. Reply again with only valid JSON.",
                    },
                ]
                try:
                    body = self._complete(messages, schema)
                    usage = body.get("usage") or {}
                    tokens_in = (tokens_in or 0) + int(usage.get("prompt_tokens") or 0)
                    tokens_out = (tokens_out or 0) + int(usage.get("completion_tokens") or 0)
                    content = (body["choices"][0]["message"].get("content") or "").strip()
                    parsed = json.loads(content)
                except (RuntimeError, KeyError, IndexError, json.JSONDecodeError) as exc:
                    error = str(exc)[:200]
                    continue
                error = _validate(parsed, schema)
                if error is None:
                    break
            if error is not None:
                invalid = True

        elapsed_ms = (time.perf_counter() - started) * 1000
        confidence_block = parsed.get(CONFIDENCE_KEY) if isinstance(parsed, dict) else None
        out: dict[str, Answer] = {}
        for question in questions:
            value = parsed.get(question.id) if isinstance(parsed, dict) else None
            in_labels = isinstance(value, str) and value in question.labels
            confidence = None
            if isinstance(confidence_block, dict):
                raw_confidence = confidence_block.get(question.id)
                if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
                    confidence = float(raw_confidence)
            out[question.id] = Answer(
                value=value if in_labels else "",
                probabilities=None,
                confidence=confidence,
                invalid=invalid or not in_labels,
                raw=parsed if isinstance(parsed, dict) else content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                meta={
                    "latency_ms": elapsed_ms,
                    "self_reported_confidence": confidence,
                    "schema_error": error,
                    "finish_reason": finish_reason,
                    "max_tokens": self.max_tokens,
                    "model": self.model,
                },
            )
        return out

    # -- Arm --------------------------------------------------------------

    def run(
        self,
        records: list[Record],
        questions: Iterable[Question],
        targets: dict[str, set[str]] | None = None,
    ) -> dict[str, dict[str, Answer]]:
        questions = tuple(questions)
        schema = response_schema(questions)
        out: dict[str, dict[str, Answer]] = {question.id: {} for question in questions}

        def work(record: Record) -> tuple[str, dict[str, Answer]]:
            return record["record_id"], self._one(record, questions, schema)

        if self.concurrency == 1:
            results = (work(record) for record in records)
            for record_id, answers in results:
                for question_id, answer in answers.items():
                    out[question_id][record_id] = answer
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                for record_id, answers in pool.map(work, records):
                    for question_id, answer in answers.items():
                        out[question_id][record_id] = answer
        return out

    def close(self) -> None:
        self._client.close()


def _validate(parsed: Any, schema: dict[str, Any]) -> str | None:
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as exc:
        return f"{exc.message} at {list(exc.absolute_path)}"
    return None
