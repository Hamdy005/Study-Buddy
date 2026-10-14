"""
LLM protocol and implementations for CRAG inside src/rag.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar, runtime_checkable
from loguru import logger
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLM(Protocol):
    """Protocol defining the LLM interface for text generation and structured outputs."""

    async def ainvoke(self, prompt: str) -> str:
        """Generate unstructured text from a prompt."""
        ...

    async def ainvoke_structured(self, prompt: str, schema: type[T]) -> T:
        """Generate structured output validated against a Pydantic schema."""
        ...


class LangChainLLM:
    """
    LLM wrapper delegating to a LangChain ChatModel with automatic retry on ValidationError.
    """

    def __init__(self, chat_model: Any = None, model_name: str | None = None):
        self._chat_model = chat_model
        self.model_name = (
            model_name
            or getattr(chat_model, "model_name", None)
            or getattr(chat_model, "model", None)
            or "ministral-8b-latest"
        )
        self.last_usage: dict[str, int] | None = None

    def _extract_usage(self, res: Any) -> None:
        usage = getattr(res, "usage_metadata", None)
        if not usage and hasattr(res, "response_metadata") and isinstance(res.response_metadata, dict):
            usage = res.response_metadata.get("token_usage") or res.response_metadata.get("usage")
        if isinstance(usage, dict):
            self.last_usage = {
                "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or 0,
                "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens") or 0,
            }

    def _get_chat_model(self) -> Any:
        if self._chat_model is not None:
            return self._chat_model
        try:
            from src.rag.rag import get_llm
            self._chat_model = get_llm()
            if hasattr(self._chat_model, "model_name"):
                self.model_name = self._chat_model.model_name
            elif hasattr(self._chat_model, "model"):
                self.model_name = self._chat_model.model
            return self._chat_model
        except Exception:
            raise RuntimeError("No chat model provided and could not import default from src.rag.rag")

    async def ainvoke(self, prompt: str) -> str:
        model = self._get_chat_model()
        if hasattr(model, "ainvoke"):
            res = await model.ainvoke(prompt)
        elif hasattr(model, "invoke"):
            res = model.invoke(prompt)
        else:
            raise TypeError("Model must support ainvoke or invoke")
        self._extract_usage(res)
        if hasattr(res, "content"):
            return str(res.content)
        return str(res)

    async def astream(self, prompt: str):
        model = self._get_chat_model()
        if hasattr(model, "astream"):
            async for chunk in model.astream(prompt):
                self._extract_usage(chunk)
                content = getattr(chunk, "content", "")
                if content:
                    yield str(content)
        else:
            ans = await self.ainvoke(prompt)
            yield ans

    async def ainvoke_structured(self, prompt: str, schema: type[T]) -> T:
        """
        Generate structured output without multi-attempt retry loops.
        Uses key-value line parsing for EvaluatorOutput (Option 2) and with_structured_output
        with single-pass JSON fallback for RouterOutput.
        """
        # Fast path for EvaluatorOutput
        if hasattr(schema, "__name__") and schema.__name__ == "EvaluatorOutput":
            try:
                raw_text = await self.ainvoke(prompt)
                parsed_lines = self._parse_evaluator_lines(raw_text)
                if parsed_lines is not None:
                    return schema.model_validate(parsed_lines)
                # Fallback to JSON if the model formatted as JSON
                parsed_json = self._extract_json(raw_text)
                return schema.model_validate(parsed_json)
            except Exception as exc:
                logger.warning(f"Evaluator structured extraction failed ({exc})")
                raise exc

        # Fast path for TriageOutput using line parsing (like EvaluatorOutput)
        if hasattr(schema, "__name__") and schema.__name__ == "TriageOutput":
            try:
                raw_text = await self.ainvoke(prompt)
                parsed_lines = self._parse_triage_lines(raw_text)
                if parsed_lines is not None:
                    return schema.model_validate(parsed_lines)
                # Fallback to JSON if the model formatted as JSON
                parsed_json = self._extract_json(raw_text)
                return schema.model_validate(parsed_json)
            except Exception as exc:
                logger.warning(f"Triage structured extraction failed ({exc})")
                raise exc

        # Fast path for RouterOutput using line parsing (like EvaluatorOutput and TriageOutput)
        if hasattr(schema, "__name__") and schema.__name__ == "RouterOutput":
            try:
                raw_text = await self.ainvoke(prompt)
                parsed_lines = self._parse_router_lines(raw_text)
                if parsed_lines is not None:
                    return schema.model_validate(parsed_lines)
                # Fallback to JSON if the model formatted as JSON
                parsed_json = self._extract_json(raw_text)
                if isinstance(parsed_json, dict):
                    if "properties" in parsed_json and isinstance(parsed_json["properties"], dict):
                        parsed_json = parsed_json["properties"]
                    if not parsed_json.get("sources"):
                        parsed_json["sources"] = ["supabase", "arxiv", "wikipedia", "web"]
                    if "search_query" in parsed_json and isinstance(parsed_json["search_query"], str):
                        parsed_json["search_query"] = parsed_json["search_query"][:300].rstrip()
                return schema.model_validate(parsed_json)
            except Exception as exc:
                logger.warning(f"Router structured extraction failed ({exc})")
                raise exc

        model = self._get_chat_model()

        # Try with_structured_output if supported
        if hasattr(model, "with_structured_output"):
            try:
                runnable = model.with_structured_output(schema, include_raw=True)
                res_dict = await runnable.ainvoke(prompt) if hasattr(runnable, "ainvoke") else runnable.invoke(prompt)
                if isinstance(res_dict, dict):
                    raw = res_dict.get("raw")
                    if raw:
                        self._extract_usage(raw)
                    parsed_val = res_dict.get("parsed")
                    if parsed_val is not None:
                        return parsed_val
                    # If LangChain parser failed, inspect raw tool_calls args directly
                    if raw and hasattr(raw, "tool_calls") and raw.tool_calls:
                        args = raw.tool_calls[0].get("args", {})
                        if isinstance(args, dict):
                            if hasattr(schema, "__name__") and schema.__name__ == "RouterOutput" and "search_query" in args and isinstance(args["search_query"], str):
                                args["search_query"] = args["search_query"][:300].rstrip()
                            return schema.model_validate(args)
                    parsing_err = res_dict.get("parsing_error")
                    raise ValueError(f"Output parser returned parsed=None (error: {parsing_err})")
            except Exception as exc:
                logger.warning(f"Structured output failed ({exc}). Trying single-pass JSON extraction...")

        # Single-pass fallback: No retry loop
        try:
            current_prompt = (
                f"{prompt}\n\n"
                f"You MUST output strictly a single valid JSON object matching this schema:\n"
                f"{json.dumps(schema.model_json_schema())}\n"
                f"Output ONLY valid JSON."
            )
            raw_text = await self.ainvoke(current_prompt)
            parsed = self._extract_json(raw_text)
            if isinstance(parsed, dict) and hasattr(schema, "__name__") and schema.__name__ == "RouterOutput":
                if "search_query" in parsed and isinstance(parsed["search_query"], str):
                    parsed["search_query"] = parsed["search_query"][:300].rstrip()
            return schema.model_validate(parsed)
        except Exception as err:
            logger.error(f"Structured output failed: {err}")
            raise err

    @staticmethod
    def _parse_evaluator_lines(text: str) -> dict[str, Any] | None:
        """Parse key-value lines for EvaluatorOutput: VERDICT, CAUSE, MISSING."""
        verdict = None
        cause = None
        missing = ""
        in_missing = False
        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cleaned = re.sub(r"[*#`]", "", line).strip()
            lower = cleaned.lower()
            if lower.startswith("verdict:"):
                in_missing = False
                val = cleaned.split(":", 1)[1].strip().lower()
                words = set(re.findall(r"\w+", val))
                if "incorrect" in words:
                    verdict = "incorrect"
                elif "ambiguous" in words:
                    verdict = "ambiguous"
                elif "correct" in words:
                    verdict = "correct"
            elif lower.startswith("cause:") or lower.startswith("likely_cause:"):
                in_missing = False
                val = cleaned.split(":", 1)[1].strip().lower()
                words = set(re.findall(r"\w+", val))
                if "not_in_source" in val or ("not" in words and "source" in words):
                    cause = "not_in_source"
                elif "bad_query" in val or ("bad" in words and "query" in words):
                    cause = "bad_query"
                elif "none" in words:
                    cause = "none"
            elif lower.startswith("missing:") or lower.startswith("missing_field:"):
                missing = cleaned.split(":", 1)[1].strip()
                in_missing = True
            elif in_missing:
                if not (lower.startswith("verdict:") or lower.startswith("cause:") or lower.startswith("likely_cause:")):
                    missing += " " + cleaned

        if verdict:
            if verdict == "correct":
                cause = "none"
                missing = ""
            else:
                cause = "bad_query"
                if not missing or missing.lower() in ("none", "n/a", "null", "unknown", "nothing"):
                    missing = "additional context needed"
            return {"verdict": verdict, "likely_cause": cause, "missing": missing.strip()}
        return None

    @staticmethod
    def _parse_triage_lines(text: str) -> dict[str, Any] | None:
        """Parse key-value lines for TriageOutput: INTENT, UNSAFE_TYPE, SUB_QUESTIONS."""
        intent = None
        unsafe_type = None
        sub_questions: list[str] = []
        in_sub_questions = False

        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cleaned = re.sub(r"[*#`]", "", line).strip()
            lower = cleaned.lower()

            if lower.startswith("intent:"):
                in_sub_questions = False
                val = cleaned.split(":", 1)[1].strip().lower()
                words = set(re.findall(r"\w+", val))
                if "chitchat" in words:
                    intent = "chitchat"
                elif "unsafe" in words:
                    intent = "unsafe"
                elif "complex" in words:
                    intent = "complex"
                elif "simple" in words:
                    intent = "simple"

            elif lower.startswith("unsafe_type:") or lower.startswith("unsafetype:") or lower.startswith("unsafe:"):
                in_sub_questions = False
                val = cleaned.split(":", 1)[1].strip().lower()
                words = set(re.findall(r"\w+", val))
                if "gibberish" in words:
                    unsafe_type = "gibberish"
                elif "nsfw" in words:
                    unsafe_type = "nsfw"
                elif "political" in words:
                    unsafe_type = "political"
                elif "religious" in words:
                    unsafe_type = "religious"
                elif "none" in words:
                    unsafe_type = "none"

            elif lower.startswith("sub_questions:") or lower.startswith("sub_queries:") or lower.startswith("subquestions:"):
                in_sub_questions = True
                val = cleaned.split(":", 1)[1].strip()
                val_lower = val.lower()
                if val_lower in ("none", "n/a", "null", "[]", "", "<none>"):
                    pass
                else:
                    if ";" in val:
                        for q in val.split(";"):
                            q_clean = q.strip().lstrip("-*123456789. ")
                            if q_clean and q_clean.lower() not in ("none", "n/a", "null"):
                                sub_questions.append(q_clean)
                    else:
                        q_clean = val.lstrip("-*123456789. ").strip()
                        if q_clean and q_clean.lower() not in ("none", "n/a", "null"):
                            sub_questions.append(q_clean)

            elif in_sub_questions:
                if any(lower.startswith(prefix) for prefix in ("intent:", "unsafe_type:", "unsafetype:", "unsafe:")):
                    in_sub_questions = False
                else:
                    q_clean = cleaned.lstrip("-*123456789. ").strip()
                    if q_clean and q_clean.lower() not in ("none", "n/a", "null"):
                        sub_questions.append(q_clean)

        if intent:
            if intent == "unsafe":
                if not unsafe_type or unsafe_type == "none":
                    unsafe_type = "gibberish"
                sub_questions = []
            else:
                unsafe_type = "none"

            if intent == "complex":
                valid_subs = [q[:200] for q in sub_questions if len(q.strip()) >= 3]
                sub_questions = valid_subs[:3]
                if not sub_questions:
                    sub_questions = ["complex study query"]
            else:
                sub_questions = []

            return {
                "intent": intent,
                "unsafe_type": unsafe_type,
                "sub_questions": sub_questions,
            }
        return None

    @staticmethod
    def _parse_router_lines(text: str) -> dict[str, Any] | None:
        """Parse key-value lines for RouterOutput: SEARCH_QUERY, SOURCES."""
        search_query = None
        sources = None
        in_query = False
        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cleaned = re.sub(r"[*#`]", "", line).strip()
            lower = cleaned.lower()
            if lower.startswith("search_query:") or lower.startswith("searchquery:") or lower.startswith("query:"):
                in_query = True
                search_query = cleaned.split(":", 1)[1].strip()
            elif lower.startswith("sources:") or lower.startswith("source:"):
                in_query = False
                val = cleaned.split(":", 1)[1].strip().lower()
                raw_srcs = re.findall(r"\b(supabase|wikipedia|arxiv|web)\b", val)
                if raw_srcs:
                    sources = list(dict.fromkeys(raw_srcs))
            elif in_query:
                if any(lower.startswith(prefix) for prefix in ("sources:", "source:", "note:", "explanation:")):
                    in_query = False
                else:
                    search_query += " " + cleaned

        if not search_query and text.strip():
            stripped = text.strip()
            if not (stripped.startswith("{") or stripped.startswith("[")):
                for raw_line in stripped.splitlines():
                    cleaned_line = re.sub(r"[*#`]", "", raw_line).strip()
                    if cleaned_line and not cleaned_line.startswith("<") and len(cleaned_line) >= 2:
                        lower_line = cleaned_line.lower()
                        if not any(lower_line.startswith(p) for p in ("here is", "sure", "response:", "output:", "note:")):
                            search_query = cleaned_line
                            break

        if search_query:
            search_query = search_query.strip("\"' ")
            if len(search_query) > 300:
                search_query = search_query[:300].rstrip()
            if len(search_query) >= 2:
                res: dict[str, Any] = {"search_query": search_query}
                res["sources"] = sources if sources else ["supabase", "arxiv", "wikipedia", "web"]
                return res
        return None

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start : end + 1])
            else:
                data = json.loads(text)

        # If model echoed the JSON schema definition containing "properties"
        if isinstance(data, dict) and "properties" in data and isinstance(data["properties"], dict) and "title" in data:
            data = data["properties"]

        return data


class MockLLM:
    """Mock LLM for testing."""

    def __init__(
        self,
        text_response: str = "Mock generated answer.",
        structured_responses: list[BaseModel] | None = None,
        fail_times: int = 0,
    ):
        self.text_response = text_response
        self.structured_responses = list(structured_responses) if structured_responses else []
        self.fail_times = fail_times
        self._fails_count = 0
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text_response

    async def astream(self, prompt: str):
        self.prompts.append(prompt)
        yield self.text_response

    async def ainvoke_structured(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        if self._fails_count < self.fail_times:
            self._fails_count += 1
            raise ValidationError.from_exception_data(
                title=schema.__name__,
                line_errors=[{"type": "value_error", "loc": ("mock",), "input": {}, "ctx": {"error": "Simulated ValidationError"}}],
            )
        for idx, resp in enumerate(self.structured_responses):
            if isinstance(resp, schema):
                return self.structured_responses.pop(idx)
            if isinstance(resp, dict):
                try:
                    val = schema.model_validate(resp)
                    self.structured_responses.pop(idx)
                    return val
                except Exception:
                    pass

        if hasattr(schema, "__name__") and schema.__name__ == "TriageOutput":
            from src.rag.schemas import TriageOutput
            return TriageOutput(intent="complex", unsafe_type="none", sub_questions=["mock query"])  # type: ignore
        if hasattr(schema, "__name__") and schema.__name__ == "RouterOutput":
            from src.rag.schemas import RouterOutput
            return RouterOutput(sources=["supabase"], search_query="mock search query")  # type: ignore
        if hasattr(schema, "__name__") and schema.__name__ == "EvaluatorOutput":
            from src.rag.schemas import EvaluatorOutput
            return EvaluatorOutput(verdict="correct", missing="", likely_cause="none")  # type: ignore

        raise RuntimeError(f"No mock response configured for schema {schema}")
