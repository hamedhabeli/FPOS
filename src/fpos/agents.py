from __future__ import annotations

import asyncio
import json
from typing import Any

try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover - allows local import in environments without the SDK
    class _MissingGenAI:
        class Client:  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ImportError("google-genai is not installed")

    genai = _MissingGenAI()  # type: ignore[assignment]

from pydantic import BaseModel, ValidationError

from .models import ArchitectAgentOutput, PMAgentOutput, ProblemState, UXAgentOutput


class BaseAgent:
    def __init__(self, client: genai.Client, model: str, name: str, system_prompt: str) -> None:
        self.client = client
        self.model = model
        self.name = name
        self.system_prompt = system_prompt

    def _schema(self, response_model: type[BaseModel]) -> dict[str, Any]:
        return response_model.model_json_schema()

    def _make_prompt(self, state: ProblemState) -> str:
        scope = " > ".join(state.scope_path) if state.scope_path else "(root)"
        return f"""
Project idea:
{state.raw_idea}

Context:
{state.context}

Constraints:
{json.dumps(state.constraints, ensure_ascii=False, indent=2)}

Stakeholders:
{json.dumps(state.stakeholders, ensure_ascii=False, indent=2)}

Assumptions:
{json.dumps(state.assumptions, ensure_ascii=False, indent=2)}

Evidence:
{json.dumps([e.model_dump() for e in state.evidence], ensure_ascii=False, indent=2)}

Scope path:
{scope}

Instructions:
- Return valid JSON only.
- Follow the schema exactly.
- Identify subproblems for the current abstraction level.
- Mark a subproblem atomic only if it can become a code-ready task.
- Provide uncertainty_delta as remaining uncertainty, from 0.0 to 1.0.
""".strip()

    def _generate_sync(self, state: ProblemState, response_model: type[BaseModel]) -> BaseModel:
        prompt = self._make_prompt(state)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": self.system_prompt,
                "response_format": {
                    "text": {
                        "mime_type": "application/json",
                        "schema": self._schema(response_model),
                    }
                },
                "temperature": 0.2,
            },
        )

        raw = (response.text or "").strip()
        if not raw:
            raise ValueError(f"{self.name} returned an empty response.")

        try:
            return response_model.model_validate_json(raw)
        except ValidationError:
            repair = self.client.models.generate_content(
                model=self.model,
                contents=f"""
Fix the following JSON so it exactly matches the schema.
Return JSON only.

BROKEN_JSON:
{raw}
""".strip(),
                config={
                    "system_instruction": self.system_prompt,
                    "response_format": {
                        "text": {
                            "mime_type": "application/json",
                            "schema": self._schema(response_model),
                        }
                    },
                    "temperature": 0.0,
                },
            )
            fixed = (repair.text or "").strip()
            return response_model.model_validate_json(fixed)

    async def generate(self, state: ProblemState, response_model: type[BaseModel]) -> BaseModel:
        return await asyncio.to_thread(self._generate_sync, state, response_model)


class PM_Agent(BaseAgent):
    def __init__(self, client: genai.Client, model: str) -> None:
        super().__init__(
            client=client,
            model=model,
            name="PM_Agent",
            system_prompt=(
                "You are a senior product manager. "
                "Focus on problem framing, value hypotheses, user segments, metrics, assumptions, "
                "risks, and decomposition into product subproblems."
            ),
        )

    async def analyze(self, state: ProblemState) -> PMAgentOutput:
        return await self.generate(state, PMAgentOutput)  # type: ignore[return-value]


class Architect_Agent(BaseAgent):
    def __init__(self, client: genai.Client, model: str) -> None:
        super().__init__(
            client=client,
            model=model,
            name="Architect_Agent",
            system_prompt=(
                "You are a software architect. "
                "Focus on technical decomposition, boundaries, dependencies, implementation paths, "
                "and risks. Prefer modularity and low coupling."
            ),
        )

    async def analyze(self, state: ProblemState) -> ArchitectAgentOutput:
        return await self.generate(state, ArchitectAgentOutput)  # type: ignore[return-value]


class UX_Agent(BaseAgent):
    def __init__(self, client: genai.Client, model: str) -> None:
        super().__init__(
            client=client,
            model=model,
            name="UX_Agent",
            system_prompt=(
                "You are a UX strategist and product designer. "
                "Focus on user flows, screens, accessibility, usability risks, and shippable UX slices."
            ),
        )

    async def analyze(self, state: ProblemState) -> UXAgentOutput:
        return await self.generate(state, UXAgentOutput)  # type: ignore[return-value]
