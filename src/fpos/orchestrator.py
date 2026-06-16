from __future__ import annotations

import hashlib
from typing import Any, Awaitable, Callable, Optional

from .agents import Architect_Agent, PM_Agent, UX_Agent, genai
from .models import (
    ATOMIC_UNCERTAINTY_THRESHOLD,
    MAX_CHILDREN_PER_NODE,
    MAX_DEPTH,
    MAX_ITERATIONS_PER_LEVEL,
    ArchitectAgentOutput,
    ConvergenceState,
    EvidenceItem,
    PMAgentOutput,
    ProblemState,
    Solution,
    SubProblem,
    Task,
    UXAgentOutput,
)


LogFn = Callable[..., Awaitable[None]]


class FPOSOrchestrator:
    def __init__(
        self,
        api_key: str,
        model: str,
        emit_log: Optional[LogFn] = None,
        project_id: Optional[str] = None,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.emit_log = emit_log
        self.project_id = project_id

        self.pm_agent = PM_Agent(self.client, model=model)
        self.architect_agent = Architect_Agent(self.client, model=model)
        self.ux_agent = UX_Agent(self.client, model=model)

    async def _log(
        self,
        *,
        depth: int,
        iteration: int,
        delta_i: float,
        delta_v: float,
        message: str,
        agent: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if self.emit_log:
            await self.emit_log(
                depth=depth,
                iteration=iteration,
                delta_i=delta_i,
                delta_v=delta_v,
                message=message,
                agent=agent,
                data=data or {},
            )

    async def run(
        self,
        raw_idea: str,
        context: str,
        constraints: list[str],
        stakeholders: list[str],
    ) -> Solution:
        state = ProblemState(
            raw_idea=raw_idea,
            context=context,
            constraints=constraints,
            stakeholders=stakeholders,
            assumptions=[],
            evidence=[],
            scope_path=[],
            uncertainty=1.0,
            iteration=0,
        )
        return await self._recursive_design_loop(state, depth=0)

    async def _recursive_design_loop(self, state: ProblemState, depth: int) -> Solution:
        if depth >= MAX_DEPTH:
            return self._force_atomic_solution(state, reason="max_depth_reached")

        best_solution: Optional[Solution] = None
        convergence = ConvergenceState(iteration=0, delta_i=state.uncertainty, delta_v=1.0, stable=False)

        for _ in range(MAX_ITERATIONS_PER_LEVEL):
            state.iteration += 1
            convergence.iteration = state.iteration

            await self._log(
                depth=depth,
                iteration=state.iteration,
                delta_i=state.uncertainty,
                delta_v=convergence.delta_v,
                message="Starting recursive reasoning pass",
                agent="orchestrator",
                data={"scope_path": state.scope_path},
            )

            pm_task = self.pm_agent.analyze(state)
            arch_task = self.architect_agent.analyze(state)
            ux_task = self.ux_agent.analyze(state)

            pm, arch, ux = await self._gather_agent_outputs(pm_task, arch_task, ux_task)

            subproblems = self._merge_subproblems([*pm.subproblems, *arch.subproblems, *ux.subproblems])[:MAX_CHILDREN_PER_NODE]

            await self._log(
                depth=depth,
                iteration=state.iteration,
                delta_i=state.uncertainty,
                delta_v=convergence.delta_v,
                message="Agent outputs merged",
                agent="orchestrator",
                data={"subproblems": [sp.model_dump() for sp in subproblems]},
            )

            state.evidence = self._update_evidence(state.evidence, self._evidence_from_outputs(pm, arch, ux))
            state.assumptions = self._update_assumptions(state.assumptions, [*pm.assumptions, *arch.technical_risks, *ux.usability_risks])
            state.uncertainty = self._estimate_uncertainty(state, pm, arch, ux)

            convergence.delta_i = state.uncertainty
            convergence.delta_v = self._estimate_value_delta(best_solution, len(subproblems))
            convergence.stable = convergence.delta_i <= convergence.threshold_i and convergence.delta_v <= convergence.threshold_v

            await self._log(
                depth=depth,
                iteration=state.iteration,
                delta_i=convergence.delta_i,
                delta_v=convergence.delta_v,
                message="Convergence state updated",
                agent="orchestrator",
            )

            if self._is_atomic(state, subproblems):
                tasks = self._atomicize(state, subproblems)
                solution = self._compose_solution(state, pm, arch, ux, tasks, [], convergence, depth)
                return solution

            child_solutions = []
            for sp in subproblems:
                child_state = state.spawn_child(sp)
                child_solution = await self._recursive_design_loop(child_state, depth + 1)
                child_solutions.append(child_solution)

            tasks = self._flatten_tasks(child_solutions)
            current_solution = self._compose_solution(
                state, pm, arch, ux, tasks, child_solutions, convergence, depth
            )

            if best_solution is None or self._is_better(current_solution, best_solution):
                best_solution = current_solution

            if convergence.stable:
                current_solution.convergence = convergence
                return current_solution

            state = self._refine_state(state, pm, arch, ux, current_solution)

        if best_solution is not None:
            return best_solution

        return self._force_atomic_solution(state, reason="no_solution_found")

    async def _gather_agent_outputs(
        self,
        pm_task,
        arch_task,
        ux_task,
    ) -> tuple[PMAgentOutput, ArchitectAgentOutput, UXAgentOutput]:
        pm, arch, ux = await pm_task, await arch_task, await ux_task
        return pm, arch, ux

    def _compose_solution(
        self,
        state: ProblemState,
        pm: PMAgentOutput,
        arch: ArchitectAgentOutput,
        ux: UXAgentOutput,
        tasks: list[Task],
        child_solutions: list[Solution],
        convergence: ConvergenceState,
        depth: int,
    ) -> Solution:
        architecture = {
            "summary": arch.architecture_summary,
            "components": arch.components,
            "dependencies": arch.dependencies,
            "implementation_paths": arch.implementation_paths,
            "ux_summary": ux.ux_summary,
            "pm_framing": pm.problem_framing,
            "depth": depth,
        }

        open_questions = list(dict.fromkeys([*pm.value_hypotheses, *pm.risks, *arch.technical_risks, *ux.usability_risks]))
        assumptions = list(dict.fromkeys([*state.assumptions, *pm.assumptions]))
        risks = list(dict.fromkeys([*pm.risks, *arch.technical_risks, *ux.usability_risks]))
        confidence = self._aggregate_confidence(pm, arch, ux, child_solutions)
        value_score = self._score_value(pm, arch, ux, tasks, child_solutions)

        return Solution(
            product_summary=self._summarize_product(state, pm, arch, ux),
            tasks=tasks,
            roadmap=self._sort_tasks(tasks),
            architecture=architecture,
            open_questions=open_questions,
            assumptions=assumptions,
            risks=risks,
            confidence=confidence,
            value_score=value_score,
            convergence=convergence,
        )

    def _summarize_product(self, state: ProblemState, pm: PMAgentOutput, arch: ArchitectAgentOutput, ux: UXAgentOutput) -> str:
        return f"FPOS plan for: {state.raw_idea}\nPM: {pm.problem_framing}\nArch: {arch.architecture_summary}\nUX: {ux.ux_summary}"

    def _score_value(
        self,
        pm: PMAgentOutput,
        arch: ArchitectAgentOutput,
        ux: UXAgentOutput,
        tasks: list[Task],
        child_solutions: list[Solution],
    ) -> float:
        coverage = min(1.0, len(tasks) / 10.0)
        confidence = (pm.confidence + arch.confidence + ux.confidence) / 3.0
        child_bonus = min(1.0, len(child_solutions) / 6.0)
        risk_penalty = min(1.0, (len(pm.risks) + len(arch.technical_risks) + len(ux.usability_risks)) / 12.0)
        simplicity = max(0.0, 1.0 - min(1.0, len(tasks) / 18.0))

        score = (
            35.0 * coverage
            + 25.0 * confidence
            + 20.0 * simplicity
            + 10.0 * child_bonus
            + 10.0 * (1.0 - risk_penalty)
        )
        return round(max(0.0, min(100.0, score)), 2)

    def _aggregate_confidence(
        self,
        pm: PMAgentOutput,
        arch: ArchitectAgentOutput,
        ux: UXAgentOutput,
        child_solutions: list[Solution],
    ) -> float:
        base = (pm.confidence + arch.confidence + ux.confidence) / 3.0
        child_conf = sum((c.confidence for c in child_solutions), 0.0) / len(child_solutions) if child_solutions else 0.0
        return round(max(0.0, min(1.0, base * 0.75 + child_conf * 0.25)), 3)

    def _is_atomic(self, state: ProblemState, subproblems: list[SubProblem]) -> bool:
        return (
            state.uncertainty <= ATOMIC_UNCERTAINTY_THRESHOLD
            and (not subproblems or all(sp.atomic for sp in subproblems))
            and self._has_clear_acceptance_criteria(state)
        )

    def _has_clear_acceptance_criteria(self, state: ProblemState) -> bool:
        return bool(state.constraints) or state.uncertainty <= 0.25

    def _atomicize(
        self,
        state: ProblemState,
        subproblems: list[SubProblem],
    ) -> list[Task]:
        source = subproblems or [
            SubProblem(
                id=self._stable_id(state.raw_idea + ":atomic"),
                title="Atomic implementation slice",
                description=state.raw_idea,
                kind="atomic",
                atomic=True,
                dependencies=list(state.constraints),
                assumptions=list(state.assumptions),
                acceptance_criteria=["Task can be assigned, coded, and tested independently."],
                risks=[],
                estimated_uncertainty=state.uncertainty,
            )
        ]
        return [self._subproblem_to_task(sp, state) for sp in source]

    def _force_atomic_solution(self, state: ProblemState, reason: str) -> Solution:
        task = Task(
            id=self._stable_id(state.raw_idea + reason),
            title="Atomic discovery task",
            description=f"Resolve the remaining unknowns for: {state.raw_idea}",
            owner="PM",
            category="discovery",
            atomic=True,
            inputs=[state.raw_idea, state.context],
            outputs=["validated scope", "confirmed assumptions", "clear next-step backlog"],
            dependencies=list(state.constraints),
            acceptance_criteria=["Unknowns are reduced enough to proceed."],
            definition_of_ready=["Problem statement is clear", "Stakeholders are identified", "Constraints are listed"],
            risk_level="medium",
            estimated_effort_points=2,
        )
        return Solution(
            product_summary=f"Forced atomic fallback due to {reason}.",
            tasks=[task],
            roadmap=[task],
            architecture={"fallback": True, "reason": reason},
            open_questions=["Remaining uncertainty exceeds safe recursive depth."],
            assumptions=list(state.assumptions),
            risks=["Fallback used because recursive convergence was not reached."],
            confidence=0.25,
            value_score=10.0,
            convergence=ConvergenceState(iteration=state.iteration, delta_i=state.uncertainty, delta_v=1.0, stable=False),
        )

    def _subproblem_to_task(self, sp: SubProblem, state: ProblemState) -> Task:
        owner, category = self._infer_owner_and_category(sp)
        return Task(
            id=sp.id or self._stable_id(sp.title + sp.description),
            title=sp.title,
            description=sp.description,
            owner=owner,
            category=category,
            atomic=sp.atomic,
            inputs=self._task_inputs(state, sp),
            outputs=self._task_outputs(sp),
            dependencies=list(dict.fromkeys([*state.constraints, *sp.dependencies])),
            acceptance_criteria=sp.acceptance_criteria or ["Meets Definition of Ready and is testable."],
            definition_of_ready=self._definition_of_ready(sp),
            risk_level=self._risk_level(sp),
            estimated_effort_points=self._estimate_effort(sp),
        )

    def _definition_of_ready(self, sp: SubProblem) -> list[str]:
        return list(dict.fromkeys([
            "Acceptance criteria are explicit",
            "Dependencies are identified",
            "Owner is clear",
            "Scope can be completed in one slice",
            *sp.acceptance_criteria[:2],
        ]))

    def _task_inputs(self, state: ProblemState, sp: SubProblem) -> list[str]:
        return list(dict.fromkeys([state.raw_idea, state.context, *state.constraints, *sp.dependencies]))

    def _task_outputs(self, sp: SubProblem) -> list[str]:
        return list(dict.fromkeys([sp.title, sp.description, *sp.acceptance_criteria]))

    def _infer_owner_and_category(self, sp: SubProblem) -> tuple[str, str]:
        mapping = {
            "discovery": ("PM", "discovery"),
            "strategy": ("PM", "discovery"),
            "architecture": ("Architect", "architecture"),
            "ux": ("UX", "design"),
            "frontend": ("Frontend", "implementation"),
            "backend": ("Backend", "implementation"),
            "data": ("Data", "implementation"),
            "qa": ("QA", "test"),
            "security": ("Security", "test"),
            "devops": ("DevOps", "release"),
            "launch": ("PM", "release"),
            "ops": ("DevOps", "ops"),
            "atomic": ("Unknown", "implementation"),
        }
        return mapping.get(sp.kind, ("Unknown", "implementation"))

    def _risk_level(self, sp: SubProblem) -> str:
        score = len(sp.risks) + int(sp.estimated_uncertainty * 10)
        if score <= 3:
            return "low"
        if score <= 7:
            return "medium"
        return "high"

    def _estimate_effort(self, sp: SubProblem) -> int:
        base = 1 + len(sp.dependencies) + len(sp.acceptance_criteria) // 2
        return int(max(1, min(13, base)))

    def _sort_tasks(self, tasks: list[Task]) -> list[Task]:
        order = {"discovery": 0, "design": 1, "architecture": 2, "implementation": 3, "test": 4, "release": 5, "ops": 6}
        return sorted(tasks, key=lambda t: (order.get(t.category, 99), t.estimated_effort_points, t.title.lower()))

    def _merge_subproblems(self, subproblems: list[SubProblem]) -> list[SubProblem]:
        merged: dict[str, SubProblem] = {}
        for sp in subproblems:
            key = self._normalize_key(f"{sp.kind}:{sp.title}")
            if key not in merged:
                merged[key] = sp
            else:
                existing = merged[key]
                merged[key] = existing.model_copy(
                    update={
                        "dependencies": list(dict.fromkeys([*existing.dependencies, *sp.dependencies])),
                        "assumptions": list(dict.fromkeys([*existing.assumptions, *sp.assumptions])),
                        "acceptance_criteria": list(dict.fromkeys([*existing.acceptance_criteria, *sp.acceptance_criteria])),
                        "risks": list(dict.fromkeys([*existing.risks, *sp.risks])),
                        "estimated_uncertainty": max(existing.estimated_uncertainty, sp.estimated_uncertainty),
                        "atomic": existing.atomic and sp.atomic,
                    }
                )
        return list(merged.values())

    def _flatten_tasks(self, child_solutions: list[Solution]) -> list[Task]:
        tasks: list[Task] = []
        for sol in child_solutions:
            tasks.extend(sol.tasks)
        return self._sort_tasks(tasks)

    def _evidence_from_outputs(
        self,
        pm: PMAgentOutput,
        arch: ArchitectAgentOutput,
        ux: UXAgentOutput,
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for metric in pm.success_metrics:
            items.append(EvidenceItem(source="PM_Agent", signal=metric, strength=0.55, reliability=0.70))
        for risk in arch.technical_risks:
            items.append(EvidenceItem(source="Architect_Agent", signal=f"technical_risk:{risk}", strength=0.50, reliability=0.75))
        for req in ux.accessibility_requirements:
            items.append(EvidenceItem(source="UX_Agent", signal=f"ux_requirement:{req}", strength=0.45, reliability=0.68))
        return items

    def _update_evidence(self, old: list[EvidenceItem], new: list[EvidenceItem]) -> list[EvidenceItem]:
        merged = list(old)
        for item in new:
            norm = self._normalize_key(item.signal)
            if not any(self._normalize_key(e.signal) == norm for e in merged):
                merged.append(item)
        return merged

    def _update_assumptions(self, old_assumptions: list[str], new_signals: list[str]) -> list[str]:
        return list(dict.fromkeys([*old_assumptions, *[s.strip() for s in new_signals if s.strip()]]))

    def _estimate_uncertainty(self, state: ProblemState, pm: PMAgentOutput, arch: ArchitectAgentOutput, ux: UXAgentOutput) -> float:
        confidences = [pm.confidence, arch.confidence, ux.confidence]
        avg_conf = sum(confidences) / len(confidences)
        unresolved_assumptions = max(0, len(state.assumptions) - len(state.evidence))
        open_subproblems = len(pm.subproblems) + len(arch.subproblems) + len(ux.subproblems)
        raw = (1.0 - avg_conf) * 0.6 + min(1.0, unresolved_assumptions / 10.0) * 0.2 + min(1.0, open_subproblems / 12.0) * 0.2
        return max(0.0, min(1.0, raw))

    def _refine_state(self, state: ProblemState, pm: PMAgentOutput, arch: ArchitectAgentOutput, ux: UXAgentOutput, solution: Solution) -> ProblemState:
        next_uncertainty = min(self._estimate_uncertainty(state, pm, arch, ux), solution.convergence.delta_i, state.uncertainty)
        return state.model_copy(
            update={
                "assumptions": list(dict.fromkeys([*state.assumptions, *pm.assumptions])),
                "uncertainty": max(0.0, min(1.0, next_uncertainty)),
                "iteration": state.iteration,
            }
        )

    def _is_better(self, candidate: Solution, current: Solution) -> bool:
        if candidate.value_score != current.value_score:
            return candidate.value_score > current.value_score
        if candidate.confidence != current.confidence:
            return candidate.confidence > current.confidence
        return len(candidate.tasks) < len(current.tasks)

    def _stable_id(self, text: str) -> str:
        return "fpos_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def _normalize_key(self, text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())
