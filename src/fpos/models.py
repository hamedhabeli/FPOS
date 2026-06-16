from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


DEFAULT_MODEL = "gemini-3.5-flash"
MAX_DEPTH = 6
MAX_ITERATIONS_PER_LEVEL = 4
MAX_CHILDREN_PER_NODE = 6
ATOMIC_UNCERTAINTY_THRESHOLD = 0.18
VALUE_STABILITY_THRESHOLD = 0.08


class ProjectStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ProjectCreateRequest(BaseModel):
    idea: str = Field(..., min_length=3)
    context: str = Field(..., min_length=3)
    constraints: List[str] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)
    model: Optional[str] = None


class ProjectCreateResponse(BaseModel):
    project_id: UUID
    status: ProjectStatus
    message: str


class ProjectLog(BaseModel):
    ts: str
    depth: int
    iteration: int
    delta_i: float
    delta_v: float = 1.0
    message: str
    agent: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    source: str
    signal: str
    strength: float = Field(..., ge=0.0, le=1.0)
    reliability: float = Field(..., ge=0.0, le=1.0)
    timestamp: Optional[str] = None


class ProblemState(BaseModel):
    raw_idea: str
    context: str
    constraints: List[str] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    scope_path: List[str] = Field(default_factory=list)
    uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    iteration: int = 0

    def spawn_child(self, subproblem: "SubProblem") -> "ProblemState":
        return ProblemState(
            raw_idea=f"{subproblem.title}: {subproblem.description}",
            context=self.context,
            constraints=list(dict.fromkeys([*self.constraints, *subproblem.dependencies])),
            stakeholders=self.stakeholders,
            assumptions=list(dict.fromkeys([*self.assumptions, *subproblem.assumptions])),
            evidence=list(self.evidence),
            scope_path=[*self.scope_path, subproblem.kind, subproblem.title],
            uncertainty=min(1.0, max(self.uncertainty, subproblem.estimated_uncertainty)),
            iteration=self.iteration,
        )


class SubProblem(BaseModel):
    id: str
    title: str
    description: str
    kind: Literal[
        "discovery",
        "strategy",
        "architecture",
        "ux",
        "frontend",
        "backend",
        "data",
        "qa",
        "security",
        "devops",
        "launch",
        "ops",
        "atomic",
    ]
    atomic: bool = False
    dependencies: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    estimated_uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)


class Task(BaseModel):
    id: str
    title: str
    description: str
    owner: Literal["PM", "Architect", "UX", "Frontend", "Backend", "QA", "DevOps", "Security", "Data", "Unknown"] = "Unknown"
    category: Literal["discovery", "design", "architecture", "implementation", "test", "release", "ops"]
    atomic: bool = True
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    definition_of_ready: List[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    estimated_effort_points: int = Field(default=1, ge=1, le=13)


class ConvergenceState(BaseModel):
    iteration: int = 0
    delta_i: float = Field(default=1.0, ge=0.0, le=1.0)
    delta_v: float = Field(default=1.0, ge=0.0, le=1.0)
    stable: bool = False
    threshold_i: float = Field(default=ATOMIC_UNCERTAINTY_THRESHOLD, ge=0.0, le=1.0)
    threshold_v: float = Field(default=VALUE_STABILITY_THRESHOLD, ge=0.0, le=1.0)


class Solution(BaseModel):
    product_summary: str
    tasks: List[Task] = Field(default_factory=list)
    roadmap: List[Task] = Field(default_factory=list)
    architecture: Dict[str, Any] = Field(default_factory=dict)
    open_questions: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    value_score: float = Field(default=0.0, ge=0.0, le=100.0)
    convergence: ConvergenceState = Field(default_factory=ConvergenceState)


class PMAgentOutput(BaseModel):
    problem_framing: str
    user_segments: List[str] = Field(default_factory=list)
    value_hypotheses: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    subproblems: List[SubProblem] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty_delta: float = Field(default=0.5, ge=0.0, le=1.0)


class ArchitectAgentOutput(BaseModel):
    architecture_summary: str
    components: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    implementation_paths: List[str] = Field(default_factory=list)
    technical_risks: List[str] = Field(default_factory=list)
    subproblems: List[SubProblem] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty_delta: float = Field(default=0.5, ge=0.0, le=1.0)


class UXAgentOutput(BaseModel):
    ux_summary: str
    user_flows: List[str] = Field(default_factory=list)
    screens: List[str] = Field(default_factory=list)
    accessibility_requirements: List[str] = Field(default_factory=list)
    usability_risks: List[str] = Field(default_factory=list)
    subproblems: List[SubProblem] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty_delta: float = Field(default=0.5, ge=0.0, le=1.0)


class ProjectStatusResponse(BaseModel):
    project_id: UUID
    status: ProjectStatus
    created_at: str
    updated_at: str
    depth: int
    iteration: int
    delta_i: float
    delta_v: float
    logs: List[ProjectLog]
    error: Optional[str] = None
    progress_message: Optional[str] = None
