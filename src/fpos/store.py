from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from .models import ProjectCreateRequest, ProjectLog, ProjectStatus, Solution


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


@dataclass
class ProjectRuntime:
    project_id: UUID
    request: ProjectCreateRequest
    api_key: str = field(repr=False)
    status: ProjectStatus = ProjectStatus.pending
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    depth: int = 0
    iteration: int = 0
    delta_i: float = 1.0
    delta_v: float = 1.0
    logs: list[ProjectLog] = field(default_factory=list)
    error: Optional[str] = None
    progress_message: Optional[str] = None
    final_solution: Optional[Solution] = None

    def snapshot(self):
        from .models import ProjectStatusResponse

        return ProjectStatusResponse(
            project_id=self.project_id,
            status=self.status,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            depth=self.depth,
            iteration=self.iteration,
            delta_i=self.delta_i,
            delta_v=self.delta_v,
            logs=self.logs,
            error=self.error,
            progress_message=self.progress_message,
        )


class ProjectStore:
    def __init__(self) -> None:
        self._items: dict[UUID, ProjectRuntime] = {}
        self._lock: Optional[asyncio.Lock] = None

    async def init(self) -> None:
        self._lock = asyncio.Lock()

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            raise RuntimeError("ProjectStore is not initialized.")
        return self._lock

    async def create(self, runtime: ProjectRuntime) -> None:
        async with self._ensure_lock():
            self._items[runtime.project_id] = runtime

    async def get(self, project_id: UUID) -> Optional[ProjectRuntime]:
        async with self._ensure_lock():
            return self._items.get(project_id)

    async def update(self, project_id: UUID, **kwargs) -> None:
        async with self._ensure_lock():
            item = self._items.get(project_id)
            if not item:
                return
            for key, value in kwargs.items():
                setattr(item, key, value)
            item.updated_at = utc_now()

    async def append_log(
        self,
        project_id: UUID,
        *,
        depth: int,
        iteration: int,
        delta_i: float,
        delta_v: float,
        message: str,
        agent: str | None = None,
        data: dict | None = None,
    ) -> None:
        async with self._ensure_lock():
            item = self._items.get(project_id)
            if not item:
                return
            item.logs.append(
                ProjectLog(
                    ts=utc_now_iso(),
                    depth=depth,
                    iteration=iteration,
                    delta_i=delta_i,
                    delta_v=delta_v,
                    message=message,
                    agent=agent,
                    data=data or {},
                )
            )
            item.depth = max(item.depth, depth)
            item.iteration = max(item.iteration, iteration)
            item.delta_i = delta_i
            item.delta_v = delta_v
            item.updated_at = utc_now()

    async def complete(self, project_id: UUID, solution: Solution) -> None:
        async with self._ensure_lock():
            item = self._items.get(project_id)
            if not item:
                return
            item.final_solution = solution
            item.status = ProjectStatus.completed
            item.delta_i = solution.convergence.delta_i
            item.delta_v = solution.convergence.delta_v
            item.updated_at = utc_now()
            item.progress_message = "Completed"

    async def fail(self, project_id: UUID, error: str) -> None:
        async with self._ensure_lock():
            item = self._items.get(project_id)
            if not item:
                return
            item.status = ProjectStatus.failed
            item.error = error
            item.updated_at = utc_now()
            item.progress_message = "Failed"

    async def set_running(self, project_id: UUID) -> None:
        await self.update(project_id, status=ProjectStatus.running, progress_message="Running")

    async def scrub_secret(self, project_id: UUID) -> None:
        async with self._ensure_lock():
            item = self._items.get(project_id)
            if item:
                item.api_key = ""
