from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import fpos.api as api
from fpos.api import app
from fpos.models import (
    ArchitectAgentOutput,
    ConvergenceState,
    PMAgentOutput,
    ProjectCreateRequest,
    ProjectStatus,
    Solution,
    SubProblem,
    Task,
    UXAgentOutput,
)
from fpos.store import ProjectRuntime, ProjectStore


@pytest.fixture(autouse=True)
def reset_store():
    api.store = ProjectStore()
    app.state.store = api.store
    import asyncio
    asyncio.run(api.store.init())
    yield


def make_solution() -> Solution:
    task = Task(
        id="task_1",
        title="Create API endpoint",
        description="Implement the project creation endpoint.",
        owner="Backend",
        category="implementation",
        atomic=True,
        inputs=["idea", "context"],
        outputs=["endpoint"],
        dependencies=[],
        acceptance_criteria=["Returns project_id"],
        definition_of_ready=["Schema defined", "Owner clear"],
        risk_level="low",
        estimated_effort_points=3,
    )
    return Solution(
        product_summary="Stub solution",
        tasks=[task],
        roadmap=[task],
        architecture={"summary": "stub"},
        open_questions=[],
        assumptions=[],
        risks=[],
        confidence=0.95,
        value_score=88.0,
        convergence=ConvergenceState(iteration=2, delta_i=0.05, delta_v=0.02, stable=True),
    )


@pytest.mark.asyncio
async def test_project_store_lifecycle():
    store = ProjectStore()
    await store.init()

    project_id = uuid4()
    runtime = ProjectRuntime(
        project_id=project_id,
        request=ProjectCreateRequest(idea="Build FPOS", context="Internal tooling"),
        api_key="secret",
    )
    await store.create(runtime)
    fetched = await store.get(project_id)
    assert fetched is not None
    assert fetched.status == ProjectStatus.pending

    await store.set_running(project_id)
    fetched = await store.get(project_id)
    assert fetched.status == ProjectStatus.running

    await store.append_log(
        project_id,
        depth=1,
        iteration=1,
        delta_i=0.7,
        delta_v=0.9,
        message="PM working",
        agent="PM",
    )
    fetched = await store.get(project_id)
    assert len(fetched.logs) == 1

    sol = make_solution()
    await store.complete(project_id, sol)
    fetched = await store.get(project_id)
    assert fetched.status == ProjectStatus.completed
    assert fetched.final_solution is not None

    await store.scrub_secret(project_id)
    fetched = await store.get(project_id)
    assert fetched.api_key == ""


def test_create_project_endpoint_returns_project_id_with_mocked_bg_execution():
    with patch("fpos.api._process_project", new=AsyncMock()) as bg:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/projects",
            headers={"X-Gemini-Key": "test-key"},
            json={
                "idea": "Build an AI product planner",
                "context": "Startup with 3 engineers",
                "constraints": ["must support BYOK"],
                "stakeholders": ["PM", "Founder"],
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "project_id" in data
        UUID(data["project_id"])
        assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_background_processing_marks_completed_without_network():
    store = ProjectStore()
    await store.init()

    project_id = uuid4()
    runtime = ProjectRuntime(
        project_id=project_id,
        request=ProjectCreateRequest(
            idea="Build an FPOS SaaS backend",
            context="Cloud service",
            constraints=["polling status endpoint"],
            stakeholders=["PM", "Architect"],
        ),
        api_key="fake-key",
    )
    await store.create(runtime)

    fake_solution = make_solution()

    with patch("fpos.orchestrator.genai.Client") as mock_client:
        mock_client.return_value = MagicMock()
        with patch("fpos.api.FPOSOrchestrator.run", new=AsyncMock(return_value=fake_solution)):
            await api._process_project(project_id, store)

    fetched = await store.get(project_id)
    assert fetched is not None
    assert fetched.status == ProjectStatus.completed
    assert fetched.final_solution is not None
    assert fetched.final_solution.confidence == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_status_and_roadmap_endpoints_after_completion():
    client = TestClient(app)

    project_id = uuid4()
    runtime = ProjectRuntime(
        project_id=project_id,
        request=ProjectCreateRequest(
            idea="Build a roadmap engine",
            context="Product team",
            constraints=[],
            stakeholders=["PM"],
        ),
        api_key="secret",
    )
    await api.store.create(runtime)
    await api.store.complete(project_id, make_solution())

    status_resp = client.get(f"/api/v1/projects/{project_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "completed"

    roadmap_resp = client.get(f"/api/v1/projects/{project_id}/roadmap")
    assert roadmap_resp.status_code == 200
    payload = roadmap_resp.json()
    assert payload["project_id"] == str(project_id)
    assert payload["result"]["roadmap"][0]["owner"] == "Backend"
