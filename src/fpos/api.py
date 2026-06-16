from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from .models import ProjectCreateRequest, ProjectCreateResponse, ProjectStatus, ProjectStatusResponse
from .orchestrator import FPOSOrchestrator
from .store import ProjectRuntime, ProjectStore


app = FastAPI(
    title="FPOS API",
    version="1.0.0",
    description="Fractal Product Operating System SaaS backend with BYOK Gemini execution.",
)

store = ProjectStore()


@app.on_event("startup")
async def startup_event() -> None:
    await store.init()
    app.state.store = store


def get_store() -> ProjectStore:
    return app.state.store


async def _process_project(project_id: UUID, store: ProjectStore) -> None:
    runtime = await store.get(project_id)
    if runtime is None:
        return

    await store.set_running(project_id)
    await store.append_log(
        project_id,
        depth=0,
        iteration=0,
        delta_i=runtime.delta_i,
        delta_v=runtime.delta_v,
        message="Project accepted and background execution started",
        agent="system",
        data={"model": runtime.request.model or "gemini-3.5-flash"},
    )

    try:
        orchestrator = FPOSOrchestrator(
            api_key=runtime.api_key,
            model=runtime.request.model or "gemini-3.5-flash",
            emit_log=lambda **kwargs: store.append_log(project_id, **kwargs),
            project_id=str(project_id),
        )

        solution = await orchestrator.run(
            raw_idea=runtime.request.idea,
            context=runtime.request.context,
            constraints=runtime.request.constraints,
            stakeholders=runtime.request.stakeholders,
        )

        await store.complete(project_id, solution)
        await store.append_log(
            project_id,
            depth=solution.convergence.iteration,
            iteration=solution.convergence.iteration,
            delta_i=solution.convergence.delta_i,
            delta_v=solution.convergence.delta_v,
            message="Project completed",
            agent="system",
            data={"tasks": len(solution.tasks), "confidence": solution.confidence, "value_score": solution.value_score},
        )

    except Exception as exc:
        await store.fail(project_id, str(exc))
        await store.append_log(
            project_id,
            depth=runtime.depth,
            iteration=runtime.iteration,
            delta_i=runtime.delta_i,
            delta_v=runtime.delta_v,
            message="Project failed",
            agent="system",
            data={"error": str(exc)},
        )
    finally:
        await store.scrub_secret(project_id)


@app.post("/api/v1/projects", response_model=ProjectCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_project(
    payload: ProjectCreateRequest,
    background_tasks: BackgroundTasks,
    x_gemini_key: str = Header(..., alias="X-Gemini-Key"),
    project_store: ProjectStore = Depends(get_store),
) -> ProjectCreateResponse:
    project_id = uuid4()

    runtime = ProjectRuntime(
        project_id=project_id,
        request=payload,
        api_key=x_gemini_key,
        status=ProjectStatus.pending,
    )
    await project_store.create(runtime)

    background_tasks.add_task(_process_project, project_id, project_store)

    return ProjectCreateResponse(
        project_id=project_id,
        status=ProjectStatus.pending,
        message="Project accepted. Poll /status for convergence updates.",
    )


@app.get("/api/v1/projects/{project_id}/status", response_model=ProjectStatusResponse)
async def get_project_status(
    project_id: UUID,
    project_store: ProjectStore = Depends(get_store),
) -> ProjectStatusResponse:
    runtime = await project_store.get(project_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return runtime.snapshot()


@app.get("/api/v1/projects/{project_id}/roadmap")
async def get_project_roadmap(
    project_id: UUID,
    project_store: ProjectStore = Depends(get_store),
) -> JSONResponse:
    runtime = await project_store.get(project_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if runtime.status != ProjectStatus.completed or runtime.final_solution is None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Project is not completed yet.",
                "status": runtime.status,
            },
        )

    return JSONResponse(
        content={
            "project_id": str(project_id),
            "status": runtime.status,
            "result": runtime.final_solution.model_dump(mode="json"),
        }
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
