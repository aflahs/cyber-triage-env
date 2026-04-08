"""
FastAPI application for the Cyber Triage Environment.

Exposes the OpenEnv HTTP API:
  POST /reset   → initial observation
  POST /step    → step result (observation, reward, done)
  GET  /state   → episode metadata
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cyber_triage_env.environment import CyberTriageEnvironment, TASKS
from cyber_triage_env.models import (
    CyberTriageAction,
    ResetResult,
    StateResult,
    StepResult,
)

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------
_env: Optional[CyberTriageEnvironment] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cyber Threat Intelligence Triage — OpenEnv",
    description=(
        "An RL environment simulating a Security Operations Center analyst "
        "triaging security alerts. Implements the full OpenEnv spec."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "name": "cyber-triage-env",
        "version": "1.0.0",
        "description": "Cyber Threat Intelligence Triage Environment for OpenEnv",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": "cyber-triage-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks_endpoint():
    return {
        "tasks": [
            {
                "task_id": tid,
                "description": cfg["description"],
                "max_steps": cfg["max_steps"],
                "difficulty": tid,
            }
            for tid, cfg in TASKS.items()
        ]
    }


@app.post("/reset", response_model=ResetResult)
def reset(request: Optional[ResetRequest] = None):
    global _env
    if request is None:
        request = ResetRequest()
    try:
        _env = CyberTriageEnvironment(task_id=request.task_id, seed=request.seed)
        result = _env.reset()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step", response_model=StepResult)
def step(action: CyberTriageAction):
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state", response_model=StateResult)
def state():
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    return _env.state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
