"""
FastAPI server for the Cyber Threat Intelligence Triage Environment.

Endpoints:
  POST /reset          → ResetResult
  POST /step           → StepResult
  GET  /state          → StateResult
  GET  /health         → {"status": "ok"}
  GET  /tasks          → list of available tasks
  GET  /openapi.json   → auto-generated OpenAPI schema
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import CyberTriageEnvironment, TASKS
from env.models import (
    CyberTriageAction,
    ResetResult,
    StateResult,
    StepResult,
)

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

# Global environment instance (single-session server)
_env: Optional[CyberTriageEnvironment] = None


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "cyber-triage-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
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
def reset(request: ResetRequest):
    global _env
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
