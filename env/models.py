"""
Typed Pydantic models for the Cyber Threat Intelligence Triage Environment.
Defines Observation, Action, and Reward models per OpenEnv spec.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    source_ip: str = Field(..., description="Source IP address of the event")
    destination: str = Field(..., description="Destination host:port")
    event_type: str = Field(..., description="Type of security event")
    severity_raw: str = Field(..., description="Raw severity from SIEM: low|medium|high|critical")
    geo_country: str = Field(..., description="Country code of source IP")
    related_cves: List[str] = Field(default_factory=list, description="Associated CVE identifiers")
    payload_snippet: Optional[str] = Field(None, description="Truncated payload or log fragment")
    user_agent: Optional[str] = Field(None, description="HTTP user-agent if applicable")
    frequency: int = Field(1, description="How many times this pattern repeated in window")
    tags: List[str] = Field(default_factory=list, description="Analyst-applied tags from SIEM")


class ThreatIntel(BaseModel):
    known_bad_ips: List[str] = Field(default_factory=list)
    known_bad_domains: List[str] = Field(default_factory=list)
    active_campaigns: List[str] = Field(default_factory=list)
    organization_profile: str = Field("", description="Target org context e.g. 'fintech, handles PII'")
    recent_cve_advisories: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core OpenEnv Models
# ---------------------------------------------------------------------------

class CyberTriageObservation(BaseModel):
    """What the agent sees at each step."""
    task_id: str = Field(..., description="Active task identifier")
    step: int = Field(..., description="Current step number")
    alerts: List[Alert] = Field(..., description="Batch of security alerts to triage")
    threat_intel: ThreatIntel = Field(..., description="Current threat intelligence context")
    step_budget: int = Field(..., description="Remaining steps in this episode")
    previous_actions: List[str] = Field(default_factory=list, description="History of agent actions")
    instructions: str = Field("", description="Task-specific instructions for the agent")
    done: bool = Field(False, description="Whether the episode is complete")


class CyberTriageAction(BaseModel):
    """What the agent can do — triage one alert per step."""
    alert_id: str = Field(..., description="ID of the alert being triaged")
    classification: str = Field(
        ...,
        description="true_positive | false_positive | uncertain"
    )
    severity: str = Field(
        ...,
        description="Agent-assessed severity: low | medium | high | critical"
    )
    action: str = Field(
        ...,
        description=(
            "Recommended response: dismiss | monitor | block_ip | "
            "escalate_to_ir | isolate_host | patch_advisory"
        )
    )
    reasoning: Optional[str] = Field(None, description="Agent's reasoning (used for partial credit)")


class CyberTriageReward(BaseModel):
    """Reward breakdown for the step."""
    total: float = Field(..., description="Total reward for this step [-1.0, 1.0]")
    classification_score: float = Field(0.0, description="Score for true/false positive classification")
    severity_score: float = Field(0.0, description="Score for severity assessment")
    action_score: float = Field(0.0, description="Score for recommended response action")
    efficiency_bonus: float = Field(0.0, description="Bonus for speed/efficiency")
    penalty: float = Field(0.0, description="Penalty for harmful or wasteful actions")
    feedback: str = Field("", description="Human-readable feedback string")


class StepResult(BaseModel):
    """Full return value of env.step()"""
    observation: CyberTriageObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Return value of env.reset()"""
    observation: CyberTriageObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """Return value of env.state()"""
    task_id: str
    step: int
    total_alerts: int
    triaged_alerts: int
    cumulative_reward: float
    done: bool
    scores: Dict[str, float] = Field(default_factory=dict)
